
import os
import importlib.resources
import warnings
from typing import Iterable

import torch
import accelerate
import tqdm.auto
from datasets_wlr import WLRZhuangEpisodeDataset
from curobo.types.robot import RobotConfig
from curobo.types.base import TensorDeviceType as _CuroboTensorDeviceType
from xvla_wlr.model_legacy import DATA_DOMAIN_ID, XVLA, XVLAProcessor, Trainer, get_peft_model, Action, Observation
from xvla_wlr_experiments.xvla_finetune_piper_v0.dataset import XVLAWLRZhuangEpisodeDataset, normalize_observation, XVLAChunkDataset
import xvla_wlr_experiments.xvla_finetune_piper_v0.assets


def main(
    # TODO example: "samples/2026-01-21_demo_clothes/episode_0/data.json"
    wlr_dataset_paths: Iterable[str],
    checkpoint_load_path: ... = "2toINF/X-VLA-SoftFold",
    checkpoint_save_step_interval: int = 100,
    checkpoint_save_path: ... = None,
    processor_checkpoint_load_path: ... = "2toINF/X-VLA-SoftFold",
    num_iterations: int = 1,
    num_timesteps_per_episode: int = 4,
    num_timesteps_per_action: int = 2,
    use_peft: bool = True,
    device: str | None = None,
    report_timestep_interval: int = 10,
):
    accelerator = accelerate.Accelerator(
        # log_with="tensorboard", 
        # project_dir=".xvla",
        # fsdp_plugin=accelerate.FullyShardedDataParallelPlugin(),
    )

    with (
        tqdm.auto.tqdm(total=1., leave=False) as pbar,
        importlib.resources.as_file(
            importlib.resources.files(xvla_wlr_experiments.xvla_finetune_piper_v0.assets) 
            / "piper-dualarm"
        ) as piper_dualarm_asset_path
    ):
        # TODO
        model = XVLA.from_pretrained(checkpoint_load_path)
        processor = XVLAProcessor.from_pretrained(processor_checkpoint_load_path, use_fast=True)

        if use_peft:
            model = get_peft_model(model)
        if device is not None:
            model = model.to(device=device)

        trainer = Trainer(model, processor, accelerator=accelerator)

        for i in range(num_iterations):
            for wlr_dataset_path in wlr_dataset_paths:
                pbar.set_description(f"Loading episode dataset: {wlr_dataset_path}")
                # TODO
                dataset = WLRZhuangEpisodeDataset(wlr_dataset_path)
                domain_id = DATA_DOMAIN_ID["AIR-AGILEX-HQ"]

                # TODO
                xvla_dataset = XVLAWLRZhuangEpisodeDataset(
                    dataset=dataset,
                    robot_config_left=RobotConfig.from_basic(
                        piper_dualarm_asset_path / "piper-dualarm.urdf",
                        base_link="common_base_link",
                        ee_link="left_link8",
                        # TODO NOTE this must be used due to a BUG in curobo
                        # TODO NOTE the curobo kernels assume all inputs 
                        # to be on the device of the current stream. cuda illegal
                        # mem access will occur when theres any mismatch.
                        tensor_args=_CuroboTensorDeviceType(device=torch.device(
                            "cuda",
                            index=torch.cuda.current_device(),
                        )),
                    ),
                    robot_config_right=RobotConfig.from_basic(
                        piper_dualarm_asset_path / "piper-dualarm.urdf",
                        base_link="common_base_link",
                        ee_link="right_link8",
                        # TODO NOTE this must be used due to a BUG in curobo
                        tensor_args=_CuroboTensorDeviceType(device=torch.device(
                            "cuda",
                            index=torch.cuda.current_device(),
                        )),
                    ),
                    domain_id=domain_id,
                    prefetch=True,
                )

                with (
                    tqdm.auto.tqdm(total=1., leave=False) as pbar_training,
                ):
                    timestep_current = 0

                    while True:
                        if timestep_current + num_timesteps_per_episode >= len(xvla_dataset):
                            break
                        observation = xvla_dataset[
                            timestep_current
                            :timestep_current + num_timesteps_per_episode
                        ]

                        action = Action.from_observation(
                            observation,
                            num_steps=num_timesteps_per_action,
                        )

                        action_next = action[1:]
                        observation_current = observation[:len(action_next)]

                        # observation_current_normalized = normalize_observation([
                        #     xvla_dataset._ik_solver_left.robot_config, 
                        #     xvla_dataset._ik_solver_right.robot_config
                        # ], observation=observation_current)
                        losses = trainer.fit(
                            # observation=observation_current_normalized,
                            observation=observation_current,
                            action=action_next,
                        )

                        timestep_current += len(observation_current)

                        if timestep_current % report_timestep_interval == 0:
                            pbar_training.update(timestep_current / len(xvla_dataset) - pbar_training.n)
                            pbar_training.set_description(
                                f"Training episode timestep: {timestep_current} of {len(xvla_dataset)}. "
                                f"Loss: {({name: x.item() for name, x in losses.items()})}"
                            )

                        if trainer._step_count % checkpoint_save_step_interval == 0:
                            path = None
                            match checkpoint_save_path:
                                case None:
                                    pass
                                case str():
                                    path = checkpoint_save_path
                                case path_gen if callable(checkpoint_save_path):
                                    path = path_gen(i)
                                case _:
                                    warnings.warn(f"Invalid checkpoint saving path: {checkpoint_save_path}. Skipping checkpointing.")
                                    continue
                            if path is not None:
                                if accelerator.is_main_process():
                                    accelerator.unwrap_model(trainer._model).save_pretrained(path)
                                    print(f"Checkpoint at iteration {i}: {path}")

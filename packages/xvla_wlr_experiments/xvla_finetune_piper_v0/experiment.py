


import os
import importlib.resources
import warnings

import torch
torch.set_float32_matmul_precision("high")
torch._dynamo.config.compiled_autograd = True
import tqdm.auto
from datasets_wlr import WLRZhuangEpisodeDataset
from curobo.types.robot import RobotConfig
from xvla_wlr.model import DATA_DOMAIN_ID, XVLA, XVLAProcessor, Trainer, get_peft_model, Action, Observation
from xvla_wlr_experiments.xvla_finetune_piper_v0.dataset import XVLAWLRZhuangEpisodeDataset
import xvla_wlr_experiments.xvla_finetune_piper_v0.assets


def main(
    # TODO example: "samples/2026-01-21_demo_clothes/episode_0/data.json"
    wlr_dataset_paths: list[str],
    checkpoint_load_path: ... = "2toINF/X-VLA-SoftFold",
    checkpoint_save_iteration_interval: int = 1,
    checkpoint_save_path: ... = None,
    processor_checkpoint_load_path: ... = "2toINF/X-VLA-SoftFold",
    num_iterations: int = 1,
    num_timesteps_per_episode: int = 4,
    num_timesteps_per_action: int = 2,
    use_peft: bool = True,
    use_torch_compile: bool = False,
    device: str = "cuda",
    logging_training_report_timestep_interval: int = 10,
):
    
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

        trainer = Trainer(model, processor)

        for i in range(num_iterations):
            for i_dataset, wlr_dataset_path in enumerate(wlr_dataset_paths):
                pbar.update((i_dataset + 1) / len(wlr_dataset_paths) - pbar.n)
                pbar.set_description(f"Loading episode dataset: {wlr_dataset_path}")
                # TODO
                dataset = WLRZhuangEpisodeDataset(wlr_dataset_path)
                domain_id = DATA_DOMAIN_ID["robomind-agilex"]

                # TODO
                xvla_dataset = XVLAWLRZhuangEpisodeDataset(
                    dataset=dataset,
                    robot_config_left=RobotConfig.from_basic(
                        piper_dualarm_asset_path / "piper-dualarm.urdf",
                        base_link="common_base_link",
                        ee_link="left_link8",
                    ),
                    robot_config_right=RobotConfig.from_basic(
                        piper_dualarm_asset_path / "piper-dualarm.urdf",
                        base_link="common_base_link",
                        ee_link="right_link8",
                    ),
                    domain_id=domain_id,
                    prefetch=True,
                )

                # TODO
                fit = torch.compile(trainer.fit, disable=not use_torch_compile)

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

                        loss = fit(
                            observation=observation_current,
                            action=action_next,
                        )

                        timestep_current += len(observation_current)

                        if timestep_current % logging_training_report_timestep_interval == 0:
                            pbar_training.update(timestep_current / len(xvla_dataset) - pbar_training.n)
                            pbar_training.set_description(f"Training episode timestep: {timestep_current} of {len(xvla_dataset)}. Loss: {loss}")

            if i % checkpoint_save_iteration_interval == 0:
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
                    model.save_pretrained(path)
                    print(f"Checkpoint at iteration {i}: {path}")

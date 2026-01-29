
import os
import dataclasses
import contextlib
import importlib.resources
import warnings
from typing import Iterable

import torch
import numpy
import accelerate
import tqdm.auto
from datasets_wlr import WLRZhuangEpisodeDataset
from curobo.types.robot import RobotConfig
from curobo.types.base import TensorDeviceType as _CuroboTensorDeviceType
from xvla_wlr.model import DATA_DOMAIN_ID, XVLA, XVLAProcessor, Trainer, get_peft_model, Action, Observation
from xvla_wlr_experiments.xvla_finetune_piper_v0.dataset import XVLAWLRZhuangEpisodeDataset, normalize_observation, XVLAChunk, XVLAChunkDataset
from xvla_wlr_experiments.xvla_finetune_piper_v0 import assets


def main(
    wlr_dataset_paths: Iterable[str] | None = None,
    checkpoint_load_path: ... = "2toINF/X-VLA-SoftFold",
    checkpoint_save_step_interval: int = 100,
    checkpoint_save_path: ... = None,
    processor_checkpoint_load_path: ... = "2toINF/X-VLA-SoftFold",
    processor_checkpoint_save_path: ... = None,
    num_iterations: int = 1,
    num_timesteps_per_episode: int = 4,
    num_timesteps_per_action: int = 2,
    use_peft: bool = True,
    device: str | None = None,
    report_step_interval: int = 10,
):
    accelerator = accelerate.Accelerator(
        # log_with="tensorboard", 
        # project_dir=".xvla",
        # fsdp_plugin=accelerate.FullyShardedDataParallelPlugin(),
    )

    with contextlib.ExitStack() as context_stack:
        pbar = context_stack.enter_context(tqdm.auto.tqdm(total=1., leave=False))
        piper_dualarm_asset_path = context_stack.enter_context(
            importlib.resources.as_file(
                importlib.resources.files(assets) 
                / "piper-dualarm"
            )
        )
        if wlr_dataset_paths is None:
            wlr_dataset_paths = [
                str(context_stack.enter_context(
                    importlib.resources.as_file(
                        importlib.resources.files(assets) 
                        / "wlr-dataset-sample"
                    )                
                ) / "data.json")
            ]

        # TODO
        model = XVLA.from_pretrained(checkpoint_load_path)
        processor = XVLAProcessor.from_pretrained(processor_checkpoint_load_path, use_fast=True)

        if use_peft:
            model = get_peft_model(model)
        if device is not None:
            model = model.to(device=device)

        trainer = Trainer(model, processor, accelerator=accelerator)

        for i_epoch in range(num_iterations):
            for wlr_dataset_path in wlr_dataset_paths:
                pbar.set_description(f"Loading episode dataset: {wlr_dataset_path}")

                xvla_dataset = XVLAWLRZhuangEpisodeDataset(
                    dataset=WLRZhuangEpisodeDataset(wlr_dataset_path),
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
                    domain_id=DATA_DOMAIN_ID["AIR-AGILEX-HQ"],
                    prefetch=True,
                )

                def xvla_dataset_chunk_collate(chunks: list[XVLAChunk]):
                    _concat = lambda arrays: (
                        torch.concatenate(arrays, dim=0)
                        if torch.is_tensor(arrays[0]) else
                        numpy.concatenate(arrays, axis=0)
                    )

                    observation = Observation(
                        text=_concat([chunk.observation.text for chunk in chunks]),
                        images=_concat([chunk.observation.images for chunk in chunks]),
                        images_mask=_concat([chunk.observation.images_mask for chunk in chunks]),
                        domain_id=_concat([chunk.observation.domain_id for chunk in chunks]),
                        ee_transform=_concat([chunk.observation.ee_transform for chunk in chunks]),
                        ee_gripper_val=_concat([chunk.observation.ee_gripper_val for chunk in chunks]),
                    )

                    action = Action(
                        ee_transforms=_concat([chunk.action.ee_transforms for chunk in chunks]),
                        ee_gripper_vals=_concat([chunk.action.ee_gripper_vals for chunk in chunks]),
                    )
                    
                    return XVLAChunk(observation=observation, action=action)

                # TODO
                xvla_dataset_chunk_loader = torch.utils.data.DataLoader(
                    XVLAChunkDataset(
                        xvla_dataset=xvla_dataset,
                        num_timesteps_per_episode=num_timesteps_per_episode,
                        num_timesteps_per_action=num_timesteps_per_action,
                    ),
                    collate_fn=xvla_dataset_chunk_collate,
                )
                xvla_dataset_chunk_loader = accelerator.prepare(xvla_dataset_chunk_loader)

                with (
                    tqdm.auto.tqdm(total=1., leave=False) as pbar_training,
                ):
                    for observation, action in xvla_dataset_chunk_loader:
                        losses = trainer.fit(
                            observation=observation,
                            action=action,
                        )

                        if trainer._step_count % report_step_interval == 0:
                            pbar_training.set_description(
                                f"Epoch: {trainer._step_count}. "
                                f"Loss: {({name: x.item() for name, x in losses.items()})}"
                            )

                        if trainer._step_count % checkpoint_save_step_interval == 0:
                            if accelerator.is_main_process:
                                def expand_checkpoint_path(path_or_path_gen):
                                    match path_or_path_gen:
                                        case None:
                                            return None
                                        case str() as path:
                                            return path
                                        case path_gen if callable(path_or_path_gen):
                                            return path_gen(trainer)
                                        case _:
                                            warnings.warn(f"Invalid checkpoint saving path: {checkpoint_save_path}")
                                            return None
                                checkpoint_save_path_ = expand_checkpoint_path(checkpoint_save_path)
                                if checkpoint_save_path_ is not None:
                                    accelerator.unwrap_model(trainer._model).save_pretrained(checkpoint_save_path_)
                                    pbar_training.set_description(f"Checkpoint at epoch {trainer._step_count}: {checkpoint_save_path_}")
                                processor_checkpoint_save_path_ = expand_checkpoint_path(processor_checkpoint_save_path)
                                if processor_checkpoint_save_path_ is not None:
                                    processor.save_pretrained(processor_checkpoint_save_path_)
                                    pbar_training.set_description(f"Processor checkpoint at epoch {trainer._step_count}: {processor_checkpoint_save_path_}")


if __name__ == "__main__":
    main()
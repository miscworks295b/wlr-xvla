
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
from xvla_wlr.agent import XVLAAgent, XVLAAction, XVLAObservation, XVLA_DOMAIN_IDS
from xvla_wlr_experiments.xvla_finetune_piper_v0.dataset import XVLAWLRZhuangEpisodeDataset, normalize_observation, XVLAChunk, XVLAChunkDataset
from xvla_wlr_experiments.xvla_finetune_piper_v0 import assets


def main(
    wlr_dataset_paths: Iterable[str] | None = None,
    checkpoint_source: ... = XVLAAgent.Config.sample(),
    checkpoint_save_step_interval: int = 100,
    checkpoint_save_target: ... = None,
    num_iterations: int = 1,
    num_iterations_per_episode: int = 10,
    num_timesteps_per_episode: int = 4,
    num_timesteps_per_action: int = 2,
    report_step_interval: int = 10,
):
    accelerator = accelerate.Accelerator(
        # TODO NOTE accelerate does not support custom collate!!!
        # dataloader_config=accelerate.utils.DataLoaderConfiguration(
        #     dispatch_batches=True,  
        #     split_batches=False,
        #     even_batches=False,
        # ),
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
        agent = XVLAAgent(checkpoint_source, accelerator=accelerator)

        for _ in range(num_iterations):
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
                    domain_id=XVLA_DOMAIN_IDS["AIR-AGILEX-HQ"],
                    prefetch=True,
                )

                def xvla_dataset_chunk_collate(chunks: list[XVLAChunk]):
                    _concat = lambda arrays: (
                        torch.concatenate(arrays, dim=0)
                        if torch.is_tensor(arrays[0]) else
                        numpy.concatenate(arrays, axis=0)
                    )

                    observation = XVLAObservation(
                        text=_concat([chunk.observation.text for chunk in chunks]),
                        images=_concat([chunk.observation.images for chunk in chunks]),
                        images_mask=_concat([chunk.observation.images_mask for chunk in chunks]),
                        domain_id=_concat([chunk.observation.domain_id for chunk in chunks]),
                        ee_transform=_concat([chunk.observation.ee_transform for chunk in chunks]),
                        ee_gripper_val=_concat([chunk.observation.ee_gripper_val for chunk in chunks]),
                    )

                    action = XVLAAction(
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

                chunks = list(xvla_dataset_chunk_loader)
                for _ in range(num_iterations_per_episode):
                    for observation, action in chunks:
                        # TODO
                        epoch, losses = agent.learn(
                            observation=observation,
                            action=action,
                        )

                        if epoch % report_step_interval == 0:
                            pbar.set_description(
                                f"Epoch: {epoch}. "
                                f"Loss: {({name: x.item() for name, x in losses.items()})}"
                            )

                        if epoch % checkpoint_save_step_interval == 0:
                            if accelerator.is_main_process:
                                def resolve_checkpoint_path(path_or_path_gen):
                                    match path_or_path_gen:
                                        case None:
                                            return None
                                        case str() as path:
                                            return path
                                        case path_gen if callable(path_or_path_gen):
                                            return path_gen(agent)
                                        case _:
                                            warnings.warn(f"Invalid checkpoint saving path: {checkpoint_save_target}")
                                            return None
                                checkpoint_save_path_ = resolve_checkpoint_path(checkpoint_save_target)
                                if checkpoint_save_path_ is not None:
                                    agent.save(checkpoint_save_path_, force=True)
                                    pbar.set_description(f"Checkpoint at epoch {epoch}: {checkpoint_save_path_}")

if __name__ == "__main__":
    main()
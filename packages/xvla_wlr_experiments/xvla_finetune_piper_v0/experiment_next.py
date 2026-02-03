
import asyncio
import threading
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
from xvla_wlr_experiments.xvla_finetune_piper_v0.dataset import XVLAWLRZhuangEpisodeDataset, normalize_observation, XVLAChunk, XVLAChunkDataset, xvla_dataset_chunk_collate
from xvla_wlr_experiments.xvla_finetune_piper_v0 import assets


async def load_datasets(
    paths: Iterable[str] | None = None,
):
    with contextlib.ExitStack() as context_stack:
        if paths is None:
            paths = [
                str(context_stack.enter_context(
                    importlib.resources.as_file(
                        importlib.resources.files(assets) 
                        / "wlr-dataset-sample"
                    )                
                ) / "data.json")
            ]

        piper_dualarm_asset_path = context_stack.enter_context(
            importlib.resources.as_file(
                importlib.resources.files(assets) 
                / "piper-dualarm"
            )
        )

        def _impl(path: str):
            return XVLAWLRZhuangEpisodeDataset(
                dataset=WLRZhuangEpisodeDataset(path),
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

        for path in paths:
            yield (await asyncio.to_thread(_impl, path), path)


async def main(
    wlr_dataset_paths: Iterable[str] | None = None,
    checkpoint_source: ... = XVLAAgent.Config.sample(),
    checkpoint_save_step_interval: int | None = 100,
    checkpoint_save_target: ... = None,
    num_iterations: int = 1,
    num_iterations_per_episode: int = 10,
    num_timesteps_per_episode: int = 4,
    num_timesteps_per_action: int = 2,
    report_step_interval: int | None = 10,
    accelerator: accelerate.Accelerator | None = None,
):
    _lock = threading.Lock()

    if accelerator is None:
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

        # TODO
        agent = XVLAAgent(checkpoint_source, accelerator=accelerator)

        for _ in range(num_iterations):
            async for dataset, path in load_datasets(wlr_dataset_paths):
                pbar.set_description(f"Using episode dataset: {path}")

                # TODO
                xvla_dataset_chunk_loader = torch.utils.data.DataLoader(
                    XVLAChunkDataset(
                        xvla_dataset=dataset,
                        num_timesteps_per_episode=num_timesteps_per_episode,
                        num_timesteps_per_action=num_timesteps_per_action,
                    ),
                    collate_fn=xvla_dataset_chunk_collate,
                )
                xvla_dataset_chunk_loader = accelerator.prepare(xvla_dataset_chunk_loader)

                chunks = list(xvla_dataset_chunk_loader)
                for _ in range(num_iterations_per_episode):
                    for observation, action in chunks:
                        def _learn_threadsafe():
                            with _lock:
                                return agent.learn(
                                    observation=observation,
                                    action=action,
                                )
                        epoch, losses = await asyncio.to_thread(_learn_threadsafe)

                        if report_step_interval is not None:
                            if epoch % report_step_interval == 0:
                                pbar.set_description(
                                    f"Epoch: {epoch}. "
                                    f"Loss: {({name: x.item() for name, x in losses.items()})}"
                                )

                        if checkpoint_save_step_interval is not None:
                            if epoch % checkpoint_save_step_interval == 0:
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
    asyncio.run(main())
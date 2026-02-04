import glob
import asyncio
import os
import argparse

import xvla_wlr_experiments.xvla_finetune_piper_v0.experiment as _experiment


current_checkpoint_path = f"{os.path.dirname(__file__)}/checkpoints/current/checkpoint.json"

checkpoint_source = _experiment.XVLAAgent.Config(
    schema="xvla-config:v0",
    model={
        "pretrained_model_name_or_path": "2toINF/X-VLA-SoftFold"
    },
    processor={
        "pretrained_model_name_or_path": "2toINF/X-VLA-SoftFold",
        "use_fast": True,
    },
    adapter=True,
    accelerator=True,
)
if os.path.exists(current_checkpoint_path):
    checkpoint_source = current_checkpoint_path


async def train():
    await _experiment.train(
        glob.glob("/liujinxin/dataset/piper/cloth_new/**/data.json", recursive=True), 
        num_iterations=2,
        num_iterations_per_episode=4,
        num_timesteps_per_episode=32,
        num_timesteps_per_action=4,
        checkpoint_source=checkpoint_source,
        checkpoint_save_target=current_checkpoint_path,
    )


async def evaluate():
    await _experiment.evaluate(
        glob.glob("/liujinxin/dataset/piper/cloth_new/**/data.json", recursive=True), 
        num_timesteps_per_episode=32,
        num_timesteps_per_action=4,
        checkpoint_source=checkpoint_source,
    )


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "command", 
        nargs="?",
        choices=["train", "evaluate"], 
        default="train",
    )
    args = argparser.parse_args()
    match args.command:
        case "train":
            asyncio.run(train())
        case "evaluate":
            asyncio.run(evaluate())

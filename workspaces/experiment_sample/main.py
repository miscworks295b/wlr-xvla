import asyncio
import contextlib
import os

import torch
import accelerate
import xvla_wlr_experiments.xvla_finetune_piper_v0.experiment as _experiment


async def main():
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

    accelerator = accelerate.Accelerator()

    # with accelerator.profile(accelerate.ProfileKwargs(activities=["cpu", "cuda"])) as prof:
    # with torch.profiler.profile(
    #     activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
    # ) as prof:
    if True:
        await _experiment.train(
            num_iterations=128,
            num_iterations_per_episode=32,
            checkpoint_source=checkpoint_source,
            # checkpoint_save_target=current_checkpoint_path,
            checkpoint_save_step_interval=10,
            # report_step_interval=10,
            # report_step_interval=None,
            # checkpoint_save_step_interval=None,
            accelerator=accelerator,
        )
    # print(prof.key_averages().table(sort_by="self_cuda_time_total"))
    # prof.export_chrome_trace(f"{os.path.dirname(__file__)}/logs/trace-{accelerator.process_index}.json")

    # TODO
    # import tracemalloc
    # snapshot = tracemalloc.take_snapshot()
    # top_stats = snapshot.statistics('lineno', cumulative=True)
    # for stat in top_stats[:10]:
    #     print(stat)

if __name__ == "__main__":
    asyncio.run(main())
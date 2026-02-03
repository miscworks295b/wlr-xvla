import os

import accelerate
from xvla_wlr_experiments.xvla_finetune_piper_v0.experiment import main, XVLAAgent


current_checkpoint_path = f"{os.path.dirname(__file__)}/checkpoints/current/checkpoint.json"

checkpoint_source = XVLAAgent.Config(
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

with accelerator.profile(accelerate.ProfileKwargs(activities=["cpu", "cuda"])) as prof:
    main(
        num_iterations=1,
        num_iterations_per_episode=2,
        checkpoint_source=checkpoint_source,
        checkpoint_save_target=current_checkpoint_path,
        # checkpoint_save_step_interval=10,
        # report_step_interval=10,
        report_step_interval=None,
        checkpoint_save_step_interval=None,
        accelerator=accelerator,
    )

print(prof.key_averages().table(sort_by="flops", row_limit=10))
prof.export_chrome_trace(f"{os.path.dirname(__file__)}/logs/trace.json")
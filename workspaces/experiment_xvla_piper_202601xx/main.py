import os
import glob

from xvla_wlr_experiments.xvla_finetune_piper_v0.experiment import main, XVLAAgent


current_checkpoint_path = f"{os.path.dirname(__file__)}/checkpoints/current/checkpoint.json"

checkpoint_source = XVLAAgent.Config(
    schema="xvla-config:v0",
    model={
        "pretrained_model_name_or_path": "2toINF/X-VLA-SoftFold"
    },
    processor={
        "pretrained_model_name_or_path": "2toINF/X-VLA-SoftFold"
    },
    adapter=True,
    accelerator=True,
)
if os.path.exists(current_checkpoint_path):
    checkpoint_source = current_checkpoint_path

main(
    glob.glob("/liujinxin/dataset/piper/cloth_new/**/data.json", recursive=True), 
    num_iterations=10,
    num_timesteps_per_episode=32,
    num_timesteps_per_action=4,
    checkpoint_source=checkpoint_source,
    checkpoint_save_target=current_checkpoint_path,
)
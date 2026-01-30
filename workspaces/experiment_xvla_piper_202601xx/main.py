import os
import glob

from xvla_wlr_experiments.xvla_finetune_piper_v0.experiment import main


current_checkpoint_path = f"{os.path.dirname(__file__)}/checkpoints/current"

checkpoint_load_path = "2toINF/X-VLA-SoftFold"
if os.path.exists(current_checkpoint_path):
    checkpoint_load_path = current_checkpoint_path

main(
    glob.glob("/liujinxin/dataset/piper/cloth_new/**/data.json", recursive=True), 
    num_iterations=10,
    num_timesteps_per_episode=32,
    num_timesteps_per_action=4,
    checkpoint_load_path=checkpoint_load_path,
    checkpoint_save_path=current_checkpoint_path,
    processor_checkpoint_load_path=checkpoint_load_path,
    processor_checkpoint_save_path=current_checkpoint_path,
)
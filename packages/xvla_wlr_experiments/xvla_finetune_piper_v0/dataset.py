import dataclasses
from typing import Annotated, NamedTuple

import torch
import torch.linalg
import numpy
import einops
import torchvision.transforms.functional
from curobo.types.robot import RobotConfig
from curobo.types.base import TensorDeviceType as _CuroboTensorDeviceType
from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig
from xvla_wlr.model import DATA_DOMAIN_ID, Observation, Action
from datasets_wlr import WLRZhuangEpisodeDataset


def compute_conservative_reach_radius(robot: RobotConfig):
    # TODO filter links
    link_transforms_positions = robot.kinematics.kinematics_config.fixed_transforms[..., :3, 3]
    return torch.sum(torch.linalg.norm(link_transforms_positions, dim=-1))


def normalize_observation(robots: list[RobotConfig], observation: Observation):
    ee_transform = observation.ee_transform.clone()
    *_, num_ees, _, _ = ee_transform.shape
    for i_ee in range(num_ees):
        translation_scale = compute_conservative_reach_radius(robots[i_ee])
        ee_transform[..., i_ee, :3, 3] /= translation_scale

    return dataclasses.replace(
        observation,
        ee_transform=ee_transform,
    )


class XVLAWLRZhuangEpisodeDataset(torch.utils.data.Dataset):
    r"""
    X-VLA adapter dataset for :class:`WLRZhuangEpisodeDataset`.
    """

    def __init__(
        self,
        dataset: WLRZhuangEpisodeDataset,
        robot_config_left: RobotConfig,
        robot_config_right: RobotConfig,
        domain_id: Annotated[int, DATA_DOMAIN_ID],
        prefetch: bool = False,
        device: torch.device | None = None,
    ):
        self._dataset = dataset
        if prefetch:
            self._dataset = self._dataset[:]
        if device is None:
            device = torch.device(
                "cuda",
                index=torch.cuda.current_device(),
            )
        self._ik_solver_left = IKSolver(
            IKSolverConfig.load_from_robot_config(
                robot_cfg=robot_config_left,
                tensor_args=_CuroboTensorDeviceType(device=device),
            ),
        )
        self._ik_solver_right = IKSolver(
            IKSolverConfig.load_from_robot_config(
                robot_cfg=robot_config_right,
                tensor_args=_CuroboTensorDeviceType(device=device),
            )
        )
        # TODO
        self._domain_id = domain_id

    def __len__(self):
        return len(self._dataset)
    
    def __getitem__(self, index):
        data = self._dataset[index]

        # TODO aug should use rand choice instead of index 0
        text = numpy.asarray(data.text[..., 0])

        images = einops.rearrange(
            [
                torchvision.transforms.functional.resize(torch.asarray(image), [224, 224])
                for image in [
                    data.image_left,
                    data.image_front,
                    data.image_right,
                ]
            ],
            "camera batch channel height width -> batch camera channel height width",
        )
        images_mask = einops.repeat(
            torch.asarray(True),
            "-> batch camera",
            **einops.parse_shape(images, "batch camera _ _ _"),
        )

        domain_id = einops.repeat(
            torch.asarray(self._domain_id),
            "-> batch",
            **einops.parse_shape(text, "batch"),
        )

        # TODO use dof_names for indexing instead of left/right !!!
        dof_positions_left = torch.asarray(data.dof_positions[..., 0:7])
        dof_positions_right = torch.asarray(data.dof_positions[..., 7:14])

        ee_transforms = [
            solver.fk(
                torch.asarray(
                    d[..., 0:6], 
                    dtype=solver.tensor_args.dtype,
                    device=solver.tensor_args.device,
                )
            )
            .ee_pose.get_matrix()
            for solver, d in [
                (self._ik_solver_left, dof_positions_left), 
                (self._ik_solver_right, dof_positions_right),
            ]
        ]
        ee_transforms = einops.rearrange(
            ee_transforms,
            "ee batch a b -> batch ee a b",
        )

        ee_gripper_vals = einops.rearrange(
            [
                d[..., 6]
                for d in [dof_positions_left, dof_positions_right]
            ],
            "ee batch -> batch ee",
        )

        observation = Observation(
            text=text,
            images=images,
            images_mask=images_mask,
            domain_id=domain_id,
            ee_transform=ee_transforms,
            ee_gripper_val=ee_gripper_vals,
        )

        return observation
    

class XVLAChunk(NamedTuple):
    observation: Observation
    action: Action


class XVLAChunkDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        xvla_dataset: torch.utils.data.Dataset[Observation],
        *,
        num_timesteps_per_episode: int,
        num_timesteps_per_action: int,
        stride: int | None = None,
    ):
        self.dataset = xvla_dataset
        self.num_timesteps_per_episode = int(num_timesteps_per_episode)
        self.num_timesteps_per_action = int(num_timesteps_per_action)

        self.stride = int(
            max(1, self.num_timesteps_per_episode - self.num_timesteps_per_action) 
            if stride is None else 
            stride
        )

    def __len__(self) -> int:
        max_start = len(self.dataset) - self.num_timesteps_per_episode
        if max_start < 0:
            return 0
        return (max_start // self.stride) + 1

    def __getitem__(self, idx: int):
        if idx < 0 or idx >= len(self):
            raise IndexError(idx)

        start = idx * self.stride
        stop = start + self.num_timesteps_per_episode

        observation = self.dataset[start:stop]

        action = Action.from_observation(
            observation,
            num_steps=self.num_timesteps_per_action,
        )

        action_next = action[1:]
        observation_current = observation[:len(action_next)]

        return XVLAChunk(
            observation=observation_current,
            action=action_next,
        )


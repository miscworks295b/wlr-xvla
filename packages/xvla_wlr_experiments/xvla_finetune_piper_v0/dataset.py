from typing import Annotated

import torch
import numpy
import einops
import torchvision.transforms.functional
from curobo.types.robot import RobotConfig
from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig
from xvla_wlr.model import DATA_DOMAIN_ID, Observation, Action
from datasets_wlr import WLRZhuangEpisodeDataset


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
    ):
        self._dataset = dataset
        if prefetch:
            self._dataset = self._dataset[:]
        self._ik_solver_left = IKSolver(
            IKSolverConfig.load_from_robot_config(
                robot_cfg=robot_config_left,
            )
        )
        self._ik_solver_right = IKSolver(
            IKSolverConfig.load_from_robot_config(
                robot_cfg=robot_config_right,
            )
        )
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
                    dtype=torch.float,
                    device="cuda",
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
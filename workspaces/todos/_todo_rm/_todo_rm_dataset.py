from typing import Annotated

import torch
import numpy
import einops
import torchvision.transforms.functional
from curobo.types.robot import RobotConfig
from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig
from xvla_wlr.model_legacy import DATA_DOMAIN_ID, Observation, Action
from datasets_wlr import WLRZhuangEpisodeDataset, WLRZhuangEpisodeData


class XVLAWLRZhuangEpisodeDataset(torch.utils.data.Dataset):
    r"""
    X-VLA adapter dataset for :class:`WLRZhuangEpisodeDataset`.
    """

    def __init__(
        self,
        dataset: WLRZhuangEpisodeDataset,
        robot_config: RobotConfig,
        domain_id: Annotated[int, DATA_DOMAIN_ID],
        num_action_steps: int | None = None,
        num_action_substeps: int = 1,
    ):
        self._dataset = dataset
        self._ik_solver = IKSolver(
            IKSolverConfig.load_from_robot_config(
                robot_cfg=robot_config,
                # position_threshold=0.005,
                self_collision_check=False,
                self_collision_opt=False,
                # regularization=False,
                use_cuda_graph=True,
            )
        )
        self._domain_id = domain_id
        self._num_action_steps = num_action_steps
        self._num_action_substeps = num_action_substeps

    def _compute_ee_transforms_and_gripper_vals(self, data: WLRZhuangEpisodeData):
        # TODO use dof_names for indexing
        dof_positions_left = torch.asarray(data.dof_positions[..., 0:7])
        dof_positions_right = torch.asarray(data.dof_positions[..., 7:14])

        ee_transforms = [
            self._ik_solver.fk(
                torch.asarray(
                    d[..., 0:6], 
                    dtype=torch.float,
                    device="cuda",
                )
            )
            .ee_pose.get_matrix()
            for d in [dof_positions_left, dof_positions_right]
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

        return ee_transforms, ee_gripper_vals
    
    def _compute_observation(self, data: WLRZhuangEpisodeData):
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

        ee_transforms, ee_gripper_vals = (
            self._compute_ee_transforms_and_gripper_vals(data)
        )

        domain_id = einops.repeat(
            torch.asarray(self._domain_id),
            "-> batch",
            **einops.parse_shape(text, "batch"),
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
    
    def _compute_action(self, data: WLRZhuangEpisodeData, num_steps: int, num_substeps: int = 1):
        ee_transforms, ee_gripper_vals = (
            self._compute_ee_transforms_and_gripper_vals(data)
        )

        ee_transforms = einops.rearrange(
            torch.asarray(ee_transforms)
            .unfold(0, size=num_steps, step=num_substeps),
            "batch ee a b time -> batch time ee a b",
        )
        ee_gripper_vals = einops.rearrange(
            torch.asarray(ee_gripper_vals)
            .unfold(0, size=num_steps, step=num_substeps),
            "batch ee time -> batch time ee",
        )        

        action = Action(
            ee_transforms=ee_transforms,
            ee_gripper_vals=ee_gripper_vals,
        )

        return action

    def __len__(self):
        length = len(self._dataset)
        if self._num_action_steps is not None:
            return (length - self._num_action_steps) // self._num_action_substeps
        return length
    
    def __getitem__(self, index):
        data = self._dataset[index]

        observation = self._compute_observation(data)
        action = None

        if self._num_action_steps is not None:
            index_next = ...
            match index:
                case slice():
                    index = index
                    _start, _stop, _step = index.indices(len(self._dataset))
                    index_next = slice(
                        _start + self._num_action_substeps,
                        # NOTE include one future substep
                        _stop + self._num_action_substeps * 2,
                        _step * self._num_action_substeps,
                    )
                case int():
                    index = index
                    # NOTE include one future substep
                    index_next = slice(
                        index + self._num_action_substeps,
                        # NOTE include one future substep
                        index + 1 + self._num_action_substeps * 2,
                        self._num_action_substeps,
                    )
                case _:                    
                    raise NotImplementedError(f"Unsupported indexing for actions: {index}")

            # TODO do not refetch
            data_next = self._dataset[index_next]

            action = self._compute_action(
                data_next,
                num_steps=self._num_action_steps,
                num_substeps=self._num_action_substeps,
            )
        
        return observation, action

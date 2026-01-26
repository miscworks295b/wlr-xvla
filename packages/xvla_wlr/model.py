import dataclasses
from typing import Annotated
from types import SimpleNamespace

import torch
import einops
import peft
import accelerate

from .xvla.models.modeling_xvla import XVLA
from .xvla.models.processing_xvla import XVLAProcessor
from .xvla.train import update_group_lrs, build_optimizer
from .xvla.datasets.domain_config import DATA_WEIGHTS, DATA_DOMAIN_ID


@dataclasses.dataclass(slots=True)
class Observation:
    @classmethod
    def sample(cls):
        return Observation(
            text=["do something"],
            images=torch.full((1, 3, 3, 224, 224), fill_value=0.),
            images_mask=torch.full((1, 3), fill_value=True),
            domain_id=torch.full((1,), fill_value=DATA_DOMAIN_ID["lift2"]),
            # TODO
            ee_transform=torch.full((1, 2, 4, 4), fill_value=1.),
            ee_gripper_val=torch.full((1, 2), fill_value=0.),
        )

    text: Annotated[str, "*batch"]
    images: Annotated[torch.FloatTensor, "*batch camera:3 channel:3 height width", ">=0.,<=1."]
    images_mask: Annotated[torch.BoolTensor, "*batch camera:3"]
    domain_id: Annotated[torch.LongTensor, "*batch"]
    # proprio
    ee_transform: Annotated[torch.FloatTensor, "*batch ee:2 4 4"]
    ee_gripper_val: Annotated[torch.FloatTensor, "*batch ee:2", ">=0.,<=1."]

    def __len__(self):
        # TODO
        return len(self.ee_transform)
    
    def __getitem__(self, index):
        return Observation(
            text=self.text[index],
            images=self.images[index],
            images_mask=self.images_mask[index],
            domain_id=self.domain_id[index],
            ee_transform=self.ee_transform[index],
            ee_gripper_val=self.ee_gripper_val[index],
        )


@dataclasses.dataclass(slots=True)
class Action:
    @classmethod
    def sample(cls):
        # TODO use torch.eye(4, 4) for transforms
        return Action(
            # TODO
            ee_transforms=torch.full((1, 30, 2, 4, 4), fill_value=1.),
            ee_gripper_vals=torch.full((1, 30, 2), fill_value=0.),
        )
    
    @classmethod
    def from_observation(
        cls, 
        observation: Observation,
        num_steps: int = 30,
        num_substeps: int = 1,
    ):
        ee_transforms = einops.rearrange(
            torch.asarray(observation.ee_transform)
            .unfold(0, size=num_steps, step=num_substeps),
            "batch ee a b time -> batch time ee a b",
        )
        ee_gripper_vals = einops.rearrange(
            torch.asarray(observation.ee_gripper_val)
            .unfold(0, size=num_steps, step=num_substeps),
            "batch ee time -> batch time ee",
        )
        return cls(
            ee_transforms=ee_transforms,
            ee_gripper_vals=ee_gripper_vals,
        )

    # proprio
    ee_transforms: Annotated[torch.FloatTensor, "*batch time ee:2 4 4"]
    ee_gripper_vals: Annotated[torch.FloatTensor, "*batch time ee:2", ">=0.,<=1."]

    def __len__(self):
        # TODO
        return len(self.ee_transforms)
    
    def __getitem__(self, index):
        return Action(
            ee_transforms=self.ee_transforms[index],
            ee_gripper_vals=self.ee_gripper_vals[index],
        )


def _encode_ee6d(
    ee_transform: Annotated[torch.FloatTensor, "*n 4 4"], 
    ee_gripper_val: Annotated[torch.FloatTensor, "*n"],
) -> Annotated[torch.FloatTensor, "*n buffer:10"]:
    r"""
    TODO doc

    :param ee_transform: End-effector 3D transformation matrix in column-vector convention.
    :param ee_gripper_val: End-effector gripper setting.
    :return: TODO
    """

    *batch_size_ee_transform, _a, _b = ee_transform.shape
    assert _a == 4 and _b == 4
    *batch_size_gripper, = ee_gripper_val.shape
    batch_size = torch.broadcast_shapes(batch_size_ee_transform, batch_size_gripper)
    # TODO
    a = torch.empty((*batch_size, 10))
    a[..., 0:3] = ee_transform[..., :3, [3]].reshape(*batch_size, -1)
    a[..., 3:9] = ee_transform[..., :3, [0, 1]].reshape(*batch_size, -1)
    a[..., 9:10] = ee_gripper_val.reshape(*batch_size, -1)
    return a


# TODO
def _decode_ee6d(a: Annotated[torch.FloatTensor, "*n buffer:10"]):
    r"""
    TODO doc
    """

    # TODO complete xform matrix
    a[..., 0:3]
    a[..., 3:9]

    ee_transform = ...
    gripper_val = a[..., 9:10]

    return ee_transform, gripper_val


def get_peft_model(model: XVLA):
    lora_config = peft.LoraConfig(
        lora_alpha=16,
        r=8,
        bias="none",
        target_modules="all-linear",
        modules_to_save=[
            "transformer.soft_prompt_hub",
            "transformer.action_encoder",
            "transformer.action_decoder",
        ],
    )
    model = peft.get_peft_model(model, lora_config)
    return model


def compute_losses(
    model: XVLA, 
    processor: XVLAProcessor,
    observation: Observation, 
    action: Action,
):
    #
    _model_input_ids = processor.encode_language(observation.text)["input_ids"]
    # TODO rm
    # _model_image_input = einops.rearrange(
    #     observation.images, 
    #     "batch camera height width channel -> batch camera channel height width",
    # )
    _model_image_input = processor.process_image(observation.images)
    _model_image_mask = observation.images_mask
    _model_domain_id = observation.domain_id
    _model_proprio = _encode_ee6d(
        ee_transform=observation.ee_transform,
        ee_gripper_val=observation.ee_gripper_val,
    )
    _model_proprio = einops.rearrange(
        _model_proprio,
        "batch ee buffer -> batch (ee buffer)",
        ee=2, 
        buffer=10,
    )

    #
    _model_action = _encode_ee6d(
        ee_transform=action.ee_transforms,
        ee_gripper_val=action.ee_gripper_vals,
    )
    _model_action = einops.rearrange(
        _model_action,
        "batch time ee buffer -> batch time (ee buffer)",
        ee=2, 
        buffer=10,
    )

    losses = model.forward(
        input_ids=_model_input_ids.to(model.device, non_blocking=True),
        image_input=_model_image_input.to(model.device, non_blocking=True),
        image_mask=_model_image_mask.to(model.device, non_blocking=True),
        domain_id=_model_domain_id.to(model.device, non_blocking=True),
        proprio=_model_proprio.to(model.device, non_blocking=True),
        action=_model_action.to(model.device, non_blocking=True),
    )

    return losses


# TODO
def compute_actions(
    model: XVLA, 
    processor: XVLAProcessor,
    observation: Observation, 
    num_denoising_steps: int = 10,
) -> Action:
    #
    _model_input_ids = processor.encode_language(observation.text)["input_ids"]
    # TODO rm
    # _model_image_input = einops.rearrange(
    #     observation.images, 
    #     "batch camera height width channel -> batch camera channel height width",
    # )
    _model_image_input = processor.process_image(observation.images)
    _model_image_mask = observation.images_mask
    _model_domain_id = observation.domain_id
    _model_proprio = _encode_ee6d(
        ee_transform=observation.ee_transform,
        ee_gripper_val=observation.ee_gripper_val,
    )
    _model_proprio = einops.rearrange(
        _model_proprio,
        "batch ee buffer -> batch (ee buffer)",
        ee=2, 
        buffer=10,
    )

    _model_actions = model.generate_actions(
        input_ids=_model_input_ids.to(model.device, non_blocking=True),
        image_input=_model_image_input.to(model.device, non_blocking=True),
        image_mask=_model_image_mask.to(model.device, non_blocking=True),
        domain_id=_model_domain_id.to(model.device, non_blocking=True),
        proprio=_model_proprio.to(model.device, non_blocking=True),
        steps=num_denoising_steps,
    )
    _model_actions = einops.rearrange(
        _model_actions,
        "batch time (ee buffer) -> batch time ee buffer",
        ee=2, 
        buffer=10,
    )
    ee_transforms, ee_gripper_vals = _decode_ee6d(_model_actions)

    return Action(
        ee_transforms=ee_transforms,
        ee_gripper_vals=ee_gripper_vals,
    )


class Trainer:
    def __init__(
        self,
        model: XVLA,
        processor: XVLAProcessor,
        accelerator: accelerate.Accelerator | None = None,
    ):
        self._model = model
        self._processor = processor
        self._optimizer = build_optimizer(
            model=self._model,
            lr=1e-2,
            weight_decay=0.,
            # betas=tuple(args.betas),
            # lr_coef_soft=args.learning_coef,
        )
        self._accelerator = accelerator
        if self._accelerator is not None:
            self._model = self._accelerator.prepare(self._model)
            self._optimizer = self._accelerator.prepare(self._optimizer)
        self._step = 0

    def fit(
        self,
        observation: Observation, 
        action: Action,
        #
        max_grad_norm: float | None = 1.,
    ):
        model = self._model
        processor = self._processor

        # TODO
        if not model.training:
            model.train(mode=True)
        losses = compute_losses(
            model, 
            processor, 
            observation=observation, 
            action=action,
        )
        total_loss = sum(losses.values())
        if self._accelerator is None:
            total_loss.backward()
        else:
            self._accelerator.backward(total_loss)

        if max_grad_norm is not None:
            if self._accelerator is None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            else:
                self._accelerator.clip_grad_norm_(model.parameters(), max_grad_norm)

        # TODO
        update_group_lrs(
            self._optimizer, 
            step=self._step, 
            args=SimpleNamespace(
                # TODO
                learning_rate=1e-4,
                # TODO
                learning_coef=1.,
                # TODO
                freeze_steps=1000,
                warmup_steps=2000,
                # TODO
                iters=1000000,
                min_lr_ratio=.1,
                use_cosine_decay=False,
            ),
        )
        self._optimizer.step()
        self._optimizer.zero_grad()

        self._step += 1

        return total_loss

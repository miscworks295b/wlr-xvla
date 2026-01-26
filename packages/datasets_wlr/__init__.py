import os
import dataclasses
import concurrent.futures

import torch
import torchvision
import numpy
import polars


# TODO typing
@dataclasses.dataclass(slots=True)
class WLRZhuangEpisodeData:
    image_left: ... # "batch channel:3 height width"
    image_front: ... # "batch channel:3 height width"
    image_right: ... # "batch channel:3 height width"
    dof_positions: ... # "batch dof:20"
    text: ... # "batch n"

    def __len__(self):
        return len(self.dof_positions)

    def __getitem__(self, index):
        return WLRZhuangEpisodeData(
            image_left=self.image_left[index],
            image_front=self.image_front[index],
            image_right=self.image_right[index],
            dof_positions=self.dof_positions[index],
            text=self.text[index],
        )


class WLRZhuangEpisodeDataset(torch.utils.data.Dataset):
    r"""
    WLR episode dataset.
    """

    def __init__(self, path: str):
        self._metadata = polars.read_json(path)
        self._base_path = os.path.dirname(path)

    def __len__(self):
        return len(self._metadata)

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.numpy(force=True)

        metadata = self._metadata[index]

        with concurrent.futures.ThreadPoolExecutor() as executor:
            def _decode_images(paths: ...):
                return list(executor.map(
                    lambda x: torchvision.io.decode_image(
                        os.path.join(self._base_path, x),
                        mode=torchvision.io.ImageReadMode.RGB,
                    ),
                    paths,
                ))
            
            image_left, image_front, image_right = list(executor.map(
                _decode_images,
                (
                    metadata["left"],
                    metadata["front"],
                    metadata["right"],
                ),
            ))

        return WLRZhuangEpisodeData(
            image_left=numpy.stack(image_left),
            image_front=numpy.stack(image_front),
            image_right=numpy.stack(image_right),
            dof_positions=numpy.stack(metadata["joint"]),
            text=numpy.stack(metadata["task"]),
        )


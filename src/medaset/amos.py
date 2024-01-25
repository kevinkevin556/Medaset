import os
from typing import Literal

import numpy as np
from monai.data import CacheDataset
from monai.transforms import (
    Compose,
    CropForegroundd,
    EnsureChannelFirstd,
    LoadImaged,
    Orientationd,
    RandAffined,
    RandCropByPosNegLabeld,
    RandShiftIntensityd,
    ScaleIntensityRanged,
    Spacingd,
    SpatialPadd,
    ToTensord,
)
from monai.transforms import Transform as MonaiTransform

from .base import BaseMixIn
from .transforms import ApplyMaskMappingd

__all__ = [
    "AmosDataset",
    "amos_train_transforms",
    "amos_val_transforms",
]


abbr = {
    "train": "Tr",
    "validation": "Va",
    "test": "Ts",
}

spatial_size = (96, 96, 96)


def get_file_number(filename):
    return int(str(filename).rsplit("_", maxsplit=1)[-1].split(".", maxsplit=1)[0])


class AmosDataset(BaseMixIn, CacheDataset):
    # Dataset info
    num_classes = 16
    max_ct_number = 500

    def __init__(
        self,
        root_dir: str,
        modality: str,
        stage: Literal["train", "validation", "test"] = "train",
        transform: MonaiTransform = None,
        mask_mapping: dict = None,
        dev: bool = False,
        cache_rate: float = 0.1,
        num_workers: int = 2,
    ):
        self.modality = modality
        self.stage = stage

        BaseMixIn.__init__(
            self,
            image_dir=os.path.join(root_dir, f"images{abbr[self.stage]}"),
            target_dir=os.path.join(root_dir, f"labels{abbr[self.stage]}"),
            num_classes=self.num_classes,
            mask_mapping=mask_mapping,
        )

        # Collect data with specified modality
        if self.modality == "ct":
            self.image_path = [p for p in self.image_path if get_file_number(p) <= self.max_ct_number]
            self.target_path = [p for p in self.target_path if get_file_number(p) <= self.max_ct_number]
        elif self.modality == "mr":
            self.image_path = [p for p in self.image_path if get_file_number(p) > self.max_ct_number]
            self.target_path = [p for p in self.target_path if get_file_number(p) > self.max_ct_number]
        elif self.modality == "ct+mr":
            pass
        else:
            raise ValueError("Invalid modality is specified. Options are {ct, mr, ct+mr}.")

        # Developing mode (20% for training data, 5% for other stage)
        if dev:
            if stage == "train":
                # at least 40 images in a training set
                n_train_dev = max(int(len(self.image_path) * 0.2), 36)
                self.image_path = self.image_path[:n_train_dev]
                self.target_path = self.target_path[:n_train_dev]
            else:
                # at least 5 image in a validation / testing set
                n_val_dev = max(int(len(self.image_path) * 0.02), 4)
                self.image_path = self.image_path[:n_val_dev]
                self.target_path = self.target_path[:n_val_dev]

        # Transformations
        mask_mapping_transform = ApplyMaskMappingd(keys=["label"], mask_mapping=self.mask_mapping)
        if isinstance(transform, MonaiTransform):
            transform = Compose([transform, mask_mapping_transform])
        elif stage == "train":
            transform = Compose([amos_train_transforms, mask_mapping_transform])
        elif stage in {"validation", "test"}:
            transform = Compose([amos_val_transforms, mask_mapping_transform])
        else:
            raise ValueError("Either stage or transform should be specified.")

        # Initialize as Monai CacheDataset
        modality_label = ["ct" if get_file_number(p) <= self.max_ct_number else "mr" for p in self.image_path]
        CacheDataset.__init__(
            self,
            data=[
                {"image": im, "label": la, "modality": mod}
                for im, la, mod in zip(self.image_path, self.target_path, modality_label)
            ],
            transform=transform,
            cache_rate=cache_rate,
            num_workers=num_workers,
        )

    def __len__(self):
        if self.stage == "test":
            return len(self.image_path)
        else:
            return len(self.target_path)


amos_train_transforms = Compose(
    [
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"], strict_check=True, channel_dim="no_channel"),
        Spacingd(keys=["image", "label"], pixdim=(1.5, 1.5, 2.0), mode=("bilinear", "nearest")),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        ScaleIntensityRanged(keys=["image"], a_min=-125, a_max=275, b_min=0.0, b_max=1.0, clip=True),
        CropForegroundd(keys=["image", "label"], source_key="image"),
        RandCropByPosNegLabeld(
            keys=["image", "label"],
            label_key="label",
            spatial_size=spatial_size,
            pos=1,
            neg=1,
            num_samples=2,
            image_key="image",
            image_threshold=0,
            allow_smaller=True,
        ),
        SpatialPadd(keys=["image", "label"], spatial_size=spatial_size),
        RandShiftIntensityd(keys=["image"], offsets=0.10, prob=0.50),
        RandAffined(
            keys=["image", "label"],
            mode=("bilinear", "nearest"),
            prob=1.0,
            spatial_size=(96, 96, 96),
            rotate_range=(0, 0, np.pi / 30),
            scale_range=(0.1, 0.1, 0.1),
        ),
        ToTensord(keys=["image", "label"]),
    ]
)


amos_val_transforms = Compose(
    [
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"], strict_check=True, channel_dim="no_channel"),
        Spacingd(keys=["image", "label"], pixdim=(1.5, 1.5, 2.0), mode=("bilinear", "nearest")),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        ScaleIntensityRanged(keys=["image"], a_min=-125, a_max=275, b_min=0.0, b_max=1.0, clip=True),
        CropForegroundd(keys=["image", "label"], source_key="image"),
        ToTensord(keys=["image", "label"]),
    ]
)

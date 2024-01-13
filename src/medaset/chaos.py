import logging
import math
import os
import warnings
from glob import glob
from pathlib import Path
from typing import Final, Literal, Sequence, Tuple, Union

import cv2
import numpy as np
import numpy.random as random
from monai.data import CacheDataset, PydicomReader
from monai.transforms import (
    Compose,
    CropForegroundd,
    EnsureChannelFirstd,
    LoadImaged,
    NormalizeIntensityd,
    Orientationd,
    RandAffined,
    RandCropByPosNegLabeld,
    RandShiftIntensityd,
    Resized,
    ResizeWithPadOrCropd,
    ScaleIntensityRanged,
    Spacingd,
    SpatialCropd,
    SpatialPadd,
    ToTensord,
)
from monai.transforms import Transform as MonaiTransform

from .base import BaseMixIn
from .image_readers import CV2Reader
from .transforms import (
    ApplyMaskMappingd,
    BackgroundifyClassesd,
    LoadDicomSliceAsVolumed,
)
from .utils import generate_dev_subset, split_train_test

__all__ = []

chaos_ct_transforms = Compose(
    [
        LoadDicomSliceAsVolumed(
            keys=["image", "label"],
            index_patterns=[r"i(.*),0000b.dcm", r"liver_GT_(.*).png"],
            keep_volume=False,
            disable_conversion_warning=True,
        ),
        EnsureChannelFirstd(keys=["image", "label"], channel_dim="no_channel"),
        ToTensord(keys=["image", "label"]),
    ]
)


class ChaosCtDataset(BaseMixIn, CacheDataset):
    def __init__(
        self,
        root_dir: str,
        stage: Literal["train", "validation", "test"] = None,
        transform: MonaiTransform = None,
        mask_mapping: dict = None,
        dev: bool = False,
        cache_rate: float = 1,
        num_workers: int = 2,
        random_seed: int = 42,
        split_ratio: Tuple[float] = (0.81, 0.09, 0.1),
        *,
        class_info: dict = dict(
            background={"value": 0, "color": "#000000", "mask_value": 0},
            liver={"value": 255, "color": "#FFFFFF", "mask_value": 1},
        ),
    ):
        # Dataset info
        self.modality: Final = "ct"
        self.class_info = class_info
        self.num_classes: Final = len(self.class_info)
        self.stage: Final = stage

        # Register dataset information using base mixin
        BaseMixIn.__init__(
            self,
            image_dir=[],
            target_dir=[],
            num_classes=self.num_classes,
            mask_mapping=mask_mapping,
        )
        for sample_dir in sorted((Path(root_dir) / "CT").glob("*")):
            self.image_path.append(str(Path(sample_dir) / "DICOM_anon"))
            self.target_path.append(str(Path(sample_dir) / "Ground"))

        # Train-test split (if required)
        self.image_path, self.target_path = split_train_test(
            self.image_path, self.target_path, stage, split_ratio, random_seed
        )
        # Developing mode (20% for training data, 5% for other stage)
        if dev:
            os.environ["MONAI_DEBUG"] = "True"  # Turn on Monai debug mode to observe details in the raised exceptions
            self.image_path, self.target_path = generate_dev_subset(
                self.image_path, self.target_path, stage, n_train_dev=10, n_val_dev=5
            )

        # Transformations
        label_to_integer = ApplyMaskMappingd(
            keys=["label"], mask_mapping={part["value"]: part["mask_value"] for part in self.class_info.values()}
        )
        mask_mapping_transform = ApplyMaskMappingd(keys=["label"], mask_mapping=self.mask_mapping)
        if isinstance(transform, MonaiTransform):
            transform = Compose([transform, label_to_integer, mask_mapping_transform])
        elif stage == "train":
            transform = Compose([chaos_ct_transforms, label_to_integer, mask_mapping_transform])
        elif (stage == "validation") or (stage == "test"):
            transform = Compose([chaos_ct_transforms, label_to_integer, mask_mapping_transform])
        else:
            raise ValueError("Either stage or transform should be specified.")

        # Initialize as Monai CacheDataset
        CacheDataset.__init__(
            self,
            data=[{"image": im, "label": la, "modality": "ct"} for im, la in zip(self.image_path, self.target_path)],
            transform=transform,
            cache_rate=cache_rate,
            num_workers=num_workers,
        )

    def __len__(self) -> int:
        return len(self.target_path)

    def __getitem__(self, index: Union[int, slice, Sequence[int]]):
        return super().__getitem__(index)


class ChaosT2spirDataset(BaseMixIn, CacheDataset):
    def __init__(
        self,
        root_dir: str,
        stage: Literal["train", "validation", "test"] = None,
        transform: MonaiTransform = None,
        mask_mapping: dict = None,
        dev: bool = False,
        cache_rate: float = 1,
        num_workers: int = 2,
        random_seed: int = 42,
        split_ratio: Tuple[float] = (0.81, 0.09, 0.1),
        *,
        class_info: dict = dict(
            background={"value": 0, "color": "#000000", "mask_value": 0},
            liver={"value": 255, "color": "#FFFFFF", "mask_value": 1},
        ),
    ):
        # Dataset info
        self.modality: Final = "ct"
        self.class_info = class_info
        self.num_classes: Final = len(self.class_info)
        self.stage: Final = stage

        # Register dataset information using base mixin
        BaseMixIn.__init__(
            self,
            image_dir=[],
            target_dir=[],
            num_classes=self.num_classes,
            mask_mapping=mask_mapping,
        )
        for sample_dir in sorted((Path(root_dir) / "CT").glob("*")):
            self.image_path.append(str(Path(sample_dir) / "DICOM_anon"))
            self.target_path.append(str(Path(sample_dir) / "Ground"))

        # Train-test split (if required)
        self.image_path, self.target_path = split_train_test(
            self.image_path, self.target_path, stage, split_ratio, random_seed
        )
        # Developing mode (20% for training data, 5% for other stage)
        if dev:
            os.environ["MONAI_DEBUG"] = "True"  # Turn on Monai debug mode to observe details in the raised exceptions
            self.image_path, self.target_path = generate_dev_subset(
                self.image_path, self.target_path, stage, n_train_dev=10, n_val_dev=5
            )

        # Transformations
        label_to_integer = ApplyMaskMappingd(
            keys=["label"], mask_mapping={part["value"]: part["mask_value"] for part in self.class_info.values()}
        )
        mask_mapping_transform = ApplyMaskMappingd(keys=["label"], mask_mapping=self.mask_mapping)
        if isinstance(transform, MonaiTransform):
            transform = Compose([transform, label_to_integer, mask_mapping_transform])
        elif stage == "train":
            transform = Compose([chaos_ct_transforms, label_to_integer, mask_mapping_transform])
        elif (stage == "validation") or (stage == "test"):
            transform = Compose([chaos_ct_transforms, label_to_integer, mask_mapping_transform])
        else:
            raise ValueError("Either stage or transform should be specified.")

        # Initialize as Monai CacheDataset
        CacheDataset.__init__(
            self,
            data=[{"image": im, "label": la, "modality": "ct"} for im, la in zip(self.image_path, self.target_path)],
            transform=transform,
            cache_rate=cache_rate,
            num_workers=num_workers,
        )

    def __len__(self) -> int:
        return len(self.target_path)

    def __getitem__(self, index: Union[int, slice, Sequence[int]]):
        return super().__getitem__(index)

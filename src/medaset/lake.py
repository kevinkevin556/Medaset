import logging
import math
import os
import warnings
from pathlib import Path
from typing import Literal, Sequence, Tuple, Union

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
    ScaleIntensityRanged,
    Spacingd,
    SpatialPadd,
    ToTensord,
)
from monai.transforms import Transform as MonaiTransform

from .base import BaseMixIn
from .image_readers import CV2Reader, IncompatiblePydicomReader
from .transforms import ApplyMaskMappingd, BackgroundifyClassesd

__all__ = [
    "SMATCTDataset",
    "smat_ct_transforms",
    "SMATMRDataset",
    "smat_mr_transforms",
    "SMATDataset",
]


smat_ct_transforms = Compose(
    [
        LoadImaged(
            keys=["image", "label"],
            reader=[PydicomReader(swap_ij=False), CV2Reader(flags=cv2.IMREAD_GRAYSCALE)],
        ),
        EnsureChannelFirstd(keys=["image", "label"], channel_dim="no_channel"),
        # Spacingd(keys=["image", "label"], pixdim=(1.5, 1.5, 2.0), mode=("bilinear", "nearest")),
        # Orientationd(keys=["image", "label"], axcodes="RAS"),
        ScaleIntensityRanged(keys=["image"], a_min=-200, a_max=200, b_min=0, b_max=1, clip=True),
        NormalizeIntensityd(keys=["image"]),
        SpatialPadd(keys=["image", "label"], spatial_size=(512, 512)),
        ToTensord(keys=["image", "label"]),
    ]
)


class SMATCTDataset(BaseMixIn, CacheDataset):
    def __init__(
        self,
        root_dir: str,
        target: Literal["vat", "tsm", "sat", "aw", "ps", "pm", "ra", "all"] = "all",
        stage: Literal["train", "validation", "test"] = None,
        transform: MonaiTransform = None,
        mask_mapping: dict = None,
        dev: bool = False,
        cache_rate: float = 1,
        num_workers: int = 2,
        random_seed: int = 42,
        split_ratio: Tuple[float] = (0.81, 0.09, 0.1),
        sm_as_whole: bool = False,
    ):
        # Class information
        self.modality = "ct"
        self.target = target
        self.stage = stage
        if target == "all":
            self.class_info = dict(
                background={"value": 0, "color": "#000000", "mask_value": 0},
                ra={"value": 255, "color": "#25CED1", "mask_value": 1},
                vat={"value": 227, "color": "#777777", "mask_value": 2},
                ps={"value": 199, "color": "#FEEFE5", "mask_value": 3},
                sat={"value": 170, "color": "#555555", "mask_value": 4},
                pm={"value": 142, "color": "#FFCF00", "mask_value": 5},
                aw={"value": 85, "color": "#EE6123", "mask_value": 6},
            )
            self.num_classes = 7
        else:
            self.class_info = {
                "background": {"value": 0.0, "color": "#000000", "mask_value": 0},
                target: {"value": 255.0, "color": "#FFFFFF", "mask_value": 1},
            }
            self.num_classes = 2

        # Register dataset information using base mixin
        BaseMixIn.__init__(
            self,
            image_dir=os.path.join(root_dir, "imagesTr"),
            target_dir=os.path.join(root_dir, f"labelsTr/{target}"),
            num_classes=self.num_classes,
            mask_mapping=mask_mapping,
        )

        # Check whether images match labels
        image_name = set([Path(p).stem for p in self.image_path])
        image_parent, image_suffix = Path(self.image_path[0]).parent, Path(self.image_path[0]).suffix
        target_name = set([Path(p).stem for p in self.target_path])
        target_parent, target_suffix = Path(self.target_path[0]).parent, Path(self.target_path[0]).suffix
        no_target_images = image_name - target_name
        no_source_labels = target_name - image_name
        if no_target_images:
            self.image_path = sorted(
                [str(image_parent / f"{name}{image_suffix}") for name in image_name - no_target_images]
            )
            warnings.warn(
                f"Some images are removed due to the lack of associated label: {sorted(no_target_images)}", UserWarning
            )
        else:
            self.image_path = sorted(self.image_path)

        if no_source_labels:
            self.target_path = sorted(
                [str(target_parent / f"{name}{target_suffix}") for name in target_name - no_source_labels]
            )
            warnings.warn(
                f"Some labels are removed due to the lack of associated image: {sorted(no_source_labels)}", UserWarning
            )
        else:
            self.target_path = sorted(self.target_path)

        # Train-test split (if required)
        assert math.isclose(sum(split_ratio), 1)
        n = len(self.target_path)
        n_train, n_test = int(n * split_ratio[0]), int(n * split_ratio[2])
        random_id = random.RandomState(seed=random_seed).permutation(n)
        self.train_indices = random_id[:n_train]
        self.valid_indices = random_id[n_train:-n_test]
        self.test_indices = random_id[-n_test:]
        if stage == "train":
            self.image_path = [self.image_path[i] for i in self.train_indices]
            self.target_path = [self.target_path[i] for i in self.train_indices]
        elif stage == "validation":
            self.image_path = [self.image_path[i] for i in self.valid_indices]
            self.target_path = [self.target_path[i] for i in self.valid_indices]
        elif stage == "test":
            self.image_path = [self.image_path[i] for i in self.test_indices]
            self.target_path = [self.target_path[i] for i in self.test_indices]
        elif stage is None:
            pass
        else:
            raise ValueError(f"Invalid stage. Expect 'train', 'val', 'test' or None. Got {stage}")

        # Developing mode (20% for training data, 5% for other stage)
        if dev:
            if stage == "train":
                # at least 10 images in a training set
                n_train_dev = 10
                self.image_path = self.image_path[:n_train_dev]
                self.target_path = self.target_path[:n_train_dev]
            else:
                # at least 5 image in a validation / testing set
                n_val_dev = 5
                self.image_path = self.image_path[:n_val_dev]
                self.target_path = self.target_path[:n_val_dev]

        # Transformations
        label_to_integer = ApplyMaskMappingd(
            keys=["label"], mask_mapping={part["value"]: part["mask_value"] for part in self.class_info.values()}
        )
        mask_mapping_transform = ApplyMaskMappingd(keys=["label"], mask_mapping=self.mask_mapping)
        if isinstance(transform, MonaiTransform):
            _transform = [transform, label_to_integer]
        elif stage == "train":
            _transform = [smat_ct_transforms, label_to_integer]
        elif (stage == "validation") or (stage == "test"):
            _transform = [smat_ct_transforms, label_to_integer]
        else:
            raise ValueError("Either stage or transform should be specified.")

        if sm_as_whole and target == "all":
            # tsm = 1, vat = 2, sat = 3
            combine_into_tsm = ApplyMaskMappingd(keys=["label"], mask_mapping={1: 1, 3: 1, 5: 1, 6: 1, 4: 3})
            _transform = _transform + [combine_into_tsm, mask_mapping_transform]
        else:
            _transform = _transform + [mask_mapping_transform]
        transform = Compose(_transform)

        # Initialize as Monai CacheDataset
        CacheDataset.__init__(
            self,
            data=[
                {"image": im, "label": la, "modality": "ct"}
                for im, la in zip(sorted(self.image_path), sorted(self.target_path))
            ],
            transform=transform,
            cache_rate=cache_rate,
            num_workers=num_workers,
        )

    def __len__(self) -> int:
        return len(self.target_path)
    
    def __getitem__(self, index: Union[int, slice, Sequence[int]]):
        # Suppress CV2Reader "unable to load" exceptions for DICOM files
        logging.disable(logging.CRITICAL)
        return super().__getitem__(index)


smat_mr_transforms = Compose(
    [
        LoadImaged(
            keys=["image", "label"],
            reader=[PydicomReader(swap_ij=False, force=True), CV2Reader(flags=cv2.IMREAD_GRAYSCALE)],
            image_only=True,
        ),
        EnsureChannelFirstd(keys=["image", "label"], channel_dim="no_channel"),
        # Spacingd(keys=["image", "label"], pixdim=(1.5, 1.5, 2.0), mode=("bilinear", "nearest")),
        # Orientationd(keys=["image", "label"], axcodes="RAS"),
        NormalizeIntensityd(keys=["image"]),
        SpatialPadd(keys=["image", "label"], spatial_size=(512, 512)),
        ToTensord(keys=["image", "label"]),
        # DeleteItemsd(keys=["image", "label"]),
    ]
)


class SMATMRDataset(BaseMixIn, CacheDataset):
    max_non_pkd_number = 121

    def __init__(
        self,
        root_dir: str,
        sequence: Literal["w", "f", "in", "op", "pdff"] = "pdff",
        target: Literal["vat", "tsm", "sat", "all"] = "all",
        stage: Literal["train", "validation", "test"] = None,
        transform: MonaiTransform = None,
        mask_mapping: dict = None,
        dev: bool = False,
        cache_rate: float = 1,
        num_workers: int = 2,
        random_seed: int = 42,
        split_ratio: Tuple[float] = (0.81, 0.09, 0.1),
        pkd_only: bool = False,
        non_pkd_only: bool = True,
    ):
        self.modality = "mr"
        self.target = target
        self.sequence = sequence
        self.stage = stage

        sequence = [sequence] if not isinstance(sequence, (list, tuple)) else sequence
        _images = []
        _seq_images = {}
        for s in sequence:
            _seq_images[s] = [file for file in (Path(root_dir) / "imagesTr").glob(f"*_{s}.dcm")]
        for files in zip(*_seq_images.values()):
            _images.append(list(files))

        if target == "all":
            self.class_info = dict(
                background={"value": 0, "color": "#000000", "mask_value": 0},
                tsm={"value": 85, "color": "#EE6123", "mask_value": 1},
                vat={"value": 255, "color": "#FEEFE5", "mask_value": 2},
                sat={"value": 170, "color": "#25CED1", "mask_value": 3},
            )
            self.num_classes = 4
        else:
            self.class_info = {
                "background": {"value": 0.0, "color": "#000000", "mask_value": 0},
                target: {"value": 255.0, "color": "#FFFFFF", "mask_value": 1},
            }
            self.num_classes = 2

        # Register dataset information using base mixin
        BaseMixIn.__init__(
            self,
            image_dir=_images,
            target_dir=os.path.join(root_dir, f"labelsTr/{target}"),
            num_classes=self.num_classes,
            mask_mapping=mask_mapping,
        )

        # Check whether images match label
        assert (pkd_only and non_pkd_only) == False, "pkd_only and non_pkd_only can't be true at the same time."
        if non_pkd_only:
            self.image_path = self.image_path[: self.max_non_pkd_number]
            self.target_path = self.target_path[: self.max_non_pkd_number]
        elif pkd_only:
            if target != "sat":
                self.image_path = self.image_path[self.max_non_pkd_number :]
                self.target_path = self.target_path[self.max_non_pkd_number :]
            else:
                raise ValueError("No SAT ground truth from pkd patients")
        else:
            if target == "all":
                warnings.warn("There is no SAT ground truth from pkd patients. Set non_pkd_only=True if necessary.")
            elif target == "sat":
                warnings.warn(
                    "There is no SAT ground truth from pkd patients. Only images from non_pkd patients will be loaded."
                )
                self.image_path = self.image_path[: self.max_non_pkd_number]
            else:
                pass

        # Train-test split (if required)
        assert math.isclose(sum(split_ratio), 1)
        n = len(self.target_path)
        n_train, n_test = int(n * split_ratio[0]), int(n * split_ratio[2])
        random_id = random.RandomState(seed=random_seed).permutation(n)
        self.train_indices = random_id[:n_train]
        self.valid_indices = random_id[n_train:-n_test]
        self.test_indices = random_id[-n_test:]
        if stage == "train":
            self.image_path = [self.image_path[i] for i in self.train_indices]
            self.target_path = [self.target_path[i] for i in self.train_indices]
        elif stage == "validation":
            self.image_path = [self.image_path[i] for i in self.valid_indices]
            self.target_path = [self.target_path[i] for i in self.valid_indices]
        elif stage == "test":
            self.image_path = [self.image_path[i] for i in self.test_indices]
            self.target_path = [self.target_path[i] for i in self.test_indices]
        elif stage is None:
            pass
        else:
            raise ValueError(f"Invalid stage. Expect 'train', 'val', 'test' or None. Got {stage}")

        # Developing mode (20% for training data, 5% for other stage)
        if dev:
            if stage == "train":
                # at least 10 images in a training set
                n_train_dev = 10
                self.image_path = self.image_path[:n_train_dev]
                self.target_path = self.target_path[:n_train_dev]
            else:
                # at least 5 image in a validation / testing set
                n_val_dev = 5
                self.image_path = self.image_path[:n_val_dev]
                self.target_path = self.target_path[:n_val_dev]

        # Transformations
        label_to_integer = ApplyMaskMappingd(
            keys=["label"], mask_mapping={part["value"]: part["mask_value"] for part in self.class_info.values()}
        )
        mask_mapping_transform = ApplyMaskMappingd(keys=["label"], mask_mapping=self.mask_mapping)
        if isinstance(transform, MonaiTransform):
            transform = Compose([transform, label_to_integer, mask_mapping_transform])
        elif stage == "train":
            transform = Compose([smat_mr_transforms, label_to_integer, mask_mapping_transform])
        elif (stage == "validation") or (stage == "test"):
            transform = Compose([smat_mr_transforms, label_to_integer, mask_mapping_transform])
        else:
            raise ValueError("Either stage or transform should be specified.")

        # Initialize as Monai CacheDataset
        CacheDataset.__init__(
            self,
            data=[
                {"image": im, "label": la, "modality": "mr"}
                for im, la in zip(sorted(self.image_path), sorted(self.target_path))
            ],
            transform=transform,
            cache_rate=cache_rate,
            num_workers=num_workers,
        )

    def __getitem__(self, index: Union[int, slice, Sequence[int]]):
        # Suppress CV2Reader "unable to load" exceptions for DICOM files
        logging.disable(logging.CRITICAL) 
        return super().__getitem__(index)


class SMATDataset(SMATCTDataset, SMATMRDataset):
    num_classes = 4

    def __init__(
        self,
        root_dir: str,
        modality: str,
        target: Literal["vat", "tsm", "sat", "all"] = "all",
        stage: Literal["train", "validation", "test"] = "train",
        transform: MonaiTransform = None,
        mask_mapping: dict = None,
        dev: bool = False,
        cache_rate: float = 1,
        num_workers: int = 2,
        random_seed: int = 42,
        split_ratio: tuple = (0.81, 0.09, 0.1),
        sm_as_whole: bool = True,
    ):
        if modality in ["CT", "ct"]:
            SMATCTDataset.__init__(
                self=self,
                root_dir=os.path.join(root_dir, "CT"),
                target=target,
                stage=stage,
                transform=transform,
                mask_mapping=mask_mapping,
                dev=dev,
                cache_rate=cache_rate,
                num_workers=num_workers,
                random_seed=random_seed,
                split_ratio=split_ratio,
                sm_as_whole=sm_as_whole,
            )
        elif modality in ["MR", "mr"]:
            SMATMRDataset.__init__(
                self=self,
                root_dir=os.path.join(root_dir, "MR"),
                sequence="pdff",
                target=target,
                stage=stage,
                transform=transform,
                mask_mapping=mask_mapping,
                dev=dev,
                cache_rate=cache_rate,
                num_workers=num_workers,
                random_seed=random_seed,
                split_ratio=split_ratio,
            )
        else:
            raise ValueError(f"Invalid modality. Got {modality}.")

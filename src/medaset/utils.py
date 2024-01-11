import json
import math
import os
import warnings
from collections.abc import Mapping, Sequence
from inspect import signature
from pathlib import Path

import cv2
import nibabel as nib
import numpy as np
import pydicom
from numpy.typing import ArrayLike


def read_image(image_path, mask_mapping=None):
    if isinstance(image_path, (str, Path)):
        image_path = Path(image_path)
        if image_path.suffix == ".npy":
            image = np.load(str(image_path))
        elif image_path.suffix == ".dcm":
            ds = pydicom.dcmread(str(image_path), force=True)
            ds.file_meta.TransferSyntaxUID = pydicom.uid.ImplicitVRLittleEndian
            image = ds.pixel_array.astype(np.int32)
        elif image_path.suffix in [".nii", ".gz"]:
            image = nib.load(str(image_path)).get_fdata()
        elif image_path.suffix in [".png", ".jpg"]:
            image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        else:
            raise OSError(
                f"The file extension of image is not supported currently. Got file extension = {image_path.suffix}"
            )
    elif isinstance(image_path, Sequence):
        channels = []
        for channel_path in image_path:
            channels.append(read_image(channel_path, mask_mapping))
        image = np.stack(channels, axis=-1)
    elif isinstance(image_path, Mapping):
        channels = image_path.copy()
        values = {k: read_image(v) for k, v in image_path.items() if isinstance(v, (str, Path))}
        channels.update(values)
        for key, val in channels.items():
            if callable(val):
                f = val
                params = {k: channels[k] for k in channels.keys() & signature(f).parameters.keys()}
                channels.update({key: f(**params)})
        image = np.stack(list(channels.values()), axis=-1)
    else:
        raise TypeError(f"Invalid input type of image path. Got type(image_path) = {str(type(image_path))}")

    if mask_mapping:
        for key, value in mask_mapping.items():
            image[image == key] = value
    return image


def get_file_paths(image_dirs, sort=True):
    support_data_type = [".npy", ".dcm", ".nii", ".gz", ".png"]
    if isinstance(image_dirs, (str, Path)):
        image_dirs = Path(image_dirs)
        if image_dirs.is_file():
            file_paths = [str(image_dirs)]
        else:
            file_paths = [str(p) for p in image_dirs.iterdir() if p.is_file() and p.suffix in support_data_type]
            if sort:
                file_paths = sorted(file_paths)
    elif isinstance(image_dirs, Sequence):
        file_paths = []
        for elem in image_dirs:
            if isinstance(elem, (str, Path)):
                file_paths += get_file_paths(elem)
            elif isinstance(elem, Mapping):
                channels = {k: get_file_paths(v) if isinstance(v, (str, Path)) else v for k, v in elem.items()}
                length = [len(c) for c in channels.values() if isinstance(c, Sequence)]
                assert min(length) == max(length), "Input sequences should have same length."
                length = length[0]
                for i in range(length):
                    file_path = elem.__class__({k: v[i] if isinstance(v, Sequence) else v for k, v in channels.items()})
                    file_paths.append(file_path)
            elif isinstance(elem, Sequence):
                file_paths += list(zip(*[get_file_paths(channel) for channel in elem]))
            else:
                raise TypeError(f"Invalid input type of image dirs. Got type(image_dirs) = {str(type(image_dirs))}")
    else:
        raise TypeError(f"Invalid input type of image dirs. Got type(image_dirs) = {str(type(image_dirs))}")
    return file_paths


def get_paths_from_json(json_path, root_dir=None):
    image_path = []
    target_path = []
    root_dir = Path(json_path).parent if root_dir is None else root_dir
    with open(json_path) as f:
        for image_pair in json.load(f)["training"]:
            if image_pair["image"].startswith("."):
                image_path.append(str(root_dir / image_pair["image"]))
            else:
                image_path.append(image_pair["image"])

            if image_pair["label"].startswith("."):
                target_path.append(str(root_dir / image_pair["label"]))
            else:
                target_path.append(image_pair["label"])
    return image_path, target_path


def apply_window(image: ArrayLike, min: int, max: int) -> ArrayLike:
    """Set window to a (CT) image, i.e. only values in the range of window
    are preserved. Values are assigned the min value if they are less than lower bound (min)
    of the window or the max value if they are greater than the upper bound (max) of the window.

    Args:
        image (ArrayLike): an image
        min (int): the lower bound of the window
        max (int): the upper bound of the window

    Returns:
        ArrayLike: the transformed image
    """
    image[image < min] = min
    image[image > max] = max
    return image


def split_train_test(image_path, target_path, stage, split_ratio, random_seed):
    assert math.isclose(sum(split_ratio), 1)
    n = len(target_path)
    n_train, n_test = int(n * split_ratio[0]), int(n * split_ratio[2])
    random_id = np.random.RandomState(seed=random_seed).permutation(n)
    train_indices = random_id[:n_train]
    valid_indices = random_id[n_train:-n_test]
    test_indices = random_id[-n_test:]
    if stage == "train":
        image_path = [image_path[i] for i in train_indices]
        target_path = [target_path[i] for i in train_indices]
    elif stage == "validation":
        image_path = [image_path[i] for i in valid_indices]
        target_path = [target_path[i] for i in valid_indices]
    elif stage == "test":
        image_path = [image_path[i] for i in test_indices]
        target_path = [target_path[i] for i in test_indices]
    elif stage is None:
        pass
    else:
        raise ValueError(f"Invalid stage. Expect 'train', 'val', 'test' or None. Got {stage}")
    return image_path, target_path


def check_image_label_pairing(image_path, target_path, raise_exception=False):
    image_name = set([Path(p).stem for p in image_path])
    image_parent, image_suffix = Path(image_path[0]).parent, Path(image_path[0]).suffix
    target_name = set([Path(p).stem for p in target_path])
    target_parent, target_suffix = Path(target_path[0]).parent, Path(target_path[0]).suffix
    no_target_images = image_name - target_name
    no_source_labels = target_name - image_name

    # Raise exception or remove images if there is no associated label
    if no_target_images:
        if raise_exception:
            raise FileNotFoundError(f"No associated label of these images: {sorted(no_target_images)}")
        else:
            image_path = sorted([str(image_parent / f"{name}{image_suffix}") for name in image_name - no_target_images])
            warnings.warn(
                f"Some images are removed due to the lack of associated label: {sorted(no_target_images)}", UserWarning
            )

    # Raise exception or remove labels if there is no associated source image
    if no_source_labels:
        if raise_exception:
            raise FileNotFoundError(f"No associated image of these labels: {sorted(no_source_labels)}")
        else:
            target_path = sorted(
                [str(target_parent / f"{name}{target_suffix}") for name in target_name - no_source_labels]
            )
            warnings.warn(
                f"Some labels are removed due to the lack of associated image: {sorted(no_source_labels)}", UserWarning
            )

    return image_path, target_path


def generate_dev_subset(image_path, target_path, stage, n_train_dev=10, n_val_dev=5):
    if stage == "train":
        image_path = image_path[:n_train_dev]
        target_path = target_path[:n_train_dev]
    else:
        image_path = image_path[:n_val_dev]
        target_path = target_path[:n_val_dev]
    return image_path, target_path

import json
import math
import re
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
        elif image_path.suffix in {".nii", ".gz"}:
            image = nib.load(str(image_path)).get_fdata()
        elif image_path.suffix in {".png", ".jpg"}:
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


def apply_window(image: ArrayLike, min_val: int, max_val: int) -> ArrayLike:
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
    image[image < min_val] = min_val
    image[image > max_val] = max_val
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


def _s2i(s):
    """
    Safely converts a string to an integer, returning the integer value if successful,
    or the original string if conversion fails.

    Args:
        s (str): The input string to be converted to an integer.

    Returns:
        int or str: If the conversion is successful, returns the integer value.
                    If the conversion fails, returns the original string without modification.
    """
    try:
        integer_value = int(s)
        return integer_value
    except ValueError:
        return s


def check_image_label_pairing(
    image_path, target_path, raise_exception=False, image_pattern=r"(.*)\..*", target_pattern=r"(.*)\..*"
):
    """
    Check and validate the pairing between image and label files based on provided patterns.

    This function verifies the pairing between image and label files by comparing their file names
    after applying the specified patterns. It extracts a unique identifier (index) from each file's
    name using the provided regular expression patterns between parentheses. The function then
    checks for matching indices between the two lists of files.

    Example:
        image_path = ["image_1.dcm", "image_2.dcm", "image_3.dcm"]
        target_path = ["label_1.png", "label_3.png"]
        check_image_label_pairing(image_path, target_path, raise_exception=True)
        # This will raise a FileNotFoundError indicating that "image_2.jpg" has no associated label.

    Note:
        - This function assumes that the provided patterns correctly extract a unique identifier
          (index) from the file names, allowing it to pair image and label files based on those
          identifiers.
        - The patterns should include parentheses to capture the index portion of the file names.
        - Use caution when setting raise_exception to False, as it will modify the input paths by
          removing unmatched files.

    Args:
        image_path (list of str): List of image file paths to be checked.
        target_path (list of str): List of label file paths to be checked.
        raise_exception (bool, optional): If True, raise exceptions when discrepancies are found.
            If False (default), issue warnings and attempt to remove unmatched files.
        image_pattern (str, optional): Regular expression pattern used to extract information from
            image file names. Default is '(.*)\..*', which captures the file name without the file extension.
        target_pattern (str, optional): Regular expression pattern used to extract information from
            label file names. Default is '(.*)\..*', which captures the file name without the file extension.

    Raises:
        FileNotFoundError: If there are image files with no associated label files or vice versa.
            The exception message will contain the list of affected file names.
        UserWarning: If raise_exception is False and discrepancies are found, warnings will be issued,
            and unmatched files will be removed from the respective paths.

    Returns:
        tuple of list of str: A tuple containing the modified image_path and target_path after checking
        and potentially removing unmatched files.

    """
    image_name = [str(Path(p).name) for p in image_path]
    image_parent = Path(image_path[0]).parent
    image_index_to_name = {_s2i(re.search(image_pattern, name).group(1)): name for name in image_name}
    image_index = set(image_index_to_name.keys())

    target_name = [str(Path(p).name) for p in target_path]
    target_parent = Path(target_path[0]).parent
    target_index_to_name = {_s2i(re.search(target_pattern, name).group(1)): name for name in target_name}
    target_index = set(target_index_to_name.keys())

    no_target_images = image_index - target_index
    no_target_images_name = sorted([image_index_to_name[index] for index in no_target_images])
    no_source_labels = target_index - image_index
    no_source_labels_name = sorted([target_index_to_name[index] for index in no_source_labels])

    # Raise exception or remove unmatched images
    if no_target_images:
        if raise_exception:
            raise FileNotFoundError(f"No associated label of these images: {no_target_images_name}")
        else:
            image_path = [str(image_parent / image_index_to_name[index]) for index in image_index - no_target_images]
            image_path.sort()
            warnings.warn(
                f"Some images are removed due to the lack of associated label: {no_target_images_name}", UserWarning
            )

    # Raise exception or remove unmatched labels
    if no_source_labels:
        if raise_exception:
            raise FileNotFoundError(f"No associated image of these labels: {no_source_labels_name}")
        else:
            target_path = [
                str(target_parent / target_index_to_name[index]) for index in target_index - no_source_labels
            ]
            target_path.sort()
            warnings.warn(
                f"Some labels are removed due to the lack of associated image: {no_source_labels_name}", UserWarning
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

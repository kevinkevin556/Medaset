import logging
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, Hashable, Mapping, Optional, Sequence, Union

import dicom2nifti
import numpy as np
import pydicom
import torch
from monai.config import DtypeLike, KeysCollection, NdarrayOrTensor, PathLike
from monai.data.image_reader import ImageReader
from monai.data.meta_tensor import MetaTensor
from monai.transforms import LoadImage, LoadImaged, MapTransform, Transform
from monai.utils import GridSamplePadMode, ensure_tuple, ensure_tuple_rep, require_pkg
from monai.utils.enums import PostFix
from PIL import Image

from .utils import check_image_label_pairing

__all__ = [
    "ApplyMaskMapping",
    "ApplyMaskMappingd",
    "BackgroundifyClasses",
    "BackgroundifyClassesd",
    "LoadDicomSliceAsVolume",
    "LoadDicomSliceAsVolumed",
]

DEFAULT_POST_FIX = PostFix.meta()


class ApplyMaskMapping(Transform):
    def __init__(self, mask_mapping: Optional[dict] = None) -> None:
        self.mask_mapping = mask_mapping

    def __call__(self, mask) -> torch.Tensor:
        if self.mask_mapping:
            for original_value, new_value in self.mask_mapping.items():
                mask[mask == original_value] = new_value
        return mask


class ApplyMaskMappingd(MapTransform):
    """
    Dictionary-based version `ApplyMaskMapping`.
    """

    def __init__(
        self,
        keys: KeysCollection,
        mask_mapping: dict,
        allow_missing_keys: bool = False,
    ) -> None:
        MapTransform.__init__(self, keys, allow_missing_keys)
        self.mask_mapper = ApplyMaskMapping(mask_mapping)

    def __call__(self, data: Mapping[Hashable, torch.Tensor]) -> Dict[Hashable, torch.Tensor]:
        d = dict(data)
        for key in self.key_iterator(d):
            d[key] = self.mask_mapper(d[key])
        return d


class BackgroundifyClasses(Transform):
    """
    Modifies ground truth masks by assigning a value of zero to specific classes,
    merging them into the background class.
    """

    def __init__(self, channel_dim: int, classes: list) -> None:
        self.channel_dim = channel_dim
        self.classes = torch.tensor(classes)

    def __call__(self, img) -> torch.Tensor:
        n_channels = img.shape[self.channel_dim]

        # Create indexing tuple to retrieve background anc the merging class
        # E.g. ix_bg = (0, slice(None), slice(None)) for the background of image = img[0, :, :] when dim = 0
        #      ix_cls = (slice(None), self.classes, slice(None), slice(None))
        #      for the merging classes = img[:, classes, :, :] when dim = 1
        ix_bg = tuple([0 if (d == self.dim) else slice(None) for d in range(len(img.shape))])
        ix_cls = tuple([self.classes if (d == self.dim) else slice(None) for d in range(len(img.shape))])
        img_t = torch.Tensor(img)

        # If n_channels == 1, the input tensor stores labels.
        # Otherwise, the input tensor is a tensor of activations (probabilities) or results from one-hot encoding.
        if n_channels == 1:
            img_t[torch.isin(img_t, self.classes)] = 0
        else:
            img_t[ix_bg] += torch.sum(img_t[ix_cls], dim=self.dim)
            img_t[ix_cls] = 0

        # Collect meta-data from original MetaTensor, or return a plain MetaTensor
        if isinstance(img, MetaTensor):
            img_t = MetaTensor(img_t, meta=img.meta)
        else:
            img_t = MetaTensor(img_t)
        return img_t


class BackgroundifyClassesd(MapTransform):
    """
    Dictionary-based version `BackgroundifyClasses`.
    """

    def __init__(
        self,
        keys: KeysCollection,
        channel_dim: int,
        classes: list,
        allow_missing_keys: bool = False,
    ) -> None:
        MapTransform.__init__(self, keys, allow_missing_keys)
        self.backgroundifier = BackgroundifyClasses(channel_dim, classes)

    def __call__(self, data: Mapping[Hashable, torch.Tensor]) -> Dict[Hashable, torch.Tensor]:
        d = dict(data)
        for key in self.key_iterator(d):
            d[key] = self.backgroundifier(d[key])
        return d


@require_pkg("dicom2nifti")
class LoadDicomSliceAsVolume(LoadImage):
    def __init__(
        self,
        reader=None,
        image_only: bool = True,
        dtype: Optional[DtypeLike] = np.float32,
        ensure_channel_first: bool = False,
        simple_keys: bool = False,
        prune_meta_pattern: Optional[str] = None,
        prune_meta_sep: str = ".",
        expanduser: bool = True,
        nifti_filename: str = "volume.nii",
        keep_volume: bool = False,
        reorient_to_las: bool = False,
        disable_conversion_warning: bool = False,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(
            reader,
            image_only,
            dtype,
            ensure_channel_first,
            simple_keys,
            prune_meta_pattern,
            prune_meta_sep,
            # expanduser,
            #   Note: The `expanduser` argument is introduced in monai 1.2.0.
            #   Medaset is currently developed under monai 1.0.1, and thus `expanduser`
            #   is left as comment.
            *args,
            **kwargs,
        )
        self.nifti_filename = nifti_filename
        self.keep_volume = keep_volume
        self.reorient_to_las = reorient_to_las
        self.disable_conversion_warning = disable_conversion_warning

    def __call__(self, dirname: Union[Sequence[PathLike], PathLike], reader=None):
        if self.disable_conversion_warning:
            logging.disable(logging.WARNING)

        # Create a volume file if a directory is specified
        if Path(dirname).is_dir():
            volume_file = Path(dirname) / self.nifti_filename
        else:
            return super().__call__(dirname, reader)

        # If the volume NIfTI file has not been created, generate one under the input directory
        # Otherwise, read it directly through the nii file.
        if not volume_file.exists():
            assert all(
                [Path(f).suffix == ".dcm" for f in Path(dirname).glob("*")]
            ), "The directory should contain only DICOM files."

            if not self.keep_volume:
                temp_dir = tempfile.TemporaryDirectory()
                volume_file = Path(temp_dir.name) / self.nifti_filename

            _ = dicom2nifti.dicom_series_to_nifti(dirname, volume_file, reorient_nifti=self.reorient_to_las)
        return super().__call__(filename=volume_file)


@require_pkg("dicom2nifti")
class LoadDicomSliceAsVolumed(LoadImaged):
    def __init__(
        self,
        keys: KeysCollection,
        reader: Optional[Union[ImageReader, str]] = None,
        dtype: DtypeLike = np.float32,
        meta_keys: Optional[KeysCollection] = None,
        meta_key_postfix: str = DEFAULT_POST_FIX,
        overwriting: bool = False,
        image_only: bool = True,
        ensure_channel_first: bool = False,
        simple_keys: bool = False,
        prune_meta_pattern: Optional[str] = None,
        prune_meta_sep: str = ".",
        allow_missing_keys: bool = False,
        expanduser: bool = True,
        nifti_filename: str = "volume.nii",
        keep_volume: bool = False,
        reorient_to_las: bool = False,
        disable_conversion_warning: bool = False,
        index_patterns: Optional[Sequence] = None,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(
            keys,
            reader,
            dtype,
            meta_keys,
            meta_key_postfix,
            overwriting,
            image_only,
            ensure_channel_first,
            simple_keys,
            prune_meta_pattern,
            prune_meta_sep,
            allow_missing_keys,
            # expanduser,
            #   Note: The `expanduser` argument is introduced in monai 1.2.0.
            #   Medaset is currently developed under monai 1.0.1, and thus `expanduser`
            #   is left as comment.
            *args,
            **kwargs,
        )
        self._loader = LoadDicomSliceAsVolume(
            reader,
            image_only,
            dtype,
            ensure_channel_first,
            simple_keys,
            prune_meta_pattern,
            prune_meta_sep,
            expanduser,
            nifti_filename,
            keep_volume,
            reorient_to_las,
            disable_conversion_warning,
            *args,
            **kwargs,
        )
        self.nifti_filename = nifti_filename
        self.keep_volume = keep_volume
        self.reorient_to_las = reorient_to_las
        self.disable_conversion_warning = disable_conversion_warning
        if index_patterns is None:
            self.index_patterns = [r"(.*)\..*" for k in self.keys]
        else:
            self.index_patterns = index_patterns

    def __call__(self, data, reader=None):
        # Make copies of input data dict
        d = dict(data)

        # Get file extension from input directories
        suffix_to_data_key = {}
        for key, dirname in d.items():
            if Path(dirname).is_dir():
                _suffix = np.unique([Path(f).suffix for f in Path(dirname).glob("*")])
                # Skip the directory with a .nii file in it
                if ".nii" in _suffix:
                    nii_files = list(Path(dirname).glob("*.nii"))
                    assert len(nii_files) == 1
                    d[key] = nii_files[0]
                    continue
                else:
                    assert len(_suffix) == 1
                    suffix_to_data_key[_suffix[0]] = key

        # If there is no need for volume generation, read files directly.
        # Otherwise, make sure there is a source directory of dicom files
        if not suffix_to_data_key:
            return super().__call__(d, reader)
        else:
            assert ".dcm" in suffix_to_data_key.keys(), "There should be a key to directory of dicom files."
        image_key = suffix_to_data_key[".dcm"]
        image_files = sorted(Path(d[image_key]).glob("*"))
        del suffix_to_data_key[".dcm"]

        # Create temp dicom files for labels
        tempdirs = []
        key_to_pattern = {key: pattern for key, pattern in zip(self.keys, self.index_patterns)}
        for suffix, key in suffix_to_data_key.items():
            # Test if image and labels are paired
            label_files = sorted(Path(d[key]).glob("*"))
            _ = check_image_label_pairing(
                image_files,
                label_files,
                raise_exception=True,
                image_pattern=key_to_pattern[image_key],
                target_pattern=key_to_pattern[key],
            )

            # Generate temp dicoms
            temp_dicom_dir = tempfile.TemporaryDirectory(prefix=f"{key}_")
            tempdirs.append(temp_dicom_dir)
            for image, label in zip(image_files, label_files):
                ds = pydicom.dcmread(image)
                # Scale the label to obtain the same scale as PixelData
                # It will be inversed when reading DICOMs during generating volumes
                rs_intercept = float(getattr(ds, "RescaleIntercept", 0))
                rs_slope = float(getattr(ds, "RescaleSlope", 1))
                label_pixel = np.asarray(Image.open(label))
                label_data = (label_pixel - rs_intercept) / rs_slope
                # Replace the original pixel data with label information and save the modified dicom
                ds.PixelData = label_data.astype(np.int16).tobytes()
                ds.save_as(Path(temp_dicom_dir.name) / image.name)

            # Update label directory
            d[key] = temp_dicom_dir.name

        # Create sample dict from loader
        output = super().__call__(d, reader)

        # If keep_volume=True, make a copy of volume nifti file in the original directory
        if self.keep_volume:
            for suffix, key in suffix_to_data_key.items():
                temp_dicom_dir = d[key]
                temp_volume_file = Path(temp_dicom_dir) / self.nifti_filename
                dst_volume_file = Path(data[key]) / self.nifti_filename
                shutil.copyfile(temp_volume_file, dst_volume_file)
                output[key].meta["filename_or_obj"] = str(dst_volume_file)

        # Clean up temparary directories
        for temp_dicom_dir in tempdirs:
            temp_dicom_dir.cleanup()
        return output

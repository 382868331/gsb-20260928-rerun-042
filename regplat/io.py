"""影像读写：NIfTI（.nii/.nii.gz），完整保留原点、间距与方向。"""

from __future__ import annotations
import numpy as np

from .volume import MedicalVolume


def load_nifti(path: str) -> MedicalVolume:
    import nibabel as nib

    img = nib.load(str(path))
    aff = img.affine
    linear = aff[:3, :3]
    spacing = tuple(float(v) for v in np.linalg.norm(linear, axis=0))
    direction = linear / np.asarray(spacing)
    data = np.asarray(img.get_fdata(caching="unchanged"), dtype=np.float64)
    if data.ndim != 3:
        raise ValueError(f"仅支持三维 NIfTI，{path} 为 {data.ndim} 维")
    return MedicalVolume(data, spacing, tuple(aff[:3, 3]), direction)


def save_nifti(vol: MedicalVolume, path: str) -> str:
    import nibabel as nib

    from pathlib import Path
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    img = nib.Nifti1Image(np.asarray(vol.data, dtype=np.float32), vol.affine_4x4())
    nib.save(img, str(path))
    return str(path)

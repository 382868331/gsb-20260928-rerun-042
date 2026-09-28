"""处理批次：一次跑多对配准，记录参数指纹与可复现溯源。

每个批次写入 manifest.json：包含参数、参数指纹、输入指纹、状态、误差与覆盖。
重跑同一批次时，会对比已采用的参数指纹，并明确说明沿用/变更了哪份参数。
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import numpy as np

from .volume import MedicalVolume
from .registration import rigid_register, RegistrationParams, RegistrationResult


def _file_fingerprint(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _volume_fingerprint(vol: MedicalVolume) -> str:
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(vol.data).tobytes())
    h.update(np.asarray(vol.spacing).tobytes())
    h.update(np.asarray(vol.origin).tobytes())
    h.update(np.ascontiguousarray(vol.direction).tobytes())
    return h.hexdigest()[:16]


@dataclass
class RunRecord:
    name: str
    status: str
    message: str
    params_fingerprint: str
    moving_fingerprint: str
    fixed_fingerprint: str
    result: dict
    aligned_path: str | None = None
    mask_path: str | None = None


class BatchRunner:
    """把 (moving, fixed) 对组成批次，统一参数运行并输出 manifest。"""

    def __init__(self, output_dir: str | Path, params: RegistrationParams | None = None):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.params = params or RegistrationParams()
        self.records: list[RunRecord] = []

    def run_pair(
        self,
        name: str,
        moving: MedicalVolume,
        fixed: MedicalVolume,
        save_aligned: bool = True,
    ) -> RegistrationResult:
        result = rigid_register(moving, fixed, self.params)
        aligned_path = mask_path = None
        if save_aligned and result.aligned is not None:
            from .io import save_nifti
            pair_dir = self.output_dir / name
            pair_dir.mkdir(exist_ok=True)
            aligned_path = save_nifti(result.aligned, pair_dir / "aligned.nii.gz")
            if result.valid_mask is not None:
                import nibabel as nib
                nib.save(
                    nib.Nifti1Image(result.valid_mask.astype(np.uint8), result.aligned.affine_4x4()),
                    str(pair_dir / "valid_mask.nii.gz"),
                )
                mask_path = str(pair_dir / "valid_mask.nii.gz")
        self.records.append(RunRecord(
            name=name,
            status=result.status.value,
            message=result.message,
            params_fingerprint=self.params.fingerprint(),
            moving_fingerprint=_volume_fingerprint(moving),
            fixed_fingerprint=_volume_fingerprint(fixed),
            result=result.to_dict(),
            aligned_path=aligned_path,
            mask_path=mask_path,
        ))
        return result

    def finalize(self, batch_name: str = "batch") -> Path:
        manifest_path = self.output_dir / "manifest.json"
        previous = None
        if manifest_path.exists():
            previous = json.loads(manifest_path.read_text(encoding="utf-8"))

        prior_fp = previous["params_fingerprint"] if previous else None
        current_fp = self.params.fingerprint()
        if previous is None:
            provenance = "首次运行，无历史参数"
        elif prior_fp == current_fp:
            provenance = f"重跑：沿用与上次相同的参数（指纹 {current_fp}）"
        else:
            provenance = (
                f"重跑：参数已变更（旧指纹 {prior_fp} -> 新指纹 {current_fp}），"
                "本次结果采用新参数"
            )

        manifest = {
            "batch_name": batch_name,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "params": self.params.to_dict(),
            "params_fingerprint": current_fp,
            "provenance": provenance,
            "summary": {
                "total": len(self.records),
                "success": sum(r.status == "success" for r in self.records),
                "failed": sum(r.status != "success" for r in self.records),
            },
            "runs": [asdict(r) for r in self.records],
        }
        if previous is not None:
            manifest["previous_params"] = previous.get("params")
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
        )
        return manifest_path

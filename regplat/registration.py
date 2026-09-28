"""多分辨率刚性配准。

设计原则：
* 相似性在物理空间计算，方向与间距不同的影像也可对齐（不做尺寸拉平）；
* 中心对齐初始化 + 粗到细金字塔 + Powell 无梯度优化；
* 信息不足（全常量/重叠太小）和未收敛都返回明确状态，
  绝不用中间结果冒充成功。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import hashlib
import json
import time
import numpy as np
from scipy.optimize import minimize
from scipy.ndimage import gaussian_filter, zoom
from scipy.spatial.transform import Rotation

from .volume import MedicalVolume
from .transforms import RigidTransform
from .resample import resample, Coverage
from .metrics import AlignmentErrors, alignment_errors


class RegistrationStatus(str, Enum):
    SUCCESS = "success"                # 收敛且达到质量门槛
    DID_NOT_CONVERGE = "did_not_converge"
    INSUFFICIENT_INFORMATION = "insufficient_information"
    INVALID_INPUT = "invalid_input"


@dataclass(frozen=True)
class RegistrationParams:
    """可序列化、可复现的配准参数。"""

    levels: tuple[int, ...] = (1, 2, 4)
    samples_per_axis: int = 28
    min_overlap_fraction: float = 0.25
    min_ncc: float = 0.6
    ftol: float = 1e-5
    max_iter: int = 200
    interpolation_order: int = 1
    init_from_centers: bool = True
    max_translation_mm: float = 150.0
    initial_transform_params: tuple[float, ...] | None = None

    def to_dict(self) -> dict:
        return {
            "method": "rigid_multiresolution_powell",
            "levels": list(self.levels),
            "samples_per_axis": self.samples_per_axis,
            "min_overlap_fraction": self.min_overlap_fraction,
            "min_ncc": self.min_ncc,
            "ftol": self.ftol,
            "max_iter": self.max_iter,
            "interpolation_order": self.interpolation_order,
            "init_from_centers": self.init_from_centers,
            "max_translation_mm": self.max_translation_mm,
            "initial_transform_params": (
                list(self.initial_transform_params)
                if self.initial_transform_params is not None
                else None
            ),
        }

    def fingerprint(self) -> str:
        blob = json.dumps(self.to_dict(), sort_keys=True).encode("utf-8")
        return hashlib.sha256(blob).hexdigest()[:16]


@dataclass
class RegistrationResult:
    status: RegistrationStatus
    transform: RigidTransform | None
    aligned: MedicalVolume | None
    valid_mask: np.ndarray | None
    coverage: Coverage | None
    errors: AlignmentErrors | None
    params: RegistrationParams
    message: str = ""
    iterations: int = 0
    objective_history: list[float] = field(default_factory=list)
    elapsed_seconds: float = 0.0

    @property
    def ok(self) -> bool:
        return self.status is RegistrationStatus.SUCCESS

    def to_dict(self) -> dict:
        d = {
            "status": self.status.value,
            "ok": self.ok,
            "message": self.message,
            "iterations": self.iterations,
            "elapsed_seconds": self.elapsed_seconds,
            "params": self.params.to_dict(),
            "params_fingerprint": self.params.fingerprint(),
        }
        if self.transform is not None:
            d["transform"] = self.transform.to_dict()
        if self.coverage is not None:
            d["coverage"] = self.coverage.to_dict()
        if self.errors is not None:
            d["errors"] = self.errors.to_dict()
        if self.objective_history:
            d["final_objective"] = self.objective_history[-1]
        return d


def _validate(moving: MedicalVolume, fixed: MedicalVolume) -> str | None:
    if not isinstance(moving, MedicalVolume) or not isinstance(fixed, MedicalVolume):
        return "moving 与 fixed 必须是 MedicalVolume"
    if np.nanstd(moving.data) < 1e-9 or np.nanstd(fixed.data) < 1e-9:
        return "影像近乎常量（std≈0），没有可供配准的结构信息"
    if min(fixed.shape) < 8 or min(moving.shape) < 8:
        return "影像每维至少需要 8 个体素才能稳定配准"
    return None


def _downsample(vol: MedicalVolume, factor: int):
    if factor <= 1:
        return vol
    # 各向异性间距下按物理尺度模糊
    sigma = 0.75 * factor * (np.array(vol.spacing) / np.array(vol.spacing).mean())
    blurred = gaussian_filter(vol.data, sigma=sigma, mode="nearest")
    zoom_factors = np.array([1 / factor] * 3)
    small = zoom(blurred, zoom_factors, order=1, mode="nearest")
    new_spacing = tuple(s * factor for s in vol.spacing)
    return MedicalVolume(small, new_spacing, vol.origin, vol.direction)


def _objective(
    p: np.ndarray,
    moving: MedicalVolume,
    fixed: MedicalVolume,
    sample_world: np.ndarray,
    order: int,
    max_translation: float,
) -> float:
    transform = RigidTransform.from_params(p)
    if np.linalg.norm(transform.translation) > max_translation:
        return 1e3 + float(np.linalg.norm(transform.translation) - max_translation)
    moving_world = transform.apply(sample_world)
    moving_vox = moving.world_to_voxel(moving_world)
    fixed_vox = fixed.world_to_voxel(sample_world)
    shape = np.array(moving.shape, dtype=np.float64)
    in_m = np.all((moving_vox >= 0) & (moving_vox <= shape - 1), axis=1)
    fshape = np.array(fixed.shape, dtype=np.float64)
    in_f = np.all((fixed_vox >= 0) & (fixed_vox <= fshape - 1), axis=1)
    valid = in_m & in_f
    overlap = valid.mean()
    if overlap < 0.15:
        return 1e3 * (1.0 - overlap) + 1.0
    from scipy.ndimage import map_coordinates
    a = map_coordinates(moving.data, moving_vox[valid].T, order=order, mode="nearest")
    b = map_coordinates(fixed.data, fixed_vox[valid].T, order=order, mode="nearest")
    a = a - a.mean()
    b = b - b.mean()
    denom = np.sqrt(np.sum(a * a) * np.sum(b * b))
    if denom < 1e-12:
        return 2.0
    ncc = float(np.sum(a * b) / denom)
    # 轻微重叠惩罚，避免退到只在极小区域优化
    return 1.0 - ncc + 0.02 * (1.0 - overlap)


def _sample_grid(vol: MedicalVolume, n: int) -> np.ndarray:
    lo, hi = vol.world_bounds()
    axis = [np.linspace(lo[d] + 0.1 * (hi[d] - lo[d]),
                       hi[d] - 0.1 * (hi[d] - lo[d]), n) for d in range(3)]
    zz, yy, xx = np.meshgrid(axis[2], axis[1], axis[0], indexing="ij")
    return np.stack([zz.ravel(), yy.ravel(), xx.ravel()], axis=1)


def rigid_register(
    moving: MedicalVolume,
    fixed: MedicalVolume,
    params: RegistrationParams | None = None,
) -> RegistrationResult:
    params = params or RegistrationParams()
    t0 = time.monotonic()
    problem = _validate(moving, fixed)
    if problem is not None:
        return RegistrationResult(
            status=RegistrationStatus.INVALID_INPUT
            if "MedicalVolume" in problem else RegistrationStatus.INSUFFICIENT_INFORMATION,
            transform=None, aligned=None, valid_mask=None, coverage=None,
            errors=None, params=params, message=problem,
            elapsed_seconds=time.monotonic() - t0,
        )

    # 初始化：两影像物理中心对齐
    if params.initial_transform_params is not None:
        current = RigidTransform.from_params(np.asarray(params.initial_transform_params, dtype=float))
    elif params.init_from_centers:
        f_center = fixed.voxel_to_world((np.array(fixed.shape, float) - 1) / 2)
        m_center = moving.voxel_to_world((np.array(moving.shape, float) - 1) / 2)
        current = RigidTransform(Rotation.identity(), m_center - f_center)
    else:
        current = RigidTransform.identity()

    history: list[float] = []
    total_iters = 0
    converged_any = False
    x0 = current.to_params()
    for level in sorted(set(params.levels), reverse=True):
        m_small = _downsample(moving, int(level))
        f_small = _downsample(fixed, int(level))
        n = max(10, params.samples_per_axis // max(1, int(round(level ** (2 / 3)))))
        samples = _sample_grid(f_small, n)
        res = minimize(
            _objective, x0,
            args=(m_small, f_small, samples, params.interpolation_order,
                  params.max_translation_mm),
            method="Powell",
            options={"xtol": params.ftol, "ftol": params.ftol,
                     "maxiter": params.max_iter, "disp": False},
        )
        history.append(float(res.fun))
        total_iters += int(getattr(res, "nit", 0))
        x0 = np.asarray(res.x, dtype=float)
        if not np.all(np.isfinite(x0)):
            return RegistrationResult(
                status=RegistrationStatus.DID_NOT_CONVERGE,
                transform=None, aligned=None, valid_mask=None, coverage=None,
                errors=None, params=params,
                message=f"第 {level} 分辨率层优化产生非有限参数",
                iterations=total_iters, objective_history=history,
                elapsed_seconds=time.monotonic() - t0,
            )
        converged_any = converged_any or bool(res.success)

    transform = RigidTransform.from_params(x0)

    # 全分辨率最终评估
    r = resample(moving, fixed, transform, order=params.interpolation_order)
    insufficient = None
    if r.coverage.fraction < params.min_overlap_fraction:
        insufficient = (
            f"覆盖比例 {r.coverage.fraction:.1%} 低于阈值 "
            f"{params.min_overlap_fraction:.0%}，对齐结果不可信"
        )
    try:
        errors = alignment_errors(r.aligned, fixed, r.valid_mask)
    except ValueError as exc:
        errors = None
        insufficient = str(exc)

    elapsed = time.monotonic() - t0
    if insufficient is not None:
        return RegistrationResult(
            status=RegistrationStatus.INSUFFICIENT_INFORMATION,
            transform=transform, aligned=r.aligned, valid_mask=r.valid_mask,
            coverage=r.coverage, errors=errors, params=params,
            message=insufficient, iterations=total_iters,
            objective_history=history, elapsed_seconds=elapsed,
        )
    if not converged_any:
        return RegistrationResult(
            status=RegistrationStatus.DID_NOT_CONVERGE,
            transform=transform, aligned=r.aligned, valid_mask=r.valid_mask,
            coverage=r.coverage, errors=errors, params=params,
            message="优化器在所有分辨率层均未报告收敛",
            iterations=total_iters, objective_history=history,
            elapsed_seconds=elapsed,
        )
    if errors.ncc < params.min_ncc:
        return RegistrationResult(
            status=RegistrationStatus.DID_NOT_CONVERGE,
            transform=transform, aligned=r.aligned, valid_mask=r.valid_mask,
            coverage=r.coverage, errors=errors, params=params,
            message=f"最终 NCC={errors.ncc:.3f} 低于阈值 {params.min_ncc}，视为未收敛",
            iterations=total_iters, objective_history=history,
            elapsed_seconds=elapsed,
        )
    return RegistrationResult(
        status=RegistrationStatus.SUCCESS,
        transform=transform, aligned=r.aligned, valid_mask=r.valid_mask,
        coverage=r.coverage, errors=errors, params=params,
        message="刚性配准收敛",
        iterations=total_iters, objective_history=history,
        elapsed_seconds=elapsed,
    )

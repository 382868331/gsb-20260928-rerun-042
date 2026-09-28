"""医学图像配准平台基础包。

当前提供：带方向/间距/原点的影像模型、刚性变换、重采样、
刚性配准（含明确收敛状态）以及带参数溯源的处理批次。
非刚性方法将在后续版本基于同一影像/变换/批次基础扩展。
"""

from .volume import MedicalVolume
from .transforms import RigidTransform
from .resample import resample
from .registration import (
    RegistrationResult,
    RegistrationParams,
    RegistrationStatus,
    rigid_register,
)
from .batch import BatchRunner, RunRecord

__all__ = [
    "MedicalVolume",
    "RigidTransform",
    "resample",
    "rigid_register",
    "RegistrationResult",
    "RegistrationParams",
    "RegistrationStatus",
    "BatchRunner",
    "RunRecord",
]

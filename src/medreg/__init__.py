"""Medical image registration platform."""

from .image import MedicalImage
from .multi import InitializationSpec, MultiRegistrationResult, register_multi
from .register import RegistrationResult, evaluate_transform, register_rigid
from .roi import RegionOfInterest
from .transform import RigidTransform

__all__ = [
    "MedicalImage",
    "RigidTransform",
    "RegionOfInterest",
    "RegistrationResult",
    "MultiRegistrationResult",
    "InitializationSpec",
    "register_rigid",
    "register_multi",
    "evaluate_transform",
]

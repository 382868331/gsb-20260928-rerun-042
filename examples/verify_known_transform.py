"""用已知位移与旋转的合成体数据核对刚性配准精度。

运行：python examples/verify_known_transform.py
"""

from __future__ import annotations
import numpy as np

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from regplat import rigid_register, RegistrationParams
from regplat.metrics import point_target_error
from tests.synthetic import known_pair


def main() -> int:
    moving, fixed, gt = known_pair(translation=(5.0, -3.0, 4.0), angle_deg=12.0)
    print(f"fixed  shape={fixed.shape} spacing={fixed.spacing} origin={tuple(round(v,2) for v in fixed.origin)}")
    print(f"moving shape={moving.shape} spacing={moving.spacing}")
    print(f"真值: 角度=12.00° 平移=[5,-3,4]mm")

    params = RegistrationParams(levels=(1, 2, 4), samples_per_axis=26)
    result = rigid_register(moving, fixed, params)
    print(f"\n状态: {result.status.value} — {result.message}")
    if not result.ok:
        return 1
    print(f"恢复: 角度={np.degrees(result.transform.rotation.magnitude()):.2f}° "
          f"平移={np.round(result.transform.translation, 3).tolist()}mm")
    print(f"参数误差: 平移={result.transform.translation_difference(gt):.3f}mm "
          f"旋转={result.transform.angular_difference(gt):.3f}°")
    print(f"覆盖: {result.coverage.fraction:.1%}  NCC={result.errors.ncc:.4f}  "
          f"RMSE={result.errors.rmse:.4f}")

    pts_fixed = np.array([[-15, -15, -15], [15, 15, 15], [0, 10, -10], [-10, 5, 8]], float)
    tre = point_target_error(pts_fixed, gt.apply(pts_fixed), result.transform)
    print(f"目标配准误差(TRE): mean={tre['tre_mean_mm']:.3f}mm max={tre['tre_max_mm']:.3f}mm")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

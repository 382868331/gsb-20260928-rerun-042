# regplat — 医学图像配准平台（刚性对齐基础）

先完成两组人工体数据的**刚性对齐**，并搭好后续扩展非刚性方法所需的三层基础：

1. **影像层** `regplat/volume.py`：`MedicalVolume` 携带数据、间距 `spacing`、
   原点 `origin` 和方向余弦 `direction`。物理坐标
   `world = origin + direction·diag(spacing)·voxel`。**不把数组尺寸拉成一样**——
   方向、间距、原点不同的影像在物理空间对齐。
2. **变换层** `regplat/transforms.py`：`RigidTransform`（旋转向量参数化 + mm 平移），
   支持复合、求逆、参数序列化。非刚性方法将实现同一接口。
3. **处理批次层** `regplat/batch.py`：多对影像统一参数运行，产物与
   `manifest.json` 落盘；参数带 SHA-256 指纹，重跑时明确解释沿用或变更了哪份参数。

## 结果包含什么

每次配准返回 `RegistrationResult`：

- `aligned`：变换并重采样到 fixed 网格的数据；域外为 `NaN`，并给 `valid_mask`；
  **原始 moving/fixed 影像不被修改**。
- `coverage`：有效体素数、覆盖比例、物理交集体积(mm³)、缺失比例。
- `errors`：MAE/RMSE、归一化互相关 NCC、强度差，仅在有效重叠区统计。
- `transform`：恢复出的刚性参数（旋转矩阵、旋转角、平移 mm）。
- `status`：**明确状态**，失败绝不冒充成功：
  - `success`：优化收敛且 NCC/覆盖达到阈值；
  - `did_not_converge`：优化器未收敛或最终 NCC 低于阈值；
  - `insufficient_information`：影像常量、重叠/有效体素太少；
  - `invalid_input`：输入不合法。

## 方法

多分辨率（默认降采样层 4×/2×/1×）Powell 无梯度优化，最大化重叠区 NCC，
物理中心对齐初始化，可选初始变换、平移上限和各阈值（见 `RegistrationParams`）。

## 使用

```python
from regplat import rigid_register, RegistrationParams
from regplat.io import load_nifti, save_nifti

moving = load_nifti("moving.nii.gz")
fixed = load_nifti("fixed.nii.gz")
result = rigid_register(moving, fixed, RegistrationParams())
if result.ok:
    save_nifti(result.aligned, "aligned.nii.gz")
    print(result.coverage.fraction, result.errors.ncc, result.transform.to_dict())
else:
    print(result.status.value, result.message)
```

命令行与批次溯源：

```bash
python -m regplat run --moving m1.nii.gz --fixed f1.nii.gz --name p0 --out out/
python -m regplat run --moving m1.nii.gz --fixed f1.nii.gz --name p0 --out out/  # 重跑
python -m regplat explain out/manifest.json
```

`manifest.json` 中 `provenance` 会写明「沿用相同参数（指纹 …）」或
「参数已变更（旧指纹 … -> 新指纹 …）」。

## 核对：已知位移与旋转

`tests/synthetic.py` 在物理空间构造含多个高斯球的体数据，并生成**间距各向异性、
体素轴旋转 90°、原点不同**的一对影像，施加已知 `(平移, 旋转)`。核对结果：

```bash
python examples/verify_known_transform.py
pytest -q
```

真值 12°、[5,-3,4]mm 时，恢复约 11.96°、[5.01,-2.99,3.99]mm，NCC≈1.0。
测试还覆盖：不同方向/间距几何、常量影像与零重叠（信息不足）、
结构无关（未收敛）、原始影像不变性、批次参数指纹与重跑溯源。

## 后续扩展点

非刚性配准只需新增与 `RigidTransform` 同接口的形变变换（位移场/B-spline），
复用 `MedicalVolume`、`resample`、覆盖/误差统计与 `BatchRunner` 溯源机制。

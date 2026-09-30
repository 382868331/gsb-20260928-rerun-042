# 医学图像刚性配准平台

在真实物理坐标中对两组人工体数据做六自由度刚性对齐，支持用户关注区域（ROI）约束、多组初始位置比较，以及参数/人工选择的版本化留痕。

## 设计要点

- `MedicalImage` 同时保存三维数组和 4×4 index-to-world affine，支持不同 spacing、origin 和方向（含轴排列与翻转）。
- `RigidTransform` 在世界坐标中表示旋转和平移，参数顺序为 `[rx, ry, rz, tx, ty, tz]`。
- 重采样在固定影像网格上进行，并返回有效重叠掩膜；不会通过拉伸数组尺寸冒充对齐。
- `RegionOfInterest`（`roi.py`）以物理世界盒子描述用户选定区域，可由索引盒子构造；每个候选同时报告**全影像**与 **ROI 内**两套误差（mse/rmse/mae/ncc）和覆盖率，大片背景一致无法掩盖目标错位。
- 优化目标以 ROI 内 NCC 为主（0.75）、全局 NCC 为辅（0.25），并要求 ROI 覆盖率与 ROI NCC 同时达标。
- `register_multi` 从多组初始位置分别优化，保留**全部候选**（成功、失败、歧义）。相近分数但几何不一致（平移/旋转超阈值）或 ROI 未完全覆盖时标记 `uncertain` 并给出原因，不静默二选一。
- 失败候选保持 `failed` 状态和诊断信息，不写出变换影像；人工选择只能选择通过验收的候选（`apply_manual_selection` 对失败/不存在的候选直接报错）。
- `run_batch` 采用 schema v2：参数哈希变化或 `manual_candidate_id` 变化都会生成新的版本目录（时间戳_参数哈希_选择方式），旧版本不覆盖；每项目另有 `versions.json` 台账。
- 人工体数据支持部分重叠视场（`partial_overlap`）、不同方向矩阵（`moving_direction` / `reoriented_direction`）和无信息平坦体（`uniform_volume`），用于核对物理位置与失败路径。

## 运行

```bash
PYTHONPATH=src .venv/bin/python -m medreg.cli output/demo \
  --parameters examples/synthetic_parameters.json
```

每次运行在 `output/demo/<item>/<version_id>/` 下另存：

- `fixed_original.nii.gz`、`moving_original.nii.gz`
- 每个候选各自的 `<candidate>_moving_transformed.nii.gz` 与 `<candidate>_overlap_mask.npy`（失败候选为缺省，不产文件）
- 自动/人工选定结果 `moving_transformed_selected.nii.gz`、`overlap_mask_selected.npy`
- 上级 `batch_report.json` 与每项目 `versions.json`（参数哈希、选择方式、不确定性原因）

## 验证

```bash
.venv/bin/python -m pytest -q
```

测试覆盖：已知变换恢复、ROI 与全局误差/覆盖分离、背景一致但 ROI 错位被拒、多初始候选保留与不确定性、物理空间不一致度量、部分重叠且方向不同的配准、无信息区域失败且不写成功影像、失败候选禁止人工选择、参数修改与人工选择另存新版本。

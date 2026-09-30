# 医学图像刚性配准平台

在真实物理坐标中完成两组人工体数据的六自由度刚性对齐，并在全影像指标之外强制局部关注区域（ROI）约束、多初始位置候选比较和版本化留档。

## 设计要点

- `MedicalImage` 同时保存三维数组和 4×4 index-to-world affine，支持不同 spacing、origin 和方向。
- `RigidTransform` 在世界坐标中表示旋转和平移，参数顺序为 `[rx, ry, rz, tx, ty, tz]`。
- 重采样在固定影像网格上进行，并返回有效重叠掩膜；不会通过简单拉伸数组尺寸冒充对齐。
- `RegionOfInterest` 以物理坐标（毫米）的轴对齐包围盒定义，可用 `center/radius` 或 `world_min/world_max` 配置；在不同 spacing、origin、方向下仍指向同一解剖位置。
- 验收同时报告并检查两组指标：
  - 全影像：NCC、MSE/RMSE/MAE、覆盖率及世界坐标包围盒；
  - ROI：ROI 内 NCC/误差与 ROI 覆盖率。大片背景一致但目标错位的候选会因 ROI NCC/覆盖率不达标而失败。
- 多分辨率 Powell 优化使用全影像与 ROI 联合 NCC（ROI 权重更高，覆盖不足直接惩罚）。
- `register_multi` 为每个初始位置保留独立候选，记录参数、指标、覆盖和诊断：
  - `agreed`：近优候选分数接近且变换在旋转/平移容限内一致；
  - `uncertain`：分数相近但变换不一致，全部候选保留并在 `uncertainty` 中说明最大旋转/平移差异；
  - `selected_manual`：人工指定候选；
  - `failed`：没有通过验收的候选。
- `evaluate_transform` 可直接评估给定变换（不做优化），用于用已知变换核对物理位置；`lock_initial` 可锁定候选初始值用于人工复核。
- 失败候选保留诊断信息和候选参数，但状态为 `failed`，不产出成功变换影像和掩膜；只有被选中且通过验收的候选写出 `moving_transformed.nii.gz`。
- 批次始终另存原始 fixed/moving；有 ROI 时另存 `roi_mask.npy`。每次运行都在 `versions/v0001、v0002 ……` 式目录下另存，不覆盖旧版本，`versions/versions.json` 记录 `initial/rerun/parameter_change/manual_selection` 触发原因、自动/人工选择方式、参数来源、参数哈希和报告路径，`LATEST` 指向最新版本。
- `batch_report.json` 记录每项实际生效参数、参数哈希、参数来源、全部候选和选中候选及输出路径。

## 运行

```bash
PYTHONPATH=src .venv/bin/python -m medreg.cli output/demo \
  --parameters examples/synthetic_parameters.json
```

参数文件可配置多个 `initializations`（每个候选独立 id、起始参数、是否锁定）、`roi`、ROI 阈值与 `manual_candidate_id`。修改参数或设置人工选择后再次运行，会自动另存新版本。

每个版本目录包括：

- `fixed_original.nii.gz`
- `moving_original.nii.gz`
- `moving_transformed.nii.gz`（仅选中且成功的候选）
- `overlap_mask.npy`（仅选中且成功的候选）
- `roi_mask.npy`（配置 ROI 时）
- `batch_report.json`

## 验证

```bash
.venv/bin/python -m pytest -q
```

测试使用已知旋转和平移生成不同间距、部分重叠和不同方向的体数据，覆盖：

- 已知变换的物理位置恢复和覆盖率/NCC；
- ROI 错位时全影像通过但 ROI 拒绝，正确对齐时 ROI 通过；
- 部分重叠与 40° 方向差异下的物理对齐；
- 无信息区域失败且不产出成功影像；
- 相似但不一致候选的 `uncertain` 判定与候选保留；
- 参数修改、重跑和人工选择分别另存版本；
- 人工选择失败候选时直接拒绝，不产生半成品版本。

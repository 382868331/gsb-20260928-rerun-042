# 医学图像刚性配准平台

当前版本先实现两组人工体数据在真实物理坐标中的六自由度刚性对齐，为后续非刚性配准保留影像、变换和处理批次边界。

## 设计要点

- `MedicalImage` 同时保存三维数组和 4×4 index-to-world affine，支持不同 spacing、origin 和方向。
- `RigidTransform` 在世界坐标中表示旋转和平移，参数顺序为 `[rx, ry, rz, tx, ty, tz]`。
- 重采样在固定影像网格上进行，并返回有效重叠掩膜；不会通过简单拉伸数组尺寸冒充对齐。
- 多分辨率 Powell 优化使用 NCC。只有优化器成功、覆盖率、NCC 和有限参数同时满足条件才标记 `success`。
- 失败结果保留 `failed` 状态、诊断信息、覆盖率和误差，不写出伪造成成功结果的变换影像。
- 批次始终另存原始 fixed/moving；成功时另存 transformed 数据和 overlap mask。
- `batch_report.json` 记录每项实际生效参数、参数哈希、参数来源和输出路径。重跑同一参数文件可通过哈希解释采用了哪份参数。

## 运行

```bash
PYTHONPATH=src .venv/bin/python -m medreg.cli output/demo \
  --parameters examples/synthetic_parameters.json
```

输出包括：

- `fixed_original.nii.gz`
- `moving_original.nii.gz`
- `moving_transformed.nii.gz`
- `overlap_mask.npy`
- `batch_report.json`

## 验证

```bash
.venv/bin/python -m pytest -q
```

测试使用已知旋转和平移生成不同间距与覆盖范围的体数据，并检查估计点变换、覆盖率、NCC 和批次参数哈希。

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


## 区域传播（propagation）

- `propagation.py` 的 `PropagationSession` 在已完成的时间序列配准之上，把某一时刻（默认为参照时刻）人工描出的区域逐时刻传播。每个目标时刻各自记录采用的配准候选、覆盖率与区域结果；失败或覆盖不足的时刻保持 `pending` 并写明原因，绝不为了凑齐轮廓而跨过缺失影像插值。
- 所有区域掩膜都落在参照影像网格上（与各时刻变换后影像同一网格），因此任意时刻的区域都能与原描画直接比较；传播只保留真实落在有效成像范围内的体素，覆盖不足时保留部分掩膜供查看但状态仍为 `pending`。
- `apply_correction(label, mask, use_as_basis=...)` 在目标时刻保存人工修正；选择作为新依据时，其后的时刻从修正结果重新传播，之前的时刻不受影响。`revert_correction` 撤回修正后自动结果按原依据重算，原描画永不被覆盖（`draw_region` 只允许一次）。
- `preview_candidate_change` 在重新选择配准候选之前列出哪些自动结果需要更新、哪些人工修正会冲突；`apply_candidate_change` 保留全部人工结果、只重算自动传播。
- `lineage()` 给出可见的传播关系（来源时刻、依据类型、候选、变换参数）；`change_curve()` 输出逐时刻区域体积，区分 original / auto / manual，pending 时刻为显式缺口（volume_mm3=None），不插值。
- `propagation_store.py` 的 `save_propagation` / `load_propagation` / `verify_propagation` 把原描画、每时刻区域掩膜（NIfTI）、修正、采用候选与变换参数、变化曲线落盘为 `propagation_report.json`；重开后自动传播一律依据序列重算并逐一核对状态、覆盖率、掩膜哈希与曲线，发现不一致即报错。

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

## 时间序列配准（series）

- `series.py` 的 `register_series` 以用户指定的 `reference_index` 时刻为参照，逐时刻独立配准：每个时刻保留自己的候选、覆盖率与误差，某个时刻失败只记为 `failed`/`rejected` 并在邻接报告中形成显式 `gap`，不会拖累或改写其他时刻的结果。
- 相邻时刻的选定变换用物理空间差异（平移 mm / 旋转 deg）比较，超过 `jump_translation_mm` / `jump_rotation_deg` 时给出 `jump` 提示；失败时刻之间不做任何插值，不用连续曲线掩盖缺口。
- `select_series_candidate` 允许人工在某一时刻改选另一个通过验收的候选，随后自动重算整条邻接提示，便于重新查看整段变化；失败候选与参照时刻不可被选择。
- ROI 在参照影像上无体素（`roi_empty_on_reference`）、ROI 内影像平坦无信息（`roi_has_no_image_information`）、或 ROI 覆盖不足（`roi_coverage_insufficient`）时，相关时刻明确 `rejected` 并记录原因。
- `series_batch.py` 的 `run_series_batch` 按 `<series>/<version_id>/<label>/` 落盘：参照原影像、每时刻 `moving_original.nii.gz`、每候选变换影像、选定结果与 `series_report.json`；人工选择或参数变化生成新版本目录并追加 `versions.json`。`verify_series_version` 重开版本目录，逐一核对候选选择、原影像与各时刻变换的一一对应，发现不一致即报错。

```bash
PYTHONPATH=src .venv/bin/python -m medreg.cli output/series --series 4 \
  --parameters examples/series_parameters.json
```

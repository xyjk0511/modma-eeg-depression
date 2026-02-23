# Phase 12: Feature Importance - Research

**Researched:** 2026-02-23
**Domain:** PCA back-projection, MNE topomap visualization
**Confidence:** HIGH

## Summary

Phase 12 back-projects LOSO fold LR coefficients from PCA space to channel space, producing interpretable 128-channel weight maps. The math is: each fold's `coef_` (shape 20,) multiplied by `pca.components_.T` (shape 128×20) yields a 128-dim channel weight vector. Averaging absolute weights across folds gives a stable ranking.

The two models are: hcue full-window (Phase 8, 128-dim P300 mean, PCA=20, C=1.0) and 250-300ms bin (Phase 11, 128-dim bin mean, PCA=20, C=0.1). Both use the same LOSO+PCA+LR pipeline already in `run_modma_erp.py`. Implementation is a new function that re-runs LOSO while collecting coef_ per fold.

MNE's `mne.viz.plot_topomap` accepts an `Info` object directly as `pos` — build `Info` from `GSN-HydroCel-128` montage (already used in Phase 10). The `vlim` parameter controls colorbar range; symmetric `(-vmax, vmax)` is recommended for RdBu_r. The `names` parameter (list of 128 strings) handles top-5 channel label annotation cleanly.

**Primary recommendation:** Add `run_feature_importance()` to `run_modma_erp.py` — re-run LOSO for both models collecting coef_ per fold, back-project, average, rank, plot side-by-side topomap, save CSV + PNG to `out_phase12/`.


---

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

**反投影来源**
- 两个模型都做：hcue 全窗口（Phase 8，BA=0.670）和 250-300ms bin（Phase 11，BA=0.673）
- 每折独立 PCA（与现有 LOSO 逻辑一致，不共享 PCA 对象）
- 两套独立 LOSO 重新运行，各自提取 coef_
- 聚合方式：每折先反投影（pca.components_.T @ coef_）得到 128-dim 权重，再跨折取平均绝对值排名

**权重排名输出**
- 打印格式：格式化表格（Rank / Channel / Avg|weight| / Sign）
- 保留正负符号（正=MDD偏高，负=HC偏高）
- 同时保存完整 128 通道排名到 CSV
- 两个模型各自独立输出表格和 CSV

**Topomap 可视化**
- 颜色映射：RdBu_r 发散色图（正权重红色=MDD偏高，负权重蓝色=HC偏高）
- 标注 top-5 通道名（在对应位置显示通道标签）
- 两个模型并排对比图，保存为单文件 PNG
- 通道位置：从数据集真实坐标文件读取（GSN-HydroCel-128 montage）

**输出目录与文件**
- 输出目录：out_phase12/
- CSV 命名：feature_importance_hcue.csv / feature_importance_bin250.csv
- Topomap PNG：topomap_hcue_vs_bin.png（并排单文件）
- 代码组织：独立入口函数 run_feature_importance()，在 __main__ 中调用

### Claude's Discretion
- colorbar 范围（对称 vmin/vmax 或自动）
- 表格列宽与对齐细节
- 通道坐标文件的具体路径查找逻辑

### Deferred Ideas (OUT OF SCOPE)
None — discussion stayed within phase scope
</user_constraints>

---

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| FIMP-01 | 从 LOSO 各折提取 LR coef_，通过 PCA 反投影（pca.components_.T @ coef_）到通道空间，输出 128 通道权重排名 | Back-projection math verified; MNE plot_topomap API confirmed in project env; existing LOSO pipeline reusable |
</phase_requirements>


---

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| mne | installed | `plot_topomap`, `make_standard_montage`, `create_info` | Already used; GSN-HydroCel-128 montage confirmed working in Phase 10 |
| matplotlib | installed | Side-by-side figure, colorbar | Already used in Phase 11 |
| numpy | installed | Back-projection math, averaging | Already used throughout |
| sklearn | installed | Pipeline components (StandardScaler, PCA, LR) | Same as existing LOSO |

**No new dependencies required.**

## Architecture Patterns

### Pattern 1: LOSO with coef_ collection

The existing `run_loso()` and `_loso_ba()` do not return per-fold model objects. Phase 12 needs a new helper that collects the back-projected weight per fold.

```python
def _loso_collect_coefs(X, y, n_components=20, C=1.0):
    """LOSO returning per-fold back-projected channel weights (n_folds, 128)."""
    loo = LeaveOneOut()
    fold_weights = []
    for tr_idx, _ in loo.split(X):
        scaler = StandardScaler()
        pca = PCA(n_components=n_components)
        clf = LogisticRegression(C=C, max_iter=2000, class_weight="balanced")
        X_pca = pca.fit_transform(scaler.fit_transform(X[tr_idx]))
        clf.fit(X_pca, y[tr_idx])
        # coef_[0] shape (n_components,); components_.T shape (128, n_components)
        fold_weights.append(pca.components_.T @ clf.coef_[0])  # (128,)
    return np.array(fold_weights)  # (n_folds, 128)
```

### Pattern 2: Aggregation and ranking

```python
avg_weights = fold_weights.mean(axis=0)          # (128,) signed
avg_abs     = np.abs(fold_weights).mean(axis=0)  # (128,) mean absolute
rank_idx    = np.argsort(avg_abs)[::-1]          # descending by importance
```

Sign from `avg_weights[rank_idx]`: positive = MDD-higher, negative = HC-higher.

### Pattern 3: MNE topomap with Info (verified in project env)

```python
# plot_topomap signature confirmed: vlim=(None,None), names=None, axes=None
montage = mne.channels.make_standard_montage("GSN-HydroCel-128")
info = mne.create_info(montage.ch_names[:128], sfreq=1000, ch_types="eeg")
info.set_montage(montage)

# Top-5 labels via names parameter (length must equal 128)
names_list = [""] * 128
for i in range(5):
    names_list[rank_idx[i]] = montage.ch_names[rank_idx[i]]

vmax = np.abs(avg_weights).max()
im, _ = mne.viz.plot_topomap(
    avg_weights, info,
    cmap="RdBu_r", vlim=(-vmax, vmax),
    names=names_list,
    axes=ax, show=False
)
plt.colorbar(im, ax=ax, shrink=0.7)
```

### Pattern 4: Side-by-side figure

```python
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
# plot hcue in axes[0], bin250 in axes[1]
fig.savefig(out_dir / "topomap_hcue_vs_bin.png", dpi=150, bbox_inches="tight")
plt.close(fig)
```

### Anti-Patterns to Avoid

- **`clf.coef_` shape `(1, 20)` in matmul:** Binary LR coef_ is 2D — use `clf.coef_[0]` (shape `(20,)`).
- **Sharing PCA across folds:** Each fold fits its own PCA on training data only (project convention).
- **Manual 3D→2D coordinate projection for annotation:** Use `names` parameter instead; MNE handles coordinate projection internally.
- **`plt.show()` without Agg backend:** Use `matplotlib.use("Agg")` before pyplot import (already done in `plot_ttest_curve`).


## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Topomap rendering | Custom matplotlib scatter/contour | `mne.viz.plot_topomap` | Handles head outline, spherical interpolation, EEG coordinate projection |
| Channel 2D positions | Manual 3D→2D projection | Pass `Info` to `plot_topomap` | MNE handles projection internally; coordinate system matches montage |
| Montage lookup | Custom coordinate file parsing | `make_standard_montage("GSN-HydroCel-128")` | Already used in Phase 10; confirmed working |

## Common Pitfalls

### Pitfall 1: coef_ shape for binary LR
**What goes wrong:** `pca.components_.T @ clf.coef_` fails — `coef_` is `(1, 20)` not `(20,)`.
**How to avoid:** Use `clf.coef_[0]` (shape `(20,)`) in the matmul.

### Pitfall 2: bin250 uses C=0.1 not C=1.0
**What goes wrong:** Using C=1.0 for bin250 produces different regularization than Phase 11's `run_bin_ablation` (which uses C=0.1).
**How to avoid:** Pass `C=0.1` to `_loso_collect_coefs` for the bin250 model.

### Pitfall 3: names list length must equal 128
**What goes wrong:** `plot_topomap` raises error if `names` length != `len(data)`.
**How to avoid:** Initialize `names_list = [""] * 128` then fill top-5 indices.

### Pitfall 4: plot_topomap return value
**What goes wrong:** `plot_topomap` returns `(im, contours_obj)` — single-value unpacking fails.
**How to avoid:** `im, _ = mne.viz.plot_topomap(...)` then `plt.colorbar(im, ax=ax)`.

## Code Examples

### Complete run_feature_importance skeleton
```python
def run_feature_importance():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = Path("out_phase12")
    out_dir.mkdir(exist_ok=True)

    # Load data
    global CONDITION
    CONDITION = "hcue"
    X, y, ids = load_erp_features()

    # Build Info once (reused for both topomaps)
    montage = mne.channels.make_standard_montage("GSN-HydroCel-128")
    info = mne.create_info(montage.ch_names[:128], sfreq=1000, ch_types="eeg")
    info.set_montage(montage)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    for ax, (label, C, X_model) in zip(axes, [
        ("hcue", 1.0, X),
        ("bin250", 0.1, _get_bin250_features(X, y)),  # extract 250-300ms bin
    ]):
        fold_weights = _loso_collect_coefs(X_model, y, n_components=20, C=C)
        avg_weights = fold_weights.mean(axis=0)
        avg_abs = np.abs(fold_weights).mean(axis=0)
        rank_idx = np.argsort(avg_abs)[::-1]

        _print_and_save_ranking(rank_idx, avg_weights, avg_abs,
                                montage.ch_names[:128], label, out_dir)

        names_list = [""] * 128
        for i in range(5):
            names_list[rank_idx[i]] = montage.ch_names[rank_idx[i]]

        vmax = np.abs(avg_weights).max()
        im, _ = mne.viz.plot_topomap(
            avg_weights, info, cmap="RdBu_r", vlim=(-vmax, vmax),
            names=names_list, axes=ax, show=False
        )
        ax.set_title(f"Feature Importance: {label}")
        plt.colorbar(im, ax=ax, shrink=0.7)

    fig.savefig(out_dir / "topomap_hcue_vs_bin.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_dir}/topomap_hcue_vs_bin.png")
```

## Open Questions

1. **bin250 n_components**
   - What we know: `_loso_ba_subset` caps `n_components = min(X.shape[1], len(y)-1)`. For 128-dim with ~52 subjects that gives 51, not 20.
   - What's unclear: Phase 11 used `permutation_test_subset` → `_loso_ba_subset` (51 components). Phase 12 should use 20 for consistency with Phase 8.
   - Recommendation: Use `n_components=20` for both models in Phase 12. Document explicitly.

2. **Colorbar range (Claude's discretion)**
   - Recommendation: Symmetric `vlim=(-vmax, vmax)` where `vmax = np.abs(avg_weights).max()`. Zero maps to white in RdBu_r; scale is interpretable.

## Sources

### Primary (HIGH confidence)
- Installed MNE — `plot_topomap` signature verified via `help()` in project environment (params: data, pos, cmap, vlim, names, axes, show)
- `run_modma_erp.py` — existing LOSO pipeline, C values per model, PCA n_components=20
- Phase 10 `get_parietal_indices()` — confirms `GSN-HydroCel-128` montage works in this project

### Secondary (MEDIUM confidence)
- Phase 11 `run_bin_ablation()` — confirms C=0.1 for 128-dim bin models

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — all libraries installed and used in project
- Architecture: HIGH — back-projection math standard; MNE API verified in environment
- Pitfalls: HIGH — coef_ shape and C value issues derived directly from existing code

**Research date:** 2026-02-23
**Valid until:** 2026-03-25

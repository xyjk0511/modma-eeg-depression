# Phase 16: Preprocessing Deferred Items - Research

**Researched:** 2026-02-24
**Domain:** EEG preprocessing (bad channel detection, data caching)
**Confidence:** HIGH

## Summary

Phase 16 closes two deferred items from Phase 2: (1) replacing amplitude-only bad channel detection with pyprep NoisyChannels (PRE-02), and (2) extending .npz caching to the ERP pipeline (ARCH-01).

The resting-state pipeline (`modma_mdd_real_experiment.py`) already has .npz caching and amplitude-based bad channel detection with spherical spline interpolation. The ERP pipeline (`run_modma_erp.py`) has neither — it reads `.raw` EGI files fresh every run with no bad channel detection beyond epoch rejection (`reject=dict(eeg=150e-6)`). The pyprep integration targets the resting-state pipeline's `load_windows()` function, replacing lines 253-256 (amplitude threshold detection) with NoisyChannels multi-criteria detection. The .npz cache for ERP targets `load_erp_features()` and `load_erp_timeseries()` in `run_modma_erp.py`.

**Primary recommendation:** Install pyprep 0.5.0, integrate NoisyChannels into `load_windows()` with `do_detrend=False` (data is already highpass filtered at 1.0 Hz), run all four detection methods, then interpolate. For ERP caching, save post-filter/post-epoch averaged data as .npz keyed by filter params hash.

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- 先用 pyprep NoisyChannels 全部方法（deviation, correlation, high-frequency noise, RANSAC）跑一遍，根据结果再决定是否裁剪方法子集
- 检测到的坏通道用球面样条插值修复（MNE 默认方法）
- 单个受试者超过 20% 通道被标记为坏时，整个受试者排除
- 只替换现有管线中的坏通道检测步骤，ICA 等其他预处理流程不变
- 缓存 EDF 加载+滤波后、ICA 之前的数据为 .npz 格式
- 缓存文件存放在 data/modma/cache/ 目录下
- 在 .npz 文件名或元数据中记录滤波参数哈希，参数变化时自动重建缓存
- 添加 --no-cache 命令行参数支持强制跳过缓存
- 优先保证受试者数 >= 35 人
- 自适应阈值：先用 20% 跑一遍，如果剩余人数 < 35，逐步放宽到 25%、30%
- 输出每个受试者的详细坏通道检测日志（坏通道数、使用的阈值、排除原因）
- 主验收：pyprep 预处理后 LOSO CV BA >= 0.670（不低于当前基线）
- 次验收兜底：若 0.640 <= BA < 0.670，必须同时满足"受试者保留率显著提升且组间保留差更小"才可接受
- A/B 对比：旧方法 vs pyprep 并排运行，生成结构化对比报告（BA、AUC、受试者数、坏通道统计）
- 未达标时自动回退到旧方法，代码保留两套路径

### Claude's Discretion
- pyprep 具体参数调优（如 RANSAC 的 fraction_bad 等）
- 缓存文件的具体命名格式和哈希算法选择
- 对比报告的具体字段和排版

### Deferred Ideas (OUT OF SCOPE)
None — discussion stayed within phase scope
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| PRE-02 | 集成 pyprep 进行 PREP 标准坏通道检测（替代纯振幅阈值） | pyprep 0.5.0 NoisyChannels API verified; integrates into `load_windows()` lines 253-264 |
| ARCH-01 | EDF 加载后缓存为 .npz，后续迭代 <1s | Resting-state already has cache; ERP pipeline needs new cache in `load_erp_features()` / `load_erp_timeseries()` |
</phase_requirements>

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| pyprep | 0.5.0 | PREP pipeline bad channel detection | Only Python implementation of PREP standard; used by MNE-BIDS-pipeline |
| mne | 1.11.0 (installed) | EEG I/O, montage, interpolation | Already in use; `interpolate_bads()` for spherical spline |
| numpy | 2.4.2 (installed) | .npz cache format, array ops | Already in use; `np.savez` / `np.load` for caching |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| hashlib | stdlib | Cache key generation | SHA-256 of filter params for cache invalidation |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| pyprep NoisyChannels | MNE `find_bad_channels_maxwell()` | Maxwell filtering is for MEG, not EEG — not applicable |
| pyprep NoisyChannels | autoreject | autoreject works at epoch level, not raw channel level; different scope |

**Installation:**
```bash
pip install pyprep
```

## Architecture Patterns

### Affected Files and Functions

```
scripts/modma/
├── modma_mdd_real_experiment.py   # PRE-02: replace bad channel detection in load_windows()
│                                  # ARCH-01: cache already exists, update cache key for pyprep
├── run_modma_erp.py               # ARCH-01: add .npz cache to load_erp_features() / load_erp_timeseries()
```

### Pattern 1: pyprep NoisyChannels Integration

**What:** Replace amplitude-threshold bad channel detection with pyprep multi-criteria detection.
**Where:** `modma_mdd_real_experiment.py`, `load_windows()`, lines 253-264.
**Current code (to replace):**
```python
# Detect bad channels on Raw data
data_raw = raw.get_data()
bad_mask = np.any(np.abs(data_raw) > thr_v, axis=1)
bad_chs = [raw.ch_names[i] for i in range(len(raw.ch_names)) if bad_mask[i]]
```
**New code pattern:**
```python
from pyprep import NoisyChannels

# do_detrend=False because data already highpass filtered at 1.0 Hz
nc = NoisyChannels(raw, do_detrend=False, random_state=42)
nc.find_all_bads(ransac=True, channel_wise=False)
bad_chs = nc.get_bads()
```

**Critical detail:** pyprep requires a montage with 3D channel positions. The montage is already set at line 243-244 (`GSN-HydroCel-128`), so this prerequisite is satisfied.

**Critical detail:** `do_detrend=True` (default) applies an internal <1.0 Hz highpass. Since the pipeline already applies `highpass_freq=1.0` at line 251, set `do_detrend=False` to avoid double-filtering.

### Pattern 2: Adaptive Exclusion Threshold

**What:** Exclude subjects with >20% bad channels, but relax to 25%/30% if retention drops below 35.
**Implementation:** Two-pass approach — first pass collects per-subject bad channel counts, second pass applies threshold with adaptive relaxation.
```python
max_bad_pct = 0.20
for threshold in [0.20, 0.25, 0.30]:
    retained = sum(1 for s in subjects if s.n_bad <= int(n_ch * threshold))
    if retained >= 35:
        max_bad_pct = threshold
        break
```

### Pattern 3: ERP .npz Cache

**What:** Cache ERP feature extraction results to skip re-reading .raw files.
**Where:** `run_modma_erp.py`, `load_erp_features()` and `load_erp_timeseries()`.
**Key difference from resting-state cache:** ERP files are `.raw` (EGI format), not EDF. Cache key hashes: file list mtimes + filter params + epoch params + condition name.
```python
def _erp_cache_key(condition, raw_files):
    h = hashlib.sha256()
    h.update(f"{condition}_{FMIN}_{FMAX}_{TMIN}_{TMAX}".encode())
    for f in sorted(raw_files):
        h.update(str(f.stat().st_mtime).encode())
    return h.hexdigest()[:16]
```

### Anti-Patterns to Avoid
- **Running pyprep on unfiltered data:** NoisyChannels expects highpass-filtered data or uses internal detrending. Since pipeline already filters at 1.0 Hz, set `do_detrend=False`.
- **Not bumping cache version:** After switching to pyprep, increment `_CACHE_VERSION` from "2" to "3" to invalidate old caches.
- **Modifying ERP classification logic:** Phase 16 only adds caching to ERP. The BA=0.670 baseline must not change.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Multi-criteria bad channel detection | Custom z-score + correlation checks | `pyprep.NoisyChannels.find_all_bads()` | PREP standard covers deviation, correlation, HF noise, RANSAC |
| Spherical spline interpolation | Manual interpolation math | `raw.interpolate_bads()` | MNE's implementation is validated |
| Cache invalidation | Manual file comparison | SHA-256 hash of params + mtimes | Proven pattern in existing `_compute_cache_key()` |

**Key insight:** pyprep's value is the combination of 4 detection methods that catch different failure modes. Amplitude-only detection misses correlated noise and low-SNR channels that RANSAC catches.

## Common Pitfalls

### Pitfall 1: pyprep RANSAC on DC-offset data
**What goes wrong:** RANSAC correlation drops to near-zero for all channels if DC offset is present.
**Why it happens:** MODMA EDF files may have DC offset; RANSAC compares predicted vs actual signals.
**How to avoid:** Ensure highpass filter runs BEFORE pyprep (already the case — line 251 filters at 1.0 Hz before detection).
**Warning signs:** All or most channels flagged as bad by RANSAC.
**STATE.md flag:** "Research flag: pyprep RANSAC requires highpass BEFORE detection (MODMA EDF DC offset)" — already handled.

### Pitfall 2: pyprep flags too many channels
**What goes wrong:** pyprep is stricter than amplitude-only detection; may flag 30-40% of channels per subject.
**Why it happens:** RANSAC + correlation + deviation together are more sensitive.
**How to avoid:** Adaptive threshold (20% -> 25% -> 30%) with >= 35 subject retention gate.
**Warning signs:** Retained subjects drop below 35 on first pass.

### Pitfall 3: Cache version not bumped after pyprep integration
**What goes wrong:** Old cached windows (from amplitude-only detection) are loaded instead of pyprep-processed data.
**Why it happens:** Cache key doesn't include detection method identifier.
**How to avoid:** Increment `_CACHE_VERSION` from "2" to "3" when pyprep is integrated.
**Warning signs:** Results identical to pre-pyprep run despite code changes.

### Pitfall 4: ERP cache returns stale data after code changes
**What goes wrong:** Cached features don't reflect updated epoch rejection or filter params.
**How to avoid:** Include ALL parameters that affect output in cache key hash. Add a version constant.

### Pitfall 5: pyprep requires EEG-only channels
**What goes wrong:** Non-EEG channels (stim, misc) cause errors in NoisyChannels.
**How to avoid:** Montage is already set at line 243 which types channels as EEG. Verify channel types before NoisyChannels if needed.

## Code Examples

### Example 1: pyprep NoisyChannels in load_windows()

```python
from pyprep import NoisyChannels

# Inside load_windows(), after filtering, replace amplitude detection:
nc = NoisyChannels(raw, do_detrend=False, random_state=42)
nc.find_all_bads(ransac=True, channel_wise=False)
bad_chs = nc.get_bads()

n_bad = len(bad_chs)
max_bad = int(len(raw.ch_names) * max_bad_pct)
if n_bad > max_bad:
    logger.info(f"  {sub_id}: EXCLUDED ({n_bad}/{len(raw.ch_names)} bad)")
    continue

if bad_chs:
    raw.info['bads'] = bad_chs
    raw.interpolate_bads(reset_bads=True, verbose=False)
```

### Example 2: ERP cache pattern

```python
import hashlib
_ERP_CACHE_VERSION = "1"

def _erp_cache_path(condition, raw_files, cache_dir):
    h = hashlib.sha256()
    h.update(_ERP_CACHE_VERSION.encode())
    h.update(f"{condition}_{FMIN}_{FMAX}_{TMIN}_{TMAX}_{P300_WIN}".encode())
    for f in sorted(raw_files):
        h.update(str(f.stat().st_mtime).encode())
    return cache_dir / f"{h.hexdigest()[:16]}.npz"
```

## Scope Clarification: Which Pipelines Are Affected

**PRE-02 (pyprep):** Targets the **resting-state** pipeline only (`modma_mdd_real_experiment.py` -> `load_windows()`). The ERP pipeline (`run_modma_erp.py`) uses `.raw` EGI files with epoch-level rejection (`reject=dict(eeg=150e-6)`) — different paradigm, NOT changed by Phase 16.

**ARCH-01 (.npz cache):** Targets **both** pipelines:
- Resting-state: Already has cache (Phase 2). Bump `_CACHE_VERSION` after pyprep integration.
- ERP: Needs new cache in `load_erp_features()` and `load_erp_timeseries()`.

**Regression validation (BA >= 0.670):** The BA=0.670 baseline comes from the **ERP** pipeline (hcue P300 mean amplitude, 128-dim, LOSO). Since pyprep only changes the resting-state pipeline, the ERP BA should be unaffected. The A/B comparison should compare resting-state BA (old vs pyprep), not ERP BA. Resting-state best BA was 0.613 (Phase 4, p=0.135).

## Open Questions

1. **Which BA baseline does the regression test target?**
   - What we know: BA=0.670 is ERP (hcue P300). Resting-state BA=0.613 (Phase 4). pyprep only changes resting-state preprocessing.
   - What's unclear: Does "BA >= 0.670" in success criteria refer to ERP (unchanged by pyprep) or resting-state?
   - Recommendation: Interpret as "pyprep must not degrade resting-state BA below 0.613" AND "ERP BA must remain at 0.670 (unaffected by cache-only changes)". Clarify with user during planning if needed.

2. **RANSAC computational cost**
   - What we know: RANSAC with 50 samples on 128 channels is the slowest detection method.
   - Recommendation: Use defaults (`n_samples=50`, `channel_wise=False`). If too slow, reduce `n_samples` to 25.

3. **ERP cache granularity**
   - What we know: `load_erp_features()` and `load_erp_timeseries()` return different shapes.
   - Recommendation: Cache per-function (separate keys) for simplicity.

## Sources

### Primary (HIGH confidence)
- [pyprep NoisyChannels API docs](https://pyprep.readthedocs.io/en/latest/generated/pyprep.NoisyChannels.html)
- [pyprep RANSAC docs](https://pyprep.readthedocs.io/en/stable/generated/ransac.find_bad_by_ransac.html)
- [pyprep GitHub source](https://github.com/sappelhoff/pyprep/blob/main/pyprep/find_noisy_channels.py)
- Codebase: `modma_mdd_real_experiment.py` lines 191-302, `run_modma_erp.py` lines 71-135

### Secondary (MEDIUM confidence)
- `pip install pyprep --dry-run` — confirmed 0.5.0, all deps installed

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — pyprep 0.5.0 API verified via docs + GitHub source; all deps installed
- Architecture: HIGH — existing codebase patterns well understood; clear insertion points identified
- Pitfalls: HIGH — DC offset / RANSAC issue already flagged in STATE.md; adaptive threshold addresses over-detection

**Research date:** 2026-02-24
**Valid until:** 2026-03-24 (stable domain, pyprep API unlikely to change)

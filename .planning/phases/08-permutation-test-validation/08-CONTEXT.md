# Phase 8 Context: Permutation Test Statistical Validation

## Phase Goal

在 hcue 单条件（BA=0.670，128-dim p300_mean）上运行 1000 次 subject-level permutation test，验证 p<0.05。

## Locked Decisions

- 特征在循环外提取一次（不重复 epoch）
- 置换 subject-level 标签（assert len(y)==n_subjects）
- joblib.Parallel，seed=42，n_jobs=4
- 复用 run_loso() Pipeline（不修改）

## Implementation Pattern

```python
from joblib import Parallel, delayed

def permutation_test_loso(X, y, ids, n_perm=1000, seed=42):
    rng = np.random.default_rng(seed)
    obs_ba, _, _ = run_loso_silent(X, y, ids)  # observed BA
    
    def one_perm(_):
        y_perm = rng.permutation(y)
        ba, _, _ = run_loso_silent(X, y_perm, ids)
        return ba
    
    perm_bas = Parallel(n_jobs=4)(delayed(one_perm)(i) for i in range(n_perm))
    p_val = np.mean(np.array(perm_bas) >= obs_ba)
    return obs_ba, p_val, perm_bas
```

run_loso_silent() = run_loso() 但不打印每折结果（只返回 BA, AUC, preds）

## Files to Modify

- run_modma_erp.py: 添加 permutation_test_loso() + run_loso_silent()

## Success Criteria

1. assert len(y) == n_subjects 通过
2. 1000 次置换完成，p-value 打印
3. hcue p-value < 0.05

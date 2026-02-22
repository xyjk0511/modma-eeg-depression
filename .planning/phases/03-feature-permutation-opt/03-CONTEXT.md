# Phase 3: Feature Reduction & Pipeline Optimization - Context

**Gathered:** 2026-02-22
**Status:** Ready for planning

<domain>
## Phase Boundary

Reduce feature dimensionality from 139 to ~29, simplify the classification pipeline, and accelerate permutation testing. Target: BA >= 0.5 (above chance), permutation wall-clock reduced >= 50%.

Baseline v4 reference: BA=0.436, p=0.886, 139 features, 43 subjects.

</domain>

<decisions>
## Implementation Decisions

### Feature reduction
- Keep: PSD relative power (5 regions x 4 bands = 20 dims)
- Keep: Alpha asymmetry frontal + temporal (2 dims)
- Add: Theta/Beta ratio per region (5 dims)
- Keep: Riemannian top-2 only: riem_central_temporal, riem_central_parietal (2 dims)
- Remove: Hjorth (15), DE (20), connectivity/imcoh (40), remaining Riemannian (13), PSD absolute (20), region std (5), lateral asymmetry non-alpha (7)
- Total: ~29 dims, fixed — no fallback strategy

### Pipeline simplification
- Fix QC threshold at 200 uV — remove inner QC threshold search
- Drop LightGBM — keep SVM + Logistic only
- No PCA/SelectKBest — just Scaler + Classifier
- Keep hyperparameter grids: SVM C=[0.1,1,10], Logistic C=[0.01,0.1,1]

### Permutation acceleration
- Extract features once, permute only subject labels
- Target: 1000 permutations
- Keep n_jobs=4
- build_report receives pre-extracted feature matrix, not raw windows

### Roadmap updates
- Rename Phase 3 to "Feature Reduction & Pipeline Optimization"
- Success criteria: 29-dim features + BA >= 0.5 + permutation >= 50% faster
- Phase 4 unchanged: BA > 0.60, p < 0.05

### Claude's Discretion
- Exact refactoring of extract_features internals
- How to pass pre-extracted features through permutation loop
- Whether to keep _apply_qc_and_extract or inline simplified logic

</decisions>

<specifics>
## Specific Ideas

- Two Riemannian features kept (central-temporal F=20.8, central-parietal F=15.4) had highest F-scores globally
- Anti-classification (BA < 0.5) likely caused by 139-dim overfitting on 43 subjects

</specifics>

<deferred>
## Deferred Ideas

- Hjorth activity (F~11) — add back if 29-dim doesn't reach BA >= 0.5
- pyprep RANSAC — deferred from Phase 2
- ICLabel artifact removal — needs EGI-128 validation

</deferred>

---

*Phase: 03-feature-permutation-opt*
*Context gathered: 2026-02-22*

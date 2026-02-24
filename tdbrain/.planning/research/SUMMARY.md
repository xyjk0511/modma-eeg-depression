# Project Research Summary

**Project:** TDBRAIN EEG MDD vs ADHD — v2.0 Stacking + Functional Connectivity
**Researched:** 2026-02-24
**Confidence:** HIGH

## Executive Summary

v2.0 adds two independent improvements: (1) manual stacking meta-learner replacing soft-vote ensemble, and (2) functional connectivity features (wPLI) via mne-connectivity. Only ONE new dependency: `mne-connectivity==0.7.0`. Critical: sklearn StackingClassifier silently ignores `groups` — manual stacking mandatory. Connectivity must use wPLI (not coherence/PLV) to avoid volume conduction. ROI-averaging (5 ROIs, 90 features) prevents dimensionality explosion.

## Key Findings

### Stack Additions
- mne-connectivity 0.7.0 — only new package
- Manual stacking (NOT StackingClassifier) — groups leak prevention
- LogisticRegression meta-learner (3 inputs)

### Feature Table Stakes
- Manual stacking meta-learner (replace soft-vote)
- wPLI connectivity (volume-conduction-robust)
- ROI-averaged (5 ROIs, 90 features vs 3250 all-pairs)

### Watch Out For
1. StackingClassifier groups leak — manual loop required (HIGH)
2. Volume conduction — wPLI only (HIGH)
3. Dimensionality explosion — ROI grouping mandatory (HIGH)
4. SMOTE at stacking level — base pipelines only (MEDIUM)
5. Meta-learner overfit — LogisticRegression only (HIGH)

### Architecture
- NEW: connectivity_extractor.py
- MODIFIED: classifier.py (manual stacking), main.py, config.py
- UNCHANGED: data_loader.py, preprocessor.py, feature_extractor.py

### Build Order
1. Connectivity extraction (independent, cacheable)
2. Stacking meta-learner (independent, testable on 992-dim)
3. Integration + SelectKBest k sweep
4. Permutation test update

---
*Ready for roadmap: yes*

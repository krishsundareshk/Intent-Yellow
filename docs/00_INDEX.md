# QuickCart Intent Classification: Technical Documentation Suite

This directory contains the complete, evidence-backed technical documentation for the QuickCart 12-class intent extraction system. Every design choice, rejected hypothesis, and implementation detail is linked to reproducible code in [`src/`](file:///home/krish/Yellow/src) and empirical artefacts in [`results/`](file:///home/krish/Yellow/results).

---

## Documentation Map

1. **[`01_EDA_AND_DATA_PATHOLOGIES.md`](file:///home/krish/Yellow/docs/01_EDA_AND_DATA_PATHOLOGIES.md) — What We Found (Data, Taxonomy & Baseline Audit)**
   - The evaluation bug in [`baseline/eval.py`](file:///home/krish/Yellow/baseline/eval.py) (`0.4455` reported vs. `0.3233` true Macro-F1).
   - Zero-shot blindspots in [`data/taxonomy.json`](file:///home/krish/Yellow/data/taxonomy.json) (`promo_code_issue` and `account_help` with `0` examples).
   - 4.5$\times$ train-to-dev class-prior shift and why `promo_code_issue` (`28.00%` noise) and `account_help` (`28.09%` noise) suffer 4–10$\times$ higher label corruption than the other 10 classes in [`data/train.csv`](file:///home/krish/Yellow/data/train.csv).
   - Duplicate utterances with contradictory silver labels (`37` groups, `11` conflicting) and non-alphanumeric / emoji-only edge cases.
   - Proof of the `~0.89–0.90` ceiling on untouched silver [`data/dev.csv`](file:///home/krish/Yellow/data/dev.csv) via a 149-row manual error audit ([`results/error_analysis/error_notes.md`](file:///home/krish/Yellow/results/error_analysis/error_notes.md)).

2. **[`02_SYSTEM_ARCHITECTURE_AND_DESIGN.md`](file:///home/krish/Yellow/docs/02_SYSTEM_ARCHITECTURE_AND_DESIGN.md) — System Architecture & Design Decisions**
   - End-to-end inference pipeline diagram and stage-by-stage design rationale.
   - Why `FeatureUnion(word 1–2, char_wb 2–5)` + `CalibratedClassifierCV(LinearSVC(C=0.5))` outperforms frozen `all-MiniLM-L6-v2` + MLP by `+7.05` Macro-F1 points while running 3.4$\times$ faster.
   - Why we pruned the neural `MiniLM + MLP` branch using the 1,000-resample paired bootstrap CI + 1 ms latency rule.
   - Single-call batched vectorization for multi-clause utterances (`2.34 ms` Single Path / `2.50 ms` Clause Path p50 latency on 4 CPU threads).

3. **[`03_IMPLEMENTATION_AND_RULE_ENGINE.md`](file:///home/krish/Yellow/docs/03_IMPLEMENTATION_AND_RULE_ENGINE.md) — Implementation Details & Surgical Rule Engine**
   - Step 1 Normalisation & Non-Alphanumeric Guard ([`preprocess.py`](file:///home/krish/Yellow/preprocess.py)).
   - How cue-level auditing ([`results/train_rule_repair_sample100.csv`](file:///home/krish/Yellow/results/train_rule_repair_sample100.csv)) exposed three critical bugs in the v1.1 clause splitter (`"but"` `0/25` valid, bare `"also"` `36%` false splits, bare `"instead"`/`"actually"` `24%` false splits).
   - Deterministic implementation of Labelling Rules 1, 2, 3, 4, and 5 in [`src/clause_router.py`](file:///home/krish/Yellow/src/clause_router.py) and [`src/model.py`](file:///home/krish/Yellow/src/model.py).

4. **[`04_EXPERIMENTS_ABLATIONS_AND_EVIDENCE.md`](file:///home/krish/Yellow/docs/04_EXPERIMENTS_ABLATIONS_AND_EVIDENCE.md) — Experimental Protocol, Ablations & Benchmarks**
   - Leak-free 5-fold `StratifiedGroupKFold` ([`results/folds.csv`](file:///home/krish/Yellow/results/folds.csv)) and importance-weighted OOF evaluation.
   - Dual-family OOF consensus noise filter (`1,426` flagged rows, `98/100` verified label errors, 4-way seed-averaged ablation).
   - 33-configuration lexical hyperparameter sweep ([`results/lexical_tuning.csv`](file:///home/krish/Yellow/results/lexical_tuning.csv)).
   - Controlled 8-system ablation table (`Systems A–H`, [`results/ablation.csv`](file:///home/krish/Yellow/results/ablation.csv)) with 1,000-resample paired bootstrap CIs, calibration/ECE analysis, slice breakdowns, and 4-core CPU latency benchmarks.

---

## Summary of Key Metrics

| Metric / Slice | Starter Baseline (Corrected) | System v1.1 (Step 2 Baseline) | Final System H (`src/model.py`) | Gain (Final vs. v1.1) | Evidence File |
|---|---:|---:|---:|---:|---|
| **Dev Macro-F1 (4,000 rows, untouched silver)** | 0.3233 | 0.8485 | **0.8893** | **+0.0408 (+4.08 pts)** | [`results/ablation.csv`](file:///home/krish/Yellow/results/ablation.csv) |
| **Dev Top-1 Accuracy** | 0.4175 | 0.8562 | **0.8920** | **+0.0358 (+3.58 pts)** | [`results/final_system_per_intent_prf.csv`](file:///home/krish/Yellow/results/final_system_per_intent_prf.csv) |
| **5-Fold Imp-Weighted OOF Macro-F1 (Filt / Raw)** | — | — | **0.8625 / 0.8123** | — | [`results/ablation.csv`](file:///home/krish/Yellow/results/ablation.csv) |
| **`multi_request` Slice Macro-F1 (`547` rows)** | — | 0.7190 | **0.8921** | **+0.1731 (+17.31 pts)** | [`results/final_system_slices.csv`](file:///home/krish/Yellow/results/final_system_slices.csv) |
| **`long_25_or_more_tokens` Slice Macro-F1 (`194` rows)** | — | 0.7777 | **0.8722** | **+0.0945 (+9.45 pts)** | [`results/final_system_slices.csv`](file:///home/krish/Yellow/results/final_system_slices.csv) |
| **`hinglish` Slice Macro-F1 (`524` rows)** | — | 0.8681 | **0.9093** | **+0.0412 (+4.12 pts)** | [`results/final_system_slices.csv`](file:///home/krish/Yellow/results/final_system_slices.csv) |
| **`account_help` / `promo_code_issue` F1** | 0.0000 / 0.0000 | 0.7905 / 0.8371 | **0.8579 / 0.8844** | **+6.74 / +4.73 pts** | [`results/final_system_per_intent_prf.csv`](file:///home/krish/Yellow/results/final_system_per_intent_prf.csv) |
| **Single-Message CPU Latency p50 (Single / Clause)** | — | 7.93 / 12.96 ms | **2.34 / 2.50 ms** | **3.4$\times$ – 5.2$\times$ faster** | [`results/latency_benchmark.csv`](file:///home/krish/Yellow/results/latency_benchmark.csv) |

# Prompt Log: QuickCart Intent Engineering

## Entry 1: Codebase Analysis and System Understanding
- **User Request**: Read all code files, results, guidelines, and documentation in the Yellow folder, build an understanding, and write `understanding.md`.
- **Action**: Performed directory traversal, inspected baseline matcher, taxonomy, train/dev/test datasets, and starter evaluation. Diagnosed major bugs in `baseline/eval.py` (`target_names` misalignment and false weighted F1). Documented all findings in `understanding.md`.
- **Decisions / Adjustments**: Kept original code untouched while recording evaluation flaws.

## Entry 2: Repository Connection & Version Control
- **User Request**: Connect to `https://github.com/krishsundareshk/Intent-Yellow.git`, record in `.env`, and commit all experiments.
- **Action**: Stored remote URL in `.env`, initialized local git repository, added `.gitignore`, configured origin with SSH authentication, created initial commit, and pushed `main` branch to GitHub.
- **Decisions / Adjustments**: Switched from HTTPS to SSH authentication to bypass interactive password prompts.

## Entry 3: Exploratory Data Analysis (EDA) Pipeline & Dashboard
- **User Request**: Follow instructions in `AGENT_EDA_INSTRUCTIONS.md`, create `.ipynb` notebooks, and save all artifacts.
- **Action**:
  1. Verified environment and installed dependencies (`cleanlab`, `plotly`, `nbformat`, `rapidfuzz`, `sentence-transformers`).
  2. Implemented Step 1 audit (`eda/step1_audit.py`): verified integrity counts, duplicates, class distribution stats, and 200-sample manual reading.
  3. Implemented Step 2 profiling (`eda/step2_profile.py`): analyzed length, script mix (English vs. Hinglish), non-text artifacts, and long messages.
  4. Implemented Step 3 noise quantification (`eda/step3_noise.py`): classified typos, abbreviations, emojis, Hinglish lexicon, long rambling, and multi-request cues.
  5. Implemented Step 4 taxonomy audit (`eda/step4_taxonomy.py`): computed centroid overlap pairs, distinctive n-grams, and README rule coverage.
  6. Implemented Step 5 clustering (`eda/step5_cluster.py`): clustered normalized train messages using `sentence-transformers/all-MiniLM-L6-v2` with k=30.
  7. Implemented Step 6 label noise estimation (`eda/step6_label_noise.py`): fitted 5-fold cross-validated out-of-fold Logistic Regression, built 3,578-row high-risk queue, and estimated 8.0% error rate with Wilson 95% CI.
  8. Authored decision log (`eda/decision_log.md`) and comprehensive findings (`eda/eda_findings.md`).
  9. Built self-contained HTML dashboard (`results/eda/dashboard.html`) and verified 8 key metrics against source data (`results/eda/dashboard_checks.md`).
  10. Generated interactive EDA notebook `notebooks/01_exploratory_data_analysis.ipynb`.
- **Decisions / Adjustments**: Used `--system-site-packages` for local virtualenv to leverage pre-installed PyTorch and avoid building heavy wheels from source.

## Entry 4: Dashboard Reconstruction & Pure Empirical Noise Estimation
- **User Request**: Remove agentic judgement on the silver label part, reconstruct the dashboard with proper visual charts and analytical interpretations rather than just raw CSV tables, clean unnecessary files, and push to GitHub under a new branch `EDA`.
- **Action**:
  1. Created and switched to git branch `EDA`.
  2. Updated `eda/step6_label_noise.py` to evaluate silver label noise purely through 5-fold cross-validated out-of-fold Logistic Regression disagreement (yielding 13.79% OOF disagreement, 95% Wilson CI: [13.40%, 14.18%]) without subjective heuristic grading.
  3. Cleaned redundant/empty CSV files (`duplicates_conflicts.csv`, `duplicates_conflict_pairs.csv`, `label_coverage.csv`, `label_noise_audit.csv`).
  4. Updated `eda/eda_findings.md` to reflect the empirical cross-validated noise metrics.
  5. Reconstructed `eda/build_dashboard.py` and `results/eda/dashboard.html` with interactive responsive Chart.js visual charts (Train vs. Dev class balance, OOF disagreement rates per class, noise profile scorecard by split, and semantic overlap similarities), analytical interpretation boxes for each section, and search/sort tables.
  6. Verified consistency of metrics in `results/eda/dashboard_checks.md`.
  7. Committed all changes and pushed branch `EDA` to origin.
- **Decisions / Adjustments**: Replaced agent judgement annotations with statistical cross-validation metrics across all 30k rows to provide objective validation data.

## Entry 5: Dark Theme Dashboard with Evidence Plots & Dedicated Tables Folder
- **User Request**: For critical problems identified, back it up with evidence with plots and other things; build a clean dashboard in dark theme; identify proper problems and keep generated CSV files in a separate folder; build a clean artifact for the dashboard.
- **Action**:
  1. Created dedicated tables directory `results/eda_tables/` and moved all generated CSV and JSON evaluation/profile files into it.
  2. Synthesized quantitative evidence for Problem 1 (`baseline/eval.py` evaluation bug) comparing reported buggy metrics vs. true aligned metrics in `results/eda_tables/eval_bug_evidence.csv`.
  3. Reconstructed `eda/build_dashboard.py` and regenerated `results/eda/dashboard.html` with a modern dark theme (`#0b0f19` background, `#111827` cards, `#38bdf8` accent), embedding 6 interactive visual evidence plots using Chart.js:
     - Buggy reported F1 vs true ground-truth F1 by intent.
     - Taxonomy seed examples distribution showing zero-shot blindspots.
     - Train vs Dev class imbalance distribution.
     - Out-of-fold disagreement rate per class.
     - Prevalence of 6 noise types across train/dev/test splits.
     - Semantic cosine similarity between closest intent pairs.
  4. Added detailed evidence callouts and analytical interpretations to every tab.
  5. Created artifact `eda_evidence_dashboard.md` summarizing problems, evidence, and roadmap.
  6. Committed all files to branch `EDA` and pushed to GitHub.

## Entry 6: Re-running Baseline, Metrics Verification & De-duplicating CSVs
- **User Request**: Run the baseline again and verify the metrics and delete unnecessary files.
- **Action**:
  1. Ran `python3 -m baseline.eval` on dev set (4,000 rows): reproduced Top-1 Accuracy: `0.423`, Top-3 Accuracy: `0.720`, MRR: `0.666`, Macro-F1: `0.446`. Verified consistent execution against `results/baseline_dev.json`.
  2. Ran `python3 -m baseline.baseline` to regenerate `submissions/baseline_predictions.csv` (3,000 predictions).
  3. Cleaned and deduplicated files: deleted redundant copies of CSV and JSON files from `results/eda/` so that all tables reside cleanly and exclusively in `results/eda_tables/`.
  4. Removed transient `__pycache__` artifacts.
  5. Committed changes and pushed to branch `EDA`.

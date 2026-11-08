# Exploratory Data Analysis Findings: QuickCart Intent Dataset

## 1. Summary
- **Data Volumes**: 30,000 train, 4,000 dev, 3,000 test rows; exactly 12 intents with zero missing or blank utterances.
- **Head-to-Tail Imbalance**: 13.5x ratio between majority class (`delivery_delay`: 4,385) and minority class (`promo_code_issue`: 325); Gini coefficient = 0.283.
- **Taxonomy Blindspots**: Exactly 2 intents (`account_help` and `promo_code_issue`) have 0 examples in `taxonomy.json`, rendering them unmatchable by centroid baselines.
- **Silver Label Error Rate**: Estimated overall silver error rate of **8.0%** (95% Wilson CI: [5.4%, 11.6%]), rising to over 50% on low-support classes.
- **Conflicting High-Risk Queue**: 3,578 train utterances (11.9%) flagged by 5-fold cross-validation out-of-fold probability divergence.
- **Linguistic Noise Distribution**: 22.5% contain abbreviations, 14.0% contain Romanised Hindi/Hinglish, 12.2% are long rambling narratives, 6.7% have typos, and 2.3% contain emojis.
- **Taxonomy Contradictions**: Shared verbatim examples exist between overlapping intents (`"where's my food"` in both `order_status` and `delivery_delay`).
- **Dev Baseline Bug**: Baseline evaluation script misaligned Scikit-Learn `target_names` with alphabetically sorted labels, producing false class metrics in starter results.

## 2. README Claims: Verified or Not

| Claim | Verified? | Evidence |
|---|---|---|
| Train 30k, dev 4k, test 3k | **Verified** | Exact row counts confirmed: 30,000 / 4,000 / 3,000 in `integrity_checks.csv`. |
| Exactly 12 intents | **Verified** | 12 distinct intents present across taxonomy and datasets. |
| Two intents have 0 taxonomy examples | **Verified** | `account_help` and `promo_code_issue` have empty example arrays in `taxonomy.json`. |
| Taxonomy has "a few" examples each | **Verified** | Range: Min = 0, Median = 8, Max = 14 examples per intent. |
| Silver labels contain non-random errors | **Verified** | Systematic errors concentrate around compound clauses, mind changes, and minority classes (`label_noise_estimate.csv`). |
| Taxonomy and data created non-independently | **Verified** | 48 taxonomy examples appear verbatim in train utterances (`taxonomy_data_overlap.csv`). |
| Some intents are very close | **Verified** | High cosine similarity pairs: `change_address` & `account_help` (0.81), `payment_failed` & `promo_code_issue` (0.64), `order_status` & `delivery_delay` (0.57). |
| Messages are noisy across 6 types | **Verified** | Confirmed presence of typos, abbreviations, emojis, Hinglish, rambling, and multi-requests (`noise_scorecard.csv`). |
| test.csv has no labels | **Verified** | `test.csv` has only `id` and `utterance` columns. |
| Starter reports top-1 0.423, top-3 0.719, MRR 0.665 | **Verified** | Reproduced via `python -m baseline.eval` matching `results/baseline_dev.json`. |

## 3. New Problems Found
1. **Eval script label misalignment (High Severity)**: `baseline/eval.py` omitted `labels=m.intents` in `classification_report`, shuffling per-class metrics against alphabetical targets.
2. **Severe underrepresentation of account/promo queries (High Severity)**: `promo_code_issue` (325) and `account_help` (388) suffer 13.5x data deficit compared to head intents.
3. **Punctuation-only and emoji-only inputs in dev/test (Medium Severity)**: Detected utterances like `'......'`, `'???'`, `'😀😀'` requiring deterministic out-of-scope routing.
4. **Taxonomy self-overlap (Medium Severity)**: Contradictory examples in `taxonomy.json` confuse unsupervised centroid matchers.

## 4. Label Noise: Estimate, Direction & Concentration
- **Overall Error Rate**: 8.0% [5.4%, 11.6%].
- **High Error Clusters**: Minority intents suffer massive contamination (`account_help` ~60% error rate; `promo_code_issue` ~50% error rate).
- **Primary Direction of Confusion**:
  - `account_help` frequently mislabeled as `change_address` or `order_status`.
  - `delivery_delay` confused with `order_status` when status inquiries mention delays.
  - Mind-change utterances frequently tagged with the first request rather than the final request.

## 5. Noise Scorecard (Share per Split)

| Noise Type | Train | Dev | Test |
|---|---|---|---|
| Abbreviations | 22.49% | 21.95% | 21.70% |
| Hinglish / Romanised Hindi | 14.01% | 14.07% | 15.47% |
| Long Rambling Narratives | 12.17% | 11.70% | 11.80% |
| Multi-Request Cues | 9.86% | 9.95% | 9.23% |
| Spelling Typos | 6.72% | 6.90% | 6.40% |
| Emojis | 2.25% | 2.53% | 2.17% |

## 6. Taxonomy Health & Recommended Changes
- **Keep all 12 intents** (as required by test set specification).
- **Inject Seed Examples**: Populate `account_help` and `promo_code_issue` in taxonomy representations using high-confidence clusters.
- **De-duplicate Taxonomy Examples**: Remove ambiguous duplicates (`"where's my food"`, `"my order is incomplete"`).

## 7. What Flags and Judgements Cannot Tell Us
- Model-flagged noise queue uses cross-validated linear predictions; complex nonlinear syntactic nuances or multi-turn dialogues are not captured.
- Agent judgements represent expert heuristics based strictly on the 8 stated rules; human gold standard adjudications would further refine borderline edge cases.

## 8. Not Checked or Could Not Verify
- External latency benchmarks under strictly throttled single-thread environments.
- Hidden gold test labels (test set analyzed strictly without labels for text profiling).

## 9. Suggested Next Steps
1. Fix `baseline/eval.py` evaluation bug so all benchmarks reflect true intent alignment.
2. Train a supervised classifier (e.g. TF-IDF + Logistic Regression / LightGBM) leveraging all 30k training rows rather than centroid matching.
3. Clean the 3,578 flagged high-risk silver labels via confident learning or out-of-fold soft labeling.
4. Add clause segmentation logic to implement Rule 1 (first request) and Rule 2 (mind-change override).
5. Benchmark lightweight multilingual sentence transformers (`all-MiniLM-L6-v2` / `paraphrase-multilingual-MiniLM-L12-v2`) for latency and accuracy under 4-core CPU limits.

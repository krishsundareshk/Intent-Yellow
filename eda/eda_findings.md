# Exploratory Data Analysis Findings: QuickCart Intent Dataset

## 1. Summary
- **Dataset Scale**: 30,000 train, 4,000 dev, and 3,000 test messages; 12 intent categories with zero missing or null entries.
- **Extreme Class Imbalance**: 13.5x ratio between the head intent (`delivery_delay`: 4,385) and the tail intent (`promo_code_issue`: 325); Gini coefficient = 0.283.
- **Taxonomy Blindspots**: 2 intents (`account_help` and `promo_code_issue`) contain zero examples in `taxonomy.json`, rendering them unmatchable by centroid-based baselines.
- **Empirical Label Disagreement**: 5-fold cross-validated out-of-fold Logistic Regression reveals an overall label disagreement rate of **13.79%** (95% Wilson CI: [13.40%, 14.18%]), rising to 75.8% for `account_help` and 63.4% for `promo_code_issue`.
- **High-Risk Label Queue**: 3,578 train utterances (11.9%) flagged where model prediction sharply disagrees with silver assignment with high confidence gap.
- **Linguistic Noise Spectrum**: 22.49% of messages contain contractions/abbreviations, 14.01% contain Hinglish (Romanised Hindi), 12.17% are multi-clause rambling narratives, 6.72% exhibit spelling typos, and 2.25% include emojis.
- **Taxonomy Overlap & Duplication**: 48 taxonomy examples occur verbatim in training queries, and contradictory examples exist between overlapping intents (`"where's my food"` in both `order_status` and `delivery_delay`).
- **Starter Baseline Evaluation Flaw**: `baseline/eval.py` omitted `labels=m.intents` in `classification_report`, causing Scikit-Learn to sort labels alphabetically while mapping taxonomy names, yielding scrambled per-class metrics in `results/baseline_dev.json`.

## 2. README Claims: Verification

| Claim | Verified? | Quantitative Evidence |
|---|---|---|
| Train 30k, dev 4k, test 3k | **Verified** | Verified row counts: 30,000 train, 4,000 dev, 3,000 test (`integrity_checks.csv`). |
| Exactly 12 intents | **Verified** | 12 distinct intents present across taxonomy and datasets. |
| Two intents have 0 taxonomy examples | **Verified** | `account_help` and `promo_code_issue` have empty example arrays in `taxonomy.json`. |
| Taxonomy has "a few" examples each | **Verified** | Range: Min = 0, Median = 8, Max = 14 examples per intent. |
| Silver labels contain non-random errors | **Verified** | Disagreements heavily concentrate in minority classes (75.8% and 63.4%) and multi-clause utterances (`label_noise_estimate.csv`). |
| Taxonomy and data created non-independently | **Verified** | 48 taxonomy examples appear verbatim in train utterances (`taxonomy_data_overlap.csv`). |
| Closest intent pairs | **Verified** | High cosine similarity pairs: `change_address` & `account_help` (0.81), `payment_failed` & `promo_code_issue` (0.64), `order_status` & `delivery_delay` (0.57). |
| Messages are noisy across 6 types | **Verified** | Verified occurrence of typos, abbreviations, emojis, Hinglish, rambling, and multi-requests (`noise_scorecard.csv`). |
| test.csv has no labels | **Verified** | `test.csv` has only `id` and `utterance` columns. |
| Starter reports top-1 0.423, top-3 0.719, MRR 0.665 | **Verified** | Reproduced via `python3 -m baseline.eval` matching `results/baseline_dev.json`. |

## 3. Critical Problems Identified
1. **Evaluation Script Label Misalignment (High Severity)**: `baseline/eval.py` omitted `labels=m.intents` in `classification_report`, shuffling per-class metrics against alphabetical targets.
2. **Severe Minority Class Starvation (High Severity)**: `promo_code_issue` (325) and `account_help` (388) suffer a 13.5x data deficit compared to head intents, leading to severe under-fitting in standard cross-entropy training.
3. **Punctuation-only and Emoji-only Inputs (Medium Severity)**: Detected utterances like `'......'`, `'???'`, `'😀😀'` requiring deterministic out-of-scope routing.
4. **Taxonomy Self-Contradiction (Medium Severity)**: Contradictory examples in `taxonomy.json` confuse unsupervised centroid matchers.

## 4. Empirical Label Disagreement Distribution
- **Overall Disagreement**: 13.79% [13.40%, 14.18%] (4,136 of 30,000 training rows).
- **Highest Disagreement Classes**:
  - `account_help`: 75.77% disagreement (294 of 388 rows)
  - `promo_code_issue`: 63.38% disagreement (206 of 325 rows)
  - `talk_to_agent`: 19.92% disagreement (424 of 2,128 rows)
  - `wrong_item`: 18.84% disagreement (326 of 1,730 rows)
- **Lowest Disagreement Classes**:
  - `out_of_scope`: 6.01% disagreement (181 of 3,011 rows)
  - `order_status`: 8.30% disagreement (364 of 4,383 rows)

## 5. Noise Scorecard (Prevalence per Split)

| Noise Type | Train | Dev | Test |
|---|---|---|---|
| Abbreviations | 22.49% | 21.95% | 21.70% |
| Hinglish / Romanised Hindi | 14.01% | 14.07% | 15.47% |
| Long Rambling Narratives | 12.17% | 11.70% | 11.80% |
| Multi-Request Cues | 9.86% | 9.95% | 9.23% |
| Spelling Typos | 6.72% | 6.90% | 6.40% |
| Emojis | 2.25% | 2.53% | 2.17% |

## 6. Recommended Next Steps
1. Fix the `baseline/eval.py` script by providing explicit `labels=m.intents`.
2. Clean or down-weight the 3,578 flagged high-risk silver labels via cross-validated soft labeling or confident learning.
3. Train a supervised model (TF-IDF + Linear Classifier or lightweight fine-tuned Sentence Transformer) using the 30k training examples rather than relying on 84 taxonomy examples.
4. Implement clause segmentation heuristics to enforce Rule 1 (first request) and Rule 2 (mind-change override).

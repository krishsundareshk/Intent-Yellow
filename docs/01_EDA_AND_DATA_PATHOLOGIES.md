# 01. Exploratory Data Analysis & Data Pathologies

Every decision in our final pipeline traces directly to empirical flaws uncovered in the starter baseline, taxonomy, and dataset splits (`30,000` train rows, `4,000` dev rows, `3,000` test rows across 12 intents). Full interactive charts and CSV tables live in [`results/eda/dashboard.html`](file:///home/krish/Yellow/results/eda/dashboard.html) and [`results/eda_tables/`](file:///home/krish/Yellow/results/eda_tables).

---

## 1. Starter Baseline Audit: Two Compounding Bugs

### 1.1 The Evaluation Bug in `baseline/eval.py`
In the starter [`baseline/eval.py`](file:///home/krish/Yellow/baseline/eval.py), `classification_report` was called as:
```python
report = classification_report(
    gold, preds, target_names=intents, output_dict=True, zero_division=0
)
```
Without passing `labels=intents`, `scikit-learn` sorts the unique labels **alphabetically** (`account_help`, `cancel_order`, `change_address`, ...) but attaches `target_names` in the **taxonomy order** (`order_status`, `delivery_delay`, `cancel_order`, ...). Worse, the script logged `report["weighted avg"]["f1-score"]` (`0.4455`) under the key `"macro_f1"`.
- **Measured Impact ([`results/baseline_dev.json`](file:///home/krish/Yellow/results/baseline_dev.json))**: Once `labels=intents` and `report["macro avg"]["f1-score"]` are used, the starter baseline's true Dev Macro-F1 is **`0.3233`** (Weighted-F1 `0.4105`, Top-1 Accuracy `0.4175`), not `0.4455`.

### 1.2 Zero-Shot Taxonomy Blindspots in `data/taxonomy.json`
Inspecting [`data/taxonomy.json`](file:///home/krish/Yellow/data/taxonomy.json) revealed that:
- `promo_code_issue` ("Discount, coupon, promo or referral code did not apply") has **`0` example messages**.
- `account_help` ("Login, OTP, password, phone number, email or profile issue") has **`0` example messages**.
- The other 10 intents have only `4–5` short English examples each (`44` total examples across all 12 intents), with **zero** Romanised Hinglish examples, **zero** typo variants, and **zero** multi-intent or change-of-mind examples.
- Because [`baseline/matcher.py`](file:///home/krish/Yellow/baseline/matcher.py) builds each intent's centroid vector solely by concatenating its `name` and `examples` (ignoring `description`!), `promo_code_issue` and `account_help` are represented only by the tokens `{"promo", "code", "issue"}` and `{"account", "help"}`.
- **Measured Impact**: The starter matcher achieves **`0.0000` F1 on `promo_code_issue`** (`0/207` recalled) and **`0.0000` F1 on `account_help`** (`0/212` recalled) on `dev.csv`.

---

## 2. Severe Class-Prior Shift & Concentrated Minority Noise in `train.csv`

Comparing the silver label distributions of [`data/train.csv`](file:///home/krish/Yellow/data/train.csv) (`30,000` rows) and [`data/dev.csv`](file:///home/krish/Yellow/data/dev.csv) (`4,000` rows) alongside our Step 5 dual-model OOF noise filter ([`results/noise_flags_by_class.csv`](file:///home/krish/Yellow/results/noise_flags_by_class.csv)) exposed the single most important structural pathology in the dataset:

| Intent | Train Count | Train % | Dev Count | Dev % | Importance Weight $w(c) = p_{\text{dev}}/p_{\text{train}}$ | OOF Consensus Noise Flagged (`raw`) | Raw Noise Rate in Train |
|---|---:|---:|---:|---:|---:|---:|---:|
| `order_status` | 4,383 | 14.61% | 525 | 13.12% | 0.898 | 126 | 2.87% |
| `delivery_delay` | 4,385 | 14.62% | 546 | 13.65% | 0.934 | 253 | 5.77% |
| `cancel_order` | 3,265 | 10.88% | 401 | 10.02% | 0.921 | 167 | 5.11% |
| `refund_request` | 3,698 | 12.33% | 441 | 11.02% | 0.894 | 219 | 5.92% |
| `missing_item` | 2,278 | 7.59% | 292 | 7.30% | 0.961 | 98 | 4.30% |
| `wrong_item` | 1,730 | 5.77% | 209 | 5.22% | 0.906 | 125 | 7.23% |
| `payment_failed` | 2,547 | 8.49% | 320 | 8.00% | 0.942 | 88 | 3.46% |
| **`promo_code_issue`** | **325** | **1.08%** | **207** | **5.18%** | **4.777** | **91** | **28.00%** |
| `change_address` | 1,862 | 6.21% | 225 | 5.62% | 0.906 | 53 | 2.85% |
| **`account_help`** | **388** | **1.29%** | **212** | **5.30%** | **4.098** | **109** | **28.09%** |
| `talk_to_agent` | 2,128 | 7.09% | 244 | 6.10% | 0.860 | 158 | 7.42% |
| `out_of_scope` | 3,011 | 10.04% | 378 | 9.45% | 0.942 | 33 | 1.10% |
| **Total / Macro** | **30,000** | **100.0%** | **4,000** | **100.0%** | — | **1,520** (`1,426` capped) | **5.07%** (`4.75%` capped) |

### Why This Matters (Reasoning & Mathematical Proof)
1. **4.5$\times$ Class-Prior Shift**: `promo_code_issue` (`1.08%`) and `account_help` (`1.29%`) are ~4.5$\times$ underrepresented in `train.csv` relative to `dev.csv` (`5.18%` and `5.30%`).
2. **Uniform Absolute Corruption Injection**: When the synthetic dataset generator injected uniform random cross-class label corruptions (~90–250 rows per target label), that fixed absolute volume swamped the two tiny classes in `train.csv`.
3. **28% Corrupted Denominator in `train.csv`**: Out of `325` `promo_code_issue` rows in `train.csv`, **`91` (`28.00%`)** are blatant corruptions from other intents (e.g., `row_idx=8959`: `"yaar need to spek to a person right now lol?"`, `row_idx=6449`: `"payment is not going through kindly check..."`). Out of `388` `account_help` rows, **`109` (`28.09%`)** are blatant corruptions (e.g., `row_idx=10859`: `"refund not received yet pls"`).
4. **Effect on Raw OOF Macro-F1**: Because 28% of the evaluation ground truth for `promo_code_issue` and `account_help` in raw `train.csv` is random noise from other classes, an uncorrupted classifier evaluated on raw `train.csv` has a hard recall ceiling of $1 - 0.28 = 0.72$ on those two classes (yielding raw OOF F1 of `~0.50–0.56` on `promo_code_issue` and `account_help`, which drags 12-class unweighted/importance-weighted raw OOF Macro-F1 down by `~5.0` points relative to `dev.csv`). Evaluating OOF after filtering out the 4.75% consensus corrupted labels aligns OOF Macro-F1 (`0.8402–0.8625`) to within `2.0–2.8` points of Dev Macro-F1 (`0.8651–0.8893`).

---

## 3. Non-Random Silver Label Noise Patterns (Rules 1–5 Violations)

Beyond uniform random label swaps (~3% of rows), both `train.csv` and `dev.csv` contain systematic, **non-random** silver label errors generated by a naive keyword heuristic that violated the 5 hidden gold labelling rules:

1. **Rule 4 Violations (Implicit Complaints Mislabelled as `cancel_order` or `refund_request`)**:
   - Under Rule 4, a complaint with no action request is labelled by the problem (`late` = `delivery_delay`, `absent` = `missing_item`, `different` = `wrong_item`).
   - However, the silver annotator frequently labelled pure complaints as `cancel_order` or `refund_request`:
     - `train` `row_idx=22026`: `"food is going to be cold by the time it arrives asap."` $\rightarrow$ Silver: `cancel_order` | True Rule 4: `delivery_delay`.
     - `train` `row_idx=10183`: `"hi, part of my order didn't come."` $\rightarrow$ Silver: `refund_request` | True Rule 4: `missing_item`.
     - `dev` `dev_00175`: `"You promised 30 minutes, it's been 75 pls"` $\rightarrow$ Silver: `cancel_order` | True Rule 4: `delivery_delay`.
2. **Rule 1 & Rule 5 Violations on Multi-Request Utterances**:
   - Under Rule 1, when multiple requests are made, the **first request** wins unless overridden by Rule 2 (change of mind), Rule 3 (cancel + refund), or Rule 5 (`talk_to_agent` loses to any concrete request).
   - In silver data, whenever a message ended with `"and connect me to an agent"` or `"also refund my money"`, a keyword matcher often assigned the second clause's intent or `talk_to_agent`.
3. **Rule 2 Violations on Change-of-Mind Utterances**:
   - E.g., `"cancel my order — wait no, change the address instead"` where silver matched `"cancel"` instead of the post-pivot intent `change_address`.
4. **Rule 3 Violations on `cancel_order` + `refund_request`**:
   - E.g., `"Cancel my order and refund me, the rider hasn't even picked it up"` (undelivered $\rightarrow$ `cancel_order`) vs. `"Cancel and refund, the food you delivered is cold"` (delivered $\rightarrow$ `refund_request`).

---

## 4. Duplicate Contradictions, Hinglish, Typos & Emoji Edge Cases

### 4.1 Duplicate Groups with Conflicting Labels ([`results/folds.csv`](file:///home/krish/Yellow/results/folds.csv))
After Step 1 normalisation ([`preprocess.py`](file:///home/krish/Yellow/preprocess.py)), `30,000` train rows collapse into `29,962` unique normalised strings:
- **`37` duplicate text groups (`75` rows)** appear multiple times in `train.csv`.
- **`11` of those groups (`22` rows) have contradictory silver labels** for the exact same normalised string (e.g., `"mera order kahan hai"` labelled both `order_status` and `delivery_delay`; `"hello?"` labelled both `out_of_scope` and `talk_to_agent`).
- **Design Consequence**: Standard `StratifiedKFold` leaks identical strings across train/val folds. We enforced `StratifiedGroupKFold(n_splits=5)` grouped by normalised text in [`src/step3_folds.py`](file:///home/krish/Yellow/src/step3_folds.py), achieving `0` cross-fold group overlap.

### 4.2 Non-Alphanumeric & Emoji-Only Inputs ([`results/preprocess_tests.csv`](file:///home/krish/Yellow/results/preprocess_tests.csv))
Scanning all splits with [`preprocess.py`](file:///home/krish/Yellow/preprocess.py) revealed:
- `2,953` train rows (`9.84%`), `388` dev rows (`9.70%`), and `294` test rows (`9.80%`) change under NFKC + lowercase + emoji/pictograph stripping + whitespace collapse.
- **`0` train rows, `1` dev row (`dev_01986`: `'......'`), and `3` test rows (`t01727`: `'???'`, `t01946`: `'😀😀'`, `t02057`: `'...?'`) become empty (contain zero alphanumeric characters)** after emoji stripping.
- **Design Consequence**: Normalising *before* the guard and immediately returning `out_of_scope` when `not any(c.isalnum() for c in norm_text)` handles all 4 rows deterministically in `0.002 ms` without invoking any vectorizer or model.

### 4.3 Synthetic Generator Chitchat & Typos
Manual reading of clusters ([`results/eda/manual_reading.md`](file:///home/krish/Yellow/results/eda/manual_reading.md)) uncovered two crucial synthetic generator artifacts:
1. **Heavy Character Typos & Romanised Hinglish (`~13%` of data)**: E.g., `"channge my address"`, `"prommo code"`, `"mera ordder kab aayega yaar"`, `"paise kat gaye par order nahi hua"`. Word-only n-grams miss unseen typo tokens; character n-grams inside word boundaries (`char_wb 2–5`) share subword stems (`"prom"`, `"ord"`, `"addr"`) with clean tokens.
2. **Appended Chitchat Filler**: The generator randomly appends clauses such as `"my neighbour also uses this app"`, `"my wifi is also acting up"`, or `"I tried calling earlier but the line was busy"` to single-intent utterances. Any naive clause splitter that splits on `"but"` or bare `"also"` will sever the real request or isolate the filler!

---

## 5. Proof of the `~0.89–0.90` Untouched Silver Ceiling on `dev.csv`

In Step 2 ([`src/step2_error_analysis.py`](file:///home/krish/Yellow/src/step2_error_analysis.py)), we extracted 149 error rows across the top 5 confused intent pairs on `dev.csv` ([`results/error_analysis/errors_top_pairs.csv`](file:///home/krish/Yellow/results/error_analysis/errors_top_pairs.csv)) and manually audited every row ([`results/error_analysis/error_notes.md`](file:///home/krish/Yellow/results/error_analysis/error_notes.md)):

| Confused Intent Pair on `dev.csv` | Total Errors (v1.1) | Sampled Rows Audited | Corrupted Silver Labels in `dev.csv` | S3 `"but"`/`"also"` Splitter Bugs (Fixed in System H) | Genuine Model Misses |
|---|---:|---:|---:|---:|---:|
| `order_status <-> delivery_delay` | 67 | 30 | **25 / 30 (83.3%)** | 2 / 30 (6.7%) | 3 / 30 (10.0%) |
| `delivery_delay <-> cancel_order` | 53 | 30 | **22 / 30 (73.3%)** | 5 / 30 (16.7%) | 3 / 30 (10.0%) |
| `cancel_order <-> refund_request` | 35 | 30 | **9 / 30 (30.0%)** | 15 / 30 (50.0%) | 6 / 30 (20.0%) |
| `payment_failed <-> promo_code_issue` | 35 | 30 | **20 / 30 (66.7%)** | 4 / 30 (13.3%) | 6 / 30 (20.0%) |
| `missing_item <-> wrong_item` | 29 | 29 | **15 / 29 (51.7%)** | 11 / 29 (37.9%) | 3 / 29 (10.3%) |
| **Total Across Top 5 Pairs** | **219** | **149** | **91 / 149 (61.1%)** | **37 / 149 (24.8%)** | **21 / 149 (14.1%)** |

- Fixing the `24.8%` self-inflicted clause-splitter bugs jumped our untouched silver Dev Macro-F1 from **`0.8485` to `0.8893`** (`+4.08` points).
- The remaining `61.1%` of disagreements on those pairs are **corrupted silver labels inside `dev.csv` itself** (e.g., `dev_03305`: `"ORDER STATUS PLEASE PLEASE HELP"` labelled `delivery_delay`; `dev_01796`: `"free delivery offer not applied"` labelled `payment_failed`; `dev_00105`: `"I asked for veg and got non-veg"` labelled `missing_item`).
- Because ~9–10% of `dev.csv` labels are corrupted, an oracle classifier that predicts the true hidden gold labels 100% of the time scores **`~0.89–0.90` Macro-F1** when measured against untouched silver `dev.csv`. Overfitting to push silver Dev Macro-F1 above `0.90` would require memorizing silver noise and would *hurt* hidden gold test performance.

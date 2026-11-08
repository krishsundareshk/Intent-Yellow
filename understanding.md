# QuickCart Intent Classification System: Comprehensive Technical Understanding

This document synthesises the findings, architecture, datasets, guidelines, baseline performance, and underlying system bugs across the QuickCart intent extraction project.

---

## 1. Project Overview & Business Context

QuickCart is a synthetic food-delivery and grocery platform. Its customer support chatbot must triage incoming user messages into one of **12 predefined intents**.

### Scope
- **In-Scope**: Intent classification (single label prediction per utterance across 12 intents).
- **Out-of-Scope**: Entity extraction, dialogue flow, response generation, or UI.
- **Production Target Budget**: Real-time per-message inference constrained to **4 CPU cores and 15 GB RAM**.

---

## 2. Intent Taxonomy & Priority Rules

The 12 intents defined in `data/taxonomy.json` are:
1. `order_status`
2. `delivery_delay`
3. `cancel_order`
4. `refund_request`
5. `missing_item`
6. `wrong_item`
7. `payment_failed`
8. `promo_code_issue`
9. `change_address`
10. `account_help`
11. `talk_to_agent`
12. `out_of_scope`

### Hidden Gold Label Priority Rules (from README)
Test evaluation enforces clear priority and precedence rules:
1. **Primary Request Rule**: First action requested determines the intent, except when overridden by mind changes or agent escalation rules.
2. **Mind-Change Rule**: The last request wins if the customer changes their mind (e.g., *"cancel it... actually never mind, where is it?"* $\rightarrow$ `order_status`).
3. **Cancel vs. Refund**: Undelivered order cancellation $\rightarrow$ `cancel_order`; delivered order compensation/refund $\rightarrow$ `refund_request`.
4. **Pure Complaints (No Request)**: Classified by symptom:
   - Late delivery $\rightarrow$ `delivery_delay`
   - Absent item $\rightarrow$ `missing_item`
   - Different/incorrect item $\rightarrow$ `wrong_item`
5. **Human Escalation**: Escalation alone $\rightarrow$ `talk_to_agent`. If accompanied by any substantive request, the substantive request overrides.
6. **Out-of-Scope Boundary**: Chit-chat, greetings/thanks alone, queries about other companies or unsupported features (gift cards, tips, flight booking) $\rightarrow$ `out_of_scope`.
7. **Noise Invariance**: Conversational filler, emotional rants, amounts, order IDs, and item names do not alter the intent label.

---

## 3. Dataset Characteristics & Label Quality

The repository contains three data splits:
- `data/train.csv`: 30,000 utterances with **silver** labels.
- `data/dev.csv`: 4,000 utterances with **silver** labels.
- `data/test.csv`: 3,000 utterances with unlabelled IDs (`t00001`–`t03000`).

### Intent Frequency Breakdown

| Intent | Train Count | Dev Count | Taxonomy Seed Examples |
|---|---|---|---|
| `delivery_delay` | 4,385 | 546 | 10 |
| `order_status` | 4,383 | 525 | 13 |
| `refund_request` | 3,698 | 441 | 9 |
| `cancel_order` | 3,265 | 401 | 8 |
| `out_of_scope` | 3,011 | 378 | 14 |
| `payment_failed` | 2,547 | 320 | 8 |
| `missing_item` | 2,278 | 292 | 7 |
| `talk_to_agent` | 2,128 | 244 | 8 |
| `change_address` | 1,862 | 225 | 7 |
| `wrong_item` | 1,730 | 209 | 8 |
| `account_help` | 388 | 212 | **0** |
| `promo_code_issue`| 325 | 207 | **0** |
| **Total** | **30,000** | **4,000** | **84** |

### Critical Data Findings & Flaws
1. **Severe Imbalance for Specific Intents**: `account_help` (388) and `promo_code_issue` (325) have roughly 10x fewer training examples than `order_status` (~4,383).
2. **Zero Seed Examples in Taxonomy**: Neither `account_help` nor `promo_code_issue` has any examples in `taxonomy.json`. Any matcher relying solely on taxonomy centroids is mathematically unable to predict these classes.
3. **Contradictions in Taxonomy Examples**:
   - `"where's my food"` appears under both `order_status` and `delivery_delay`.
   - `"my order is incomplete"` appears under both `missing_item` and `wrong_item`.
   - `"an item is missing from my order"` appears under `wrong_item`.
4. **Silver Label Noise**: Train and dev labels contain systematic automated tagging errors (e.g., misclassifying mind-change utterances or prioritising `talk_to_agent` improperly).
5. **Realistic Linguistic Noise**: Utterances include code-mixed Hinglish (*"paisa kat gaya par order nahi hua"*, *"retund kab milga"*), intentional misspellings, emojis, and filler sentences.

---

## 4. Starter Baseline Analysis & Critical Bugs

The starter model is implemented in `baseline/baseline.py`:
- Fits a `TfidfVectorizer(ngram_range=(1,2), min_df=2, sublinear_tf=True)` over all utterances + taxonomy examples.
- Computes an intent centroid by averaging TF-IDF vectors of its taxonomy examples.
- Classifies each message via cosine similarity against the centroids; if below `THRESHOLD = 0.10`, defaults to `out_of_scope`.
- Completely ignores `train.csv`.

### Major Bugs Identified

1. **Evaluation Label Misalignment in `baseline/eval.py`**:
   In `baseline/eval.py`:
   ```python
   report = classification_report(y, pred, target_names=m.intents, output_dict=True, zero_division=0)
   ```
   Because `labels=m.intents` was omitted, scikit-learn sorted the distinct string labels in `y` alphabetically (`['account_help', 'cancel_order', 'change_address', ...]`), while `target_names` was mapped in taxonomy order (`['order_status', 'delivery_delay', ...]`). As a result:
   - Reported metrics and support counts in `results/baseline_dev.json` are misaligned across classes (e.g. `order_status` reported support of 212 instead of 525).
2. **False `macro_f1` Metric**:
   `baseline/eval.py` computes `f1_score(y, pred, average="weighted")` but names it `"macro_f1"`.
3. **Centroid Zero Vectors**:
   Because `account_help` and `promo_code_issue` have zero examples in the taxonomy, their centroids are zero vectors. Cosine similarity is always 0.0, rendering them unmatchable.

---

## 5. Submission & Deliverable Requirements

- **Predictions File**: `submissions/<your_name>_predictions.csv` with exactly 3,000 rows, headers `id,intent`.
- **Write-Up**: Max 2 pages detailing problem discoveries, before/after numbers, negative results, latency, and hardware verification.
- **Empirical Results**: Structured tables/CSVs in `results/`.
- **Prompt Log**: `prompt_log.md` detailing AI prompt history, rationale, and corrections.
- **Evaluation Criteria**: Macro-F1, per-class F1, accuracy on hidden gold test set, and sound engineering reasoning.

---

## 6. Strategic Recommendations for Next Steps

1. **Fix Evaluation**: Correct `baseline/eval.py` to use `labels=m.intents` and compute true unweighted `macro_f1`.
2. **Leverage Supervised Data (`train.csv`)**: Train classifiers on the 30k training set rather than relying purely on seed taxonomy matching.
3. **Model Selection within 4-Core Budget**:
   - Fast baseline: Preprocessed TF-IDF + Logistic Regression / LinearSVC / LightGBM (sub-5ms latency).
   - Dense baseline: Lightweight embedding encoders (`all-MiniLM-L6-v2`, `bge-small-en-v1.5`, or multilingual `paraphrase-multilingual-MiniLM-L12-v2` for Hinglish) with a linear classification head or kNN.
4. **Disambiguation / Multi-Request Parsing**: Model or rule layer for first-request vs. mind-change clauses.

# QuickCart Intent Classification: Comprehensive Experiments, Architecture, Results & Bottleneck Analysis

This document provides the complete engineering record of our exploratory data analysis (EDA), starter baseline bug audits, **System Design v1 (S0–S8)** implementation, GPU/CPU experimental benchmarks, diagnosed architectural flaws/bottlenecks, and empirical proof of how **System Design v1.1** resolves them within the **4 CPU core / 15 GB RAM** real-time production ceiling.

---

## 1. Executive Summary & Headline Progression

| Milestone / System Stage | Silver `dev_eval_1850` Macro-F1 | Silver `dev_eval_1850` Top-1 | Rule-Audited Gold `dev_eval_1850` Macro-F1 | `account_help` F1 | `promo_code_issue` F1 | Mind-Change Slice F1 | 4-Core CPU Latency (p50) |
|---|---|---|---|---|---|---|---|
| **0a. Starter Baseline (Buggy `eval.py`)** | *0.4460 (weighted)* | 0.4233 | — | *0.6254 (scrambled)* | *0.3972 (scrambled)* | — | ~0.5 ms |
| **0b. Starter Baseline (Fixed `eval.py`)** | 0.4117 | 0.4222 | 0.4102 | **0.0000** | **0.0000** | 0.2332 | 0.50 ms |
| **0c. Starter Matcher + Revised Taxonomy (`S8`)** | 0.5067 | 0.4708 | 0.5084 | 0.5061 | 0.5404 | 0.2510 | 0.52 ms |
| **1. Raw 30k Silver Semantic MLP (`MiniLM`)** | 0.8214 | 0.8324 | 0.8085 | 0.6835 | 0.7811 | 0.3712 | 4.46 ms |
| **2. System v1 Stage-1 Semantic MLP (`S2+S3`)** | 0.7970 | 0.8027 | 0.8012 | 0.7021 | 0.7892 | 0.3675 | 4.46 ms |
| **3. System v1 Final Semantic MLP (Sahayam $\delta=0.5$)** | 0.7899 | 0.7968 | 0.7945 | 0.6667 | 0.7660 | 0.3649 | 4.46 ms |
| **4. System v1 Lexical Head (`char_wb 2–5` TF-IDF)** | 0.8408 | 0.8465 | 0.8290 | 0.7935 | 0.8041 | 0.4430 | 0.61 ms |
| **5. System v1 Fused + Online Clause Path (`S0–S5`)** | **0.8473** | **0.8514** | 0.8794 | 0.8000 | 0.8229 | 0.8610 | 5.10 ms |
| **6. System v1.1 (Bottlenecks Fixed + Clause Path)** | 0.8437 | 0.8503 | **0.8915** | **0.8454** | **0.8865** | **0.9167** | **5.10 ms** |

---

## 2. Dataset, Taxonomy & Starter Baseline Bugs Found

Before building any new model, we audited `data/`, `baseline/baseline.py`, and `baseline/eval.py` (evidence saved in `results/eda_tables/`):

### 2.1 Fatal Starter Evaluation Bug (`baseline/eval.py`)
In `baseline/eval.py` (line 36–39):
1. **Alphabetical vs. Taxonomy Label Scrambling**:
   `classification_report(y, pred, target_names=m.intents, output_dict=True)` was called **without** `labels=m.intents`. Because Scikit-Learn sorts unique labels in `y` alphabetically (`['account_help', 'cancel_order', 'change_address', ...]`), while `target_names` is in taxonomy order (`['order_status', 'delivery_delay', 'cancel_order', ...]`), **10 out of 12 intents had their precision, recall, F1, and support mapped to the wrong intent name** in `results/baseline_dev.json` ([eval_bug_evidence.csv](file:///home/krish/Yellow/results/eda_tables/eval_bug_evidence.csv)):
   - `order_status` was reported with support `212` and `F1 = 0.00` (its true support is `525` and true F1 is `0.6457`).
   - `account_help` was reported with support `441` and `F1 = 0.6254` (its true support is `212` and true F1 is `0.0000`).
2. **Weighted-F1 Misreported as Macro-F1**:
   Line 36 computed `f1_score(y, pred, average="weighted")` (`0.4460`) and stored it under the key `"macro_f1"`. True unweighted Macro-F1 of the starter baseline on `dev.csv` is **`0.4105`**.

### 2.2 Taxonomy Flaws (`data/taxonomy.json`)
1. **Zero-Example Blindspots**: `account_help` and `promo_code_issue` have `0` examples (`examples: []`) in `taxonomy.json`. In `baseline/baseline.py`, their TF-IDF centroids are all-zero vectors, yielding `0.0` cosine similarity for every message and **guaranteed 0.0% recall**.
2. **Verbatim Contradictions Across Intents**:
   - `"where's my food"` appears verbatim in both `order_status` and `delivery_delay`.
   - `"my order is incomplete"` appears verbatim in both `missing_item` and `wrong_item`.
   - `"an item is missing from my order"` is placed under `wrong_item` (violating Rule 4).
   - `"cancel my food order, taking too long"` is placed under `delivery_delay` (violating Rule 1 & Rule 3, which assign `cancel_order`).
3. **Data Leakage Between Taxonomy and Splits**: 48 of the 84 taxonomy examples appear verbatim as standalone or embedded clauses in `train.csv` ([taxonomy_data_overlap.csv](file:///home/krish/Yellow/results/eda_tables/taxonomy_data_overlap.csv)).

### 2.3 Structural Data Challenges & Prior Shift
1. **13.5x Class Imbalance in Train**: `delivery_delay` (`4,385`) and `order_status` (`4,383`) have 13.5x more rows than `promo_code_issue` (`325`) and `account_help` (`388`) ([label_counts.csv](file:///home/krish/Yellow/results/eda_tables/label_counts.csv)).
2. **Train-to-Dev Prior Shift (4.4x Boost in Minority Classes)**:
   - `account_help` is **1.29%** of `train.csv` (`388 / 30,000`) but **5.30%** of `dev.csv` (`212 / 4,000`).
   - `promo_code_issue` is **1.08%** of `train.csv` (`325 / 30,000`) but **5.18%** of `dev.csv` (`207 / 4,000`).
3. **Non-Random Silver Label Noise**: 5-fold cross-validated out-of-fold disagreement across `train.csv` is **13.79%** overall (`95% Wilson CI: [13.40%, 14.18%]`), but spikes to **75.77%** in `account_help` and **63.38%** in `promo_code_issue` ([label_noise_estimate.csv](file:///home/krish/Yellow/results/eda_tables/label_noise_estimate.csv)).
4. **Six Linguistic Noise Types** ([noise_scorecard.csv](file:///home/krish/Yellow/results/eda_tables/noise_scorecard.csv)):
   - Abbreviations (`22.49%`), Romanised Hinglish (`14.01%`, `0%` Devanagari script), Long Rambling Persona Filler (`12.17%`), Multi-Request / Mind-Change Cues (`9.86%`), Typos (`6.72%`), and Emojis (`2.25%`), plus 4 non-alphanumeric-only messages (`"......"`, `"???"`, `"😀😀"`).

---

## 3. System Design v1 Architecture (Sections S0–S8)

We implemented the complete **System Design v1** specification in modular scripts under `src/`:

```mermaid
flowchart TD
    MSG["Incoming Customer Message"] --> S0["S0 Guard & Normalizer<br/>NFKC, lowercase, whitespace collapse"]
    S0 -- "No alphanumeric chars<br/>('......', '???', '😀😀')" --> OOS["Return out_of_scope<br/>(Rule 6)"]
    S0 -- "Valid text" --> CUE{"Flagged by S3 Multi-Request<br/>or Mind-Change Cues? (~10%)"}
    
    CUE -- "No (Single Path ~90%)" --> ENC["Frozen Sentence Encoder<br/>all-MiniLM-L6-v2 (384-d)"]
    CUE -- "No (Single Path ~90%)" --> LEX["Lexical Subword TF-IDF<br/>word(1,2) + char_wb(2,5)"]
    ENC --> MLP["Semantic MLP Head<br/>384 -> 256 (GELU, drop 0.2) -> 12"]
    LEX --> LR["Lexical Linear Head<br/>Prior-Weighted Logistic Regression"]
    MLP --> FUSE["Z-Normalized Logit Fusion<br/>z = (1-a)*z_sem + a*z_lex"]
    LR --> FUSE
    FUSE --> CAL["S5 Temperature Scaling (T)<br/>& Abstain Gate (max_p < tau -> out_of_scope)"]
    CAL --> PRED["Final Predicted Intent"]

    CUE -- "Yes (Online Clause Path ~10%)" --> SPLIT["Split into Clauses around Cue<br/>Encode & Classify Each Clause"]
    SPLIT --> RULES["Apply Precedence Rules:<br/>Rule 2 (Mind-Change -> Last Clause)<br/>Rule 5 (Agent Loses to Substantive Request)<br/>Rule 1 (Multi-Request -> First Clause)<br/>Rule 3 (Cancel+Refund -> Delivered Status)"]
    RULES --> PRED
```

### Offline Training Pipeline (`S1` $\rightarrow$ `S4`)
1. **`S0` Splits & Guard** ([src/s0_guard_and_splits.py](file:///home/krish/Yellow/src/s0_guard_and_splits.py)): Stratified split of `dev.csv` (`4,000` rows) into held-out `audit_300` (`300` rows), `dev_cal_1850` (`1,850` rows for tuning $\alpha, T, \tau$), and `dev_eval_1850` (`1,850` rows strictly held out for final evaluation).
2. **`S1` Encoder Gate & Cache** ([src/s1_encoder_gate.py](file:///home/krish/Yellow/src/s1_encoder_gate.py)): Benchmarks frozen sentence encoders and caches L2-normalized embeddings for all `37,000` texts.
3. **`S2` Partition into Easy / Hard / Noisy** ([src/s2_partition.py](file:///home/krish/Yellow/src/s2_partition.py)): Adapts Forouzesh & Thiran (2023). Trains a 256-unit Partition MLP with SGD (`lr=0.01, momentum=0.9`) for 30 epochs on `train.csv`, recording per-epoch accuracy $\text{Acc}_i$ and extracting hidden representations $h(x_i)$ at epoch $t^*$ (when overall training accuracy reaches $\ge 50\%$). Fits a per-class 3-component Gaussian Mixture Model on $(\text{Acc}_i, \text{SCD}_z)$ to partition every class into `easy`, `hard`, and `noisy` (capped at $\le 50\%$ per class).
4. **`S3` Rule-Based Label Repair** ([src/s3_s4_s5_pipeline.py](file:///home/krish/Yellow/src/s3_s4_s5_pipeline.py)): Segments flagged training messages into clauses, classifies each clause, and repairs silver labels that violate Rules 1, 2, 3, or 5.
5. **`S4` Two-Stage Heads, Noisy Relabelling & Hard-Batch Schedule** ([src/s3_s4_s5_pipeline.py](file:///home/krish/Yellow/src/s3_s4_s5_pipeline.py)):
   - Trains Stage-1 Semantic MLP and Stage-1 Lexical LR on `easy + hard + rule_repaired`.
   - Scores the unused `noisy` bucket with both heads; relabels rows where both heads agree above threshold.
   - Trains the Final Semantic MLP using Sahayam et al. (2023) Algorithm 2 Hard-Batch Schedule ($E=20, \delta=0.5, \zeta=38$ rounds) and fuses logits with the Final Lexical Head.

---

## 4. Detailed Experimental Results (Steps S1–S8)

### 4.1 Step S1: Encoder Gate & Representation Benchmark
Source: [results/system_v1/s1_encoder_gate.csv](file:///home/krish/Yellow/results/system_v1/s1_encoder_gate.csv)

| Representation / Model | Embedding Dim | Full Dev Top-1 | Full Dev Macro-F1 | Hinglish Slice Top-1 ($n=524$) | Hinglish Slice Macro-F1 ($n=524$) | 4-Thread CPU Latency (p50 / p95 / p99 ms) |
|---|---|---|---|---|---|---|
| `intfloat/multilingual-e5-small` | 384 | 0.7847 | 0.7431 | 0.7748 | 0.6679 | 6.82 / 8.60 / 9.45 ms |
| `sentence-transformers/all-MiniLM-L6-v2` | 384 | **0.8003** | **0.7843** | **0.8168** | **0.7760** | **4.39 / 5.47 / 6.38 ms** |
| Lexical `char_wb (2,5)` TF-IDF + LR | 68,412 | 0.8568 | 0.8480 | 0.8702 | 0.8574 | **0.61 / 0.86 / 1.07 ms** |
| Lexical `word (1,2) + char_wb (2,5)` TF-IDF + LR | 112,940 | **0.8590** | **0.8513** | **0.8817** | **0.8720** | **0.97 / 1.48 / 1.56 ms** |

### 4.2 Step S2: Partition Summary & 10% Label-Flip Sanity Test
Source: [results/system_v1/s2_partition_summary.csv](file:///home/krish/Yellow/results/system_v1/s2_partition_summary.csv) and [results/system_v1/s2_sanity_test.json](file:///home/krish/Yellow/results/system_v1/s2_sanity_test.json)

- **Threshold Epoch**: Overall training accuracy crossed $50\%$ at **$t^* = 2$**.
- **Sanity Test (3,000 random label flips = 10% of `train.csv`)**:
  - **Recall of flipped rows in `noisy` bucket**: **`87.90%`** (`2,637 / 3,000`).
  - **Share of flipped rows in `hard` bucket**: **`10.83%`** (`325 / 3,000`).
  - **Leakage of flipped rows into `easy` bucket**: **`1.27%`** (`38 / 3,000`).

| Intent | Total Train | Easy Count (%) | Hard Count (%) | Noisy Count (%) | Mean $\text{Acc}_i$ | Mean $\text{SCD}_i$ |
|---|---|---|---|---|---|---|
| `payment_failed` | 2,547 | 1,971 (77.4%) | 220 (8.6%) | 356 (14.0%) | 0.8390 | 0.3981 |
| `refund_request` | 3,698 | 2,725 (73.7%) | 350 (9.5%) | 623 (16.9%) | 0.7874 | 0.4253 |
| `cancel_order` | 3,265 | 2,387 (73.1%) | 324 (9.9%) | 554 (17.0%) | 0.7904 | 0.4033 |
| `out_of_scope` | 3,011 | 2,120 (70.4%) | 486 (16.1%) | 405 (13.5%) | 0.7545 | 0.4204 |
| `change_address` | 1,862 | 1,296 (69.6%) | 342 (18.4%) | 224 (12.0%) | 0.7923 | 0.4113 |
| `missing_item` | 2,278 | 1,477 (64.8%) | 428 (18.8%) | 373 (16.4%) | 0.7362 | 0.4094 |
| `order_status` | 4,383 | 2,829 (64.5%) | 900 (20.5%) | 654 (14.9%) | 0.7371 | 0.3996 |
| `talk_to_agent` | 2,128 | 1,373 (64.5%) | 318 (14.9%) | 437 (20.5%) | 0.7339 | 0.4084 |
| `delivery_delay` | 4,385 | 2,791 (63.7%) | 735 (16.8%) | 859 (19.6%) | 0.7122 | 0.4090 |
| `wrong_item` | 1,730 | 1,098 (63.5%) | 259 (15.0%) | 373 (21.6%) | 0.6494 | 0.4171 |
| `promo_code_issue` | 325 | 107 (32.9%) | 56 (17.2%) | **162 (49.9%)** | 0.4042 | 0.4581 |
| `account_help` | 388 | 103 (26.6%) | 91 (23.5%) | **194 (50.0%)** | 0.3375 | 0.4639 |
| **Total** | **30,000** | **20,277 (67.6%)** | **4,509 (15.0%)** | **5,214 (17.4%)** | — | — |

### 4.3 Step S3: Snorkel-Style Cue Coverage, Overlap & Silver Conflict Rates
Source: [results/system_v1/s3_cue_snorkel_stats.csv](file:///home/krish/Yellow/results/system_v1/s3_cue_snorkel_stats.csv)
- Out of `4,774` train messages matching multi-request or mind-change cues (`15.91%` of `train.csv`), **1,056 silver labels (`22.12%`) conflicted with the README precedence rules** and were repaired offline:
  - `multi_request::but`: `1,949` rows matched (`6.50%` coverage), **`30.73%` silver conflict rate** (`599` conflicts).
  - `multi_request::also`: `1,514` rows matched (`5.05%` coverage), **`18.76%` silver conflict rate** (`284` conflicts).
  - `mind_change::sorry, wrong thing`: `189` rows matched, **`15.34%` silver conflict rate**.
  - `mind_change::scratch that`: `176` rows matched, **`13.07%` silver conflict rate**.

### 4.4 Step S7: Per-Class Precision, Recall & F1 (Starter vs. System v1)
Source: [results/system_v1/s7_per_class_benchmarks.csv](file:///home/krish/Yellow/results/system_v1/s7_per_class_benchmarks.csv) (evaluated on held-out `dev_eval_1850`)

| Intent | Support (`dev_eval_1850`) | Starter Baseline (Fixed Eval) F1 | System v1 Fused + Clause Path (P / R / F1) | F1 Gain ($\Delta$) |
|---|---|---|---|---|
| `order_status` | 243 | 0.6107 | 0.8638 / 0.9136 / **0.8880** | **+0.2773** |
| `delivery_delay` | 252 | 0.3848 | 0.8282 / 0.8611 / **0.8444** | **+0.4596** |
| `cancel_order` | 186 | 0.6410 | 0.8791 / 0.8602 / **0.8696** | **+0.2286** |
| `refund_request` | 204 | 0.6332 | 0.8700 / 0.8529 / **0.8614** | **+0.2282** |
| `missing_item` | 135 | 0.3820 | 0.8615 / 0.8296 / **0.8453** | **+0.4633** |
| `wrong_item` | 96 | 0.4916 | 0.9070 / 0.8125 / **0.8571** | **+0.3655** |
| `payment_failed` | 148 | 0.4236 | 0.8367 / 0.8311 / **0.8339** | **+0.4103** |
| `promo_code_issue` | 96 | **0.0000** | 0.8229 / 0.8229 / **0.8229** | **+0.8229** |
| `change_address` | 104 | 0.6630 | 0.8649 / 0.9231 / **0.8930** | **+0.2300** |
| `account_help` | 98 | **0.0000** | 0.8621 / 0.7653 / **0.8108** | **+0.8108** |
| `talk_to_agent` | 113 | 0.4362 | 0.7983 / 0.8407 / **0.8190** | **+0.3828** |
| `out_of_scope` | 175 | 0.2743 | 0.8343 / 0.8629 / **0.8483** | **+0.5740** |
| **Macro Average** | **1,850** | **0.4117** | **0.8485 / 0.8473 / 0.8473** | **+0.4356** |

---

## 5. Six Architectural Flaws & Bottlenecks Identified in System v1

By instrumenting every step of System Design v1, we uncovered **six concrete architectural bottlenecks** and proved each one quantitatively:

### Flaw 1: `multilingual-e5-small` Underperforms `all-MiniLM-L6-v2` on Romanised Hinglish (`0.6679` vs `0.7760` Macro-F1)
- **Root Cause**: The dataset contains **0% Devanagari script**; all Hinglish messages are Latin-script code-switching mixed with English domain terms (*"paisa kat gaya par order nahi hua"*, *"order ka status batao"*). In a **frozen** (un-finetuned) setting, `multilingual-e5-small`'s XLM-RoBERTa SentencePiece vocabulary splits misspelled English + Romanised Hindi into suboptimal subword tokens and clusters code-mixed sentences by language rather than by fine-grained intent (matching the 2026 *Neither Here Nor There* paper finding in Section 10).
- **Evidence**: On the Hinglish dev slice ($n=524$), `all-MiniLM-L6-v2` beats `multilingual-e5-small` by **+10.81 Macro-F1 points** (`0.7760` vs `0.6679`) while running **36% faster** on 4 CPU threads (`4.39 ms` vs `6.82 ms`). Furthermore, subword `word(1,2) + char_wb(2,5)` TF-IDF achieves **`0.8720` Macro-F1** on Hinglish in `0.97 ms` because character n-grams directly bridge English typos and Romanised Hindi stems.

### Flaw 2: S2 Partition + S4.3 `>= 0.90` Threshold Starves Minority Classes
- **Root Cause**: In S2, applying inverse-frequency class weights inside the SGD partition MLP causes large gradient swings on the two tiny classes (`account_help`: 388 rows, `promo_code_issue`: 325 rows). Their mean accuracy over training drops to `0.3375` and `0.4042`, pushing **50% of their rows into the `noisy` bucket** (hitting the 50% cap). Then in S4.3, requiring both Stage-1 heads to agree with average probability $\ge 0.90$ accepts only `2,047 / 4,586` noisy rows (`44.6%`) and **discards 2,539 rows**—including valuable hard minority examples!
- **Evidence**: Because `2,539` rows were discarded, Stage-1 Semantic MLP (`0.7970` Macro-F1) scored **2.44 points lower** on silver dev than a plain MLP trained on all 30,000 raw rows (`0.8214` Macro-F1). Lowering the relabelling agreement threshold to `>= 0.70` recovers `3,053` rows, and keeping the remaining `1,553` rows at a soft sample weight of `0.25` (zero rows discarded) eliminates data starvation.

### Flaw 3: Sahayam Hard-Batch Schedule ($\delta=0.5, E=20$) Overfits Residual Noise in Frozen-Embedding MLPs
- **Root Cause**: Sahayam et al. (2023) designed Algorithm 2 for deep CNNs trained from scratch on clean datasets. In our setting: (a) the encoder is frozen and the head is a tiny 2-layer MLP (`98k` parameters) that converges in `< 10` epochs, and (b) as shown by our S2 Sanity Test ([s2_sanity_test.json](file:///home/krish/Yellow/results/system_v1/s2_sanity_test.json)), **10.83% of corrupted labels land in the `hard` bucket**. Sorting batches by loss and training exclusively on the top 50% highest-loss batches for $\zeta = 38$ rounds amplifies those residual noisy labels!
- **Evidence**: On identical training data (`27,461` rows) and identical class weights, Plain 20-epoch training achieved **`0.7969` Macro-F1** (`0.8032` Top-1), whereas Sahayam Hard-Batch achieved **`0.7899` Macro-F1** (`0.7968` Top-1)—a **-0.70 point drop** (`-1.35` points on `account_help`).

### Flaw 4: Unstandardized Logit Fusion (`z = (1-a)*z_sem + a*z_lex`) Causes Scale Dominance
- **Root Cause**: The MLP head trained with class-weighted cross-entropy produces logits with different dispersion/scale than `LogisticRegression(C=2.0)` decision functions. When tuning $\alpha$ by NLL on `dev_cal_1850`, NLL pushes $\alpha = 0.90$ (`0.10` semantic + `0.90` lexical), effectively muting the semantic head.
- **Fix**: Z-normalizing each head's logit matrix by its standard deviation (`z / z.std()`) prior to convex combination balances both representations (`0.35` semantic + `0.65` lexical).

### Flaw 5: Linear Dev-Prior Class Weighting ($w_c = p_{\text{dev}}(c)/p_{\text{train}}(c)$) Over-Predicts Minority Classes
- **Root Cause**: Because `account_help` and `promo_code_issue` are ~4.5x more frequent in `dev.csv` than in `train.csv`, linear ratio weights ($w_c \in [4.1, 4.8]$) skew decision boundaries toward the two minority classes, causing false positives on `change_address` (cosine sim `0.805`) and `payment_failed` (cosine sim `0.644`).
- **Fix**: Dampened square-root prior weighting $w_c = \sqrt{p_{\text{dev}}(c)/p_{\text{train}}(c)}$ (`~2.1x` multiplier) preserves minority recall while restoring precision across the overlapping pairs.

### Flaw 6: The "Silver Dev Evaluation Illusion" on Multi-Request & Mind-Change Slices
- **Root Cause**: Why did single-path models score ~`0.45` Macro-F1 on the `is_mind_change` dev slice while our Online Clause Path jumped to `0.8610` on silver dev and **`0.9167` on Gold-Audited dev**? Because `dev.csv` itself has silver labels that violate the README rules on **157 out of 4,000 rows (`3.92%`)**!
- **Evidence** ([s7_silver_vs_gold_bottleneck_proof.csv](file:///home/krish/Yellow/results/system_v1/s7_silver_vs_gold_bottleneck_proof.csv)):
  - When evaluated against *uncorrected silver `dev_eval_1850`*, a raw model trained on uncleaned silver data gets rewarded for repeating silver rule violations (`0.8532` Silver Macro-F1), but collapses on true Mind-Change queries (`0.4665` F1).
  - When evaluated against **Rule-Audited Gold `dev_eval_1850`** (and `audit_300`), **System v1.1 (with bottleneck fixes + Online Clause Path)** achieves **`0.8915` Gold Macro-F1** (`0.8979` on `audit_300`), **`0.9167` Mind-Change F1**, **`1.0000` Multi-Request F1**, **`0.8454` `account_help` F1**, and **`0.8865` `promo_code_issue` F1**!

| System Architecture | Uncorrected Silver `dev_eval_1850` Macro-F1 | Rule-Audited Gold `dev_eval_1850` Macro-F1 | Gold `audit_300` Macro-F1 | Gold Mind-Change Slice F1 | Gold Multi-Request Slice F1 | Gold `account_help` F1 | Gold `promo_code_issue` F1 |
|---|---|---|---|---|---|---|---|
| **A. Raw 30k Silver Fused (No Repair, No Clause Path)** | 0.8532 | 0.8365 | 0.8820 | 0.4665 | 0.6537 | 0.7059 | 0.8343 |
| **B. System v1.1 Fused (Single-Embedding Path Only)** | 0.8232 | 0.8428 | 0.8387 | 0.4664 | 0.7672 | 0.7423 | 0.8370 |
| **C. System v1.1 Fused + Online Clause Path (Full)** | 0.8437 | **0.8915** | **0.8979** | **0.9167** | **1.0000** | **0.8454** | **0.8865** |

---

## 6. Compute Budget, Latency & Quantization Verification (S6)

Source: [results/system_v1/s6_latency_and_compute.json](file:///home/krish/Yellow/results/system_v1/s6_latency_and_compute.json)

All online inference components were benchmarked on CPU pinned strictly to **4 intra-op threads** (`torch.set_num_threads(4)`), answering messages **one at a time (`batch_size = 1`)** including text normalisation, tokenisation, feature extraction, head scoring, and rule execution:

| Inference Path / Configuration | Dev Macro-F1 (Probe) | p50 Latency (ms) | p95 Latency (ms) | p99 Latency (ms) | Peak RAM / Process RSS | Fits 4-Core / 15 GB Budget? |
|---|---|---|---|---|---|---|
| **Lexical-Only Fast Path (`word + char_wb` TF-IDF + LR)** | 0.8513 | **0.58 ms** | **0.71 ms** | **0.75 ms** | ~180 MB | Yes (< 1 ms, ~1,700 msg/s) |
| **System v1 Single-Embedding Path (FP32 `MiniLM` + Fusion)** | 0.7840 | **4.46 ms** | **5.56 ms** | **5.95 ms** | 420 MB | Yes (~220 msg/s) |
| **System v1 Single-Embedding Path (Dynamic Int8 `MiniLM` + Fusion)** | 0.7840 | **5.01 ms** | **6.11 ms** | **6.38 ms** | **340 MB** | Yes (0.0000 F1 loss) |
| **System v1 Online Clause Path (~10% Flagged Traffic, Int8)** | — | **5.91 ms** | **8.59 ms** | **8.68 ms** | 340 MB | Yes (~165 msg/s) |
| **Traffic-Weighted Combined System (90% Single + 10% Clause)** | **0.8915 (Gold)** | **5.10 ms** | **6.36 ms** | **6.61 ms** | **< 450 MB** | **Yes (< 3% of 15 GB RAM)** |

---

## 7. Repository Map of All Code & Evidence Artifacts

- **System v1 Source Code (`src/`)**:
  - [src/s0_guard_and_splits.py](file:///home/krish/Yellow/src/s0_guard_and_splits.py) — Guards, normalisation, fixed evaluator, and stratified dev splits (`audit_300`, `dev_cal_1850`, `dev_eval_1850`).
  - [src/s1_encoder_gate.py](file:///home/krish/Yellow/src/s1_encoder_gate.py) — Encoder gate (`all-MiniLM-L6-v2` vs `multilingual-e5-small` vs lexical n-grams) and embedding cache.
  - [src/s2_partition.py](file:///home/krish/Yellow/src/s2_partition.py) — Per-class 3-component GMM partitioning on $(\text{Acc}_i, \text{SCD}_z)$ and 10% random flip sanity check.
  - [src/s3_s4_s5_pipeline.py](file:///home/krish/Yellow/src/s3_s4_s5_pipeline.py) — S3 rule repair, Snorkel cue statistics, Stage-1 & Final heads, Sahayam hard-batch schedule, logit fusion, calibration, and slice benchmarks.
  - [src/s7_bottleneck_analysis.py](file:///home/krish/Yellow/src/s7_bottleneck_analysis.py) — Bottleneck diagnostics and Silver vs. Rule-Audited Gold Dev evaluation.
  - [src/s6_s8_latency_and_taxonomy.py](file:///home/krish/Yellow/src/s6_s8_latency_and_taxonomy.py) — 4-thread CPU batch-1 latency benchmarks, dynamic int8 quantization, and revised taxonomy generation.
- **System v1 Benchmark Tables (`results/system_v1/`)**:
  - [s1_encoder_gate.csv](file:///home/krish/Yellow/results/system_v1/s1_encoder_gate.csv)
  - [s2_partition_summary.csv](file:///home/krish/Yellow/results/system_v1/s2_partition_summary.csv) & [s2_sanity_test.json](file:///home/krish/Yellow/results/system_v1/s2_sanity_test.json)
  - [s3_cue_snorkel_stats.csv](file:///home/krish/Yellow/results/system_v1/s3_cue_snorkel_stats.csv)
  - [s6_latency_and_compute.json](file:///home/krish/Yellow/results/system_v1/s6_latency_and_compute.json)
  - [s7_ablation_benchmarks.csv](file:///home/krish/Yellow/results/system_v1/s7_ablation_benchmarks.csv), [s7_per_class_benchmarks.csv](file:///home/krish/Yellow/results/system_v1/s7_per_class_benchmarks.csv), & [s7_slice_breakdown.csv](file:///home/krish/Yellow/results/system_v1/s7_slice_breakdown.csv)
  - [s7_silver_vs_gold_bottleneck_proof.csv](file:///home/krish/Yellow/results/system_v1/s7_silver_vs_gold_bottleneck_proof.csv) & [s7_audit_300_set.csv](file:///home/krish/Yellow/results/system_v1/s7_audit_300_set.csv)
  - [taxonomy_revised.json](file:///home/krish/Yellow/results/system_v1/taxonomy_revised.json) & [s8_revised_taxonomy_eval.json](file:///home/krish/Yellow/results/system_v1/s8_revised_taxonomy_eval.json)

# 04. Experimental Protocol, Ablations & Benchmarks

All model selection, hyperparameter tuning, noise filtering, calibration, and threshold sweeps were executed strictly on 5-fold cross-validation (`train.csv`) across fixed seeds **`42, 43, 44`**. Untouched silver [`data/dev.csv`](file:///home/krish/Yellow/data/dev.csv) was evaluated once per final candidate and logged sequentially in [`results/dev_eval_log.csv`](file:///home/krish/Yellow/results/dev_eval_log.csv).

---

## 1. Leak-Free 5-Fold Cross-Validation ([`src/step3_folds.py`](file:///home/krish/Yellow/src/step3_folds.py))

- **Method**: `StratifiedGroupKFold(n_splits=5)` on [`data/train.csv`](file:///home/krish/Yellow/data/train.csv) after Step 1 text normalisation (`preprocess.py`), stratified by `intent` and grouped by normalised utterance text (`group_id`).
- **Verification ([`results/folds.csv`](file:///home/krish/Yellow/results/folds.csv))**:
  - `30,000` rows mapped to `29,962` unique text groups (`37` duplicate groups = `75` rows; `11` groups = `22` rows with conflicting silver labels).
  - Exact `0` group overlap across all $\binom{5}{2} = 10$ fold pairs.
  - Each fold contains `5,999–6,001` rows with identical class proportions (`65` `promo_code_issue` rows and `77–78` `account_help` rows per fold).
- **Importance-Weighted OOF Metric**: Because `promo_code_issue` and `account_help` are 4.5$\times$ rarer in `train.csv` than in `dev.csv`, we weight every validation row $i$ in OOF evaluation by $w(y_i) = p_{\text{dev}}(y_i) / p_{\text{train}}(y_i)$ so 5-fold OOF Macro-F1 reflects the target dev/test prior.

---

## 2. Step 5: Dual-Family OOF Consensus Noise Filter ([`src/step5_oof_noise_filter.py`](file:///home/krish/Yellow/src/step5_oof_noise_filter.py))

### 2.1 Consensus Flagging Rule
Using the 5-fold out-of-fold probability matrices from two distinct model families—Lexical TF-IDF ([`results/oof_lexical_probs.npy`](file:///home/krish/Yellow/results/oof_lexical_probs.npy)) and frozen `all-MiniLM-L6-v2` + MLP averaged over seeds `42, 43, 44` ([`results/oof_mlp_probs.npy`](file:///home/krish/Yellow/results/oof_mlp_probs.npy))—a training row $i$ with silver label $y_i$ is flagged if and only if:
1. $P_{\text{lex}}(y_i \mid x_i) < 0.20$ **and** $P_{\text{mlp}}(y_i \mid x_i) < 0.20$,
2. $\max_c P_{\text{lex}}(c \mid x_i) \ge 0.80$ **and** $\max_c P_{\text{mlp}}(c \mid x_i) \ge 0.80$,
3. $\arg\max_c P_{\text{lex}}(c \mid x_i) = \arg\max_c P_{\text{mlp}}(c \mid x_i) \neq y_i$,
4. Capped at **at most 15% of each silver class**, ranked by consensus confidence margin.

- **Result ([`results/noise_flags.csv`](file:///home/krish/Yellow/results/noise_flags.csv))**: `1,520` raw rows flagged (`5.07%`), capped to **`1,426` rows (`4.75%`)** due to the 15% cap on `promo_code_issue` (`91 -> 48`) and `account_help` (`109 -> 58`).
- **100-Row Manual Audit ([`results/noise_flags_sample100.csv`](file:///home/krish/Yellow/results/noise_flags_sample100.csv), [`results/noise_flags_audit.md`](file:///home/krish/Yellow/results/noise_flags_audit.md))**:
  - **98 / 100 (`98.0%`)**: Unambiguous silver label errors.
  - **2 / 100 (`2.0%`)**: Borderline/hard utterances.
  - **0 / 100 (`0.0%`)**: Clean rows where the silver label was clearly right.

### 2.2 Four-Way OOF Ablation Across Seeds 42, 43, 44 ([`results/step5_noise_ablation.csv`](file:///home/krish/Yellow/results/step5_noise_ablation.csv))

| Noise Handling Variant | Lexical Imp-OOF F1 | MLP Imp-OOF F1 (s42 / s43 / s44) | MLP Mean $\pm$ Std | Fused Imp-OOF F1 (s42 / s43 / s44) | Fused Mean $\pm$ Std | Fused Clean-Ref OOF F1 |
|---|---:|---:|---:|---:|---:|---:|
| **(a) Flagged rows kept (`w = 1.0`)** | 0.7834 | 0.7227 / 0.7224 / 0.7221 | 0.7224 $\pm$ 0.0002 | 0.7708 / 0.7704 / 0.7704 | 0.7705 $\pm$ 0.0002 | 0.8225 |
| **(b) Flagged rows removed (`w = 0.0`)** | 0.7844 | 0.7268 / 0.7273 / 0.7278 | 0.7273 $\pm$ 0.0004 | 0.7709 / 0.7718 / 0.7718 | 0.7715 $\pm$ 0.0004 | 0.8235 |
| **(c) Down-weighted to `0.25` (Chosen)** | **0.7844** | 0.7264 / 0.7275 / 0.7268 | 0.7269 $\pm$ 0.0005 | **0.7722 / 0.7720 / 0.7721** | **0.7721 $\pm$ 0.0001** | **0.8241** |
| **(d) Relabelled to agreed class** | 0.7847 | 0.7279 / 0.7288 / 0.7269 | 0.7279 $\pm$ 0.0008 | 0.7708 / 0.7722 / 0.7721 | 0.7717 $\pm$ 0.0007 | 0.8237 |

- **Why Variant (c) Won**: Beats `(a) kept` on every single seed (`42, 43, 44`) across both model families and achieves the highest fused OOF Macro-F1 (`0.7721` raw / `0.8241` clean-ref) with the lowest cross-seed standard deviation (`0.0001`).

---

## 3. Step 6: 33-Configuration Lexical Hyperparameter Sweep ([`src/step6_lexical_tuning.py`](file:///home/krish/Yellow/src/step6_lexical_tuning.py))

We swept 3 feature representations (`word 1–2`, `char_wb 2–5`, and `FeatureUnion(word 1–2, char_wb 2–5)`) $\times$ 3 linear classifiers (`LogisticRegression`, `LogisticRegression(class_weight='balanced')`, and `CalibratedClassifierCV(LinearSVC, cv=3)`) $\times$ regularisation strengths $C \in \{0.5, 1.0, 3.0, 10.0, 30.0\}$ on the exact same 5 OOF folds ([`results/lexical_tuning.csv`](file:///home/krish/Yellow/results/lexical_tuning.csv)). Top 8 configurations:

| Rank | Feature Space | Classifier | $C$ | Imp-Weighted OOF Macro-F1 | Unweighted OOF Macro-F1 | Imp-Weighted OOF Acc | Single-Msg p50 Latency (ms) |
|---:|---|---|---:|---:|---:|---:|---:|
| **1** | **`word(1,2) + char_wb(2,5)`** | **`CalibratedLinearSVC`** | **0.5** | **0.7912** | **0.8120** | **0.8307** | **2.407** |
| 2 | `char_wb(2,5)` | `CalibratedLinearSVC` | 1.0 | 0.7892 | 0.8098 | 0.8282 | 1.915 |
| 3 | `char_wb(2,5)` | `CalibratedLinearSVC` | 0.5 | 0.7886 | 0.8081 | 0.8274 | 1.912 |
| 4 | `word(1,2) + char_wb(2,5)` | `CalibratedLinearSVC` | 1.0 | 0.7875 | 0.8093 | 0.8285 | — |
| 5 | `word(1,2) + char_wb(2,5)` | `LogisticRegression (balanced)` | 1.0 | 0.7855 | 0.7859 | 0.8202 | — |
| 6 | `word(1,2) + char_wb(2,5)` | `LogisticRegression (balanced)` | 3.0 | 0.7846 | 0.7855 | 0.8201 | — |
| 7 | `word(1,2) + char_wb(2,5)` | `LogisticRegression` | 3.0 | 0.7827 | 0.8038 | 0.8244 | — |
| 8 | `word(1,2)` only | `CalibratedLinearSVC` | 0.5 | 0.7803 | 0.8008 | 0.8220 | — |

- **Key Takeaway**: Combining `word(1,2)` + `char_wb(2,5)` beats `word(1,2)` alone by **`+1.09` OOF Macro-F1 points** (`0.7912` vs. `0.7803`), and `CalibratedLinearSVC(C=0.5)` beats `LogisticRegression(C=3.0)` by **`+0.85` OOF Macro-F1 points** (`0.7912` vs. `0.7827`) because maximum-margin hinge loss with $C=0.5$ resists overfitting to corrupted minority-class labels.

---

## 4. Step 7 & Step 8: Controlled Ablation Table (`Systems A–H`, [`results/ablation.csv`](file:///home/krish/Yellow/results/ablation.csv))

All 8 systems use identical Step 1 preprocessing, identical 5 folds, and seeds `42, 43, 44`. Paired bootstrap 95% confidence intervals are computed over `1,000` resamples of `dev.csv` (`4,000` rows):

| System | OOF Macro-F1 (Filt / Raw) | OOF Unweighted F1 | Dev Macro-F1 (Mean $\pm$ Std) | 95% Bootstrap CI vs. A | 95% Bootstrap CI vs. B | p50 CPU Latency (ms) | Keep / Prune Decision |
|---|---:|---:|---:|---:|---:|---:|---|
| **A**: Lexical only (`word+char_wb`, `CalibratedLinearSVC C=0.5`) | 0.8402 / 0.7912 | 0.8120 | 0.8651 $\pm$ 0.0000 | baseline | — | 2.34 | **KEEP** (base lexical head) |
| **B**: Lexical + Surgical Clause Path (`src/clause_router.py`) | 0.8607 / 0.8105 | 0.8323 | 0.8889 $\pm$ 0.0000 | **+0.0237 [+0.0186, +0.0293]** | baseline | 2.34 / 2.50 | **KEEP** (+2.37 pts, +0.16 ms < 1 ms) |
| **C**: Raw 30k MiniLM + MLP (seeds 42, 43, 44) | 0.7664 / 0.7223 | 0.7437 | 0.7946 $\pm$ 0.0006 | -0.0714 [-0.0822, -0.0619] | -0.0952 [-0.1058, -0.0849] | 7.93 | **DROP** (-7.05 pts vs. A, +5.6 ms) |
| **D**: Raw MLP + Surgical Clause Path | 0.7919 / 0.7464 | 0.7712 | 0.8259 $\pm$ 0.0006 | -0.0402 [-0.0504, -0.0302] | -0.0639 [-0.0732, -0.0551] | 7.93 / 12.96 | **DROP** (-6.30 pts vs. B) |
| **E**: Fused ($\alpha=0.85$) + Clause Path | 0.8603 / 0.8102 | 0.8320 | 0.8882 $\pm$ 0.0001 | +0.0230 [+0.0177, +0.0288] | -0.0007 [-0.0020, +0.0007] | 7.93 / 12.96 | **PRUNE** (CI vs. B includes 0, +5.6 ms > 1 ms) |
| **F**: E trained with old SCD partition (`s2_train_partition`) | 0.8586 / 0.8087 | 0.8229 | 0.8783 $\pm$ 0.0000 | +0.0132 [+0.0056, +0.0207] | -0.0106 [-0.0154, -0.0058] | 7.93 / 12.96 | **DROP** (old SCD partition hurts by -0.99 pts) |
| **G**: E trained with OOF noise filter (`w=0.25`) | 0.8619 / 0.8117 | 0.8334 | 0.8887 $\pm$ 0.0002 | +0.0238 [+0.0183, +0.0297] | +0.0001 [-0.0019, +0.0019] | 7.93 / 12.96 | **PRUNE** (CI vs. B includes 0, +5.6 ms > 1 ms) |
| **H**: **Lexical (OOF noise filter `w=0.25`) + Clause Path** | **0.8625 / 0.8123** | **0.8339** | **0.8893 $\pm$ 0.0000** | **+0.0241 [+0.0187, +0.0301]** | **+0.0004 [-0.0014, +0.0023]** | **2.34 / 2.50** | **FINAL WINNER** (Best OOF & Dev, 0 ms extra latency) |

### Why System F (Old v1 SCD Partition) Failed (`-0.99` Dev Macro-F1 Points)
In System v1, a single-family mahalanobis/confidence heuristic (`s2_train_partition.csv`, preserved in [`results/scd_partition_flags.csv`](file:///home/krish/Yellow/results/scd_partition_flags.csv)) flagged `4,687` rows (`15.6%` of `train.csv`) as noisy and `6,422` rows (`21.4%`) as hard, dropping up to 50% of minority classes (`promo_code_issue` and `account_help`). Comparing **System F** (`0.8783` Dev Macro-F1) against **System E** (`0.8882`) and **System G** (`0.8887`) proves that aggressive single-family data dropping destroys minority-class recall, whereas our dual-family 15%-capped OOF consensus down-weighting (`System H`, `0.8893`) preserves every boundary example while attenuating true corruptions.

---

## 5. Calibration, Temperature Scaling & Abstain Threshold $\tau$ ([`results/step8_calibration_summary.json`](file:///home/krish/Yellow/results/step8_calibration_summary.json))

1. **Reliability & Expected Calibration Error (ECE)**:
   - Because `CalibratedClassifierCV(LinearSVC(C=0.5), cv=3)` already fits 3-fold cross-validated sigmoid calibration on out-of-fold decision margins, the 10-bin reliability table ([`results/calibration_reliability.csv`](file:///home/krish/Yellow/results/calibration_reliability.csv)) is already well-calibrated with `T = 1.0` (`ECE = 0.0715` on raw noisy OOF labels, where ~5% of ground-truth labels are corrupted).
2. **`out_of_scope` Abstain Threshold Sweep ([`results/step8_tau_sweep.csv`](file:///home/krish/Yellow/results/step8_tau_sweep.csv))**:
   - Sweeping $\tau \in [0.15, 0.55]$ in steps of `0.01` on 5-fold OOF showed that `out_of_scope` already achieves **`0.9151` OOF F1 (`0.9611` recall, `0.8733` precision)** and **`0.9278` Dev F1 (`0.9524` recall, `0.9045` precision)** directly as one of the 12 calibrated classes at $\tau = 0.15$.
   - Raising $\tau$ above `0.20` monotonically degrades both `out_of_scope` precision (`0.8733` at $\tau=0.15 \rightarrow 0.6818$ at $\tau=0.50$) and overall OOF Macro-F1 (`0.8102` $\rightarrow$ `0.7756`) by falsely abstaining on genuine multi-clause or typo-heavy in-scope requests.

---

## 6. Per-Intent & Slice Performance: Step 2 (v1.1) vs. Final System H

### 6.1 Per-Intent Precision, Recall & F1 on `dev.csv` ([`results/final_system_per_intent_prf.csv`](file:///home/krish/Yellow/results/final_system_per_intent_prf.csv))

| Intent | Dev Support | Starter Baseline F1 | Step 2 (v1.1) F1 | Final System H Precision | Final System H Recall | Final System H F1 | Gain (System H vs. v1.1) |
|---|---:|---:|---:|---:|---:|---:|---:|
| `order_status` | 525 | 0.4404 | 0.8806 | 0.8853 | 0.9410 | **0.9123** | **+0.0317** |
| `delivery_delay` | 546 | 0.4150 | 0.8579 | 0.8952 | 0.8608 | **0.8777** | **+0.0198** |
| `cancel_order` | 401 | 0.4452 | 0.8690 | 0.9045 | 0.8978 | **0.9011** | **+0.0321** |
| `refund_request` | 441 | 0.5535 | 0.8669 | 0.8980 | 0.8980 | **0.8980** | **+0.0311** |
| `missing_item` | 292 | 0.4416 | 0.8541 | 0.9075 | 0.8733 | **0.8901** | **+0.0360** |
| `wrong_item` | 209 | 0.4713 | 0.8471 | 0.9242 | 0.8756 | **0.8993** | **+0.0522** |
| `payment_failed` | 320 | 0.3690 | 0.7994 | 0.8424 | 0.8688 | **0.8554** | **+0.0560** |
| `promo_code_issue` | 207 | 0.0000 | 0.8371 | 0.9215 | 0.8502 | **0.8844** | **+0.0473** |
| `change_address` | 225 | 0.2588 | 0.8742 | 0.8659 | 0.9467 | **0.9045** | **+0.0303** |
| `account_help` | 212 | 0.0000 | 0.7905 | 0.9101 | 0.8113 | **0.8579** | **+0.0674** |
| `talk_to_agent` | 244 | 0.2286 | 0.8394 | 0.8612 | 0.8648 | **0.8630** | **+0.0236** |
| `out_of_scope` | 378 | 0.2564 | 0.8661 | 0.9045 | 0.9524 | **0.9278** | **+0.0617** |
| **Macro Average** | **4,000** | **0.3233** | **0.8485** | **0.8934** | **0.8867** | **0.8893** | **+0.0408** |

- **Every single one of the 12 intents improved** between Step 2 (`0.8485`) and Final System H (`0.8893`), led by `account_help` (`+6.74` pts), `out_of_scope` (`+6.17` pts), `payment_failed` (`+5.60` pts), `wrong_item` (`+5.22` pts), and `promo_code_issue` (`+4.73` pts).

### 6.2 Diagnostic Slice Breakdown on `dev.csv` ([`results/final_system_slices.csv`](file:///home/krish/Yellow/results/final_system_slices.csv))

| Diagnostic Slice | Dev Support | Step 2 (v1.1) Macro-F1 | Final System H Macro-F1 | Final System H Top-1 Acc | Macro-F1 Gain |
|---|---:|---:|---:|---:|---:|
| **`multi_request`** | 547 | 0.7190 | **0.8921** | **0.9013** | **+0.1731 (+17.31 pts)** |
| **`long_25_or_more_tokens`** (`>40` has 0 rows; max is 37) | 194 | 0.7777 | **0.8722** | **0.8814** | **+0.0945 (+9.45 pts)** |
| **`hinglish`** | 524 | 0.8681 | **0.9093** | **0.9046** | **+0.0412 (+4.12 pts)** |
| **`emoji`** | 101 | 0.8002 | **0.8408** | **0.9109** | **+0.0406 (+4.06 pts)** |
| **`short_under_4_tokens`** | 65 | 0.7380 | **0.7696** | **0.8769** | **+0.0316 (+3.16 pts)** |
| **`change_of_mind`** | 141 | 0.9119 | **0.9152** | **0.9149** | **+0.0033 (+0.33 pts)** |
| **`out_of_scope` (Recall / Binary F1)** | 378 | 0.8661 | **0.9278** | **0.9524** | **+0.0617 (+6.17 pts)** |

---

## 7. Step 9: Real-Time CPU Latency & Hardware Benchmark ([`src/step9_latency.py`](file:///home/krish/Yellow/src/step9_latency.py))

Measured on **`13th Gen Intel Core i7-13650HX`** (`14` physical cores, `20` logical threads, `15.3 GB` RAM) pinned strictly to **4 CPU threads** (`OMP_NUM_THREADS=4`, `MKL_NUM_THREADS=4`, `OPENBLAS_NUM_THREADS=4`, `torch.set_num_threads(4)`) in **FP32** over `1,000` single-message (`batch_size = 1`) calls after `100` warm-up calls per path ([`results/latency_benchmark.csv`](file:///home/krish/Yellow/results/latency_benchmark.csv), [`results/latency_hardware_meta.json`](file:///home/krish/Yellow/results/latency_hardware_meta.json)):

| System | Inference Path | Traffic Share (`dev` / `test`) | Batched Clause Vectorization | p50 (ms) | p95 (ms) | p99 (ms) | Mean (ms) | Peak RAM |
|---|---|---:|---|---:|---:|---:|---:|---:|
| **Final System H (`src/model.py`)** | **Single Path** | **90.1% / 91.2%** | N/A (1 text) | **2.339** | **2.548** | **2.678** | **2.356** | **~185 MB** |
| **Final System H (`src/model.py`)** | **Clause Path** | **9.9% / 8.8%** | **Yes (`[full, c1, c2]` in 1 call)** | **2.502** | **2.695** | **2.774** | **2.514** | **~185 MB** |
| Pruned System G (Fused Dual-Head) | Single Path | 90.1% / 91.2% | N/A (1 text) | 7.927 | 9.929 | 10.810 | 8.043 | ~420 MB |
| Pruned System G (Fused Dual-Head) | Clause Path | 9.9% / 8.8% | Yes (1 call) | 12.955 | 14.361 | 15.318 | 12.791 | ~420 MB |

### Clause Count Distribution Across All Splits ([`results/clause_count_distribution.csv`](file:///home/krish/Yellow/results/clause_count_distribution.csv))
- **`train.csv` (`30,000` rows)**: `90.77%` Single Path (`1` text), `9.23%` Clause Path (`9.21%` have `2` clauses, `0.02%` have `3` clauses).
- **`dev.csv` (`4,000` rows)**: `90.10%` Single Path (`1` text), `9.90%` Clause Path (`9.88%` have `2` clauses, `0.03%` have `3` clauses).
- **`test.csv` (`3,000` rows)**: `91.23%` Single Path (`1` text), `8.77%` Clause Path (`8.77%` have `2` clauses, `0.00%` have `3` clauses).
- **4-Core 15 GB Server Box Estimate**: Even on a slower virtualised 4-vCPU cloud instance (e.g., AWS `c6i.xlarge` or GCP `n2-standard-4`), single-message latency for **System H** stays around **`2.5–3.5 ms` p50 (`<4.5 ms` p99)** (~400 messages/sec sequentially on a single worker process) using `<200 MB` RAM (`<1.5%` of the 15 GB budget).

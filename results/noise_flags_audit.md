# Step 5: OOF Noise Filter & Rule-Repair Cue Audit

## 1. Per-Class OOF Noise Flag Distribution (`results/noise_flags_by_class.csv`)

A training row $i$ is flagged when **both** the 5-fold Lexical head and the 5-fold MiniLM+MLP head assign $P(y_{\text{silver}} \mid x_i) < 0.20$, **both** assign $\max_c P(c \mid x_i) \ge 0.80$, and **both agree** on the top predicted intent ($\hat{y}_{\text{lex}} = \hat{y}_{\text{mlp}} \neq y_{\text{silver}}$). Flagged rows are capped at **15% per silver class** ranked by confidence margin.

| Intent | Train Support | Raw Flagged | Raw Flagged % | 15% Cap | Capped Flagged | Capped Flagged % |
|---|---:|---:|---:|---:|---:|---:|
| `order_status` | 4,383 | 126 | 2.87% | 657 | 126 | 2.87% |
| `delivery_delay` | 4,385 | 253 | 5.77% | 657 | 253 | 5.77% |
| `cancel_order` | 3,265 | 167 | 5.11% | 489 | 167 | 5.11% |
| `refund_request` | 3,698 | 219 | 5.92% | 554 | 219 | 5.92% |
| `missing_item` | 2,278 | 98 | 4.30% | 341 | 98 | 4.30% |
| `wrong_item` | 1,730 | 125 | 7.23% | 259 | 125 | 7.23% |
| `payment_failed` | 2,547 | 88 | 3.46% | 382 | 88 | 3.46% |
| `promo_code_issue` | 325 | **91** | **28.00%** | 48 | **48** | **14.77%** |
| `change_address` | 1,862 | 53 | 2.85% | 279 | 53 | 2.85% |
| `account_help` | 388 | **109** | **28.09%** | 58 | **58** | **14.95%** |
| `talk_to_agent` | 2,128 | 158 | 7.42% | 319 | 158 | 7.42% |
| `out_of_scope` | 3,011 | 33 | 1.10% | 451 | 33 | 1.10% |
| **Total** | **30,000** | **1,520** | **5.07%** | — | **1,426** | **4.75%** |

### Key Structural Discovery: Why OOF Macro-F1 Is ~6 Points Lower Than Dev Macro-F1
Look at `promo_code_issue` (`325` rows, **28.00% raw flagged**) and `account_help` (`388` rows, **28.09% raw flagged**):
- In `train.csv`, `promo_code_issue` (1.08%) and `account_help` (1.29%) are **4.5x rarer** than in `dev.csv` (5.18% and 5.30%), yet they received roughly the same absolute number (~90–110 rows) of random cross-class label corruptions as the larger classes.
- Consequently, **28% of all `promo_code_issue` and `account_help` silver labels in `train.csv` are corrupted utterances from other intents** (e.g., `"OTP not coming for payment"`, `"the order was supposed to be here already"`, `"refund not received yet pls"`).
- Because 28% of the ground-truth denominator for those two intents in `train.csv` is random noise, any model evaluated against raw `train.csv` OOF labels has a hard recall ceiling of ~0.72 on `promo_code_issue` and `account_help` (yielding OOF F1 ~0.42–0.56 on those two classes vs. ~0.80–0.84 on `dev.csv`).

---

## 2. Manual Audit of 100 Random Flagged Train Rows (`results/noise_flags_sample100.csv`)

We inspected all 100 randomly sampled rows (`random_state=42`) from the 1,426 capped OOF-flagged rows:
- **98 / 100 (98.0%) are unambiguous silver label errors** where the silver label contradicts the plain meaning and labelling rules:
  - *Cross-domain corruptions*: `row_idx=15121` (`"CHANGGE MY TRAIN BOOKING DATE"`, silver=`cancel_order`, agreed=`out_of_scope`), `row_idx=20005` (`"movie tickets for tonight please help!!"`, silver=`refund_request`, agreed=`out_of_scope`), `row_idx=28856` (`"Hello, book a table at a restaurant."`, silver=`talk_to_agent`, agreed=`out_of_scope`), `row_idx=23743` (`"I am bored help me"`, silver=`order_status`, agreed=`out_of_scope`).
  - *Rule 4 complaints mislabelled as `cancel_order` or `refund_request`*: `row_idx=22026` (`"food is going to be cold by the time it arrives asap."`, silver=`cancel_order`, agreed=`delivery_delay`), `row_idx=10183` (`"hi, part of my order didn't come."`, silver=`refund_request`, agreed=`missing_item`), `row_idx=20951` (`"My eggs were not in the grocery bag"`, silver=`refund_request`, agreed=`missing_item`), `row_idx=500` (`"The pizza is not the one I picked"`, silver=`refund_request`, agreed=`wrong_item`).
  - *Direct swaps in `promo_code_issue` and `account_help`*: `row_idx=8959` (`"yaar need to spek to a person right now lol?"`, silver=`promo_code_issue`, agreed=`talk_to_agent`), `row_idx=6449` (`"payment is not going through kindly check..."`, silver=`promo_code_issue`, agreed=`payment_failed`), `row_idx=10859` (`"refund not received yet pls"`, silver=`account_help`, agreed=`refund_request`), `row_idx=25445` (`"can you deliver to a different address help me."`, silver=`account_help`, agreed=`change_address`).
- **2 / 100 (2.0%) are borderline/hard rows**:
  - `row_idx=2226` (`"hi quickcart, delivery partner not moving kindly check."`, silver=`order_status`, agreed=`delivery_delay`) and `row_idx=6540` (`"hello, hello please help."`, silver=`talk_to_agent`, agreed=`out_of_scope`).
- **0 / 100 (0.0%) are clean rows where the silver label is clearly right.**

---

## 3. Four-Way OOF Ablation Across Seeds 42, 43, 44 (`results/step5_noise_ablation.csv`)

| Variant | Lexical Imp-OOF F1 | MLP Imp-OOF F1 (s42 / s43 / s44) | MLP Mean $\pm$ Std | Fused Imp-OOF F1 (s42 / s43 / s44) | Fused Mean $\pm$ Std | Beats `(a) kept` on all 3 seeds? |
|---|---:|---:|---:|---:|---:|---|
| `(a) kept` (baseline) | 0.7834 | 0.7227 / 0.7224 / 0.7221 | 0.7224 $\pm$ 0.0002 | 0.7708 / 0.7704 / 0.7704 | 0.7705 $\pm$ 0.0002 | — |
| `(b) removed` | 0.7844 | 0.7268 / 0.7273 / 0.7278 | 0.7273 $\pm$ 0.0004 | 0.7709 / 0.7718 / 0.7718 | 0.7715 $\pm$ 0.0004 | **Yes** (MLP & Fused) |
| `(c) down-weighted (0.25)` | 0.7844 | 0.7264 / 0.7275 / 0.7268 | 0.7269 $\pm$ 0.0005 | **0.7722 / 0.7720 / 0.7721** | **0.7721 $\pm$ 0.0001** | **Yes** (MLP & Fused) |
| `(d) relabelled` | **0.7847** | **0.7279 / 0.7288 / 0.7269** | **0.7279 $\pm$ 0.0008** | 0.7708 / 0.7722 / 0.7721 | 0.7717 $\pm$ 0.0007 | **Yes** (MLP) |

- **Winner**: `(c) down-weighted to 0.25` achieves the highest Fused importance-weighted OOF Macro-F1 (`0.7721 ± 0.0001`), beating `(a) kept` (`0.7705 ± 0.0002`) on **all three seeds** (`0.7722 > 0.7708`, `0.7720 > 0.7704`, `0.7721 > 0.7704`), while `(d) relabelled` achieves the highest standalone MLP OOF Macro-F1 (`0.7279 ± 0.0008`).

---

## 4. Audit of 100 Rule-Repair Candidate Rows by Cue (`results/train_rule_repair_sample100.csv`)

We sampled 100 training rows triggered by clause cues (25 per cue type) to audit cue precision:

| Cue Category | Triggers in Train (30k) | Sample Audited | Valid Primary Clause Extracted | False / Destructive Splits | Root Cause of False Splits |
|---|---:|---:|---:|---:|---|
| `contrastive_but` (`"but"`, `"par"`, `"lekin"`) | 1,401 | 25 | **0 / 25 (0%)** | **25 / 25 (100%)** | Contrastive single intents (`"I ordered 3 things but only got 2"` $\rightarrow$ `"I ordered 3 things"`, `"delivered but my shake is absent"` $\rightarrow$ `"delivered"`) + boilerplate filler (`"I tried calling earlier but the line was busy"`). |
| `but_first_override` (`"but first"`, `"but before that"`) | 378 | 25 | **25 / 25 (100%)** | **0 / 25 (0%)** | Always follows a `talk_to_agent` prefix (`"connect me to an agent, but first order status please"` $\rightarrow$ `"order status please"`). |
| `change_of_mind` (incl. bare `"instead"`, `"actually"`) | 909 | 25 | **19 / 25 (76%)** | **6 / 25 (24%)** | Bare `"instead"` splits `"I got milk instead of curd"` (`wrong_item`) into `"of curd"` (4/25); bare `"actually"` splits `"get me someone who can actually help"` (`talk_to_agent`) into `"help"` (2/25). Explicit markers (`"actually never mind"`, `"actually forget it"`, `"forget that"`, `"scratch that"`) are **19/19 (100%)** valid. |
| `additive_multi_request` (incl. bare `"also"`) | 1,682 | 25 | **16 / 25 (64%)** | **9 / 25 (36%)** | Bare `"also"` splits boilerplate chitchat `"my neighbour also uses this app"` into `"my neighbour"` and `"my wifi is also acting up"` into `"my wifi is"` (9/25). Explicit conjunctions (`", also "`, `" and also "`, `" plus "`) are **16/16 (100%)** valid multi-requests. |

### Action Taken from the Cue Audit
1. **Removed `"but"`, `"par"`, `"lekin"`** from multi-request splitters (`0%` valid multi-request rate).
2. **Removed bare `"instead"` and bare `"actually"`**, keeping only explicit reversal phrases (`"actually never mind"`, `"actually forget it"`, `"no wait, forget that"`, `"forget that"`, `"never mind"`, `"nevermind"`, `"scratch that"`, `"ignore what i just said"`, `"wait no"`).
3. **Replaced bare `"also"` with `", also "` and `" and also "`**, eliminating false splits on `"my neighbour also uses this app"` and `"my wifi is also acting up"`.
4. **Train-time rule relabelling is turned OFF** (kept strictly as an inference-time clause routing path), because on non-corrupted rows the silver label already reflects the true primary clause.

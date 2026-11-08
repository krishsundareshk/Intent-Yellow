# Agent instructions: EDA for the QuickCart intent task

You are helping with a take-home task: intent classification, 12 intents, a synthetic food-delivery app. This session is **EDA only**. Do not build the final system, do not tune thresholds, do not edit data or the taxonomy.

The plan below follows a standard EDA practice for intent datasets, adapted to this dataset. Follow the steps in order.

## Goals

1. Quantify noise (label noise and text noise) with counts and confidence intervals.
2. Expose taxonomy weaknesses: overlap, gaps, missing examples, rule coverage.
3. Check what the task README claims about the data (list in the "README claims" section).
4. Leave tables in `results/eda/` and one short findings file a human can read in five minutes.
5. **End goal: one self-contained EDA dashboard** (`results/eda/dashboard.html`) that presents all of the above.

## Hard rules

- Read-only on `data/train.csv`, `data/dev.csv`, `data/test.csv`, `data/taxonomy.json`. Never overwrite, reorder or "fix" them. Write cleaned or flagged copies to `results/eda/`.
- Do not train on test, and do not predict or infer test labels. Test is used only for text profiling (length, language, noise flags, overlap with train/dev). Fit every classifier, vocabulary and clustering on train (and dev where stated) only.
- Everything runs locally. No hosted APIs, no uploading data anywhere. Open-source models downloaded and run locally are allowed for embeddings (for example `sentence-transformers/all-MiniLM-L6-v2` and a multilingual MiniLM). Record the exact model name and version.
- Fix random seeds (use 42 unless stated). Make every table reproducible by a script in `eda/` (one script per step, named `step1_audit.py`, `step2_profile.py`, and so on).
- Report what you measured. If something cannot be computed, say so and why. Do not round or reword numbers to look tidy.
- When you judge a label yourself (manual reading, audit), apply the labelling rules in the last section. Mark every judgement you made as `agent_judgement`. A human will spot-check them.
- This dataset has no channel, timestamp or session metadata. Where the standard plan says "split by channel or time", split by `train/dev/test`, intent label, language group and message length instead.

## Repo layout (expected)

```
data/        taxonomy.json  train.csv  dev.csv  test.csv
baseline/    baseline.py  eval.py
results/     baseline_dev.json
```

`train.csv` and `dev.csv`: `intent, utterance` (silver labels, not gold). `test.csv`: `id, utterance`. `taxonomy.json` layout is not known in advance: inspect it first and write a loader that handles what you find.

## Steps

### Step 0. Setup
- Create a virtual environment, install `requirements.txt`, then `pandas numpy scikit-learn scipy matplotlib rapidfuzz sentence-transformers cleanlab` (skip any that fail to install and say which).
- Check that the files in the layout exist. If any is missing, stop and report which.

### Step 1. Data audit
Build one row-level table `results/eda/row_level_dataset.csv` with: `row_id`, `split`, `raw_message`, `normalized_message` (lowercased, whitespace collapsed; keep raw in its own column), `assigned_label` (blank for test), `label_count` (1 for train/dev), `n_chars`, `n_words`. Later steps add flag columns to this table.

Integrity checks (`integrity_checks.csv`, one row per check: `check, split, expected, observed, status`):
- Row counts against the README (30,000 / 4,000 / 3,000), columns, null or blank utterances.
- Messages containing only whitespace, punctuation, emojis or random characters.
- Exact and normalised duplicate texts, within each split and across splits.
- Same text with different labels (`duplicates_conflicts.csv`; also list which label pairs conflict most: this is direct, lower-bound evidence of label noise and shows which way it goes).
- Labels not in the taxonomy, label spelling variants, intents missing from train.
- Messages per label (`label_counts.csv`: counts, share, rank, cumulative share for train and dev; imbalance ratio, normalised entropy, Gini; chi-square test dev vs train; a verdict on whether the distribution is long-tailed, with the metric you used).

Then:
- Draw a reproducible random sample of 200 train rows (seed 42) into `sample_for_reading.csv`. Read all of them **before writing any rule or flag**. Write what you notice (noise types, odd labels, style) in `manual_reading.md`.

### Step 2. Profile the messages (before any cleaning)
One table per item, split by train/dev/test and by label where useful:
- **Length**: characters, words, sentences/clauses; percentiles; distribution plot. (`profile_length.csv`)
- **Repetition**: duplicate rate, near-duplicate rate (character or token similarity above a stated threshold), most repeated n-grams. (`profile_repetition.csv`)
- **Language/script**: Devanagari share, English, romanised Hindi, mixed. (`profile_language_script.csv`)
- **Non-text**: emoji count, URLs, mentions, digits and order-ID patterns, punctuation density. (`profile_nontext.csv`)
- **Casing/spelling**: uppercase ratio, elongated letters, out-of-vocabulary rate against the train vocabulary and an English wordlist. (`profile_casing_spelling.csv`)
- **Label coverage**: frequency and entropy per label, restating Step 1 counts. (`label_coverage.csv`)
- **Long messages**: take the longest 5% by words. Split into sentences or clauses and record whether the first actionable request differs from the last one (`long_message_first_vs_last.csv`, at least 50 rows, agent_judgement). This matters because rule 1 (first request wins) and rule 2 (last request wins on a change of mind) give different labels.

### Step 3. Quantify the six noise types
Give each type its own flag column in `row_level_dataset.csv` (boolean, plus a severity score where it makes sense). Do not build a single "noisy" flag. Validate every flag: sample 20 flagged and 20 unflagged rows, judge them, and record precision and the false negatives you saw (`noise_flag_validation.csv`).

1. **Typos.** Build a domain vocabulary from taxonomy examples plus frequent train tokens (default: frequency of 20 or more; state what you used). Flag tokens within edit distance 1 or 2 of a vocabulary word but not in it, repeated characters, missing spaces. Separate meaning-changing typos from harmless ones on a sample. Produce `normalization_map.csv` for recurring variants (keep the original text).
2. **Abbreviations.** Extract frequent short tokens (4 characters or fewer) not in the English wordlist. Compare their frequency across labels, because some may be intent signals. Draft `abbreviation_dictionary_draft.csv` (for example `pls -> please`) and check each expansion on examples. Check overlap with item names so that expansions do not damage entities.
3. **Emojis.** Count per message, top emojis, emoji by label, emoji-only and emoji-plus-punctuation messages (`emoji_stats.csv`). Test whether emojis relate to particular labels or multi-request messages. Recommend a policy: translate, remove or keep as tokens.
4. **Hinglish.** Detect Devanagari, romanised Hindi tokens and code-switching. Build a lexicon (`hinglish_lexicon.csv`): seed it with common romanised Hindi function words, then add data-driven candidates (frequent train tokens not in the English wordlist) and review them by hand. Report the share of English, Devanagari, romanised Hindi and mixed messages per split, and the label distribution per language group.
5. **Long rambling messages.** Define "long" using both word count and clause count. Detect filler, repetition and multiple topics. Annotate 30 long messages: primary request, secondary request, background, complaint, irrelevant content (`long_message_annotation.csv`). Report what share contain a clear actionable request.
6. **Multi-request messages.** Flag with cue patterns ("also", "plus", "and then", "and also", numbered lists, question marks, Hinglish equivalents like "aur"). Review 50 flagged and 50 unflagged rows and record the actual number of distinct requests (`multi_request_review.csv`). For 30 of the multi-request rows, write down which label the README rules would give and compare it with the silver label: this directly measures how well the silver labels follow rules 1, 2, 3 and 5.

Outputs: `noise_scorecard.csv` (count and share per noise type per split) and `noise_by_label_matrix.csv` (noise type by intent, train).

### Step 4. Audit the taxonomy
Treat the taxonomy as a hypothesis. Produce `taxonomy_health.csv` with one row per intent:
`label, n_examples, definition (your reading of it), positive_examples, negative_examples, top_overlapping_label, uncovered_cluster_themes, recommended_action` (keep, merge, split, rename or retire) and a one-line reason.

Check:
- **Coverage**: intents with zero taxonomy examples (the README says two), and message types the taxonomy cannot express.
- **Overlap**: closest intent pairs by TF-IDF centroid cosine from the taxonomy and from train (`intent_overlap_pairs.csv`), and which label pairs conflict in Step 1. Read 5 train messages from each side of the 3 closest pairs and say whether a new human agent could tell them apart.
- **Granularity**: does any label hold operationally different actions?
- **Actionability**: does each label map to one clear system action?
- **Label consistency**: top distinctive n-grams per label (log-odds) and near-miss examples from other labels (`label_top_ngrams.csv`).
- **Edge-case rules**: for each of the 8 README rules, say whether the taxonomy examples cover it, with examples (`taxonomy_rule_coverage.csv`). For rule 6, check specifically for gift cards, loyalty points, tax invoices, tipping riders and restaurant feedback.
- **Taxonomy and data overlap**: how many taxonomy examples appear verbatim or nearly verbatim in train or dev, and whether the labels agree (README hint: they were not created independently).

### Step 5. Cluster to discover gaps
- Embed `normalized_message` for train with a small sentence-embedding model (state the model). Cluster with k-means (try k = 12, 20, 30; choose by silhouette and say so) or HDBSCAN.
- Cross-tabulate clusters against assigned labels (`cluster_vs_label_crosstab.csv`). Read 10 messages from each of the 10 largest clusters (`cluster_samples.md`).
- Interpret: a cluster with one dominant label validates that intent; a cluster split across labels marks an unclear boundary; a large cluster with no consistent label may be a missing intent; a cluster of long or multi-request messages may need special handling.
- Summarise in `cluster_summary.csv`.

### Step 6. Detect likely label noise
1. Train TF-IDF + logistic regression (5-fold, seed 42) on train. Repeat with embeddings + logistic regression if Step 5 embeddings exist. Get out-of-fold probabilities for every train row.
2. Flag a row if the predicted label differs from the assigned label and the predicted probability is 0.8 or higher, or if the probability of the assigned label is below 0.2. State your thresholds. Optionally also run `cleanlab` and compare its list to yours. Write the ranked list to `label_quality_queue.csv`. **Do not delete or relabel anything.**
3. Review the highest-risk rows in stratified batches: 10 per assigned label, plus 5 random low-risk rows per label as a control. Assign one issue category to each: wrong label, ambiguous message, missing intent, duplicate with conflicting label, genuinely noisy text, or label correct. Record in `label_noise_audit.csv` (agent_judgement, and the README rule you applied).
4. **Noise estimate.** Draw a separate simple random sample of 300 train rows (seed 43). Judge each against the README rules. Report the silver error rate with a Wilson 95% confidence interval, overall and per assigned label (`label_noise_estimate.csv`). Build a noise transition matrix (assigned label to your judged label) in `noise_transition_matrix.csv`, and say whether errors concentrate in particular intents or slices (long, multi-request, Hinglish, emoji). Also report how precise the model-flagged queue was against your judgements.

### Step 7. Decision log and findings
- **Decision log** `decision_log.md`: explicit rules for single-label versus multi-label (the README requires exactly one label per message, with precedence rules, so record how multi-request messages are handled), primary-intent precedence, normalisation policy for typos, abbreviations, emojis and Hinglish, and out-of-scope handling.
- **Findings** `eda_findings.md` with exactly these sections, under 2 pages, tables over prose:
  1. Summary (max 8 bullets, each with a number)
  2. README claims: verified or not (table, one line of evidence each)
  3. New problems found (numbered: what, how many rows or intents, evidence file, severity high/medium/low)
  4. Label noise: estimate with CI, direction, where it concentrates
  5. Noise scorecard (six types, per split)
  6. Taxonomy health and recommended changes
  7. What the flags and judgements cannot tell us (flag precision, agent-judgement limits)
  8. Not checked or could not verify
  9. Suggested next steps (max 5, each tied to a finding)

### Step 8. EDA dashboard (the end goal)
Build `results/eda/dashboard.html` with `eda/build_dashboard.py`. The script reads only the CSV and Markdown outputs of steps 1 to 7. No recomputation, and no numbers typed into the page by hand. Also write `results/eda/dashboard_README.md` with the one command that rebuilds it.

**Format requirements**
- One self-contained HTML file. No CDN and no network needed to view it: embed the chart library (for example plotly with the JS included) or embed charts as base64 images. Keep it under about 15 MB. Embed aggregates and at most 200 rows of the label-quality queue, not the full row-level table.
- Opens by double-click. Readable at laptop width, and usable on a phone if that is cheap.
- A tab bar or side menu for the sections below. A header with title, build date, random seeds, model names, and row counts per split.
- Tables are sortable and searchable (small inline JS is fine).

**Sections**

| # | Section | Contents | Source files |
|---|---|---|---|
| 1 | Overview | Tiles: rows per split, intents, imbalance ratio and long-tail verdict, silver error rate with 95% CI, share of texts with conflicting labels, README claims verified vs not. Top 5 findings. | `label_counts.csv`, `label_noise_estimate.csv`, `integrity_checks.csv`, `eda_findings.md` |
| 2 | Data integrity | Checks table with pass/FAIL colouring, duplicates and cross-split overlap, conflicting label pairs chart | `integrity_checks.csv`, `duplicates_conflicts.csv` |
| 3 | Label distribution | Train vs dev bars, cumulative share curve, balance stats (ratio, entropy, Gini, chi-square) | `label_counts.csv` |
| 4 | Text profile | Length histograms by split, language/script mix, non-text rates, repetition | `profile_*.csv` |
| 5 | Noise scorecard | Six noise types per split (count and share), noise-by-label heatmap, flag validation precision, top emojis, Hinglish lexicon top terms | `noise_scorecard.csv`, `noise_by_label_matrix.csv`, `noise_flag_validation.csv`, `emoji_stats.csv`, `hinglish_lexicon.csv` |
| 6 | Taxonomy health | Per-intent table with recommended action, rule coverage grid (8 rules), intent similarity heatmaps (taxonomy-based and train-based), closest pairs | `taxonomy_health.csv`, `taxonomy_rule_coverage.csv`, `intent_overlap_pairs.csv` |
| 7 | Clusters | Cluster by label heatmap, cluster summary, sample messages from the top clusters | `cluster_*.csv`, `cluster_samples.md` |
| 8 | Label noise | Error rate with CI per label, noise transition matrix, flagged queue (top 200, filter by label and issue category), model-flag precision against the audit | `label_noise_estimate.csv`, `noise_transition_matrix.csv`, `label_quality_queue.csv`, `label_noise_audit.csv` |
| 9 | README claims | Table of each claim, verdict and evidence | `eda_findings.md` |
| 10 | Decision log and caveats | Rendered `decision_log.md`, plus what the flags and judgements cannot tell us | `decision_log.md`, `eda_findings.md` |

**Design requirements**
- Every chart has a title that says what it shows, the sample size n, labelled axes, and the source file name in small text.
- Same colour for the same intent in every chart. Colour-blind safe palette. Do not rely on colour alone to carry meaning.
- Show a visible badge "agent judgement, not human verified" on anything that comes from your own labelling (audit, noise estimate, transition matrix). Show "keyword proxy" on proxy flags.
- Test text appears only in profiling views. There are no test labels anywhere.

**Verification before you finish**
- Render the page in a headless browser (for example Playwright with Chromium), screenshot every section, and confirm there are no empty or broken charts and no console errors.
- Check at least 8 numbers shown on the dashboard against the source CSVs. Record the checks in `results/eda/dashboard_checks.md`.
- If you cannot render it, say so, and do a structural check of the HTML and the embedded data instead.

Out of scope this session: revising the taxonomy, cleaning the data, training the final model, building a golden review set, and re-running the EDA after revisions. List these as next steps instead.

## README claims to verify
| Claim | Report |
|---|---|
| train 30,000, dev 4,000, test 3,000 rows | Actual counts |
| 12 intents | Actual count and names |
| Two intents have nothing to match against in the taxonomy | Which intents, and their train row counts |
| Taxonomy has "a few" examples each | Min, median, max |
| Silver labels contain non-random errors | Conflict counts and pairs, noise estimate and transition matrix |
| Taxonomy examples and data were not created independently | Verbatim and near-verbatim overlap, label agreement |
| Some intents are close | Closest pairs, taxonomy-based vs train-based |
| Messages are noisy (typos, abbreviations, emoji, Hinglish, long, multi-request) | Noise scorecard |
| test.csv has no labels | Columns present |
| Starter reports top-1 0.423, top-3 0.719, MRR 0.665 on dev | Run `python -m baseline.eval`, compare with `results/baseline_dev.json`, report any difference. Do not debug the baseline this session |

## Labelling rules (use these to judge labels)
1. One label per message. If several things are asked, the first request decides, except for rules 2 and 5.
2. If the customer changes their mind, the last request wins.
3. Cancel plus refund: not yet delivered is `cancel_order`; delivered is `refund_request`.
4. A complaint with no request is labelled by the problem: late is `delivery_delay`, absent is `missing_item`, different is `wrong_item`.
5. Asking only for a human is `talk_to_agent`. If another request is present, the other request wins.
6. Not about QuickCart orders, accounts or payments is `out_of_scope`: small talk, gibberish, other companies, unsupported services (gift cards, loyalty points, tax invoices, tipping riders, restaurant feedback). A greeting or thanks alone is `out_of_scope`. A greeting plus a real request takes the request's label.
7. Filler, emotion and unrelated detail never change the label.
8. Order IDs, amounts and item names never change the label.

## Log, commit and report
- Append to `prompt_log.md` (create it if missing): the instruction you were given, a one-line note on what you did, and anything you got wrong, changed or rejected. Do not tidy it.
- Commit scripts and `results/eda/` in small commits with clear messages (for example `eda: step 1 audit and integrity checks`). Do not commit data files that are not already tracked.
- Final reply to the human: what ran, what failed and how you worked around it, the path to `dashboard.html`, the 5 most important findings with numbers, any README claim that did not hold, and what you are unsure about. Point to files instead of pasting tables.

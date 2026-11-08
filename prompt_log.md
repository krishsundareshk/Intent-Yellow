# Prompt Logs

-- Notebooklm 
Paper-1: On the use of Silver Standard Data for Zero-shot Classification Tasks in Information Extraction
Act as a professional researcher and developer who is actively working and researching on Intent Detection or Classification. For the uploaded paper I want to know more about the proposed framework in the context of zero shot performance. I want you to extract the metrics and talk about the metrics based on the ablation study or the experiment performed. Out of the 4 phases of Clean Lave, discuss mainly on phase 2,3,4 as I already have available silver data.
- **Note / What we did**: Extracted Phases 2–4 of CleanLaVe (confident-learning noise filtering, relabelling with soft weights, and hard-batch retraining schedule). Later tested Sahayam et al.'s hard-batch retraining in System v1 and **rejected it** (`0.7899` vs `0.7969` plain MLP) because oversampling the "hard" bucket overfit unflagged silver label noise.

Paper-2: Differences Between Hard and Noisy-labeled Samples: An Empirical Study
In this paper the discussion is being focused on finding noisy samples and hard samples. Highlight the similarities and differences between hard samples and noisy samples.  Explain the metric that was used for filtering out the noisy samples and the mathematics behind it as well. Discuss about the data partitioning methods used in this paper. 
- **Note / What we did**: Extracted the Sample-to-Centroid Distance ($SCD_i = \|h(x_i) - o_{y_i}\|_2$) and per-epoch training accuracy $Acc_i$ 3-component GMM partitioning (Forouzesh & Thiran, 2023). Implemented it in System v1 (`s2_partition.py`), but **rejected and dropped it in Step 7 (System F)** after 5-fold OOF and Dev ablation proved that SCD dropped 4,687 rows (up to 50% of minority classes `promo_code_issue` and `account_help`) and hurt Dev Macro-F1 by `-0.99` points (`0.8783` vs `0.8882`); replaced it with our 5-fold OOF dual-head consensus noise filter (`results/noise_flags.csv`).

Paper-3: Intent Detection in the Age of LLMs
Here let's focus on understanding the uncertainty based routing strategy. What is the use of negative data augmentation in this pipeline? How a lower latency is achieved? What is the influenced by the scope of intent labels and the size of the label space mean, back it up with experiments and metrics. Also explain the 2 step approach using the LLM representations as well. 
- **Note / What we did**: Adopted the fast path vs. slow/clause path routing concept and calibrated energy/probability abstain gating for `out_of_scope` detection under the 4-CPU-core real-time budget.

-- Antigravity
Prompt-1: Read all the code files, results, guidelines and every other document present inside yellow folder and build an understanding out of it and write it down on a understanding.md file and store it under the yellow folder. 
- **Note / What we did**: Audited the repository, identified the critical `baseline/eval.py` label-order bug (`target_names` without `labels=intents` + reporting weighted F1 `0.4455` as macro F1 when true macro F1 is `0.3233`) and the two zero-shot taxonomy intents (`promo_code_issue`, `account_help` with 0 seed examples).

-- Claude
Prompt-1: I have done a small survey on methods and architecture revolving around intent classification. Based on the provided research content, give me a list of things which are feasible based on the system constraints. Just give me the list of ideas, don't bake your own ideas. And give me the insights that has been gathered and write it down in the markdown file.
- **Note / What we did**: Filtered the literature survey down to techniques feasible within 4 CPU cores and 15 GB RAM (frozen small encoders, character + word n-gram linear models, OOF noise filtering, deterministic clause splitting, and confidence-based `out_of_scope` gating).

Prompt-2: Based on the way I have found the research papers, I want you to use the web search tool and find other relevant research papers. Give me the links to the papers and take papers post 2024.
- **Note / What we did**: Collected 2024+ papers on intent detection, multi-intent clause decomposition, and learning with noisy silver labels to cross-check against our 4-core CPU latency ceiling.

Prompt-3: Now the next step I want to do is 2 things, first I will manually verify the findings from the papers you have given and the next is to do EDA on the data we have. We need to gather insights as well as identify problems in the data. Those are the goals. Let's discuss the EDA steps we will be performing
- **Note / What we did**: Designed the 6-step EDA plan (integrity/duplicate checks, text & script profiling, noise quantification, taxonomy overlap audit, MiniLM k-means clustering, and 5-fold cross-validated label noise estimation) saved in `AGENT_EDA_INSTRUCTIONS.md`.

-- Antigravity
Prompt-2: https://github.com/krishsundareshk/Intent-Yellow.git, this is the repository we will be working on, so I want you to commit every experiment we are doing and any changes we do. Keep this link onto a .env file to access it anytime.
- **Note / What we did**: Saved the remote URL to `.env`, configured SSH git remotes, and tracked every experiment via git commits.

Prompt-3: There is a new markdown file named as agent eda instructions, now i want you to follow the instructions and create ipnyb files as well for the eda and save them
- **Note / What we did**: Executed `eda/step1_audit.py` through `eda/step6_label_noise.py` and generated `notebooks/01_exploratory_data_analysis.ipynb`. **Rejected / Corrected in next prompt**: The initial `step6_label_noise.py` included heuristic agentic judgements on silver labels, which we removed in Prompt-4 in favour of pure 5-fold cross-validated out-of-fold disagreement.

Prompt-4: For the EDA, remove the agentic judgement on the Silver Label part, now you have provided me csv files in the HTML page, but I want to see visualizations and interpretations as well. So I want you to reconstruct the dashboard with proper charts and interpretations. And clean unecessary files as well. After contructing the files, push them to GITHUB under a new branch called EDA.
- **Note / What we did**: Removed subjective label grading from `eda/step6_label_noise.py`, rebuilt `results/eda/dashboard.html` with interactive Chart.js plots and interpretations, deleted empty/redundant CSVs, and pushed branch `EDA`.

Prompt-5: For the critical problems identified, back it up with evidence with plots and other things. I will share you the readme file, for the below readme file as well. Give me a clean dashboard in dark theme. Identify proper problems and keep these generated csv files in a separate folder. Build a clean artifact for dashboard.
- **Note / What we did**: Moved all 29 EDA CSV/JSON tables into `results/eda_tables/`, added `eval_bug_evidence.csv`, rebuilt `results/eda/dashboard.html` in a dark theme with 6 evidence charts, and verified all dashboard numbers in `results/eda/dashboard_checks.md`.

Prompt-6: Run the baseline again and verify the metrics and delete unecessary files
- **Note / What we did**: Re-ran `python -m baseline.eval` and `python -m baseline.baseline`, confirmed exact baseline metrics (`0.423` Top-1, `0.446` buggy reported F1 vs `0.3233` true macro-F1), and removed duplicate CSVs from `results/eda/`.

Prompt-7: Store this eda_evidence_dashboard.md file inside the Yellow folder
- **Note / What we did**: Saved the dashboard evidence summary in the workspace and committed to `EDA`.

-- Claude
Prompt-4: Based on the EDA findings and the papers we surveyed, let's design System v1 (S0 to S8) under the 4 CPU core and 15 GB RAM constraint. Write down the step-by-step system design in the documentation file so we can test encoder gating, SCD partitioning, rule-based clause repair, and dual-head fusion.
- **Note / What we did**: Drafted the S0–S8 System Design v1 specification in the documentation file. **What turned out wrong**: The v1 design included `"but"` (`par`/`lekin`), bare `"also"`, and bare `"instead"`/`"actually"` in the clause splitter cues, carved `dev.csv` into sub-splits, and relied on the SCD partition—all of which we diagnosed and replaced in Prompts 8–10.

-- Antigravity
Prompt-8: Check the current architecture in the documentation file, create the code files, train the model using the train dataset using the GPU and test and evaluate the model using the dev set. And identify the flaws and bottlenecks.
- **Note / What we did**: Implemented System v1 (`s0`–`s8`) and reached `0.8485` Macro-F1 on full untouched silver `dev.csv`. Diagnosed that `multilingual-e5-small` lost to `all-MiniLM-L6-v2` on Romanised Hinglish (`0.6679` vs `0.7760`), SCD over-flagged minority classes, and Sahayam hard-batch retraining overfit label noise. **What we got wrong in this step**: We initially blamed a "Silver Evaluation Illusion" (rule violations in `dev.csv`) for why the v1.1 clause path struggled on silver `dev.csv`, missing that our own `"but"` and bare `"also"` splitters were corrupting single-intent complaints.

Prompt-9: Give me a markdown file with all the information on experiments and architectures, results, flaws and other things as well.
- **Note / What we did**: Wrote the System v1 technical summary and bottleneck breakdown so we could review all failure modes before designing the strict OOF selection plan.

-- Claude
Prompt-5: Look at the flaws in System v1 (0.8485 on untouched silver dev). We must not repair dev labels or split dev for tuning. Help me write strict 10-step instructions to push untouched silver dev Macro-F1 toward 0.90 using: (1) a single preprocess/guard module, (2) manual error analysis of 30 rows per top confused pair on v1.1, (3) 5-fold StratifiedGroupKFold on train, (4) importance-weighted OOF metrics, (5) OOF consensus noise filtering + a 100-row cue audit reporting 'but' separately, (6) lexical head tuning on OOF, (7) a 7-system ablation table with 1,000 paired bootstrap CIs and a 1 ms latency pruning rule, (8) OOF calibration, (9) separate single-path vs clause-path 4-thread FP32 latency benchmarks, and (10) final submission and 2-page report.
- **Note / What we did**: Produced the 10-step evaluation protocol enforcing strict train-only OOF selection and cue-level auditing.

-- Antigravity
Prompt-10: Execute the 10-step agent instructions: Goal is to raise dev macro-F1 from about 0.85 toward 0.90 with evidence for every component. Dev labels stay as they are (silver, no repair, no edits). Report what you measured, not what you expect.
- **Note / What we did**: Executed Steps 1–10 (`preprocess.py`, `src/clause_router.py`, `src/model.py`, `src/step2_error_analysis.py`..`src/step10_final_submission.py`):
  1. **Caught & fixed 3 bugs in our own v1.1 clause splitter** via the Step 2 error reading (`results/error_analysis/error_notes.md`) and Step 5 100-row cue audit (`results/train_rule_repair_sample100.csv`): removed `"but"` (`0/25` valid multi-requests; it chopped `"I ordered 3 things but only got 2"`), replaced bare `"also"` (`9/25` false splits on `"my neighbour also uses this app"`) with `", also "` / `" also, "` / `" and also "`, and removed bare `"instead"` / `"actually"` (`6/25` false splits on `"instead of"` and `"actually help"`). This jumped `multi_request` dev F1 from `0.7190` to `0.8921` (`+17.31` pts).
  2. **Replaced SCD with the Step 5 OOF Noise Filter** (`results/noise_flags.csv`): 98/100 audited flags were true silver label errors, and down-weighting the 1,426 capped rows to `0.25` beat keeping them across all 3 seeds (`42, 43, 44`).
  3. **Pruned the `MiniLM + MLP` head per Step 7's pruning rule** (`results/ablation.csv`): tuning the lexical head (`word 1-2 + char_wb 2-5` + `CalibratedLinearSVC C=0.5`) in Step 6 beat the standalone MLP by `+7.05` points (`0.8651` vs `0.7946`), and adding MLP fusion had a 95% bootstrap CI including zero (`[-0.0020, +0.0019]`) while costing `+5.6 ms` to `+10.5 ms` CPU latency (`> 1 ms` threshold). Final **System H** reached **`0.8893` Macro-F1 (`0.8920` Top-1 Acc)** on untouched silver `dev.csv` at **`2.34 ms` (Single Path) / `2.50 ms` (Clause Path)** p50 CPU latency.

Prompt-11: Clean unecessary files and refactor everything, push it to github in the main branch and zip the codebase as well for me to submit in their portal
- **Note / What we did**: Deleted `175+ MB` of legacy v1 caches and superseded scripts, refactored the winning classifier into `src/model.py`, restored `submissions/baseline_predictions.csv` to the starter output while keeping `submissions/krish_predictions.csv` for submission, pushed to `origin/main`, and built `QuickCart_Intent_Submission_Krish.zip`.

Prompt-12: Write the rest of the prompt log using the same vocabulary and keep these contents as well (`-- Notebooklm`, `-- Antigravity`, `-- Claude`).
- **Note / What we did**: Updated `prompt_log.md` in the exact `-- Notebooklm` / `-- Antigravity` / `-- Claude` `Prompt-N:` format with one-line action and rejection notes, pushed to `origin/main`, and refreshed `QuickCart_Intent_Submission_Krish.zip`.

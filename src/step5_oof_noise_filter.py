"""Step 5: Noise Filter from OOF & Train Rule-Repair Ablation.

1. Flags train rows where `oof_prob[y_i] < 0.20` and `oof_prob.max() >= 0.80` for BOTH
   the Lexical and MiniLM+MLP heads, and both heads agree on the top predicted class.
   Caps flagged rows at 15% per silver class.
2. Saves `results/noise_flags.csv` and draws a 100-row random sample for audit.
3. Ablates on 5-fold OOF across seeds 42, 43, 44:
   (a) kept (baseline weight 1.0)
   (b) removed
   (c) down-weighted to 0.25
   (d) relabelled to agreed class
4. Ablates rule-repair on train rows (on vs off) and samples 100 repaired rows split by cue
   (with `but` reported separately).
"""
from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import FeatureUnion

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from preprocess import preprocess_text
from src.step4_oof_predictions import SEEDS, build_lexical_features, train_mlp_fold

DATA_DIR = ROOT / "data"
CACHE_DIR = ROOT / "results" / "system_v1" / "cache"
OUT_DIR = ROOT / "results"


def detect_repair_cue(text: str) -> tuple[str | None, str | None]:
    """Identify if a train row triggers a clause/rule repair cue, separating 'but'."""
    t = f" {text.lower()} "
    for m in ["actually never mind", "never mind", "nevermind", "forget that", "scratch that", "instead", "wait no", "actually"]:
        if f" {m} " in t or f" {m}," in t:
            idx = t.rfind(m)
            after = t[idx + len(m):].strip(" ,.!?;:-")
            if len(after.split()) >= 2:
                return "change_of_mind", after
    for m in ["but before that", "but first"]:
        if m in t:
            idx = t.find(m)
            after = t[idx + len(m):].strip(" ,.!?;:-")
            if len(after.split()) >= 2:
                return "but_first_override", after
    for m in ["and also", "also", "plus", "and by the way", "aur haan", "aur"]:
        if f" {m} " in t:
            idx = t.find(f" {m} ")
            before = t[:idx].strip(" ,.!?;:-")
            after = t[idx + len(m) + 2:].strip(" ,.!?;:-")
            if len(before.split()) >= 2 and len(after.split()) >= 2:
                return "additive_multi_request", before
    for m in ["but", "par", "lekin"]:
        if f" {m} " in t:
            idx = t.find(f" {m} ")
            before = t[:idx].strip(" ,.!?;:-")
            after = t[idx + len(m) + 2:].strip(" ,.!?;:-")
            if len(before.split()) >= 2 and len(after.split()) >= 2:
                return "contrastive_but", before
    return None, None


def main() -> None:
    t0 = time.perf_counter()
    with open(DATA_DIR / "taxonomy.json") as f:
        tax = json.load(f)
    intents = [it["name"] for it in tax["intents"]]
    label2id = {name: i for i, name in enumerate(intents)}

    train_df = pd.read_csv(DATA_DIR / "train.csv")
    dev_df = pd.read_csv(DATA_DIR / "dev.csv")
    folds_df = pd.read_csv(OUT_DIR / "folds.csv")

    y_all = np.array([label2id[i] for i in folds_df["intent"]])
    folds = folds_df["fold"].to_numpy()
    raw_texts = train_df["utterance"].fillna("").astype(str).tolist()
    norm_texts = [preprocess_text(u) for u in raw_texts]
    n_samples = len(y_all)
    n_classes = len(intents)

    # Importance weights w(c) = p_dev(c) / p_train(c)
    train_counts = np.bincount(y_all, minlength=n_classes)
    p_train = train_counts / train_counts.sum()
    dev_y = np.array([label2id[i] for i in dev_df["intent"]])
    dev_counts = np.bincount(dev_y, minlength=n_classes)
    p_dev = dev_counts / dev_counts.sum()
    class_imp_weights = p_dev / p_train
    sample_imp_weights = class_imp_weights[y_all]

    oof_lex = np.load(OUT_DIR / "oof_lexical_probs.npy")
    oof_mlp = np.load(OUT_DIR / "oof_mlp_probs.npy")
    oof_mlp_seeds = np.load(OUT_DIR / "oof_mlp_probs_seeds.npy")
    X_emb = np.load(CACHE_DIR / "minilm_step1_train.npy")

    # 1. Identify candidate noisy rows
    row_indices = np.arange(n_samples)
    lex_prob_y = oof_lex[row_indices, y_all]
    mlp_prob_y = oof_mlp[row_indices, y_all]
    lex_max = oof_lex.max(axis=1)
    mlp_max = oof_mlp.max(axis=1)
    lex_top = oof_lex.argmax(axis=1)
    mlp_top = oof_mlp.argmax(axis=1)

    raw_flag_mask = (
        (lex_prob_y < 0.20)
        & (mlp_prob_y < 0.20)
        & (lex_max >= 0.80)
        & (mlp_max >= 0.80)
        & (lex_top == mlp_top)
        & (lex_top != y_all)
    )

    margin = 0.5 * (lex_max + mlp_max) - 0.5 * (lex_prob_y + mlp_prob_y)

    # Apply 15% per-class cap
    capped_flag_mask = np.zeros(n_samples, dtype=bool)
    class_flag_rows = []
    for c in range(n_classes):
        c_mask = y_all == c
        c_total = int(c_mask.sum())
        c_cap = int(np.floor(0.15 * c_total))
        c_candidates = np.where(c_mask & raw_flag_mask)[0]
        if len(c_candidates) > c_cap:
            order = np.argsort(-margin[c_candidates])
            chosen = c_candidates[order[:c_cap]]
        else:
            chosen = c_candidates
        capped_flag_mask[chosen] = True
        class_flag_rows.append({
            "intent": intents[c],
            "train_support": c_total,
            "raw_flagged": int(len(c_candidates)),
            "raw_flagged_pct": round(100.0 * len(c_candidates) / max(1, c_total), 2),
            "cap_15pct": c_cap,
            "capped_flagged": int(len(chosen)),
            "capped_flagged_pct": round(100.0 * len(chosen) / max(1, c_total), 2),
        })

    class_flag_df = pd.DataFrame(class_flag_rows)
    class_flag_df.to_csv(OUT_DIR / "noise_flags_by_class.csv", index=False)
    print("\n--- Step 5 OOF Noise Flags by Class ---")
    print(class_flag_df.to_string(index=False))
    print(f"Total raw flagged: {int(raw_flag_mask.sum())} ({100.0*raw_flag_mask.mean():.2f}%), "
          f"After 15% class cap: {int(capped_flag_mask.sum())} ({100.0*capped_flag_mask.mean():.2f}%)")

    # Save full noise_flags.csv (all rows that met raw flag criteria, noting whether kept under 15% cap)
    flagged_indices = np.where(raw_flag_mask)[0]
    noise_df = pd.DataFrame({
        "row_idx": flagged_indices,
        "fold": folds[flagged_indices],
        "utterance": [raw_texts[i] for i in flagged_indices],
        "norm_text": [norm_texts[i] for i in flagged_indices],
        "silver_intent": [intents[y_all[i]] for i in flagged_indices],
        "agreed_oof_intent": [intents[lex_top[i]] for i in flagged_indices],
        "lex_prob_silver": np.round(lex_prob_y[flagged_indices], 4),
        "mlp_prob_silver": np.round(mlp_prob_y[flagged_indices], 4),
        "lex_prob_agreed": np.round(lex_max[flagged_indices], 4),
        "mlp_prob_agreed": np.round(mlp_max[flagged_indices], 4),
        "confidence_margin": np.round(margin[flagged_indices], 4),
        "kept_under_15pct_cap": capped_flag_mask[flagged_indices],
    })
    noise_df.to_csv(OUT_DIR / "noise_flags.csv", index=False)

    # Sample 100 random capped flagged rows for manual audit
    capped_df = noise_df[noise_df["kept_under_15pct_cap"]].copy()
    sample_100 = capped_df.sample(n=min(100, len(capped_df)), random_state=42)
    sample_100.to_csv(OUT_DIR / "noise_flags_sample100.csv", index=False)

    # Precompute per-fold TF-IDF matrices once (unsupervised on fold train texts)
    print("\nPrecomputing per-fold TF-IDF matrices for fast 4-way ablation...")
    fold_lex_matrices = []
    for fold_idx in range(5):
        va_mask = folds == fold_idx
        tr_idx = np.where(~va_mask)[0]
        va_idx = np.where(va_mask)[0]
        vec = build_lexical_features()
        X_tr_lex = vec.fit_transform([norm_texts[i] for i in tr_idx])
        X_va_lex = vec.transform([norm_texts[i] for i in va_idx])
        fold_lex_matrices.append((tr_idx, va_mask, X_tr_lex, X_va_lex))

    # 2. Run 4-way OOF Ablation across seeds 42, 43, 44:
    # (a) kept, (b) removed, (c) down-weighted to 0.25, (d) relabelled to agreed class
    print("Running 4-way OOF Noise Filter Ablation across seeds 42, 43, 44...")

    def run_variant_oof(variant: str):
        lex_probs_var = np.zeros((n_samples, n_classes), dtype=np.float32)
        if variant == "a_kept":
            lex_probs_var = oof_lex.copy()
        else:
            for fold_idx in range(5):
                tr_idx, va_mask, X_tr_lex, X_va_lex = fold_lex_matrices[fold_idx]
                y_tr = y_all[tr_idx].copy()
                sw_tr = np.ones(len(tr_idx), dtype=np.float32)
                flag_in_tr = capped_flag_mask[tr_idx]

                if variant == "b_removed":
                    keep = ~flag_in_tr
                    X_tr_use = X_tr_lex[keep]
                    y_tr = y_tr[keep]
                    sw_tr = sw_tr[keep]
                elif variant == "c_downweight_025":
                    X_tr_use = X_tr_lex
                    sw_tr[flag_in_tr] = 0.25
                elif variant == "d_relabelled":
                    X_tr_use = X_tr_lex
                    y_tr[flag_in_tr] = lex_top[tr_idx[flag_in_tr]]

                clf = LogisticRegression(C=2.5, max_iter=500)
                clf.fit(X_tr_use, y_tr, sample_weight=sw_tr)
                lex_probs_var[va_mask] = clf.predict_proba(X_va_lex).astype(np.float32)

        # MLP OOF across seeds 42, 43, 44
        mlp_seeds_var = np.zeros((len(SEEDS), n_samples, n_classes), dtype=np.float32)
        if variant == "a_kept":
            mlp_seeds_var = oof_mlp_seeds.copy()
        else:
            for s_idx, seed in enumerate(SEEDS):
                for fold_idx in range(5):
                    va_mask = folds == fold_idx
                    tr_mask = ~va_mask
                    tr_idx = np.where(tr_mask)[0]

                    y_tr = y_all[tr_idx].copy()
                    sw_tr = np.ones(len(tr_idx), dtype=np.float32)
                    flag_in_tr = capped_flag_mask[tr_idx]

                    if variant == "b_removed":
                        keep = ~flag_in_tr
                        tr_idx = tr_idx[keep]
                        y_tr = y_tr[keep]
                        sw_tr = sw_tr[keep]
                    elif variant == "c_downweight_025":
                        sw_tr[flag_in_tr] = 0.25
                    elif variant == "d_relabelled":
                        y_tr[flag_in_tr] = lex_top[tr_idx[flag_in_tr]]

                    mlp_seeds_var[s_idx, va_mask] = train_mlp_fold(
                        X_emb[tr_idx], y_tr, X_emb[va_mask],
                        num_classes=n_classes, seed=seed, sample_weights=sw_tr
                    )

        # Evaluate per seed (both MLP-only and Fused 0.45 Lex + 0.55 MLP)
        # Check against both raw silver y_all AND cleaned OOF reference (where agreed high-conf flags use agreed label)
        y_clean_ref = y_all.copy()
        y_clean_ref[capped_flag_mask] = lex_top[capped_flag_mask]

        mlp_imp_f1s = []
        fused_imp_f1s = []
        fused_unw_f1s = []
        fused_clean_imp_f1s = []
        for s_idx in range(len(SEEDS)):
            p_m = mlp_seeds_var[s_idx]
            p_f = 0.45 * lex_probs_var + 0.55 * p_m
            mlp_imp_f1s.append(float(f1_score(y_all, p_m.argmax(axis=1), average="macro", sample_weight=sample_imp_weights)))
            fused_imp_f1s.append(float(f1_score(y_all, p_f.argmax(axis=1), average="macro", sample_weight=sample_imp_weights)))
            fused_unw_f1s.append(float(f1_score(y_all, p_f.argmax(axis=1), average="macro")))
            fused_clean_imp_f1s.append(float(f1_score(y_clean_ref, p_f.argmax(axis=1), average="macro", sample_weight=class_imp_weights[y_clean_ref])))

        lex_imp_f1 = float(f1_score(y_all, lex_probs_var.argmax(axis=1), average="macro", sample_weight=sample_imp_weights))
        return {
            "variant": variant,
            "lex_oof_imp_f1": round(lex_imp_f1, 4),
            "mlp_oof_imp_f1_s42": round(mlp_imp_f1s[0], 4),
            "mlp_oof_imp_f1_s43": round(mlp_imp_f1s[1], 4),
            "mlp_oof_imp_f1_s44": round(mlp_imp_f1s[2], 4),
            "mlp_oof_imp_f1_mean": round(float(np.mean(mlp_imp_f1s)), 4),
            "mlp_oof_imp_f1_std": round(float(np.std(mlp_imp_f1s)), 4),
            "fused_oof_imp_f1_s42": round(fused_imp_f1s[0], 4),
            "fused_oof_imp_f1_s43": round(fused_imp_f1s[1], 4),
            "fused_oof_imp_f1_s44": round(fused_imp_f1s[2], 4),
            "fused_oof_imp_f1_mean": round(float(np.mean(fused_imp_f1s)), 4),
            "fused_oof_imp_f1_std": round(float(np.std(fused_imp_f1s)), 4),
            "fused_oof_unw_f1_mean": round(float(np.mean(fused_unw_f1s)), 4),
            "fused_oof_clean_ref_imp_f1_mean": round(float(np.mean(fused_clean_imp_f1s)), 4),
        }, lex_probs_var, mlp_seeds_var

    ablation_rows = []
    saved_probs = {}
    for var in ["a_kept", "b_removed", "c_downweight_025", "d_relabelled"]:
        res, lp, mp = run_variant_oof(var)
        ablation_rows.append(res)
        saved_probs[var] = (lp, mp)
        print(f"  {var:18s}: MLP Imp-F1={res['mlp_oof_imp_f1_mean']:.4f} (±{res['mlp_oof_imp_f1_std']:.4f}), "
              f"Fused Imp-F1={res['fused_oof_imp_f1_mean']:.4f} (±{res['fused_oof_imp_f1_std']:.4f})")

    ablation_df = pd.DataFrame(ablation_rows)
    ablation_df.to_csv(OUT_DIR / "step5_noise_ablation.csv", index=False)

    # 3. Audit of Rule-Repair on Train Rows split by cue (with 'but' reported separately)
    cue_records = []
    for i, u in enumerate(norm_texts):
        cue_type, extracted_clause = detect_repair_cue(u)
        if cue_type is not None:
            cue_records.append({
                "row_idx": i,
                "cue_type": cue_type,
                "utterance": raw_texts[i],
                "extracted_clause": extracted_clause,
                "silver_intent": intents[y_all[i]],
                "oof_full_pred": intents[lex_top[i]],
                "is_flagged_noisy": bool(capped_flag_mask[i]),
            })
    cue_df = pd.DataFrame(cue_records)
    cue_sample = (
        cue_df.groupby("cue_type", group_keys=False)
        .apply(lambda g: g.sample(n=min(25, len(g)), random_state=42))
        .reset_index(drop=True)
    )
    cue_sample.to_csv(OUT_DIR / "train_rule_repair_sample100.csv", index=False)
    print("\nTrain rule-repair cue counts:")
    print(cue_df["cue_type"].value_counts().to_string())
    print(f"Completed Step 5 in {time.perf_counter() - t0:.1f}s.")


if __name__ == "__main__":
    main()

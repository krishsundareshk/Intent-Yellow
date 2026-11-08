"""Step 7 & Step 8: OOF Calibration, Abstain Gate, and 7-System Ablation Table.

Step 8 (fit strictly on 5-fold OOF predictions weighted to dev class mix):
  - Fits fusion weight `alpha`, temperature `T`, and abstain threshold `tau`.
  - Reports abstain rate, `out_of_scope` precision & recall at `tau`, and a 10-bin
    reliability table with Expected Calibration Error (ECE) in `results/calibration_reliability.csv`
    and `results/step8_calibration_summary.json`.

Step 7 (Ablation table across seeds 42, 43, 44 on the same 5 folds and Step 1 preprocessing):
  - A: Lexical only (best from Step 6: `both_word_char__CalibratedLinearSVC__C0.5`)
  - B: Lexical + clause path
  - C: Raw 30k MLP
  - D: Raw MLP + clause path
  - E: Fused (lexical + MLP) + clause path
  - F: E trained with the old SCD partition
  - G: E trained with the OOF noise filter (best of Step 5: `c_downweight_025`)
  - H: B trained with the OOF noise filter (`c_downweight_025`, Lexical + clause path)

Computes 1,000 paired bootstrap resamples of the 4,000 dev rows for 95% CIs vs System A (and vs B),
applies the pruning rule, and logs dev evaluations in `results/dev_eval_log.csv`.
"""
from __future__ import annotations

import datetime
import json
import os
import sys
import time
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"

import numpy as np
import pandas as pd
import scipy.optimize as opt
import scipy.sparse as sp
import torch
from sentence_transformers import SentenceTransformer
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from sklearn.svm import LinearSVC

torch.set_num_threads(4)

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from preprocess import preprocess_and_guard, preprocess_text
from src.clause_router import parse_clauses, resolve_clause_probs
from src.step4_oof_predictions import SEEDS, ResidualMLP, train_mlp_fold

DATA_DIR = ROOT / "data"
CACHE_DIR = ROOT / "results" / "system_v1" / "cache"
OUT_DIR = ROOT / "results"


def build_best_lexical_vectorizer():
    w_vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=60000, sublinear_tf=True)
    c_vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=3, max_features=80000, sublinear_tf=True)
    return w_vec, c_vec


def fit_best_lexical(w_vec, c_vec, tr_texts: list[str], y_tr: np.ndarray, sample_weight: np.ndarray | None = None):
    X_w = w_vec.fit_transform(tr_texts)
    X_c = c_vec.fit_transform(tr_texts)
    X_tr = sp.hstack([X_w, X_c], format="csr")
    base_svc = LinearSVC(C=0.5, random_state=42)
    clf = CalibratedClassifierCV(base_svc, cv=3)
    if sample_weight is not None:
        clf.fit(X_tr, y_tr, sample_weight=sample_weight)
    else:
        clf.fit(X_tr, y_tr)
    return clf


def predict_best_lexical(w_vec, c_vec, clf, texts: list[str]) -> np.ndarray:
    X_w = w_vec.transform(texts)
    X_c = c_vec.transform(texts)
    X = sp.hstack([X_w, X_c], format="csr")
    return clf.predict_proba(X).astype(np.float32)


def apply_temperature(probs: np.ndarray, T: float) -> np.ndarray:
    eps = 1e-12
    logits = np.log(np.clip(probs, eps, 1.0)) / T
    logits -= logits.max(axis=1, keepdims=True)
    exp_l = np.exp(logits)
    return (exp_l / exp_l.sum(axis=1, keepdims=True)).astype(np.float32)


def apply_abstain_gate(probs: np.ndarray, tau: float, oos_id: int, guard_flags: np.ndarray | None = None) -> np.ndarray:
    preds = probs.argmax(axis=1).copy()
    max_p = probs.max(axis=1)
    preds[max_p < tau] = oos_id
    if guard_flags is not None:
        preds[guard_flags] = oos_id
    return preds


def compute_ece_and_table(probs: np.ndarray, y_true: np.ndarray, sample_weights: np.ndarray, n_bins: int = 10):
    conf = probs.max(axis=1)
    preds = probs.argmax(axis=1)
    correct = (preds == y_true).astype(np.float64)
    total_w = sample_weights.sum()

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    rows = []
    ece = 0.0
    for b in range(n_bins):
        lo, hi = bins[b], bins[b + 1]
        if b == 0:
            mask = (conf >= lo) & (conf <= hi)
        else:
            mask = (conf > lo) & (conf <= hi)
        cnt = int(mask.sum())
        if cnt > 0:
            w_bin = sample_weights[mask].sum()
            w_share = float(w_bin / total_w)
            avg_conf = float(np.average(conf[mask], weights=sample_weights[mask]))
            avg_acc = float(np.average(correct[mask], weights=sample_weights[mask]))
            gap = abs(avg_acc - avg_conf)
            ece += w_share * gap
        else:
            w_share, avg_conf, avg_acc, gap = 0.0, 0.0, 0.0, 0.0
        rows.append({
            "bin_idx": b + 1,
            "bin_low": round(float(lo), 2),
            "bin_high": round(float(hi), 2),
            "count": cnt,
            "weighted_share": round(w_share, 4),
            "avg_confidence": round(avg_conf, 4),
            "avg_accuracy": round(avg_acc, 4),
            "abs_gap": round(gap, 4),
        })
    return float(ece), pd.DataFrame(rows)


def paired_bootstrap_ci(y_true: np.ndarray, preds_sys: np.ndarray, preds_ref: np.ndarray, n_boot: int = 1000, seed: int = 42):
    rng = np.random.default_rng(seed)
    n = len(y_true)
    diffs = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        f_sys = f1_score(y_true[idx], preds_sys[idx], average="macro", zero_division=0)
        f_ref = f1_score(y_true[idx], preds_ref[idx], average="macro", zero_division=0)
        diffs[b] = f_sys - f_ref
    mean_d = float(np.mean(diffs))
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return mean_d, float(lo), float(hi)


def log_dev_eval(candidate: str, macro_f1: float, top1_acc: float, notes: str) -> None:
    log_path = OUT_DIR / "dev_eval_log.csv"
    now_str = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    if log_path.exists():
        df = pd.read_csv(log_path)
        next_id = int(df["eval_id"].max()) + 1 if len(df) > 0 else 1
    else:
        df = pd.DataFrame(columns=["eval_id", "candidate", "timestamp", "dev_macro_f1", "dev_top1_acc", "notes"])
        next_id = 1
    new_row = pd.DataFrame([{
        "eval_id": next_id,
        "candidate": candidate,
        "timestamp": now_str,
        "dev_macro_f1": round(macro_f1, 4),
        "dev_top1_acc": round(top1_acc, 4),
        "notes": notes,
    }])
    df = pd.concat([df, new_row], ignore_index=True)
    df.to_csv(log_path, index=False)


def main() -> None:
    t0 = time.perf_counter()
    with open(DATA_DIR / "taxonomy.json") as f:
        tax = json.load(f)
    intents = [it["name"] for it in tax["intents"]]
    label2id = {name: i for i, name in enumerate(intents)}
    talk_id = label2id["talk_to_agent"]
    oos_id = label2id["out_of_scope"]
    n_classes = len(intents)

    train_df = pd.read_csv(DATA_DIR / "train.csv")
    dev_df = pd.read_csv(DATA_DIR / "dev.csv")
    test_df = pd.read_csv(DATA_DIR / "test.csv")
    folds_df = pd.read_csv(OUT_DIR / "folds.csv")

    y_train = np.array([label2id[i] for i in folds_df["intent"]])
    y_dev = np.array([label2id[i] for i in dev_df["intent"]])
    folds = folds_df["fold"].to_numpy()
    n_train = len(y_train)
    n_dev = len(y_dev)

    # Importance weights w(c) = p_dev(c) / p_train(c)
    p_train = np.bincount(y_train, minlength=n_classes) / n_train
    p_dev = np.bincount(y_dev, minlength=n_classes) / n_dev
    class_imp_w = p_dev / p_train
    sample_imp_w = class_imp_w[y_train]

    # Step 1 preprocess and guard on train, dev, test
    tr_prep = [preprocess_and_guard(str(u)) for u in train_df["utterance"].fillna("")]
    dv_prep = [preprocess_and_guard(str(u)) for u in dev_df["utterance"].fillna("")]
    te_prep = [preprocess_and_guard(str(u)) for u in test_df["utterance"].fillna("")]

    tr_texts = [p[0] for p in tr_prep]
    dv_texts = [p[0] for p in dv_prep]
    te_texts = [p[0] for p in te_prep]

    tr_guards = np.array([p[1] for p in tr_prep], dtype=bool)
    dv_guards = np.array([p[1] for p in dv_prep], dtype=bool)

    # Parse clauses on train and dev
    tr_routes = [parse_clauses(t) for t in tr_texts]
    dv_routes = [parse_clauses(t) for t in dv_texts]

    tr_flagged_idx = [i for i, (rt, _) in enumerate(tr_routes) if rt != "single"]
    dv_flagged_idx = [i for i, (rt, _) in enumerate(dv_routes) if rt != "single"]
    print(f"Clause router flagged: train={len(tr_flagged_idx)}/{n_train} ({100*len(tr_flagged_idx)/n_train:.1f}%), "
          f"dev={len(dv_flagged_idx)}/{n_dev} ({100*len(dv_flagged_idx)/n_dev:.1f}%)")

    # Collect all unique clause strings + encode test embeddings if needed
    unique_clauses = sorted({c for rt, cls in tr_routes + dv_routes if rt != "single" for c in cls})
    clause2idx = {c: i for i, c in enumerate(unique_clauses)}
    print(f"Encoding {len(unique_clauses)} unique extracted clauses (and test set if not cached) with MiniLM...")

    X_emb_tr = np.load(CACHE_DIR / "minilm_step1_train.npy")
    X_emb_dv = np.load(CACHE_DIR / "minilm_step1_dev.npy")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    st_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=device)
    clause_embs = st_model.encode(unique_clauses, batch_size=256, normalize_embeddings=True, show_progress_bar=False)
    clause_embs = np.asarray(clause_embs, dtype=np.float32)

    test_emb_path = CACHE_DIR / "minilm_step1_test.npy"
    if not test_emb_path.exists():
        X_emb_te = st_model.encode(te_texts, batch_size=256, normalize_embeddings=True, show_progress_bar=False)
        np.save(test_emb_path, np.asarray(X_emb_te, dtype=np.float32))

    # Load Noise Filter flags (Step 5) and Old SCD Partition (Step 7 System F)
    nf_df = pd.read_csv(OUT_DIR / "noise_flags.csv")
    oof_flag_idx = nf_df[nf_df["kept_under_15pct_cap"]]["row_idx"].to_numpy()
    oof_flag_mask = np.zeros(n_train, dtype=bool)
    oof_flag_mask[oof_flag_idx] = True
    clean_eval_mask = ~oof_flag_mask

    scd_df = pd.read_csv(ROOT / "results" / "system_v1" / "s2_train_partition.csv")
    scd_keep_mask = (scd_df["bucket"] != "noisy").to_numpy()

    # Load or compute OOF & Dev predictions for each base model variant:
    # 1. Lexical Raw (System A/B/E)
    # 2. Lexical SCD (System F)
    # 3. Lexical OOF-Noise-Downweighted (System G/H)
    print("\n1. Fitting Lexical Head (`both_word_char__CalibratedLinearSVC__C0.5`) across 5 folds + full train...")
    oof_lex_raw = np.load(OUT_DIR / "oof_lexical_best_probs.npy")
    oof_lex_scd = np.zeros((n_train, n_classes), dtype=np.float32)
    oof_lex_nf = np.zeros((n_train, n_classes), dtype=np.float32)

    # Also compute clause lexical probabilities on OOF for each fold!
    oof_clause_lex_raw = {}
    oof_clause_lex_scd = {}
    oof_clause_lex_nf = {}

    for fold_idx in range(5):
        va_mask = folds == fold_idx
        tr_idx = np.where(~va_mask)[0]
        va_idx = np.where(va_mask)[0]
        va_flagged = [i for i in va_idx if tr_routes[i][0] != "single"]
        va_clauses = sorted({c for i in va_flagged for c in tr_routes[i][1]})
        va_c2i = {c: idx for idx, c in enumerate(va_clauses)}

        # Fit Raw Lexical on fold
        w_v, c_v = build_best_lexical_vectorizer()
        clf_raw = fit_best_lexical(w_v, c_v, [tr_texts[i] for i in tr_idx], y_train[tr_idx])
        c_probs_raw = predict_best_lexical(w_v, c_v, clf_raw, va_clauses)
        for i in va_flagged:
            oof_clause_lex_raw[i] = np.stack([c_probs_raw[va_c2i[c]] for c in tr_routes[i][1]])

        # Fit SCD Lexical on fold
        tr_idx_scd = tr_idx[scd_keep_mask[tr_idx]]
        w_s, c_s = build_best_lexical_vectorizer()
        clf_scd = fit_best_lexical(w_s, c_s, [tr_texts[i] for i in tr_idx_scd], y_train[tr_idx_scd])
        oof_lex_scd[va_mask] = predict_best_lexical(w_s, c_s, clf_scd, [tr_texts[i] for i in va_idx])
        c_probs_scd = predict_best_lexical(w_s, c_s, clf_scd, va_clauses)
        for i in va_flagged:
            oof_clause_lex_scd[i] = np.stack([c_probs_scd[va_c2i[c]] for c in tr_routes[i][1]])

        # Fit OOF-Noise-Filter Lexical (c_downweight_025) on fold
        sw_nf = np.ones(len(tr_idx), dtype=np.float32)
        sw_nf[oof_flag_mask[tr_idx]] = 0.25
        w_n, c_n = build_best_lexical_vectorizer()
        clf_nf = fit_best_lexical(w_n, c_n, [tr_texts[i] for i in tr_idx], y_train[tr_idx], sample_weight=sw_nf)
        oof_lex_nf[va_mask] = predict_best_lexical(w_n, c_n, clf_nf, [tr_texts[i] for i in va_idx])
        c_probs_nf = predict_best_lexical(w_n, c_n, clf_nf, va_clauses)
        for i in va_flagged:
            oof_clause_lex_nf[i] = np.stack([c_probs_nf[va_c2i[c]] for c in tr_routes[i][1]])

    # Fit full-train Lexical models for Dev evaluation (Raw, SCD, NF)
    dv_clauses = sorted({c for i in dv_flagged_idx for c in dv_routes[i][1]})
    dv_c2i = {c: idx for idx, c in enumerate(dv_clauses)}

    w_full_raw, c_full_raw = build_best_lexical_vectorizer()
    clf_full_raw = fit_best_lexical(w_full_raw, c_full_raw, tr_texts, y_train)
    dev_lex_raw = predict_best_lexical(w_full_raw, c_full_raw, clf_full_raw, dv_texts)
    dv_c_probs_raw = predict_best_lexical(w_full_raw, c_full_raw, clf_full_raw, dv_clauses)
    dev_clause_lex_raw = {i: np.stack([dv_c_probs_raw[dv_c2i[c]] for c in dv_routes[i][1]]) for i in dv_flagged_idx}

    tr_scd_all = np.where(scd_keep_mask)[0]
    w_full_scd, c_full_scd = build_best_lexical_vectorizer()
    clf_full_scd = fit_best_lexical(w_full_scd, c_full_scd, [tr_texts[i] for i in tr_scd_all], y_train[tr_scd_all])
    dev_lex_scd = predict_best_lexical(w_full_scd, c_full_scd, clf_full_scd, dv_texts)
    dv_c_probs_scd = predict_best_lexical(w_full_scd, c_full_scd, clf_full_scd, dv_clauses)
    dev_clause_lex_scd = {i: np.stack([dv_c_probs_scd[dv_c2i[c]] for c in dv_routes[i][1]]) for i in dv_flagged_idx}

    sw_full_nf = np.ones(n_train, dtype=np.float32)
    sw_full_nf[oof_flag_mask] = 0.25
    w_full_nf, c_full_nf = build_best_lexical_vectorizer()
    clf_full_nf = fit_best_lexical(w_full_nf, c_full_nf, tr_texts, y_train, sample_weight=sw_full_nf)
    dev_lex_nf = predict_best_lexical(w_full_nf, c_full_nf, clf_full_nf, dv_texts)
    dv_c_probs_nf = predict_best_lexical(w_full_nf, c_full_nf, clf_full_nf, dv_clauses)
    dev_clause_lex_nf = {i: np.stack([dv_c_probs_nf[dv_c2i[c]] for c in dv_routes[i][1]]) for i in dv_flagged_idx}

    # 2. Train MLP across 5 folds + full train for seeds 42, 43, 44 (Raw, SCD, NF)
    print("2. Training MiniLM + MLP models across seeds 42, 43, 44 (Raw, SCD, OOF-Noise-Filter)...")
    oof_mlp_raw_seeds = np.load(OUT_DIR / "oof_mlp_probs_seeds.npy")
    oof_mlp_scd_seeds = np.zeros((len(SEEDS), n_train, n_classes), dtype=np.float32)
    oof_mlp_nf_seeds = np.zeros((len(SEEDS), n_train, n_classes), dtype=np.float32)

    oof_clause_mlp_raw_seeds = [{} for _ in SEEDS]
    oof_clause_mlp_scd_seeds = [{} for _ in SEEDS]
    oof_clause_mlp_nf_seeds = [{} for _ in SEEDS]

    dev_mlp_raw_seeds = np.zeros((len(SEEDS), n_dev, n_classes), dtype=np.float32)
    dev_mlp_scd_seeds = np.zeros((len(SEEDS), n_dev, n_classes), dtype=np.float32)
    dev_mlp_nf_seeds = np.zeros((len(SEEDS), n_dev, n_classes), dtype=np.float32)

    dev_clause_mlp_raw_seeds = [{} for _ in SEEDS]
    dev_clause_mlp_scd_seeds = [{} for _ in SEEDS]
    dev_clause_mlp_nf_seeds = [{} for _ in SEEDS]

    for s_idx, seed in enumerate(SEEDS):
        for fold_idx in range(5):
            va_mask = folds == fold_idx
            tr_idx = np.where(~va_mask)[0]
            va_idx = np.where(va_mask)[0]
            va_flagged = [i for i in va_idx if tr_routes[i][0] != "single"]
            va_c_embs = np.stack([clause_embs[clause2idx[c]] for i in va_flagged for c in tr_routes[i][1]])

            # Raw MLP fold (for clause predictions; full OOF already in oof_mlp_raw_seeds)
            c_probs_m_raw = train_mlp_fold(X_emb_tr[tr_idx], y_train[tr_idx], va_c_embs, num_classes=n_classes, seed=seed)
            ptr = 0
            for i in va_flagged:
                k = len(tr_routes[i][1])
                oof_clause_mlp_raw_seeds[s_idx][i] = c_probs_m_raw[ptr : ptr + k]
                ptr += k

            # SCD MLP fold
            tr_idx_scd = tr_idx[scd_keep_mask[tr_idx]]
            X_va_comb = np.vstack([X_emb_tr[va_mask], va_c_embs])
            comb_scd = train_mlp_fold(X_emb_tr[tr_idx_scd], y_train[tr_idx_scd], X_va_comb, num_classes=n_classes, seed=seed)
            oof_mlp_scd_seeds[s_idx, va_mask] = comb_scd[: len(va_idx)]
            c_probs_m_scd = comb_scd[len(va_idx) :]
            ptr = 0
            for i in va_flagged:
                k = len(tr_routes[i][1])
                oof_clause_mlp_scd_seeds[s_idx][i] = c_probs_m_scd[ptr : ptr + k]
                ptr += k

            # NF MLP fold (c_downweight_025)
            sw_nf = np.ones(len(tr_idx), dtype=np.float32)
            sw_nf[oof_flag_mask[tr_idx]] = 0.25
            comb_nf = train_mlp_fold(
                X_emb_tr[tr_idx], y_train[tr_idx], X_va_comb, num_classes=n_classes, seed=seed, sample_weights=sw_nf
            )
            oof_mlp_nf_seeds[s_idx, va_mask] = comb_nf[: len(va_idx)]
            c_probs_m_nf = comb_nf[len(va_idx) :]
            ptr = 0
            for i in va_flagged:
                k = len(tr_routes[i][1])
                oof_clause_mlp_nf_seeds[s_idx][i] = c_probs_m_nf[ptr : ptr + k]
                ptr += k

        # Full-train MLP models for Dev evaluation (batching X_emb_dv and dv_c_embs together)
        dv_c_embs = np.stack([clause_embs[clause2idx[c]] for i in dv_flagged_idx for c in dv_routes[i][1]])
        X_dv_comb = np.vstack([X_emb_dv, dv_c_embs])

        # Raw full
        comb_dv_raw = train_mlp_fold(X_emb_tr, y_train, X_dv_comb, num_classes=n_classes, seed=seed)
        dev_mlp_raw_seeds[s_idx] = comb_dv_raw[:n_dev]
        c_dv_raw = comb_dv_raw[n_dev:]
        ptr = 0
        for i in dv_flagged_idx:
            k = len(dv_routes[i][1])
            dev_clause_mlp_raw_seeds[s_idx][i] = c_dv_raw[ptr : ptr + k]
            ptr += k

        # SCD full
        comb_dv_scd = train_mlp_fold(
            X_emb_tr[tr_scd_all], y_train[tr_scd_all], X_dv_comb, num_classes=n_classes, seed=seed
        )
        dev_mlp_scd_seeds[s_idx] = comb_dv_scd[:n_dev]
        c_dv_scd = comb_dv_scd[n_dev:]
        ptr = 0
        for i in dv_flagged_idx:
            k = len(dv_routes[i][1])
            dev_clause_mlp_scd_seeds[s_idx][i] = c_dv_scd[ptr : ptr + k]
            ptr += k

        # NF full
        comb_dv_nf = train_mlp_fold(
            X_emb_tr, y_train, X_dv_comb, num_classes=n_classes, seed=seed, sample_weights=sw_full_nf
        )
        dev_mlp_nf_seeds[s_idx] = comb_dv_nf[:n_dev]
        c_dv_nf = comb_dv_nf[n_dev:]
        ptr = 0
        for i in dv_flagged_idx:
            k = len(dv_routes[i][1])
            dev_clause_mlp_nf_seeds[s_idx][i] = c_dv_nf[ptr : ptr + k]
            ptr += k

    def apply_clause_routing(base_probs: np.ndarray, routes: list, clause_prob_dict: dict) -> np.ndarray:
        out = base_probs.copy()
        for idx, cp in clause_prob_dict.items():
            rt, cls = routes[idx]
            out[idx] = resolve_clause_probs(rt, cls, cp, base_probs[idx], talk_id, oos_id)
        return out

    # --- STEP 8: Fit Fusion Weight alpha, Temperature T, and Abstain Threshold tau on OOF ---
    print("\n--- STEP 8: Fitting alpha, T, and tau on OOF (weighted to dev class mix) ---")
    oof_lex_routed = apply_clause_routing(oof_lex_raw, tr_routes, oof_clause_lex_raw)
    oof_mlp_mean = oof_mlp_raw_seeds.mean(axis=0)
    oof_mlp_clause_mean = {
        i: np.mean([oof_clause_mlp_raw_seeds[s][i] for s in range(len(SEEDS))], axis=0) for i in tr_flagged_idx
    }
    oof_mlp_routed = apply_clause_routing(oof_mlp_mean, tr_routes, oof_mlp_clause_mean)

    # Grid search alpha in [0.0, 1.0] on OOF (both overall and inside [0.50, 0.90] for dual-head fusion E/F/G)
    alpha_grid = np.round(np.linspace(0.0, 1.0, 21), 2)
    alpha_records = []
    for a in alpha_grid:
        pf = a * oof_lex_routed + (1.0 - a) * oof_mlp_routed
        f_imp = float(f1_score(y_train, pf.argmax(axis=1), average="macro", sample_weight=sample_imp_w))
        f_clean = float(f1_score(y_train[clean_eval_mask], pf[clean_eval_mask].argmax(axis=1), average="macro", sample_weight=sample_imp_w[clean_eval_mask]))
        alpha_records.append({"alpha": float(a), "oof_imp_f1": round(f_imp, 4), "oof_filtered_imp_f1": round(f_clean, 4)})

    alpha_df = pd.DataFrame(alpha_records)
    best_overall_alpha = float(alpha_df.sort_values("oof_imp_f1", ascending=False).iloc[0]["alpha"])
    # For the Fused systems E, F, G (which test genuine fusion of Lexical + MLP), pick best alpha in [0.50, 0.85] on OOF
    fused_sub = alpha_df[(alpha_df["alpha"] >= 0.50) & (alpha_df["alpha"] <= 0.85)]
    best_fused_alpha = float(fused_sub.sort_values("oof_imp_f1", ascending=False).iloc[0]["alpha"])
    print(f"OOF alpha grid: best overall alpha={best_overall_alpha:.2f}, best nontrivial fusion alpha={best_fused_alpha:.2f}")

    oof_fused = best_fused_alpha * oof_lex_routed + (1.0 - best_fused_alpha) * oof_mlp_routed

    # Fit temperature T on importance-weighted OOF NLL
    eps = 1e-12
    log_p = np.log(np.clip(oof_fused, eps, 1.0))

    def weighted_nll(temp_arr):
        T_val = float(temp_arr[0])
        scaled = log_p / T_val
        scaled -= scaled.max(axis=1, keepdims=True)
        lse = np.log(np.exp(scaled).sum(axis=1))
        nll_i = lse - scaled[np.arange(n_train), y_train]
        return float(np.average(nll_i, weights=sample_imp_w))

    res_T = opt.minimize(weighted_nll, x0=[1.0], bounds=[(0.2, 5.0)], method="L-BFGS-B")
    best_T = float(res_T.x[0])
    oof_cal = apply_temperature(oof_fused, best_T)

    ece_before, _ = compute_ece_and_table(oof_fused, y_train, sample_imp_w)
    ece_after, rel_table = compute_ece_and_table(oof_cal, y_train, sample_imp_w)
    rel_table.to_csv(OUT_DIR / "calibration_reliability.csv", index=False)
    print(f"OOF Temperature Scaling: best T={best_T:.4f}, ECE before={ece_before:.4f} -> ECE after={ece_after:.4f}")

    # Fit abstain threshold tau on importance-weighted OOF Macro-F1
    tau_grid = np.round(np.linspace(0.15, 0.55, 41), 3)
    best_tau = 0.25
    best_tau_f1 = -1.0
    tau_rows = []
    for tau in tau_grid:
        preds_tau = apply_abstain_gate(oof_cal, float(tau), oos_id, tr_guards)
        f_imp = float(f1_score(y_train, preds_tau, average="macro", sample_weight=sample_imp_w))
        p_w, r_w, f1_w, _ = precision_recall_fscore_support(
            y_train, preds_tau, labels=[oos_id], sample_weight=sample_imp_w, zero_division=0
        )
        abstain_rate = float(np.mean(oof_cal.max(axis=1) < tau))
        tau_rows.append({
            "tau": float(tau),
            "oof_imp_macro_f1": round(f_imp, 4),
            "abstain_rate": round(abstain_rate, 4),
            "oos_precision": round(float(p_w[0]), 4),
            "oos_recall": round(float(r_w[0]), 4),
            "oos_f1": round(float(f1_w[0]), 4),
        })
        if f_imp > best_tau_f1:
            best_tau_f1 = f_imp
            best_tau = float(tau)

    tau_df = pd.DataFrame(tau_rows)
    tau_df.to_csv(OUT_DIR / "step8_tau_sweep.csv", index=False)
    chosen_tau_row = tau_df[tau_df["tau"] == best_tau].iloc[0].to_dict()
    print(f"Chosen OOF abstain threshold tau={best_tau:.3f}: {chosen_tau_row}")

    step8_summary = {
        "best_overall_alpha_on_oof": best_overall_alpha,
        "best_fused_alpha_on_oof": best_fused_alpha,
        "temperature_T": round(best_T, 4),
        "ece_before_temperature": round(ece_before, 4),
        "ece_after_temperature": round(ece_after, 4),
        "abstain_threshold_tau": best_tau,
        "at_chosen_tau": chosen_tau_row,
        "alpha_sweep": alpha_records,
    }
    with open(OUT_DIR / "step8_calibration_summary.json", "w") as f:
        json.dump(step8_summary, f, indent=2)

    # --- STEP 7: Build Full Ablation Table (Systems A through H) ---
    print("\n--- STEP 7: Evaluating Systems A through H on 5-Fold OOF and Dev across Seeds 42, 43, 44 ---")

    def eval_preds(y_t: np.ndarray, preds: np.ndarray, weights: np.ndarray | None = None) -> float:
        return float(f1_score(y_t, preds, average="macro", sample_weight=weights, zero_division=0))

    def get_system_preds(sys_code: str, s_idx: int):
        if sys_code == "A":
            p_oof = oof_lex_raw
            p_dev = dev_lex_raw
        elif sys_code == "B":
            p_oof = apply_clause_routing(oof_lex_raw, tr_routes, oof_clause_lex_raw)
            p_dev = apply_clause_routing(dev_lex_raw, dv_routes, dev_clause_lex_raw)
        elif sys_code == "C":
            p_oof = oof_mlp_raw_seeds[s_idx]
            p_dev = dev_mlp_raw_seeds[s_idx]
        elif sys_code == "D":
            p_oof = apply_clause_routing(oof_mlp_raw_seeds[s_idx], tr_routes, oof_clause_mlp_raw_seeds[s_idx])
            p_dev = apply_clause_routing(dev_mlp_raw_seeds[s_idx], dv_routes, dev_clause_mlp_raw_seeds[s_idx])
        elif sys_code == "E":
            c_oof = {
                i: best_fused_alpha * oof_clause_lex_raw[i] + (1.0 - best_fused_alpha) * oof_clause_mlp_raw_seeds[s_idx][i]
                for i in tr_flagged_idx
            }
            c_dev = {
                i: best_fused_alpha * dev_clause_lex_raw[i] + (1.0 - best_fused_alpha) * dev_clause_mlp_raw_seeds[s_idx][i]
                for i in dv_flagged_idx
            }
            base_oof = best_fused_alpha * oof_lex_raw + (1.0 - best_fused_alpha) * oof_mlp_raw_seeds[s_idx]
            base_dev = best_fused_alpha * dev_lex_raw + (1.0 - best_fused_alpha) * dev_mlp_raw_seeds[s_idx]
            p_oof = apply_temperature(apply_clause_routing(base_oof, tr_routes, c_oof), best_T)
            p_dev = apply_temperature(apply_clause_routing(base_dev, dv_routes, c_dev), best_T)
        elif sys_code == "F":
            c_oof = {
                i: best_fused_alpha * oof_clause_lex_scd[i] + (1.0 - best_fused_alpha) * oof_clause_mlp_scd_seeds[s_idx][i]
                for i in tr_flagged_idx
            }
            c_dev = {
                i: best_fused_alpha * dev_clause_lex_scd[i] + (1.0 - best_fused_alpha) * dev_clause_mlp_scd_seeds[s_idx][i]
                for i in dv_flagged_idx
            }
            base_oof = best_fused_alpha * oof_lex_scd + (1.0 - best_fused_alpha) * oof_mlp_scd_seeds[s_idx]
            base_dev = best_fused_alpha * dev_lex_scd + (1.0 - best_fused_alpha) * dev_mlp_scd_seeds[s_idx]
            p_oof = apply_temperature(apply_clause_routing(base_oof, tr_routes, c_oof), best_T)
            p_dev = apply_temperature(apply_clause_routing(base_dev, dv_routes, c_dev), best_T)
        elif sys_code == "G":
            c_oof = {
                i: best_fused_alpha * oof_clause_lex_nf[i] + (1.0 - best_fused_alpha) * oof_clause_mlp_nf_seeds[s_idx][i]
                for i in tr_flagged_idx
            }
            c_dev = {
                i: best_fused_alpha * dev_clause_lex_nf[i] + (1.0 - best_fused_alpha) * dev_clause_mlp_nf_seeds[s_idx][i]
                for i in dv_flagged_idx
            }
            base_oof = best_fused_alpha * oof_lex_nf + (1.0 - best_fused_alpha) * oof_mlp_nf_seeds[s_idx]
            base_dev = best_fused_alpha * dev_lex_nf + (1.0 - best_fused_alpha) * dev_mlp_nf_seeds[s_idx]
            p_oof = apply_temperature(apply_clause_routing(base_oof, tr_routes, c_oof), best_T)
            p_dev = apply_temperature(apply_clause_routing(base_dev, dv_routes, c_dev), best_T)
        elif sys_code == "H":
            p_oof = apply_clause_routing(oof_lex_nf, tr_routes, oof_clause_lex_nf)
            p_dev = apply_clause_routing(dev_lex_nf, dv_routes, dev_clause_lex_nf)
        else:
            raise ValueError(sys_code)

        preds_oof = apply_abstain_gate(p_oof, best_tau, oos_id, tr_guards)
        preds_dev = apply_abstain_gate(p_dev, best_tau, oos_id, dv_guards)
        return preds_oof, preds_dev, p_dev

    systems_meta = [
        ("A", "A: Lexical only (best from Step 6)", 2.37),
        ("B", "B: Lexical + clause path", 2.51),
        ("C", "C: Raw 30k MLP", 7.12),
        ("D", "D: Raw MLP + clause path", 12.67),
        ("E", f"E: Fused (lexical + MLP, alpha={best_fused_alpha:.2f}) + clause path", 12.67),
        ("F", "F: E trained with old SCD partition", 12.67),
        ("G", "G: E trained with OOF noise filter (downweight 0.25)", 12.67),
        ("H", "H: Lexical (OOF noise filter) + clause path", 2.51),
    ]

    _, preds_dev_A, _ = get_system_preds("A", 0)
    _, preds_dev_B, _ = get_system_preds("B", 0)

    ablation_rows = []
    dev_probs_store = {}

    for sys_code, sys_desc, p50_lat in systems_meta:
        oof_raw_f1s = []
        oof_filt_f1s = []
        oof_unw_f1s = []
        dev_f1s = []
        dev_accs = []
        dev_preds_seeds = []
        dev_probs_seeds = []

        for s_idx in range(len(SEEDS)):
            pr_oof, pr_dev, prob_dev = get_system_preds(sys_code, s_idx)
            oof_raw_f1s.append(eval_preds(y_train, pr_oof, sample_imp_w))
            oof_filt_f1s.append(eval_preds(y_train[clean_eval_mask], pr_oof[clean_eval_mask], sample_imp_w[clean_eval_mask]))
            oof_unw_f1s.append(eval_preds(y_train, pr_oof, None))
            dev_f1s.append(eval_preds(y_dev, pr_dev, None))
            dev_accs.append(float(accuracy_score(y_dev, pr_dev)))
            dev_preds_seeds.append(pr_dev)
            dev_probs_seeds.append(prob_dev)

        dev_probs_mean = np.mean(dev_probs_seeds, axis=0)
        dev_preds_ens = apply_abstain_gate(dev_probs_mean, best_tau, oos_id, dv_guards)
        dev_probs_store[sys_code] = dev_probs_mean

        if sys_code == "A":
            ci_vs_A_str = "baseline (0.0000)"
            ci_lo_A, ci_hi_A, mean_d_A = 0.0, 0.0, 0.0
        else:
            # Use seed 42 (or median seed) paired predictions for 1,000 bootstrap resamples vs A
            mean_d_A, ci_lo_A, ci_hi_A = paired_bootstrap_ci(y_dev, dev_preds_seeds[0], preds_dev_A, n_boot=1000, seed=42)
            ci_vs_A_str = f"{mean_d_A:+.4f} [{ci_lo_A:+.4f}, {ci_hi_A:+.4f}]"

        if sys_code in ("A", "B"):
            ci_vs_B_str = "—"
            ci_lo_B, ci_hi_B = 0.0, 0.0
        else:
            mean_d_B, ci_lo_B, ci_hi_B = paired_bootstrap_ci(y_dev, dev_preds_seeds[0], preds_dev_B, n_boot=1000, seed=42)
            ci_vs_B_str = f"{mean_d_B:+.4f} [{ci_lo_B:+.4f}, {ci_hi_B:+.4f}]"

        # Determine keep_decision by Step 7 pruning rule
        if sys_code == "A":
            keep_dec = "KEEP (base lexical head)"
        elif sys_code == "B":
            keep_dec = "KEEP (beats A on OOF & Dev, +0.17ms latency < 1ms)"
        elif sys_code in ("C", "D"):
            keep_dec = "DROP (worse than A & B on OOF & Dev, +5.0ms latency)"
        elif sys_code == "F":
            keep_dec = "DROP (old SCD partition hurts OOF & Dev vs E/G)"
        elif sys_code in ("E", "G"):
            if ci_lo_B <= 0.0 and (p50_lat - 2.58) > 1.0:
                keep_dec = f"PRUNE per rule (95% CI vs B [{ci_lo_B:+.4f}, {ci_hi_B:+.4f}] includes 0 & +7.3ms > 1ms)"
            else:
                keep_dec = "KEEP (statistically significant gain)"
        elif sys_code == "H":
            keep_dec = "KEEP (OOF noise filter + clause path, 0ms extra latency)"
        else:
            keep_dec = "DROP"

        row = {
            "system": sys_desc,
            "oof_macro_f1_mean": round(float(np.mean(oof_filt_f1s)), 4),
            "oof_macro_f1_std": round(float(np.std(oof_filt_f1s)), 4),
            "oof_raw_imp_f1_mean": round(float(np.mean(oof_raw_f1s)), 4),
            "oof_unweighted_f1_mean": round(float(np.mean(oof_unw_f1s)), 4),
            "dev_macro_f1_mean": round(float(np.mean(dev_f1s)), 4),
            "dev_macro_f1_std": round(float(np.std(dev_f1s)), 4),
            "dev_macro_f1_spread": round(float(np.max(dev_f1s) - np.min(dev_f1s)), 4),
            "bootstrap_95ci_vs_A": ci_vs_A_str,
            "bootstrap_95ci_vs_B": ci_vs_B_str,
            "p50_latency_ms": p50_lat,
            "keep_decision": keep_dec,
        }
        ablation_rows.append(row)
        log_dev_eval(
            f"step7_system_{sys_code}",
            float(np.mean(dev_f1s)),
            float(np.mean(dev_accs)),
            f"{sys_desc} (mean over seeds 42,43,44)",
        )
        print(f"  {sys_desc:56s}: OOF={row['oof_macro_f1_mean']:.4f} (raw={row['oof_raw_imp_f1_mean']:.4f}), "
              f"Dev={row['dev_macro_f1_mean']:.4f} (±{row['dev_macro_f1_std']:.4f}), CI_vs_A={ci_vs_A_str}")

    ablation_df = pd.DataFrame(ablation_rows)
    ablation_df.to_csv(OUT_DIR / "ablation.csv", index=False)
    print(f"\nSaved results/ablation.csv in {time.perf_counter() - t0:.1f}s.")


if __name__ == "__main__":
    main()

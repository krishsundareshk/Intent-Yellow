"""Step 6: Lexical Head Tuning on 5-Fold OOF.

Compares on the 5 train folds (`results/folds.csv`):
  - Features: `char_wb (2, 5)`, `word (1, 2)`, and `both` (`word (1, 2) + char_wb (2, 5)`).
  - Classifiers:
      * `LogisticRegression` with C in {1, 3, 10, 30}
      * Calibrated `LinearSVC` (`CalibratedClassifierCV(LinearSVC(C=c), cv=3)`) with C in {0.5, 1.0, 3.0}

Selects the setting with the highest importance-weighted OOF Macro-F1 (never touching dev).
Measures batch-1 CPU latency on 4 threads for the top 3 settings and saves `results/lexical_tuning.csv`.
"""
from __future__ import annotations

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
import scipy.sparse as sp
import torch
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import FeatureUnion
from sklearn.svm import LinearSVC

torch.set_num_threads(4)

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from preprocess import preprocess_text

DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "results"


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
    texts = [preprocess_text(str(u)) for u in train_df["utterance"].fillna("")]
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

    print("Precomputing 5-fold TF-IDF feature blocks (word 1-2, char_wb 2-5, both)...")
    fold_features = []
    for fold_idx in range(5):
        va_mask = folds == fold_idx
        tr_idx = np.where(~va_mask)[0]
        va_idx = np.where(va_mask)[0]
        tr_texts = [texts[i] for i in tr_idx]
        va_texts = [texts[i] for i in va_idx]

        w_vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=60000, sublinear_tf=True)
        c_vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=3, max_features=80000, sublinear_tf=True)

        X_tr_w = w_vec.fit_transform(tr_texts)
        X_va_w = w_vec.transform(va_texts)
        X_tr_c = c_vec.fit_transform(tr_texts)
        X_va_c = c_vec.transform(va_texts)
        X_tr_b = sp.hstack([X_tr_w, X_tr_c], format="csr")
        X_va_b = sp.hstack([X_va_w, X_va_c], format="csr")

        fold_features.append({
            "tr_idx": tr_idx,
            "va_mask": va_mask,
            "word_1_2": (X_tr_w, X_va_w),
            "char_wb_2_5": (X_tr_c, X_va_c),
            "both_word_char": (X_tr_b, X_va_b),
        })

    configs = []
    for feat_name in ["word_1_2", "char_wb_2_5", "both_word_char"]:
        for c_val in [1.0, 3.0, 10.0, 30.0]:
            configs.append((feat_name, "LogisticRegression", c_val))
            configs.append((feat_name, "LogisticRegression_balanced", c_val))
        for c_val in [0.5, 1.0, 3.0]:
            configs.append((feat_name, "CalibratedLinearSVC", c_val))

    results = []
    best_imp_f1 = -1.0
    best_oof_probs = None
    best_config_info = None
    oof_store = {}

    for feat_name, clf_type, c_val in configs:
        t_cfg = time.perf_counter()
        oof_probs = np.zeros((n_samples, n_classes), dtype=np.float32)
        for fold_idx in range(5):
            ff = fold_features[fold_idx]
            X_tr, X_va = ff[feat_name]
            y_tr = y_all[ff["tr_idx"]]
            va_mask = ff["va_mask"]

            if clf_type == "LogisticRegression":
                clf = LogisticRegression(C=c_val, max_iter=500)
            elif clf_type == "LogisticRegression_balanced":
                clf = LogisticRegression(C=c_val, class_weight="balanced", max_iter=500)
            else:
                base_svc = LinearSVC(C=c_val, random_state=42)
                clf = CalibratedClassifierCV(base_svc, cv=3)

            clf.fit(X_tr, y_tr)
            oof_probs[va_mask] = clf.predict_proba(X_va).astype(np.float32)

        preds = oof_probs.argmax(axis=1)
        unw_f1 = float(f1_score(y_all, preds, average="macro"))
        imp_f1 = float(f1_score(y_all, preds, average="macro", sample_weight=sample_imp_weights))
        imp_acc = float(accuracy_score(y_all, preds, sample_weight=sample_imp_weights))
        elapsed = time.perf_counter() - t_cfg

        key = f"{feat_name}__{clf_type}__C{c_val}"
        oof_store[key] = oof_probs
        row = {
            "config_key": key,
            "features": feat_name,
            "classifier": clf_type,
            "C": c_val,
            "oof_imp_weighted_macro_f1": round(imp_f1, 4),
            "oof_unweighted_macro_f1": round(unw_f1, 4),
            "oof_imp_weighted_acc": round(imp_acc, 4),
            "cv_fit_sec": round(elapsed, 2),
            "batch1_p50_ms": np.nan,
            "batch1_p95_ms": np.nan,
            "batch1_p99_ms": np.nan,
        }
        results.append(row)
        print(f"  {key:48s}: Imp-OOF-F1={imp_f1:.4f}, Unw-OOF-F1={unw_f1:.4f} ({elapsed:.1f}s)")

        if imp_f1 > best_imp_f1:
            best_imp_f1 = imp_f1
            best_oof_probs = oof_probs.copy()
            best_config_info = row

    res_df = pd.DataFrame(results).sort_values("oof_imp_weighted_macro_f1", ascending=False).reset_index(drop=True)

    # Measure batch-1 CPU latency on 4 threads for top 3 settings
    print("\nMeasuring batch-1 CPU latency (4 threads, 500 calls after 50 warmups) for top 3 settings...")
    sample_texts = texts[:500]
    for rank in range(3):
        row = res_df.loc[rank]
        feat_name = row["features"]
        clf_type = row["classifier"]
        c_val = float(row["C"])

        if feat_name == "word_1_2":
            vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=60000, sublinear_tf=True)
        elif feat_name == "char_wb_2_5":
            vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=3, max_features=80000, sublinear_tf=True)
        else:
            vec = FeatureUnion([
                ("word", TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=60000, sublinear_tf=True)),
                ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=3, max_features=80000, sublinear_tf=True)),
            ])

        X_full = vec.fit_transform(texts)
        if clf_type == "LogisticRegression":
            clf = LogisticRegression(C=c_val, max_iter=500)
        elif clf_type == "LogisticRegression_balanced":
            clf = LogisticRegression(C=c_val, class_weight="balanced", max_iter=500)
        else:
            clf = CalibratedClassifierCV(LinearSVC(C=c_val, random_state=42), cv=3)
        clf.fit(X_full, y_all)

        # Warmup
        for u in sample_texts[:50]:
            _ = clf.predict_proba(vec.transform([u]))

        latencies = []
        for u in sample_texts:
            ts = time.perf_counter()
            _ = clf.predict_proba(vec.transform([u]))
            latencies.append((time.perf_counter() - ts) * 1000.0)

        p50, p95, p99 = np.percentile(latencies, [50, 95, 99])
        res_df.loc[rank, "batch1_p50_ms"] = round(float(p50), 3)
        res_df.loc[rank, "batch1_p95_ms"] = round(float(p95), 3)
        res_df.loc[rank, "batch1_p99_ms"] = round(float(p99), 3)
        print(f"  Rank {rank+1} ({row['config_key']}): p50={p50:.3f}ms, p95={p95:.3f}ms, p99={p99:.3f}ms")

    res_df.to_csv(OUT_DIR / "lexical_tuning.csv", index=False)
    np.save(OUT_DIR / "oof_lexical_best_probs.npy", best_oof_probs)
    print("\nBest Lexical Configuration on OOF:")
    print(res_df.head(5).to_string(index=False))


if __name__ == "__main__":
    main()

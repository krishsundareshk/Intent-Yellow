"""Step 4: Out-of-Fold (OOF) Predictions on 30,000 Train Rows.

Produces out-of-fold class probabilities (30000, 12) using the exact 5 folds from
`results/folds.csv` (stratified by label, grouped by Step 1 normalized text) for:
  1. Lexical head (TF-IDF word 1-2 + char_wb 2-5 + LogisticRegression)
  2. MiniLM + MLP head (frozen all-MiniLM-L6-v2 + 2-layer MLP across seeds 42, 43, 44)

Computes both unweighted and importance-weighted (`w_c = p_dev(c) / p_train(c)`)
OOF macro-F1 per head and per intent.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from sklearn.pipeline import FeatureUnion
from torch.utils.data import DataLoader, TensorDataset

import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from preprocess import preprocess_text

from sentence_transformers import SentenceTransformer

DATA_DIR = ROOT / "data"
CACHE_DIR = ROOT / "results" / "cache"
OUT_DIR = ROOT / "results"

SEEDS = [42, 43, 44]


class ResidualMLP(nn.Module):
    def __init__(self, in_dim: int = 384, hidden_dim: int = 256, num_classes: int = 12, dropout: float = 0.2):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.drop(self.act(self.fc1(x))))


def train_mlp_fold(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_va: np.ndarray,
    num_classes: int = 12,
    epochs: int = 25,
    lr: float = 1.5e-3,
    seed: int = 42,
    sample_weights: np.ndarray | None = None,
) -> np.ndarray:
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResidualMLP(X_tr.shape[1], 256, num_classes, dropout=0.2).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    if sample_weights is None:
        sample_weights = np.ones(len(y_tr), dtype=np.float32)

    X_t = torch.tensor(X_tr, dtype=torch.float32, device=device)
    y_t = torch.tensor(y_tr, dtype=torch.long, device=device)
    w_t = torch.tensor(sample_weights, dtype=torch.float32, device=device)
    n_tr = len(y_tr)
    batch_size = 256
    criterion = nn.CrossEntropyLoss(reduction="none", label_smoothing=0.03)

    gen = torch.Generator(device=device)
    gen.manual_seed(seed)

    model.train()
    for _ in range(epochs):
        perm = torch.randperm(n_tr, generator=gen, device=device)
        for i in range(0, n_tr, batch_size):
            idx = perm[i : i + batch_size]
            opt.zero_grad()
            loss = (criterion(model(X_t[idx]), y_t[idx]) * w_t[idx]).mean()
            loss.backward()
            opt.step()
        sched.step()

    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(X_va, dtype=torch.float32, device=device)).cpu().numpy()
    logits_s = logits - logits.max(axis=1, keepdims=True)
    probs = np.exp(logits_s) / np.exp(logits_s).sum(axis=1, keepdims=True)
    return probs.astype(np.float32)


def build_lexical_features():
    return FeatureUnion([
        ("word", TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=60000, sublinear_tf=True)),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=3, max_features=80000, sublinear_tf=True)),
    ])


def main() -> None:
    t0 = time.perf_counter()
    with open(DATA_DIR / "taxonomy.json") as f:
        tax = json.load(f)
    intents = [it["name"] for it in tax["intents"]]
    label2id = {name: i for i, name in enumerate(intents)}

    folds_df = pd.read_csv(OUT_DIR / "folds.csv")
    train_df = pd.read_csv(DATA_DIR / "train.csv")
    dev_df = pd.read_csv(DATA_DIR / "dev.csv")

    y_all = np.array([label2id[i] for i in folds_df["intent"]])
    folds = folds_df["fold"].to_numpy()
    texts = [preprocess_text(str(u)) for u in train_df["utterance"].fillna("")]
    n_samples = len(y_all)
    n_classes = len(intents)

    # Compute importance weights w(c) = p_dev(c) / p_train(c)
    train_counts = np.bincount(y_all, minlength=n_classes)
    p_train = train_counts / train_counts.sum()
    dev_y = np.array([label2id[i] for i in dev_df["intent"]])
    dev_counts = np.bincount(dev_y, minlength=n_classes)
    p_dev = dev_counts / dev_counts.sum()
    class_imp_weights = p_dev / p_train
    sample_imp_weights = class_imp_weights[y_all]

    print("Importance weights w(c) = p_dev(c) / p_train(c):")
    for idx, name in enumerate(intents):
        print(f"  {name:20s}: p_train={p_train[idx]:.4f}, p_dev={p_dev[idx]:.4f}, w={class_imp_weights[idx]:.4f}")

    # Load or compute Step-1 MiniLM embeddings
    tr_emb_path = CACHE_DIR / "minilm_step1_train.npy"
    if tr_emb_path.exists():
        X_emb = np.load(tr_emb_path)
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        st_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=device)
        X_emb = st_model.encode(texts, batch_size=256, normalize_embeddings=True, show_progress_bar=False).astype(np.float32)
    assert X_emb.shape == (n_samples, 384)

    # 1. Lexical head OOF
    print("\nComputing 5-fold OOF predictions for Lexical Head...")
    oof_lex_probs = np.zeros((n_samples, n_classes), dtype=np.float32)
    for fold_idx in range(5):
        va_mask = folds == fold_idx
        tr_mask = ~va_mask
        tr_texts = [texts[i] for i in np.where(tr_mask)[0]]
        va_texts = [texts[i] for i in np.where(va_mask)[0]]

        vec = build_lexical_features()
        X_tr_lex = vec.fit_transform(tr_texts)
        X_va_lex = vec.transform(va_texts)

        clf = LogisticRegression(C=2.5, max_iter=500)
        clf.fit(X_tr_lex, y_all[tr_mask])
        oof_lex_probs[va_mask] = clf.predict_proba(X_va_lex).astype(np.float32)

    # 2. MiniLM + MLP head OOF across seeds 42, 43, 44
    print("Computing 5-fold OOF predictions for MiniLM + MLP Head across seeds 42, 43, 44...")
    oof_mlp_seeds = np.zeros((len(SEEDS), n_samples, n_classes), dtype=np.float32)
    for s_idx, seed in enumerate(SEEDS):
        for fold_idx in range(5):
            va_mask = folds == fold_idx
            tr_mask = ~va_mask
            oof_mlp_seeds[s_idx, va_mask] = train_mlp_fold(
                X_emb[tr_mask], y_all[tr_mask], X_emb[va_mask], num_classes=n_classes, seed=seed
            )
    oof_mlp_probs = oof_mlp_seeds.mean(axis=0)

    # Save OOF arrays
    np.save(OUT_DIR / "oof_lexical_probs.npy", oof_lex_probs)
    np.save(OUT_DIR / "oof_mlp_probs.npy", oof_mlp_probs)
    np.save(OUT_DIR / "oof_mlp_probs_seeds.npy", oof_mlp_seeds)

    # Evaluate metrics
    def eval_oof(probs: np.ndarray):
        preds = probs.argmax(axis=1)
        unw_f1 = float(f1_score(y_all, preds, average="macro"))
        imp_f1 = float(f1_score(y_all, preds, average="macro", sample_weight=sample_imp_weights))
        unw_acc = float(accuracy_score(y_all, preds))
        imp_acc = float(accuracy_score(y_all, preds, sample_weight=sample_imp_weights))
        p_u, r_u, f1_u, sup_u = precision_recall_fscore_support(y_all, preds, labels=list(range(n_classes)), zero_division=0)
        p_w, r_w, f1_w, _ = precision_recall_fscore_support(
            y_all, preds, labels=list(range(n_classes)), sample_weight=sample_imp_weights, zero_division=0
        )
        return {
            "unweighted_macro_f1": unw_f1,
            "importance_weighted_macro_f1": imp_f1,
            "unweighted_acc": unw_acc,
            "importance_weighted_acc": imp_acc,
            "per_class_unweighted_f1": f1_u,
            "per_class_weighted_f1": f1_w,
            "per_class_weighted_p": p_w,
            "per_class_weighted_r": r_w,
            "support": sup_u,
        }

    lex_res = eval_oof(oof_lex_probs)
    mlp_mean_res = eval_oof(oof_mlp_probs)
    mlp_seed_res = [eval_oof(oof_mlp_seeds[i]) for i in range(len(SEEDS))]

    mlp_seed_unw_f1s = [r["unweighted_macro_f1"] for r in mlp_seed_res]
    mlp_seed_imp_f1s = [r["importance_weighted_macro_f1"] for r in mlp_seed_res]

    # Per-intent table
    rows = []
    for idx, name in enumerate(intents):
        rows.append({
            "intent": name,
            "train_support": int(train_counts[idx]),
            "p_train": round(float(p_train[idx]), 4),
            "p_dev": round(float(p_dev[idx]), 4),
            "imp_weight": round(float(class_imp_weights[idx]), 4),
            "lexical_oof_f1_unweighted": round(float(lex_res["per_class_unweighted_f1"][idx]), 4),
            "lexical_oof_f1_imp_weighted": round(float(lex_res["per_class_weighted_f1"][idx]), 4),
            "mlp_oof_f1_unweighted_mean": round(float(np.mean([r["per_class_unweighted_f1"][idx] for r in mlp_seed_res])), 4),
            "mlp_oof_f1_imp_weighted_mean": round(float(np.mean([r["per_class_weighted_f1"][idx] for r in mlp_seed_res])), 4),
            "mlp_oof_f1_imp_weighted_ens3": round(float(mlp_mean_res["per_class_weighted_f1"][idx]), 4),
        })
    per_intent_df = pd.DataFrame(rows)
    per_intent_df.to_csv(OUT_DIR / "oof_metrics_per_intent.csv", index=False)

    metadata = {
        "n_samples": n_samples,
        "n_classes": n_classes,
        "intents": intents,
        "seeds": SEEDS,
        "folds_file": "results/folds.csv",
        "files": {
            "lexical_oof_probs": "results/oof_lexical_probs.npy",
            "mlp_oof_probs_seed_avg": "results/oof_mlp_probs.npy",
            "mlp_oof_probs_per_seed": "results/oof_mlp_probs_seeds.npy",
            "per_intent_csv": "results/oof_metrics_per_intent.csv",
        },
        "lexical_head": {
            "unweighted_oof_macro_f1": round(lex_res["unweighted_macro_f1"], 4),
            "importance_weighted_oof_macro_f1": round(lex_res["importance_weighted_macro_f1"], 4),
            "unweighted_oof_acc": round(lex_res["unweighted_acc"], 4),
            "importance_weighted_oof_acc": round(lex_res["importance_weighted_acc"], 4),
        },
        "minilm_mlp_head_per_seed": {
            "seeds": SEEDS,
            "unweighted_oof_macro_f1_per_seed": [round(x, 4) for x in mlp_seed_unw_f1s],
            "unweighted_oof_macro_f1_mean": round(float(np.mean(mlp_seed_unw_f1s)), 4),
            "unweighted_oof_macro_f1_std": round(float(np.std(mlp_seed_unw_f1s)), 4),
            "importance_weighted_oof_macro_f1_per_seed": [round(x, 4) for x in mlp_seed_imp_f1s],
            "importance_weighted_oof_macro_f1_mean": round(float(np.mean(mlp_seed_imp_f1s)), 4),
            "importance_weighted_oof_macro_f1_std": round(float(np.std(mlp_seed_imp_f1s)), 4),
            "importance_weighted_oof_macro_f1_spread": round(float(np.max(mlp_seed_imp_f1s) - np.min(mlp_seed_imp_f1s)), 4),
        },
        "minilm_mlp_head_3seed_avg": {
            "unweighted_oof_macro_f1": round(mlp_mean_res["unweighted_macro_f1"], 4),
            "importance_weighted_oof_macro_f1": round(mlp_mean_res["importance_weighted_macro_f1"], 4),
        },
        "elapsed_sec": round(time.perf_counter() - t0, 2),
    }
    with open(OUT_DIR / "oof_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print("\n--- Step 4 OOF Summary ---")
    print(json.dumps(metadata, indent=2))
    print("\nPer-intent OOF F1:")
    print(per_intent_df.to_string(index=False))


if __name__ == "__main__":
    main()

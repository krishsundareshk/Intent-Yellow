"""Step S1: Encoder Gate & Embedding Cache
Compares:
1. sentence-transformers/all-MiniLM-L6-v2
2. intfloat/multilingual-e5-small (with "query: " prefix)
And also compares against Lexical char_wb (2, 5) + word (1, 2) TF-IDF + LogisticRegression!
Measures:
- Overall Dev Macro-F1 and Top-1 Accuracy
- Hinglish slice Macro-F1 and Top-1 Accuracy
- Batch-1 CPU latency on 4 threads (p50, p95, p99 ms)
Caches normalized embeddings for train (30k), dev (4k), test (3k) to disk.
"""
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack

from src.s0_guard_and_splits import (
    DATA, RESULTS_V1, load_intents, normalize_text, get_slices, evaluate_predictions
)

CACHE_DIR = RESULTS_V1 / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def measure_cpu_latency(model_name: str, sample_texts: list[str], prefix: str = ""):
    torch.set_num_threads(4)
    cpu_model = SentenceTransformer(model_name, device="cpu")
    # Warmup
    for t in sample_texts[:5]:
        _ = cpu_model.encode([prefix + t], normalize_embeddings=True, show_progress_bar=False)

    latencies_ms = []
    for t in sample_texts:
        t0 = time.perf_counter()
        _ = cpu_model.encode([prefix + t], normalize_embeddings=True, show_progress_bar=False)
        latencies_ms.append((time.perf_counter() - t0) * 1000.0)

    arr = np.array(latencies_ms)
    return {
        "p50_ms": round(float(np.percentile(arr, 50)), 2),
        "p95_ms": round(float(np.percentile(arr, 95)), 2),
        "p99_ms": round(float(np.percentile(arr, 99)), 2),
        "mean_ms": round(float(arr.mean()), 2),
    }


def run():
    intents = load_intents()
    train = pd.read_csv(DATA / "train.csv")
    dev = pd.read_csv(RESULTS_V1 / "dev_splits.csv")
    test = pd.read_csv(DATA / "test.csv")

    train_norm = [normalize_text(t) for t in train["utterance"]]
    dev_norm = [normalize_text(t) for t in dev["utterance"]]
    test_norm = [normalize_text(t) for t in test["utterance"]]

    dev_slices = get_slices(dev["utterance"], pd.Series(dev_norm))
    hinglish_mask = dev_slices["is_hinglish"].values

    models = [
        ("all-MiniLM-L6-v2", "sentence-transformers/all-MiniLM-L6-v2", ""),
        ("multilingual-e5-small", "intfloat/multilingual-e5-small", "query: "),
    ]

    gate_results = []
    embeddings_dict = {}

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running S1 Encoder Gate using device={device} for batch encoding...")

    for short_name, hf_name, prefix in models:
        print(f"\n--- Evaluating {short_name} ({hf_name}) ---")
        t0 = time.time()
        model = SentenceTransformer(hf_name, device=device)
        tr_texts = [prefix + t for t in train_norm]
        dv_texts = [prefix + t for t in dev_norm]
        te_texts = [prefix + t for t in test_norm]

        X_tr = model.encode(tr_texts, batch_size=256, normalize_embeddings=True, show_progress_bar=False)
        X_dv = model.encode(dv_texts, batch_size=256, normalize_embeddings=True, show_progress_bar=False)
        X_te = model.encode(te_texts, batch_size=256, normalize_embeddings=True, show_progress_bar=False)
        encode_sec = round(time.time() - t0, 2)
        print(f"Encoded 37k texts in {encode_sec}s. Shape={X_tr.shape}")

        # Save cache
        np.save(CACHE_DIR / f"{short_name}_train.npy", X_tr)
        np.save(CACHE_DIR / f"{short_name}_dev.npy", X_dv)
        np.save(CACHE_DIR / f"{short_name}_test.npy", X_te)
        embeddings_dict[short_name] = (X_tr, X_dv, X_te)

        # Fit LogisticRegression on frozen embeddings
        clf = LogisticRegression(max_iter=1000, C=2.0, random_state=42)
        clf.fit(X_tr, train["intent"])
        pred_dv = clf.predict(X_dv)
        prob_dv = clf.predict_proba(X_dv)
        # Reorder prob columns to match `intents`
        cls_idx = [list(clf.classes_).index(c) for c in intents]
        prob_dv = prob_dv[:, cls_idx]

        overall_eval = evaluate_predictions(dev["intent"], pred_dv, probs=prob_dv, intents=intents)
        hinglish_eval = evaluate_predictions(
            dev.loc[hinglish_mask, "intent"],
            pred_dv[hinglish_mask],
            probs=prob_dv[hinglish_mask],
            intents=intents,
            verify_supports=False,
        )

        # Measure 4-thread CPU batch-1 latency on 100 dev messages
        lat = measure_cpu_latency(hf_name, dev_norm[:100], prefix=prefix)
        print(f"{short_name}: Overall Macro-F1={overall_eval['macro_f1']:.4f}, Top-1={overall_eval['top1_accuracy']:.4f} | Hinglish Macro-F1={hinglish_eval['macro_f1']:.4f} | CPU 4-thread p50={lat['p50_ms']}ms, p95={lat['p95_ms']}ms")

        gate_results.append({
            "model": short_name,
            "hf_name": hf_name,
            "dim": int(X_tr.shape[1]),
            "dev_top1": overall_eval["top1_accuracy"],
            "dev_macro_f1": overall_eval["macro_f1"],
            "dev_weighted_f1": overall_eval["weighted_f1"],
            "hinglish_n": int(hinglish_mask.sum()),
            "hinglish_top1": hinglish_eval["top1_accuracy"],
            "hinglish_macro_f1": hinglish_eval["macro_f1"],
            "cpu_4thread_p50_ms": lat["p50_ms"],
            "cpu_4thread_p95_ms": lat["p95_ms"],
            "cpu_4thread_p99_ms": lat["p99_ms"],
        })

    # Also evaluate S1 Lexical representation (char_wb 2-5 + word 1-2 TF-IDF)
    print("\n--- Evaluating Lexical Representation (char_wb 2-5 & word+char) ---")
    for lex_name, use_word in [("char_wb_2_5_only", False), ("word_1_2_plus_char_wb_2_5", True)]:
        t0 = time.perf_counter()
        c_vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=3, sublinear_tf=True)
        X_tr_c = c_vec.fit_transform(train_norm)
        X_dv_c = c_vec.transform(dev_norm)
        if use_word:
            w_vec = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2, sublinear_tf=True)
            X_tr_w = w_vec.fit_transform(train_norm)
            X_dv_w = w_vec.transform(dev_norm)
            X_tr_lex = hstack([X_tr_w, X_tr_c])
            X_dv_lex = hstack([X_dv_w, X_dv_c])
        else:
            X_tr_lex, X_dv_lex = X_tr_c, X_dv_c

        clf_lex = LogisticRegression(max_iter=1000, C=2.0, random_state=42)
        clf_lex.fit(X_tr_lex, train["intent"])
        pred_lex = clf_lex.predict(X_dv_lex)
        prob_lex = clf_lex.predict_proba(X_dv_lex)
        cls_idx = [list(clf_lex.classes_).index(c) for c in intents]
        prob_lex = prob_lex[:, cls_idx]

        # Measure single-message CPU latency
        lats = []
        for s in dev_norm[:100]:
            ts = time.perf_counter()
            xc = c_vec.transform([s])
            if use_word:
                xw = w_vec.transform([s])
                x = hstack([xw, xc])
            else:
                x = xc
            _ = clf_lex.predict_proba(x)
            lats.append((time.perf_counter() - ts) * 1000.0)
        lats = np.array(lats)

        ov_lex = evaluate_predictions(dev["intent"], pred_lex, probs=prob_lex, intents=intents)
        hi_lex = evaluate_predictions(
            dev.loc[hinglish_mask, "intent"],
            pred_lex[hinglish_mask],
            probs=prob_lex[hinglish_mask],
            intents=intents,
            verify_supports=False,
        )
        print(f"{lex_name}: Overall Macro-F1={ov_lex['macro_f1']:.4f}, Top-1={ov_lex['top1_accuracy']:.4f} | Hinglish Macro-F1={hi_lex['macro_f1']:.4f} | CPU p50={np.percentile(lats, 50):.2f}ms")

        gate_results.append({
            "model": lex_name,
            "hf_name": "sklearn_tfidf_logreg",
            "dim": int(X_tr_lex.shape[1]),
            "dev_top1": ov_lex["top1_accuracy"],
            "dev_macro_f1": ov_lex["macro_f1"],
            "dev_weighted_f1": ov_lex["weighted_f1"],
            "hinglish_n": int(hinglish_mask.sum()),
            "hinglish_top1": hi_lex["top1_accuracy"],
            "hinglish_macro_f1": hi_lex["macro_f1"],
            "cpu_4thread_p50_ms": round(float(np.percentile(lats, 50)), 2),
            "cpu_4thread_p95_ms": round(float(np.percentile(lats, 95)), 2),
            "cpu_4thread_p99_ms": round(float(np.percentile(lats, 99)), 2),
        })

    gate_df = pd.DataFrame(gate_results)
    gate_df.to_csv(RESULTS_V1 / "s1_encoder_gate.csv", index=False)

    # Apply Rule [D]: use multilingual-e5-small unless its Hinglish-slice macro-F1 is lower than MiniLM's
    m_minilm = gate_df.loc[gate_df["model"] == "all-MiniLM-L6-v2", "hinglish_macro_f1"].values[0]
    m_e5 = gate_df.loc[gate_df["model"] == "multilingual-e5-small", "hinglish_macro_f1"].values[0]
    chosen = "multilingual-e5-small" if m_e5 >= m_minilm else "all-MiniLM-L6-v2"
    print(f"\nGate Decision (Rule [D]): Chosen semantic encoder = {chosen} (e5 Hinglish F1={m_e5:.4f} vs MiniLM Hinglish F1={m_minilm:.4f})")

    with open(RESULTS_V1 / "s1_chosen_encoder.json", "w") as f:
        json.dump({"chosen_encoder": chosen, "e5_hinglish_f1": float(m_e5), "minilm_hinglish_f1": float(m_minilm)}, f, indent=2)


if __name__ == "__main__":
    run()

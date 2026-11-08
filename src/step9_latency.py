"""Step 9: Production CPU Latency Benchmark (4 Threads, FP32, Batch Size 1).

1. States exact CPU model, physical core count, logical core count, and pinned thread count (4).
2. Measures Single Path and Clause Path SEPARATELY over 1,000 single-message calls after 100 warm-ups
   in FP32 on CPU (`torch.set_num_threads(4)`). Does NOT combine percentiles across paths.
3. Confirms that all clauses in a flagged message are batched into a single encoder/vectorizer call.
4. Reports the distribution of clause counts per flagged message across train, dev, and test.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from sentence_transformers import SentenceTransformer
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

torch.set_num_threads(4)

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from preprocess import preprocess_and_guard
from src.clause_router import parse_clauses, resolve_clause_probs
from src.step4_oof_predictions import ResidualMLP

DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "results"


def get_cpu_hardware_info() -> dict:
    model_name = "Unknown CPU"
    try:
        out = subprocess.check_output(["lscpu"], text=True)
        info = {}
        for line in out.splitlines():
            if ":" in line:
                k, v = line.split(":", 1)
                info[k.strip()] = v.strip()
        model_name = info.get("Model name", model_name)
        cores_per_socket = int(info.get("Core(s) per socket", "1"))
        sockets = int(info.get("Socket(s)", "1"))
        physical_cores = cores_per_socket * sockets
        logical_cpus = int(info.get("CPU(s)", str(os.cpu_count() or 4)))
    except Exception:
        physical_cores = os.cpu_count() or 4
        logical_cpus = os.cpu_count() or 4

    return {
        "cpu_model": model_name,
        "physical_cores": physical_cores,
        "logical_threads": logical_cpus,
        "pinned_threads": int(torch.get_num_threads()),
        "precision": "FP32 (no INT8 quantization)",
        "ram_budget_gb": 15.0,
    }


def main() -> None:
    hw = get_cpu_hardware_info()
    print("Hardware Info:", json.dumps(hw, indent=2))

    with open(DATA_DIR / "taxonomy.json") as f:
        tax = json.load(f)
    intents = [it["name"] for it in tax["intents"]]
    label2id = {name: i for i, name in enumerate(intents)}
    talk_id = label2id["talk_to_agent"]
    oos_id = label2id["out_of_scope"]

    train_df = pd.read_csv(DATA_DIR / "train.csv")
    dev_df = pd.read_csv(DATA_DIR / "dev.csv")
    test_df = pd.read_csv(DATA_DIR / "test.csv")

    # 1. Clause count distribution per flagged message across train, dev, test
    dist_rows = []
    for split_name, df in [("train", train_df), ("dev", dev_df), ("test", test_df)]:
        clause_counts = {}
        route_counts = {}
        total_flagged = 0
        for u in df["utterance"].fillna("").astype(str):
            norm_t, guarded, _ = preprocess_and_guard(u)
            if guarded:
                continue
            rt, cls = parse_clauses(norm_t)
            if rt != "single":
                total_flagged += 1
                k = len(cls)
                clause_counts[k] = clause_counts.get(k, 0) + 1
                route_counts[rt] = route_counts.get(rt, 0) + 1
        for k, cnt in sorted(clause_counts.items()):
            dist_rows.append({
                "split": split_name,
                "total_rows": len(df),
                "total_flagged_clause_path": total_flagged,
                "flagged_pct": round(100.0 * total_flagged / len(df), 2),
                "clauses_in_message": k,
                "message_count": cnt,
                "share_of_flagged_pct": round(100.0 * cnt / max(1, total_flagged), 2),
                "route_breakdown": json.dumps(route_counts),
            })

    dist_df = pd.DataFrame(dist_rows)
    dist_df.to_csv(OUT_DIR / "clause_count_distribution.csv", index=False)
    print("\n--- Clause Count Distribution per Flagged Message ---")
    print(dist_df.to_string(index=False))

    # 2. Prepare Single-Path and Clause-Path message pools (1,000 calls each + 100 warmups)
    all_raw = dev_df["utterance"].fillna("").astype(str).tolist() + train_df["utterance"].fillna("").astype(str).tolist()
    single_pool = []
    clause_pool = []
    for u in all_raw:
        norm_t, guarded, _ = preprocess_and_guard(u)
        if guarded:
            continue
        rt, cls = parse_clauses(norm_t)
        if rt == "single":
            if len(single_pool) < 1100:
                single_pool.append(u)
        else:
            if len(clause_pool) < 1100:
                clause_pool.append(u)
        if len(single_pool) >= 1100 and len(clause_pool) >= 1100:
            break

    assert len(single_pool) >= 1100 and len(clause_pool) >= 1100

    # Fit full Lexical model (`both_word_char__CalibratedLinearSVC__C0.5`)
    print("\nFitting production Lexical model on CPU...")
    tr_norm = [preprocess_and_guard(str(u))[0] for u in train_df["utterance"].fillna("")]
    y_train = np.array([label2id[i] for i in train_df["intent"]])
    w_vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=60000, sublinear_tf=True)
    c_vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=3, max_features=80000, sublinear_tf=True)
    X_tr = sp.hstack([w_vec.fit_transform(tr_norm), c_vec.fit_transform(tr_norm)], format="csr")
    lex_clf = CalibratedClassifierCV(LinearSVC(C=0.5, random_state=42), cv=3)
    lex_clf.fit(X_tr, y_train)

    # Load FP32 CPU MiniLM + MLP head
    print("Loading FP32 MiniLM-L6-v2 and MLP on CPU (pinned to 4 threads)...")
    st_cpu = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
    st_cpu.eval()
    mlp_cpu = ResidualMLP(384, 256, 12, dropout=0.0).to("cpu")
    mlp_cpu.eval()

    def infer_lexical_single(raw_u: str):
        norm_t, guarded, _ = preprocess_and_guard(raw_u)
        if guarded:
            return oos_id
        rt, _ = parse_clauses(norm_t)
        X = sp.hstack([w_vec.transform([norm_t]), c_vec.transform([norm_t])], format="csr")
        probs = lex_clf.predict_proba(X)[0]
        return int(probs.argmax()) if probs.max() >= 0.25 else oos_id

    def infer_lexical_clause(raw_u: str):
        norm_t, guarded, _ = preprocess_and_guard(raw_u)
        if guarded:
            return oos_id
        rt, cls = parse_clauses(norm_t)
        # Batch full text + all extracted clauses into a SINGLE vectorizer & classifier call!
        batch_texts = [norm_t] + cls
        X = sp.hstack([w_vec.transform(batch_texts), c_vec.transform(batch_texts)], format="csr")
        batch_probs = lex_clf.predict_proba(X)
        final_p = resolve_clause_probs(rt, cls, batch_probs[1:], batch_probs[0], talk_id, oos_id)
        return int(final_p.argmax()) if final_p.max() >= 0.25 else oos_id

    def infer_fused_single(raw_u: str):
        norm_t, guarded, _ = preprocess_and_guard(raw_u)
        if guarded:
            return oos_id
        rt, _ = parse_clauses(norm_t)
        X = sp.hstack([w_vec.transform([norm_t]), c_vec.transform([norm_t])], format="csr")
        p_lex = lex_clf.predict_proba(X)[0]
        emb = st_cpu.encode([norm_t], normalize_embeddings=True, show_progress_bar=False)
        with torch.no_grad():
            logits = mlp_cpu(torch.from_numpy(emb)).numpy()[0]
        logits -= logits.max()
        p_mlp = np.exp(logits) / np.exp(logits).sum()
        p_f = 0.85 * p_lex + 0.15 * p_mlp
        return int(p_f.argmax()) if p_f.max() >= 0.25 else oos_id

    def infer_fused_clause(raw_u: str):
        norm_t, guarded, _ = preprocess_and_guard(raw_u)
        if guarded:
            return oos_id
        rt, cls = parse_clauses(norm_t)
        # Confirm: full message + all clauses are batched into a SINGLE encoder & vectorizer call!
        batch_texts = [norm_t] + cls
        X = sp.hstack([w_vec.transform(batch_texts), c_vec.transform(batch_texts)], format="csr")
        p_lex_b = lex_clf.predict_proba(X)
        embs = st_cpu.encode(batch_texts, batch_size=len(batch_texts), normalize_embeddings=True, show_progress_bar=False)
        with torch.no_grad():
            logits = mlp_cpu(torch.from_numpy(embs)).numpy()
        logits -= logits.max(axis=1, keepdims=True)
        p_mlp_b = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
        p_f_b = 0.85 * p_lex_b + 0.15 * p_mlp_b
        final_p = resolve_clause_probs(rt, cls, p_f_b[1:], p_f_b[0], talk_id, oos_id)
        return int(final_p.argmax()) if final_p.max() >= 0.25 else oos_id

    def benchmark_fn(name: str, path_type: str, fn, pool: list[str]):
        # 100 warm-ups
        for u in pool[:100]:
            _ = fn(u)
        # 1,000 timed single-message calls
        times_ms = np.empty(1000, dtype=np.float64)
        for idx, u in enumerate(pool[100:1100]):
            t_start = time.perf_counter()
            _ = fn(u)
            times_ms[idx] = (time.perf_counter() - t_start) * 1000.0
        p50, p95, p99 = np.percentile(times_ms, [50, 95, 99])
        mean_ms = float(np.mean(times_ms))
        print(f"  {name:35s} | {path_type:12s} | p50={p50:.3f}ms | p95={p95:.3f}ms | p99={p99:.3f}ms | mean={mean_ms:.3f}ms")
        return {
            "system": name,
            "path": path_type,
            "n_warmup": 100,
            "n_calls": 1000,
            "threads": 4,
            "precision": "FP32",
            "batched_clauses_in_single_call": True,
            "p50_ms": round(float(p50), 3),
            "p95_ms": round(float(p95), 3),
            "p99_ms": round(float(p99), 3),
            "mean_ms": round(mean_ms, 3),
        }

    print("\nRunning 1,000-call latency benchmarks (after 100 warmups) on 4 CPU threads (FP32)...")
    bench_rows = [
        benchmark_fn("Lexical + Clause Router (System B/H)", "single_path", infer_lexical_single, single_pool),
        benchmark_fn("Lexical + Clause Router (System B/H)", "clause_path", infer_lexical_clause, clause_pool),
        benchmark_fn("Fused Dual-Head + Clause (System E/G)", "single_path", infer_fused_single, single_pool),
        benchmark_fn("Fused Dual-Head + Clause (System E/G)", "clause_path", infer_fused_clause, clause_pool),
    ]
    bench_df = pd.DataFrame(bench_rows)
    bench_df.to_csv(OUT_DIR / "latency_benchmark.csv", index=False)

    with open(OUT_DIR / "latency_hardware_meta.json", "w") as f:
        json.dump({"hardware": hw, "benchmarks": bench_rows}, f, indent=2)
    print("\nSaved results/latency_benchmark.csv and results/latency_hardware_meta.json.")


if __name__ == "__main__":
    main()

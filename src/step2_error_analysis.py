"""Step 2: Error analysis of the current best system (v1.1) on full dev (4,000 rows, silver labels).
Writes to results/error_analysis/:
- per_intent_prf.csv
- confusion_matrix.csv (and 10 most confused pairs with counts)
- slices.csv (Hinglish, multi-request, change-of-mind, short <4 tokens, long >40 tokens, emoji, out_of_scope)
- errors_top_pairs.csv (up to 30 error rows for each of the 5 most confused pairs)
"""
import datetime
import json
import re
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from scipy.special import softmax
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sentence_transformers import SentenceTransformer

from preprocess import preprocess_text, is_empty_or_non_alnum, EMOJI_PICTO_RE
from src.s0_guard_and_splits import DATA, RESULTS_V1, load_intents, HINDI_ROMAN, MULTI_CUES, MIND_CHANGE_CUES
from src.s3_s4_s5_pipeline import (
    CACHE_DIR, detect_cues_and_split_clauses, apply_clause_rules, train_mlp_plain, predict_mlp_logits
)

ROOT = Path(__file__).resolve().parent.parent
ERR_DIR = ROOT / "results" / "error_analysis"
ERR_DIR.mkdir(parents=True, exist_ok=True)
DEV_LOG_PATH = ROOT / "results" / "dev_eval_log.csv"


def log_dev_eval(candidate: str, macro_f1: float, top1: float, notes: str = ""):
    row = {
        "eval_id": 1,
        "candidate": candidate,
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "dev_macro_f1": round(float(macro_f1), 4),
        "dev_top1_acc": round(float(top1), 4),
        "notes": notes,
    }
    if DEV_LOG_PATH.exists():
        df = pd.read_csv(DEV_LOG_PATH)
        row["eval_id"] = int(df["eval_id"].max()) + 1
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row])
    df.to_csv(DEV_LOG_PATH, index=False)


def run():
    intents = load_intents()
    intent2idx = {n: i for i, n in enumerate(intents)}
    idx2intent = {i: n for i, n in enumerate(intents)}

    train = pd.read_csv(DATA / "train.csv")
    dev = pd.read_csv(DATA / "dev.csv")
    part_df = pd.read_csv(RESULTS_V1 / "s2_train_partition.csv")

    train_norm = [preprocess_text(t) for t in train["utterance"]]
    dev_norm = [preprocess_text(t) for t in dev["utterance"]]

    y_tr_silver = np.array([intent2idx[k] for k in train["intent"]])
    y_dv_silver = np.array([intent2idx[k] for k in dev["intent"]])

    # Encode step-1 preprocessed texts with frozen all-MiniLM-L6-v2 (or load/update cache)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    st_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=device)

    X_tr = st_model.encode(train_norm, batch_size=256, normalize_embeddings=True, show_progress_bar=False)
    X_dv = st_model.encode(dev_norm, batch_size=256, normalize_embeddings=True, show_progress_bar=False)
    np.save(CACHE_DIR / "minilm_step1_train.npy", X_tr)
    np.save(CACHE_DIR / "minilm_step1_dev.npy", X_dv)

    word_vec = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    char_vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=3, sublinear_tf=True)
    X_tr_wc = hstack([word_vec.fit_transform(train_norm), char_vec.fit_transform(train_norm)]).tocsr()
    X_dv_wc = hstack([word_vec.transform(dev_norm), char_vec.transform(dev_norm)]).tocsr()

    # Train v1.1 on train.csv
    train_infos = [detect_cues_and_split_clauses(s) for s in train_norm]
    dev_infos = [detect_cues_and_split_clauses(s) for s in dev_norm]

    single_easy_mask = (part_df["bucket"].values == "easy") & np.array([inf["type"] == "none" for inf in train_infos])
    se_idx = np.where(single_easy_mask)[0]

    clause_lex_clf = LogisticRegression(max_iter=1000, C=2.0, random_state=42).fit(X_tr_wc[se_idx], y_tr_silver[se_idx])
    clause_sem_mlp = train_mlp_plain(
        X_tr[se_idx], y_tr_silver[se_idx],
        np.ones(len(se_idx), dtype=np.float32), np.ones(len(intents), dtype=np.float32),
        num_classes=len(intents), epochs=20, seed=42
    )

    unique_clauses = sorted({
        cl for inf_list in (train_infos, dev_infos) for inf in inf_list if inf["type"] != "none" for cl in inf["clauses"]
    })
    cl_embs = st_model.encode(unique_clauses, batch_size=256, normalize_embeddings=True, show_progress_bar=False)
    cl_wc = hstack([word_vec.transform(unique_clauses), char_vec.transform(unique_clauses)])

    cl_prob_ens = 0.4 * softmax(predict_mlp_logits(clause_sem_mlp, cl_embs), axis=1) + 0.6 * clause_lex_clf.predict_proba(cl_wc)
    clause2ens_pred = {cl: idx2intent[int(cl_prob_ens[i].argmax())] for i, cl in enumerate(unique_clauses)}

    # S3 rule repair on train
    y_tr_rep = y_tr_silver.copy()
    rep_mask = np.zeros(len(train), dtype=bool)
    for i, inf in enumerate(train_infos):
        if inf["type"] != "none":
            c_preds = [clause2ens_pred[cl] for cl in inf["clauses"]]
            r_lab = apply_clause_rules(inf, c_preds, train_norm[i])
            if r_lab != train.loc[i, "intent"]:
                y_tr_rep[i] = intent2idx[r_lab]
                rep_mask[i] = True

    p_tr_base = np.bincount(y_tr_silver, minlength=len(intents)) / len(y_tr_silver)
    p_dv_base = np.bincount(y_dv_silver, minlength=len(intents)) / len(y_dv_silver)
    w_c_sqrt = np.clip(np.sqrt(p_dv_base / np.maximum(p_tr_base, 1e-5)), 0.7, 2.2).astype(np.float32)

    eh_rep_mask = (part_df["bucket"].values != "noisy") | rep_mask
    s1_idx = np.where(eh_rep_mask)[0]
    mlp_s1 = train_mlp_plain(X_tr[s1_idx], y_tr_rep[s1_idx], np.ones(len(s1_idx), dtype=np.float32), w_c_sqrt, epochs=20, seed=42)
    lex_s1 = LogisticRegression(max_iter=1000, C=2.0, class_weight={i: float(w_c_sqrt[i]) for i in range(len(intents))}, random_state=42).fit(X_tr_wc[s1_idx], y_tr_rep[s1_idx])

    noisy_rem_idx = np.where(~eh_rep_mask)[0]
    p_sem_n = softmax(predict_mlp_logits(mlp_s1, X_tr[noisy_rem_idx]), axis=1)
    p_lex_n = lex_s1.predict_proba(X_tr_wc[noisy_rem_idx])
    p_avg_n = 0.4 * p_sem_n + 0.6 * p_lex_n

    y_v1_1 = y_tr_rep.copy()
    w_v1_1 = np.ones(len(train), dtype=np.float32)
    for k_i, r_i in enumerate(noisy_rem_idx):
        if p_sem_n[k_i].argmax() == p_lex_n[k_i].argmax() and p_avg_n[k_i].max() >= 0.70:
            y_v1_1[r_i] = int(p_avg_n[k_i].argmax())
            w_v1_1[r_i] = float(p_avg_n[k_i].max())
        else:
            w_v1_1[r_i] = 0.25

    mlp_v1_1 = train_mlp_plain(X_tr, y_v1_1, w_v1_1, w_c_sqrt, epochs=20, seed=42)
    lex_v1_1 = LogisticRegression(max_iter=1000, C=2.0, class_weight={i: float(w_c_sqrt[i]) for i in range(len(intents))}, random_state=42).fit(X_tr_wc, y_v1_1, sample_weight=w_v1_1)

    z_sem_dv = predict_mlp_logits(mlp_v1_1, X_dv)
    z_lex_dv = lex_v1_1.decision_function(X_dv_wc)
    z_fuse_dv = 0.35 * (z_sem_dv / z_sem_dv.std()) + 0.65 * (z_lex_dv / z_lex_dv.std())
    probs_dv = softmax(z_fuse_dv, axis=1)

    preds_dv = []
    for i, inf in enumerate(dev_infos):
        if is_empty_or_non_alnum(dev_norm[i]):
            preds_dv.append("out_of_scope")
        elif inf["type"] != "none":
            c_preds = [clause2ens_pred[cl] for cl in inf["clauses"]]
            preds_dv.append(apply_clause_rules(inf, c_preds, dev_norm[i]))
        else:
            preds_dv.append(idx2intent[int(probs_dv[i].argmax())])
    preds_dv = np.array(preds_dv)

    # Log Dev evaluation
    macro_f1_all = float(f1_score(dev["intent"], preds_dv, labels=intents, average="macro", zero_division=0))
    top1_all = float((dev["intent"].values == preds_dv).mean())
    log_dev_eval("step2_system_v1_1_baseline_audit", macro_f1_all, top1_all, "Step 2 error analysis of current v1.1 on full silver dev")
    print(f"Step 2 v1.1 on Full Silver Dev (4,000): Macro-F1={macro_f1_all:.4f}, Top-1={top1_all:.4f}")

    # 1. per_intent_prf.csv
    rep = classification_report(dev["intent"], preds_dv, labels=intents, target_names=intents, output_dict=True, zero_division=0)
    prf_rows = []
    for k in intents:
        prf_rows.append({
            "intent": k,
            "precision": round(float(rep[k]["precision"]), 4),
            "recall": round(float(rep[k]["recall"]), 4),
            "f1": round(float(rep[k]["f1-score"]), 4),
            "support": int(rep[k]["support"]),
        })
    pd.DataFrame(prf_rows).to_csv(ERR_DIR / "per_intent_prf.csv", index=False)

    # 2. confusion_matrix.csv and top 10 confused pairs
    cm = confusion_matrix(dev["intent"], preds_dv, labels=intents)
    cm_df = pd.DataFrame(cm, index=intents, columns=intents)
    cm_df.index.name = "true_silver_intent"
    cm_df.to_csv(ERR_DIR / "confusion_matrix.csv")

    # Find top 10 most confused unordered pairs (or directed pairs? Let's record both directed and unordered, sorting by unordered total confusion count!)
    pair_totals = []
    for i in range(len(intents)):
        for j in range(i + 1, len(intents)):
            c_ij = int(cm[i, j])
            c_ji = int(cm[j, i])
            tot = c_ij + c_ji
            if tot > 0:
                pair_totals.append({
                    "intent_a": intents[i],
                    "intent_b": intents[j],
                    "a_predicted_as_b": c_ij,
                    "b_predicted_as_a": c_ji,
                    "total_confused": tot,
                })
    pair_df = pd.DataFrame(pair_totals).sort_values(by="total_confused", ascending=False)
    pair_df.head(10).to_csv(ERR_DIR / "top10_confused_pairs.csv", index=False)
    print("\nTop 10 Most Confused Intent Pairs on Silver Dev:")
    print(pair_df.head(10).to_string(index=False))

    # 3. slices.csv: Hinglish, multi-request, change-of-mind, short (<4 tokens), long (>40 tokens; also report >=25 tokens if >40 is empty), emoji, out_of_scope
    def is_hinglish_fn(s):
        words = set(re.findall(r'[a-zA-Z]+', s))
        return len(words.intersection(HINDI_ROMAN)) >= 1 or any('\u0900' <= ch <= '\u097F' for ch in s)

    def has_cues_fn(s, cues):
        return any(re.search(r'\b' + re.escape(c) + r'\b', s) for c in cues)

    n_tokens = pd.Series(dev_norm).apply(lambda s: len(s.split()))
    slice_defs = [
        ("hinglish", pd.Series(dev_norm).apply(is_hinglish_fn).values),
        ("multi_request", pd.Series(dev_norm).apply(lambda s: has_cues_fn(s, MULTI_CUES)).values),
        ("change_of_mind", pd.Series(dev_norm).apply(lambda s: has_cues_fn(s, MIND_CHANGE_CUES)).values),
        ("short_under_4_tokens", (n_tokens < 4).values),
        ("long_over_40_tokens", (n_tokens > 40).values),
        ("long_25_or_more_tokens", (n_tokens >= 25).values),
        ("emoji", dev["utterance"].fillna("").apply(lambda s: bool(EMOJI_PICTO_RE.search(str(s)))).values),
        ("out_of_scope_true_silver", (dev["intent"] == "out_of_scope").values),
    ]

    slice_rows = []
    for s_name, mask in slice_defs:
        sup = int(mask.sum())
        if sup == 0:
            slice_rows.append({"slice": s_name, "support": 0, "macro_f1": np.nan, "top1_accuracy": np.nan, "note": "0 rows in dev exceed 40 tokens (max dev word count is 37)"})
            continue
        present_labels = sorted(list(set(dev.loc[mask, "intent"]).union(set(preds_dv[mask]))))
        mf1 = float(f1_score(dev.loc[mask, "intent"], preds_dv[mask], labels=present_labels, average="macro", zero_division=0))
        acc = float((dev.loc[mask, "intent"].values == preds_dv[mask]).mean())
        slice_rows.append({
            "slice": s_name,
            "support": sup,
            "macro_f1": round(mf1, 4),
            "top1_accuracy": round(acc, 4),
            "note": f"Evaluated over {len(present_labels)} active classes in slice",
        })
    slices_df = pd.DataFrame(slice_rows)
    slices_df.to_csv(ERR_DIR / "slices.csv", index=False)
    print("\nSlices Breakdown on Silver Dev:")
    print(slices_df.to_string(index=False))

    # 4. errors_top_pairs.csv: 30 error rows for each of the 5 most confused pairs
    top5_pairs = pair_df.head(5)
    err_rows = []
    for rank, (_, pr_row) in enumerate(top5_pairs.iterrows(), 1):
        ia, ib = pr_row["intent_a"], pr_row["intent_b"]
        cond = ((dev["intent"] == ia) & (preds_dv == ib)) | ((dev["intent"] == ib) & (preds_dv == ia))
        sub_idx = np.where(cond)[0]
        # Take up to 30 rows (deterministic order)
        for r_i in sub_idx[:30]:
            p_vec = probs_dv[r_i]
            top_p_idx = int(p_vec.argmax())
            err_rows.append({
                "pair_rank": rank,
                "pair_name": f"{ia} <-> {ib}",
                "dev_row_idx": int(r_i),
                "raw_text": dev.loc[r_i, "utterance"],
                "normalized_text": dev_norm[r_i],
                "silver_label": dev.loc[r_i, "intent"],
                "predicted_label": preds_dv[r_i],
                "prob_silver_label": round(float(p_vec[intent2idx[dev.loc[r_i, "intent"]]]), 4),
                "prob_predicted_label": round(float(p_vec[intent2idx[preds_dv[r_i]]]), 4),
                "prob_top_fused": round(float(p_vec[top_p_idx]), 4),
                "clause_rule_triggered": dev_infos[r_i]["type"],
            })
    err_df = pd.DataFrame(err_rows)
    err_df.to_csv(ERR_DIR / "errors_top_pairs.csv", index=False)
    print(f"Saved results/error_analysis/errors_top_pairs.csv with {len(err_df)} rows.")


if __name__ == "__main__":
    run()

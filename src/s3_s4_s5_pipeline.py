"""Steps S3, S4, S5, S6, S7:
- S3: Rule-based label repair (offline) + Snorkel-style cue coverage/overlap/conflict table,
      and online clause path for flagged messages.
- S4: Stage-1 semantic head (MLP: dim -> 256 GELU -> dropout 0.2 -> 12, AdamW lr=1e-3, wd=1e-4)
      and lexical head (char_wb 2-5 TF-IDF + class-weighted LR; plus comparison with word+char_wb).
      Relabelling noisy bucket (accept if both heads agree at >=0.9 avg prob, and rule agrees if flagged).
      Final semantic head with Sahayam et al. (2023) Algorithm 2 hard-batch schedule (E=20, delta=0.5, zeta=38)
      and dev-prior class weights w_c = clip(p_dev(c) / p_train(c), 0.5, 5.0).
      Logit fusion z = (1 - a) * z_semantic + a * z_lexical tuned on dev_cal_1850 by NLL.
- S5: Temperature scaling (T) and abstain gate (tau) tuned on dev_cal_1850.
- S7: Full evaluation across starter (fixed), stage-1, final semantic, lexical, fused, and clause-path variants
      on dev_eval_1850, audit_300, and full dev (4,000), plus slice breakdowns and latency measurements.
"""
import json
import re
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.optimize import minimize_scalar
from scipy.special import softmax, log_softmax
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

from src.s0_guard_and_splits import (
    DATA, RESULTS_V1, load_intents, normalize_text, is_non_alphanumeric, get_slices, evaluate_predictions
)

CACHE_DIR = RESULTS_V1 / "cache"

# Seed cue lists from Section 16.2 S3
CLAUSE_SPLITTERS = [
    "and also", "and by the way", "but before that", "also", "plus", "but", "aur", "lekin"
]
MIND_CHANGE_CUES = [
    "on second thought", "actually never mind", "actually forget it", "sorry, wrong thing",
    "ignore what i just said", "no wait, forget that", "scratch that", "never mind",
    "forget that", "forget it", "rehne do", "chhod do", "actually", "nvm"
]
CANCEL_CUES = ["cancel", "abort", "withdraw"]
REFUND_CUES = ["refund", "money back", "paisa wapas", "return my money"]
DELIVERED_CUES = ["delivered", "received", "arrived", "came", "got my order", "was cold", "inedible", "damaged"]
NOT_YET_CUES = ["before it is prepared", "not yet", "just placed", "5 minutes ago", "by mistake", "accidentally", "taking too long"]


class SemanticMLPHead(nn.Module):
    """S4.1 Semantic Head: dim -> 256 (GELU) -> Dropout(0.2) -> 12 outputs."""
    def __init__(self, in_dim: int, hidden_dim: int = 256, num_classes: int = 12, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        return self.net(x)


def train_mlp_plain(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    sample_weights: np.ndarray,
    class_weights: np.ndarray,
    num_classes: int = 12,
    epochs: int = 20,
    batch_size: int = 32,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    seed: int = 42,
):
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = SemanticMLPHead(X_tr.shape[1], 256, num_classes, dropout=0.2).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    cw_t = torch.tensor(class_weights, dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=cw_t, reduction="none")

    X_t = torch.tensor(X_tr, dtype=torch.float32, device=device)
    y_t = torch.tensor(y_tr, dtype=torch.long, device=device)
    w_t = torch.tensor(sample_weights, dtype=torch.float32, device=device)

    n = len(X_tr)
    rng = np.random.default_rng(seed)

    for ep in range(epochs):
        model.train()
        perm = rng.permutation(n)
        for i in range(0, n, batch_size):
            b_idx = perm[i : i + batch_size]
            optimizer.zero_grad()
            logits = model(X_t[b_idx])
            losses = criterion(logits, y_t[b_idx]) * w_t[b_idx]
            loss = losses.mean()
            loss.backward()
            optimizer.step()

    model.eval()
    return model


def train_mlp_sahayam_hard_batch(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    sample_weights: np.ndarray,
    class_weights: np.ndarray,
    num_classes: int = 12,
    E: int = 20,
    delta: float = 0.5,
    batch_size: int = 32,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    seed: int = 42,
):
    """Sahayam et al. (2023) Algorithm 2 Hard-Batch Schedule:
    - Fixed class-stratified mini-batches of 32.
    - Round 0 (Epoch 1): train on all N batches and store each batch's loss.
    - Then zeta = int((E - 1) / delta) rounds (zeta = 38 for E=20, delta=0.5),
      each sorting batches by stored loss descending and training on the top delta * N batches,
      updating their stored losses.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = SemanticMLPHead(X_tr.shape[1], 256, num_classes, dropout=0.2).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    cw_t = torch.tensor(class_weights, dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=cw_t, reduction="none")

    X_t = torch.tensor(X_tr, dtype=torch.float32, device=device)
    y_t = torch.tensor(y_tr, dtype=torch.long, device=device)
    w_t = torch.tensor(sample_weights, dtype=torch.float32, device=device)

    # Build fixed class-stratified mini-batches of 32
    rng = np.random.default_rng(seed)
    # Interleave indices by class so every batch of 32 is stratified
    class_buckets = [rng.permutation(np.where(y_tr == c)[0]).tolist() for c in range(num_classes)]
    stratified_order = []
    while any(class_buckets):
        for c in range(num_classes):
            if class_buckets[c]:
                stratified_order.append(class_buckets[c].pop())
    stratified_order = np.array(stratified_order)

    batches = [
        stratified_order[i : i + batch_size]
        for i in range(0, len(stratified_order), batch_size)
        if len(stratified_order[i : i + batch_size]) >= 8
    ]
    N = len(batches)
    batch_losses = np.zeros(N, dtype=np.float32)

    # Epoch 1: train on all N batches and record loss
    model.train()
    for b_id, b_idx in enumerate(batches):
        optimizer.zero_grad()
        logits = model(X_t[b_idx])
        losses = criterion(logits, y_t[b_idx]) * w_t[b_idx]
        loss = losses.mean()
        loss.backward()
        optimizer.step()
        batch_losses[b_id] = float(loss.detach().cpu().item())

    # zeta = (E - 1) / delta rounds
    zeta = int(round((E - 1) / delta))
    top_k = max(1, int(round(delta * N)))

    for r in range(zeta):
        top_batch_ids = np.argsort(-batch_losses)[:top_k]
        # Shuffle the selected hard batches so parameter updates don't cycle in deterministic order
        rng.shuffle(top_batch_ids)
        for b_id in top_batch_ids:
            b_idx = batches[b_id]
            optimizer.zero_grad()
            logits = model(X_t[b_idx])
            losses = criterion(logits, y_t[b_idx]) * w_t[b_idx]
            loss = losses.mean()
            loss.backward()
            optimizer.step()
            batch_losses[b_id] = float(loss.detach().cpu().item())

    model.eval()
    return model


def predict_mlp_logits(model: nn.Module, X: np.ndarray) -> np.ndarray:
    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        x_t = torch.tensor(X, dtype=torch.float32, device=device)
        return model(x_t).cpu().numpy()


def detect_cues_and_split_clauses(norm_text: str):
    """Detects mind-change, multi-request, and cancel+refund cues, and splits clauses."""
    # Check mind-change cues first (sorted longest first)
    mc_hit = None
    for cue in MIND_CHANGE_CUES:
        m = re.search(r'\b' + re.escape(cue) + r'\b', norm_text)
        if m:
            mc_hit = cue
            # Split around the mind-change cue
            parts = re.split(r'\b' + re.escape(cue) + r'\b', norm_text, maxsplit=1)
            clauses = [p.strip(" ,.!?;:") for p in parts if len(p.strip(" ,.!?;:")) >= 3]
            return {"type": "mind_change", "cue": cue, "clauses": clauses if len(clauses) >= 2 else [norm_text]}

    # Check multi-request splitters
    for cue in CLAUSE_SPLITTERS:
        m = re.search(r'\b' + re.escape(cue) + r'\b', norm_text)
        if m:
            parts = re.split(r'\b' + re.escape(cue) + r'\b', norm_text)
            clauses = [p.strip(" ,.!?;:") for p in parts if len(p.strip(" ,.!?;:")) >= 3]
            if len(clauses) >= 2:
                return {"type": "multi_request", "cue": cue, "clauses": clauses}

    # Check cancel + refund co-occurrence (Rule 3)
    has_cancel = any(re.search(r'\b' + re.escape(c) + r'\b', norm_text) for c in CANCEL_CUES)
    has_refund = any(re.search(r'\b' + re.escape(c) + r'\b', norm_text) for c in REFUND_CUES)
    if has_cancel and has_refund:
        return {"type": "cancel_refund", "cue": "cancel+refund", "clauses": [norm_text]}

    return {"type": "none", "cue": None, "clauses": [norm_text]}


def apply_clause_rules(info: dict, clause_preds: list[str], norm_text: str) -> str:
    """Applies Rules 1, 2, 3, and 5 to clause predictions."""
    ctype = info["type"]
    if ctype == "cancel_refund":
        # Rule 3: delivered -> refund_request; not yet delivered -> cancel_order
        if any(c in norm_text for c in DELIVERED_CUES):
            return "refund_request"
        if any(c in norm_text for c in NOT_YET_CUES):
            return "cancel_order"
        return clause_preds[0]

    # Filter out out_of_scope greetings/filler clauses if another real intent is present
    real_preds = [p for p in clause_preds if p != "out_of_scope"]
    if not real_preds:
        return clause_preds[-1]

    # Rule 5: talk_to_agent loses to any other substantive request
    non_agent_preds = [p for p in real_preds if p != "talk_to_agent"]

    if ctype == "mind_change":
        # Rule 2: last request wins on a change of mind
        # If the last clause is a substantive request (or talk_to_agent when customer changed mind TO talk_to_agent),
        # wait: if customer says "otp not coming ignore what i said, need to speak to a person", last clause is talk_to_agent!
        return real_preds[-1]

    if ctype == "multi_request":
        # Check special phrase "but before that" -> the clause AFTER "but before that" is actually the first action!
        if info["cue"] == "but before that" and len(real_preds) >= 2:
            cand = [p for p in real_preds[1:] if p != "talk_to_agent"]
            return cand[0] if cand else real_preds[-1]
        # Rule 1 & Rule 5: first non-agent request wins
        if non_agent_preds:
            return non_agent_preds[0]
        return real_preds[0]

    return clause_preds[0]


def compute_snorkel_cue_stats(norm_texts: list[str], silver_labels: list[str], rule_labels: list[str], infos: list[dict]):
    """Computes Chandrakala et al. Snorkel-style coverage, overlap, and conflict rates per cue."""
    n = len(norm_texts)
    cue_groups = {}
    for i, inf in enumerate(infos):
        if inf["type"] != "none":
            key = f"{inf['type']}::{inf['cue']}"
            cue_groups.setdefault(key, []).append(i)

    # Check how many messages match >1 cue across all cues
    all_cues = [( "mind_change", c) for c in MIND_CHANGE_CUES] + [("multi_request", c) for c in CLAUSE_SPLITTERS]
    match_counts = np.zeros(n, dtype=int)
    cue_matches = {}
    for ctype, c in all_cues:
        pat = re.compile(r'\b' + re.escape(c) + r'\b')
        hits = [i for i, s in enumerate(norm_texts) if pat.search(s)]
        cue_matches[(ctype, c)] = hits
        for i in hits:
            match_counts[i] += 1

    rows = []
    for (ctype, c), hits in cue_matches.items():
        if not hits:
            continue
        cov = len(hits) / n
        ovl = sum(1 for i in hits if match_counts[i] > 1) / n
        conflicts = sum(1 for i in hits if rule_labels[i] != silver_labels[i])
        conf_rate = conflicts / len(hits)
        rows.append({
            "cue_category": ctype,
            "cue_pattern": c,
            "matched_rows": len(hits),
            "coverage_share": round(cov, 5),
            "overlap_share": round(ovl, 5),
            "silver_conflict_count": conflicts,
            "silver_conflict_rate": round(conf_rate, 4),
        })
    return pd.DataFrame(rows).sort_values(by="matched_rows", ascending=False)


def fit_temperature(logits: np.ndarray, y_true_idx: np.ndarray) -> float:
    """Fits scalar temperature T > 0 on calibration logits by minimizing NLL."""
    def nll_fn(T):
        if T <= 0.05:
            return 1e9
        scaled = logits / T
        log_probs = log_softmax(scaled, axis=1)
        return -float(log_probs[np.arange(len(y_true_idx)), y_true_idx].mean())

    res = minimize_scalar(nll_fn, bounds=(0.1, 5.0), method="bounded")
    return round(float(res.x), 4)


def run():
    intents = load_intents()
    intent2idx = {name: i for i, name in enumerate(intents)}
    idx2intent = {i: name for i, name in enumerate(intents)}
    oos_idx = intent2idx["out_of_scope"]

    with open(RESULTS_V1 / "s1_chosen_encoder.json") as f:
        chosen_info = json.load(f)
    chosen = chosen_info["chosen_encoder"]
    hf_name = "intfloat/multilingual-e5-small" if chosen == "multilingual-e5-small" else "sentence-transformers/all-MiniLM-L6-v2"
    prefix = "query: " if chosen == "multilingual-e5-small" else ""

    print(f"Running S3-S7 Pipeline with encoder={chosen} ({hf_name})...")
    X_tr = np.load(CACHE_DIR / f"{chosen}_train.npy")
    X_dv = np.load(CACHE_DIR / f"{chosen}_dev.npy")
    X_te = np.load(CACHE_DIR / f"{chosen}_test.npy")

    train = pd.read_csv(DATA / "train.csv")
    dev = pd.read_csv(RESULTS_V1 / "dev_splits.csv")
    test = pd.read_csv(DATA / "test.csv")
    part_df = pd.read_csv(RESULTS_V1 / "s2_train_partition.csv")

    train_norm = [normalize_text(t) for t in train["utterance"]]
    dev_norm = [normalize_text(t) for t in dev["utterance"]]
    test_norm = [normalize_text(t) for t in test["utterance"]]

    y_tr_silver = np.array([intent2idx[k] for k in train["intent"]])
    y_dv_silver = np.array([intent2idx[k] for k in dev["intent"]])

    buckets = part_df["bucket"].values
    easy_hard_mask = (buckets == "easy") | (buckets == "hard")
    noisy_mask = (buckets == "noisy")

    # 1. Train initial Stage-1 model on easy + hard to bootstrap S3 clause labeling
    print("\n--- Step S4.2a: Initial Stage-1 Model on Easy + Hard (for S3 Rule Repair) ---")
    eh_indices = np.where(easy_hard_mask)[0]
    counts_eh = np.bincount(y_tr_silver[eh_indices], minlength=len(intents)).astype(np.float32)
    cw_inv = (len(eh_indices) / (len(intents) * np.maximum(counts_eh, 1.0))).astype(np.float32)
    cw_inv = np.clip(cw_inv, 0.5, 5.0)

    init_mlp = train_mlp_plain(
        X_tr[eh_indices],
        y_tr_silver[eh_indices],
        sample_weights=np.ones(len(eh_indices), dtype=np.float32),
        class_weights=cw_inv,
        num_classes=len(intents),
        epochs=20,
        seed=42,
    )

    # Also fit Lexical Head on Easy + Hard (both spec's char_wb 2-5 and word+char_wb for comparison)
    print("Fitting Lexical TF-IDF (char_wb 2-5) and (word 1-2 + char_wb 2-5)...")
    char_vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=3, sublinear_tf=True)
    X_tr_char = char_vec.fit_transform(train_norm)
    X_dv_char = char_vec.transform(dev_norm)
    X_te_char = char_vec.transform(test_norm)

    word_vec = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    X_tr_word = word_vec.fit_transform(train_norm)
    X_dv_word = word_vec.transform(dev_norm)
    X_te_word = word_vec.transform(test_norm)

    X_tr_wc = hstack([X_tr_word, X_tr_char]).tocsr()
    X_dv_wc = hstack([X_dv_word, X_dv_char]).tocsr()
    X_te_wc = hstack([X_te_word, X_te_char]).tocsr()

    # 2. Step S3: Offline Rule-Based Label Repair using lexical+semantic clause classifier
    print("\n--- Step S3: Rule-Based Label Repair (Offline) ---")
    # For fast, accurate clause classification on 30k train + 4k dev, we use our fast word+char lexical classifier
    # AND the frozen sentence encoder on all unique clauses!
    from sentence_transformers import SentenceTransformer
    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    st_model = SentenceTransformer(hf_name, device=device_str)

    train_infos = [detect_cues_and_split_clauses(s) for s in train_norm]
    dev_infos = [detect_cues_and_split_clauses(s) for s in dev_norm]
    test_infos = [detect_cues_and_split_clauses(s) for s in test_norm]

    # Collect all unique clauses from flagged rows in train, dev, test
    unique_clauses = set()
    for inf_list in (train_infos, dev_infos, test_infos):
        for inf in inf_list:
            if inf["type"] != "none":
                for cl in inf["clauses"]:
                    unique_clauses.add(cl)
    unique_clauses = sorted(unique_clauses)
    print(f"Encoding {len(unique_clauses)} unique clauses from flagged multi-request/mind-change rows...")
    clause_embs = st_model.encode(
        [prefix + c for c in unique_clauses],
        batch_size=256,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    clause_logits_init = predict_mlp_logits(init_mlp, clause_embs)
    clause2pred_init = {
        cl: idx2intent[int(clause_logits_init[i].argmax())]
        for i, cl in enumerate(unique_clauses)
    }

    # Compute S3 repaired labels on train
    y_tr_repaired = y_tr_silver.copy()
    repaired_mask = np.zeros(len(train), dtype=bool)
    rule_labels_str = []

    for i, inf in enumerate(train_infos):
        if inf["type"] == "none":
            rule_labels_str.append(train.loc[i, "intent"])
            continue
        c_preds = [clause2pred_init[cl] for cl in inf["clauses"]]
        r_lab = apply_clause_rules(inf, c_preds, train_norm[i])
        rule_labels_str.append(r_lab)
        if r_lab != train.loc[i, "intent"]:
            y_tr_repaired[i] = intent2idx[r_lab]
            repaired_mask[i] = True

    snorkel_df = compute_snorkel_cue_stats(train_norm, train["intent"].tolist(), rule_labels_str, train_infos)
    snorkel_df.to_csv(RESULTS_V1 / "s3_cue_snorkel_stats.csv", index=False)
    print(f"S3 Rule Repair: {repaired_mask.sum()} train labels repaired out of {sum(1 for x in train_infos if x['type']!='none')} flagged rows.")

    # 3. Step S4.2: Train Stage-1 Semantic Head & Stage-1 Lexical Head on (Easy + Hard + Rule-Repaired)
    stage1_mask = easy_hard_mask | repaired_mask
    s1_idx = np.where(stage1_mask)[0]
    print(f"\n--- Step S4.2: Training Stage-1 Heads on {len(s1_idx)} rows (Easy + Hard + Rule-Repaired) ---")

    # Compute class weights w_c = clip(p_dev(c) / p_train(c), 0.5, 5.0)
    # Using dev_cal_1850 priors so dev_eval_1850 is strictly untouched!
    cal_mask = (dev["split_role"] == "dev_cal_1850").values
    eval_mask = (dev["split_role"] == "dev_eval_1850").values
    audit_mask = (dev["split_role"] == "audit_300").values

    p_train = np.bincount(y_tr_repaired[s1_idx], minlength=len(intents)) / len(s1_idx)
    p_dev_cal = np.bincount(y_dv_silver[cal_mask], minlength=len(intents)) / cal_mask.sum()
    w_c_prior = np.clip(p_dev_cal / np.maximum(p_train, 1e-5), 0.5, 5.0).astype(np.float32)

    stage1_mlp = train_mlp_plain(
        X_tr[s1_idx],
        y_tr_repaired[s1_idx],
        sample_weights=np.ones(len(s1_idx), dtype=np.float32),
        class_weights=w_c_prior,
        num_classes=len(intents),
        epochs=20,
        seed=42,
    )

    # Fit Stage-1 Lexical Heads:
    # (A) Strict spec: char_wb (2, 5) only
    cw_dict = {i: float(w_c_prior[i]) for i in range(len(intents))}
    lex_char_clf = LogisticRegression(max_iter=1000, C=2.0, class_weight=cw_dict, random_state=42)
    lex_char_clf.fit(X_tr_char[s1_idx], y_tr_repaired[s1_idx])

    # (B) Enhanced lexical: word (1, 2) + char_wb (2, 5)
    lex_wc_clf = LogisticRegression(max_iter=1000, C=2.0, class_weight=cw_dict, random_state=42)
    lex_wc_clf.fit(X_tr_wc[s1_idx], y_tr_repaired[s1_idx])

    # 4. Step S4.3: Relabel Noisy Bucket
    print("\n--- Step S4.3: Relabelling Noisy Bucket ---")
    noisy_indices = np.where(noisy_mask & (~repaired_mask))[0]
    sem_noisy_probs = softmax(predict_mlp_logits(stage1_mlp, X_tr[noisy_indices]), axis=1)
    lex_noisy_probs = lex_char_clf.predict_proba(X_tr_char[noisy_indices])

    avg_noisy_probs = 0.5 * (sem_noisy_probs + lex_noisy_probs)
    sem_preds = sem_noisy_probs.argmax(axis=1)
    lex_preds = lex_noisy_probs.argmax(axis=1)
    avg_max_prob = avg_noisy_probs.max(axis=1)
    avg_pred = avg_noisy_probs.argmax(axis=1)

    accepted_noisy_idx = []
    accepted_noisy_labels = []
    accepted_noisy_weights = []

    for k_i, row_i in enumerate(noisy_indices):
        if sem_preds[k_i] == lex_preds[k_i] and avg_max_prob[k_i] >= 0.90:
            pred_str = idx2intent[int(avg_pred[k_i])]
            if train_infos[row_i]["type"] != "none" and rule_labels_str[row_i] != pred_str:
                continue
            accepted_noisy_idx.append(row_i)
            accepted_noisy_labels.append(int(avg_pred[k_i]))
            accepted_noisy_weights.append(float(avg_max_prob[k_i]))

    print(f"Noisy bucket relabelling (at >=0.90 threshold): Accepted {len(accepted_noisy_idx)} / {len(noisy_indices)} noisy rows ({len(accepted_noisy_idx)/max(len(noisy_indices),1):.2%}).")

    # Also check how many would be accepted at 0.75 and 0.80 (to quantify the threshold bottleneck!)
    acc_80 = sum(1 for k_i in range(len(noisy_indices)) if sem_preds[k_i] == lex_preds[k_i] and avg_max_prob[k_i] >= 0.80)
    acc_70 = sum(1 for k_i in range(len(noisy_indices)) if sem_preds[k_i] == lex_preds[k_i] and avg_max_prob[k_i] >= 0.70)
    print(f"  [Bottleneck check] Accepted at >=0.80: {acc_80} rows; at >=0.70: {acc_70} rows.")

    # 5. Step S4.4: Final Semantic Head Training (Plain vs Sahayam Hard-Batch Schedule)
    final_idx = np.concatenate([s1_idx, np.array(accepted_noisy_idx, dtype=int)])
    final_y = np.concatenate([y_tr_repaired[s1_idx], np.array(accepted_noisy_labels, dtype=int)])
    final_w = np.concatenate([np.ones(len(s1_idx), dtype=np.float32), np.array(accepted_noisy_weights, dtype=np.float32)])

    p_train_fin = np.bincount(final_y, minlength=len(intents)) / len(final_y)
    w_c_fin = np.clip(p_dev_cal / np.maximum(p_train_fin, 1e-5), 0.5, 5.0).astype(np.float32)

    print(f"\n--- Step S4.4: Training Final Semantic Head on {len(final_idx)} rows (Sahayam Hard-Batch E=20, delta=0.5) ---")
    final_mlp_sahayam = train_mlp_sahayam_hard_batch(
        X_tr[final_idx],
        final_y,
        sample_weights=final_w,
        class_weights=w_c_fin,
        num_classes=len(intents),
        E=20,
        delta=0.5,
        batch_size=32,
        seed=42,
    )

    # Also train Final Semantic Head with Plain schedule on the exact same data to isolate Sahayam hard-batch effect!
    final_mlp_plain = train_mlp_plain(
        X_tr[final_idx],
        final_y,
        sample_weights=final_w,
        class_weights=w_c_fin,
        num_classes=len(intents),
        epochs=20,
        batch_size=32,
        seed=42,
    )

    # Refit Final Lexical Heads on final_idx
    cw_fin_dict = {i: float(w_c_fin[i]) for i in range(len(intents))}
    lex_char_final = LogisticRegression(max_iter=1000, C=2.0, class_weight=cw_fin_dict, random_state=42)
    lex_char_final.fit(X_tr_char[final_idx], final_y, sample_weight=final_w)

    lex_wc_final = LogisticRegression(max_iter=1000, C=2.0, class_weight=cw_fin_dict, random_state=42)
    lex_wc_final.fit(X_tr_wc[final_idx], final_y, sample_weight=final_w)

    # 6. Step S4.5: Logit Fusion on dev_cal_1850
    print("\n--- Step S4.5 & S5: Logit Fusion, Temperature Scaling & Abstain Gate on dev_cal_1850 ---")
    z_sem_dv = predict_mlp_logits(final_mlp_sahayam, X_dv)
    z_lex_char_dv = lex_char_final.decision_function(X_dv_char)
    z_lex_wc_dv = lex_wc_final.decision_function(X_dv_wc)

    # Search alpha in 0.0 .. 1.0 (step 0.1) by NLL on dev_cal_1850
    best_a_char, best_nll_char = 0.0, 1e9
    for a in np.linspace(0.0, 1.0, 11):
        z_fuse = (1.0 - a) * z_sem_dv[cal_mask] + a * z_lex_char_dv[cal_mask]
        T_a = fit_temperature(z_fuse, y_dv_silver[cal_mask])
        probs_a = softmax(z_fuse / T_a, axis=1)
        nll = log_loss(y_dv_silver[cal_mask], probs_a, labels=list(range(len(intents))))
        if nll < best_nll_char:
            best_nll_char = nll
            best_a_char = round(float(a), 2)

    best_a_wc, best_nll_wc = 0.0, 1e9
    for a in np.linspace(0.0, 1.0, 11):
        z_fuse = (1.0 - a) * z_sem_dv[cal_mask] + a * z_lex_wc_dv[cal_mask]
        T_a = fit_temperature(z_fuse, y_dv_silver[cal_mask])
        probs_a = softmax(z_fuse / T_a, axis=1)
        nll = log_loss(y_dv_silver[cal_mask], probs_a, labels=list(range(len(intents))))
        if nll < best_nll_wc:
            best_nll_wc = nll
            best_a_wc = round(float(a), 2)

    print(f"Optimal fusion alpha (char_wb only): a={best_a_char} (cal NLL={best_nll_char:.4f})")
    print(f"Optimal fusion alpha (word+char_wb): a={best_a_wc} (cal NLL={best_nll_wc:.4f})")

    # Calibrate Temperature T and Abstain Gate tau on dev_cal_1850 for both char_wb fusion and word+char_wb fusion
    z_fused_char_dv = (1.0 - best_a_char) * z_sem_dv + best_a_char * z_lex_char_dv
    T_char = fit_temperature(z_fused_char_dv[cal_mask], y_dv_silver[cal_mask])
    probs_fused_char_dv = softmax(z_fused_char_dv / T_char, axis=1)

    z_fused_wc_dv = (1.0 - best_a_wc) * z_sem_dv + best_a_wc * z_lex_wc_dv
    T_wc = fit_temperature(z_fused_wc_dv[cal_mask], y_dv_silver[cal_mask])
    probs_fused_wc_dv = softmax(z_fused_wc_dv / T_wc, axis=1)

    def optimize_tau(probs_cal, y_cal_str):
        best_tau, best_f1 = 0.30, -1.0
        tau_curve = []
        base_preds = [idx2intent[i] for i in probs_cal.argmax(axis=1)]
        max_p = probs_cal.max(axis=1)
        for tau in np.round(np.arange(0.30, 0.91, 0.05), 2):
            gated = ["out_of_scope" if mp < tau else bp for bp, mp in zip(base_preds, max_p)]
            f1 = evaluate_predictions(y_cal_str, gated, intents=intents, verify_supports=False)["macro_f1"]
            tau_curve.append({"tau": float(tau), "cal_macro_f1": f1})
            if f1 > best_f1:
                best_f1 = f1
                best_tau = float(tau)
        return best_tau, best_f1, tau_curve

    tau_char, cal_f1_char, tau_curve_char = optimize_tau(probs_fused_char_dv[cal_mask], dev.loc[cal_mask, "intent"])
    tau_wc, cal_f1_wc, tau_curve_wc = optimize_tau(probs_fused_wc_dv[cal_mask], dev.loc[cal_mask, "intent"])
    print(f"Calibrated Gate (char_wb fusion): T={T_char}, best_tau={tau_char} (cal Macro-F1={cal_f1_char:.4f})")
    print(f"Calibrated Gate (word+char fusion): T={T_wc}, best_tau={tau_wc} (cal Macro-F1={cal_f1_wc:.4f})")

    # 7. Online Clause Path on dev for flagged messages
    # Update clause predictions using final fused model
    clause_z_sem = predict_mlp_logits(final_mlp_sahayam, clause_embs)
    clause_z_wc = lex_wc_final.decision_function(hstack([word_vec.transform(unique_clauses), char_vec.transform(unique_clauses)]))
    clause_z_char = lex_char_final.decision_function(char_vec.transform(unique_clauses))

    clause2pred_char = {
        cl: idx2intent[int(((1.0 - best_a_char) * clause_z_sem[i] + best_a_char * clause_z_char[i]).argmax())]
        for i, cl in enumerate(unique_clauses)
    }
    clause2pred_wc = {
        cl: idx2intent[int(((1.0 - best_a_wc) * clause_z_sem[i] + best_a_wc * clause_z_wc[i]).argmax())]
        for i, cl in enumerate(unique_clauses)
    }

    def get_final_preds(probs_dv, tau_val, clause_map, use_clause_path=False):
        base_p = [idx2intent[i] for i in probs_dv.argmax(axis=1)]
        max_p = probs_dv.max(axis=1)
        out = []
        for i, (bp, mp) in enumerate(zip(base_p, max_p)):
            # S0 Guard: non-alphanumeric -> out_of_scope
            if is_non_alphanumeric(dev_norm[i]):
                out.append("out_of_scope")
                continue
            if use_clause_path and dev_infos[i]["type"] != "none":
                c_preds = [clause_map[cl] for cl in dev_infos[i]["clauses"]]
                out.append(apply_clause_rules(dev_infos[i], c_preds, dev_norm[i]))
                continue
            out.append("out_of_scope" if mp < tau_val else bp)
        return np.array(out)

    # Also train an uncleaned baseline on ALL 30,000 raw silver rows to prove whether S2+S3+S4 helped or hurt!
    print("\n--- Training Ablation Baselines on Raw 30,000 Silver Rows (No Partitioning) ---")
    p_tr_raw = np.bincount(y_tr_silver, minlength=len(intents)) / len(y_tr_silver)
    w_c_raw = np.clip(p_dev_cal / np.maximum(p_tr_raw, 1e-5), 0.5, 5.0).astype(np.float32)

    raw_mlp_unweighted = train_mlp_plain(
        X_tr, y_tr_silver, np.ones(len(X_tr), dtype=np.float32), np.ones(len(intents), dtype=np.float32),
        num_classes=len(intents), epochs=20, seed=42
    )
    raw_mlp_prior_weighted = train_mlp_plain(
        X_tr, y_tr_silver, np.ones(len(X_tr), dtype=np.float32), w_c_raw,
        num_classes=len(intents), epochs=20, seed=42
    )
    raw_lex_wc = LogisticRegression(max_iter=1000, C=2.0, class_weight={i: float(w_c_raw[i]) for i in range(len(intents))}, random_state=42)
    raw_lex_wc.fit(X_tr_wc, y_tr_silver)

    # Collect comprehensive evaluation table on dev_eval_1850 and full dev_4000
    from baseline.baseline import TaxonomyMatcher
    starter_preds, starter_scores = TaxonomyMatcher().predict(dev["utterance"].tolist())
    starter_preds = np.array(starter_preds)

    models_to_eval = [
        ("0_Starter_Baseline_FixedEval", starter_preds, starter_scores),
        ("1_Raw_Silver_Semantic_MLP_Unweighted", np.array([idx2intent[i] for i in predict_mlp_logits(raw_mlp_unweighted, X_dv).argmax(1)]), softmax(predict_mlp_logits(raw_mlp_unweighted, X_dv), axis=1)),
        ("2_Raw_Silver_Semantic_MLP_DevPriorWeighted", np.array([idx2intent[i] for i in predict_mlp_logits(raw_mlp_prior_weighted, X_dv).argmax(1)]), softmax(predict_mlp_logits(raw_mlp_prior_weighted, X_dv), axis=1)),
        ("3_Stage1_Semantic_MLP_EasyHardRepaired", np.array([idx2intent[i] for i in predict_mlp_logits(stage1_mlp, X_dv).argmax(1)]), softmax(predict_mlp_logits(stage1_mlp, X_dv), axis=1)),
        ("4_Final_Semantic_MLP_PlainSchedule", np.array([idx2intent[i] for i in predict_mlp_logits(final_mlp_plain, X_dv).argmax(1)]), softmax(predict_mlp_logits(final_mlp_plain, X_dv), axis=1)),
        ("5_Final_Semantic_MLP_SahayamHardBatch", np.array([idx2intent[i] for i in z_sem_dv.argmax(1)]), softmax(z_sem_dv, axis=1)),
        ("6_Lexical_Head_CharWB_Only", np.array([idx2intent[i] for i in z_lex_char_dv.argmax(1)]), softmax(z_lex_char_dv, axis=1)),
        ("7_Lexical_Head_Word_Plus_CharWB", np.array([idx2intent[i] for i in z_lex_wc_dv.argmax(1)]), softmax(z_lex_wc_dv, axis=1)),
        ("8_SystemV1_Fused_CharWB_CalibratedGate", get_final_preds(probs_fused_char_dv, tau_char, clause2pred_char, use_clause_path=False), probs_fused_char_dv),
        ("9_SystemV1_Fused_CharWB_WithClausePath", get_final_preds(probs_fused_char_dv, tau_char, clause2pred_char, use_clause_path=True), probs_fused_char_dv),
        ("10_SystemV1_Enhanced_WordChar_CalibratedGate", get_final_preds(probs_fused_wc_dv, tau_wc, clause2pred_wc, use_clause_path=False), probs_fused_wc_dv),
        ("11_SystemV1_Enhanced_WordChar_WithClausePath", get_final_preds(probs_fused_wc_dv, tau_wc, clause2pred_wc, use_clause_path=True), probs_fused_wc_dv),
    ]

    ablation_rows = []
    per_class_comparison = []

    for m_name, m_preds, m_probs in models_to_eval:
        ev_1850 = evaluate_predictions(dev.loc[eval_mask, "intent"], m_preds[eval_mask], probs=m_probs[eval_mask], intents=intents)
        ev_full = evaluate_predictions(dev["intent"], m_preds, probs=m_probs, intents=intents)
        ev_audit = evaluate_predictions(dev.loc[audit_mask, "intent"], m_preds[audit_mask], probs=m_probs[audit_mask], intents=intents)

        ablation_rows.append({
            "model_variant": m_name,
            "dev_eval1850_macro_f1": ev_1850["macro_f1"],
            "dev_eval1850_top1": ev_1850["top1_accuracy"],
            "dev_eval1850_top3": ev_1850["top3_accuracy"],
            "dev_eval1850_mrr": ev_1850["mrr"],
            "dev_full4000_macro_f1": ev_full["macro_f1"],
            "dev_full4000_top1": ev_full["top1_accuracy"],
            "audit300_silver_macro_f1": ev_audit["macro_f1"],
            "account_help_f1_eval1850": ev_1850["per_class"]["account_help"]["f1"],
            "promo_code_f1_eval1850": ev_1850["per_class"]["promo_code_issue"]["f1"],
        })

        for k in intents:
            per_class_comparison.append({
                "model_variant": m_name,
                "intent": k,
                "precision": ev_1850["per_class"][k]["precision"],
                "recall": ev_1850["per_class"][k]["recall"],
                "f1": ev_1850["per_class"][k]["f1"],
                "support": ev_1850["per_class"][k]["support"],
            })

    ablation_df = pd.DataFrame(ablation_rows)
    ablation_df.to_csv(RESULTS_V1 / "s7_ablation_benchmarks.csv", index=False)
    pd.DataFrame(per_class_comparison).to_csv(RESULTS_V1 / "s7_per_class_benchmarks.csv", index=False)

    print("\n=== S7 Ablation Benchmarks (on held-out dev_eval_1850 & dev_full_4000) ===")
    print(ablation_df[["model_variant", "dev_eval1850_macro_f1", "dev_eval1850_top1", "account_help_f1_eval1850", "promo_code_f1_eval1850"]].to_string(index=False))

    # Slice breakdown on dev_eval_1850 for Starter vs Stage-1 vs SystemV1 (CharWB) vs SystemV1 (Word+CharWB) vs ClausePath
    dev_slices = get_slices(dev["utterance"], pd.Series(dev_norm))
    slice_rows = []
    key_models = [models_to_eval[0], models_to_eval[3], models_to_eval[5], models_to_eval[8], models_to_eval[9], models_to_eval[10], models_to_eval[11]]
    for s_col in ["is_hinglish", "is_long", "is_multi_request", "is_mind_change", "is_emoji"]:
        s_mask = eval_mask & dev_slices[s_col].values
        for m_name, m_preds, m_probs in key_models:
            ev_s = evaluate_predictions(dev.loc[s_mask, "intent"], m_preds[s_mask], probs=m_probs[s_mask], intents=intents, verify_supports=False)
            slice_rows.append({
                "slice": s_col,
                "n_eval1850": int(s_mask.sum()),
                "model_variant": m_name,
                "macro_f1": ev_s["macro_f1"],
                "top1_accuracy": ev_s["top1_accuracy"],
            })
    slice_df = pd.DataFrame(slice_rows)
    slice_df.to_csv(RESULTS_V1 / "s7_slice_breakdown.csv", index=False)
    print("\n=== S7 Slice Breakdown (on dev_eval_1850) ===")
    print(slice_df.pivot(index="model_variant", columns="slice", values="macro_f1").to_string())

    # Save calibration & fusion metadata
    meta = {
        "chosen_encoder": chosen,
        "accepted_noisy_count_90": len(accepted_noisy_idx),
        "accepted_noisy_count_80": acc_80,
        "accepted_noisy_count_70": acc_70,
        "total_noisy_bucket": int(len(noisy_indices)),
        "repaired_train_rows": int(repaired_mask.sum()),
        "fusion_alpha_char_wb": best_a_char,
        "fusion_alpha_word_char": best_a_wc,
        "temperature_char_wb": T_char,
        "temperature_word_char": T_wc,
        "gate_tau_char_wb": tau_char,
        "gate_tau_word_char": tau_wc,
    }
    with open(RESULTS_V1 / "s4_s5_pipeline_meta.json", "w") as f:
        json.dump(meta, f, indent=2)


if __name__ == "__main__":
    run()

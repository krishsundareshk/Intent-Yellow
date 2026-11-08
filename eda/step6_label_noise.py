"""Step 6: Detect likely label noise
1. Train TF-IDF + Logistic Regression (5-fold, seed 42) on train. Get out-of-fold probabilities.
2. Flag rows with predicted prob >= 0.8 when pred != assigned, or prob(assigned) < 0.2.
   Produce label_quality_queue.csv.
3. Compute out-of-fold cross-validated error rates with Wilson 95% CI.
   Produce label_noise_estimate.csv, noise_transition_matrix.csv.
"""
from pathlib import Path
import json
import math
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RESULTS_EDA = ROOT / "results" / "eda"

def wilson_ci(k, n, confidence=0.95):
    if n == 0:
        return 0.0, 0.0, 0.0
    z = 1.95996  # for 95% CI
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    margin = (z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))) / denom
    return round(p, 4), round(max(0.0, center - margin), 4), round(min(1.0, center + margin), 4)

def run():
    print("Running Step 6: Empirical Label Noise Detection...")
    train = pd.read_csv(DATA / "train.csv")
    train["clean"] = train["utterance"].apply(lambda s: " ".join(str(s).lower().split()))
    
    # 5-fold cross validation with TF-IDF + LogisticRegression
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    labels = sorted(train["intent"].unique())
    label2idx = {l: i for i, l in enumerate(labels)}
    idx2label = {i: l for i, l in enumerate(labels)}
    y = np.array([label2idx[l] for l in train["intent"]])
    
    oof_probs = np.zeros((len(train), len(labels)))

    print("Fitting 5-fold out-of-fold LogisticRegression...")
    for fold, (train_idx, val_idx) in enumerate(skf.split(train["clean"], y)):
        vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
        X_tr = vec.fit_transform(train["clean"].iloc[train_idx])
        X_val = vec.transform(train["clean"].iloc[val_idx])
        clf = LogisticRegression(max_iter=1000, C=3.0, random_state=42)
        clf.fit(X_tr, y[train_idx])
        oof_probs[val_idx] = clf.predict_proba(X_val)

    pred_idx = oof_probs.argmax(axis=1)
    pred_labels = [idx2label[i] for i in pred_idx]
    max_probs = oof_probs.max(axis=1)
    assigned_probs = np.array([oof_probs[i, y[i]] for i in range(len(train))])

    # Threshold criteria:
    # (pred != assigned and max_prob >= 0.8) OR (assigned_prob < 0.2)
    is_mismatch = (np.array(pred_labels) != np.array(train["intent"]))
    is_high_conf = (max_probs >= 0.8)
    is_low_assigned = (assigned_probs < 0.2)
    flag_risk = (is_mismatch & is_high_conf) | is_low_assigned

    queue_df = pd.DataFrame({
        "train_idx": range(len(train)),
        "utterance": train["utterance"],
        "assigned_label": train["intent"],
        "predicted_label": pred_labels,
        "max_prob": np.round(max_probs, 4),
        "assigned_prob": np.round(assigned_probs, 4),
        "confidence_gap": np.round(max_probs - assigned_probs, 4),
        "flag_risk": flag_risk
    })
    
    # Sort queue by confidence gap descending
    queue_ranked = queue_df[queue_df["flag_risk"]].sort_values(by="confidence_gap", ascending=False)
    queue_ranked.to_csv(RESULTS_EDA / "label_quality_queue.csv", index=False)
    print(f"Saved label_quality_queue.csv ({len(queue_ranked)} high-risk rows flagged)")

    # Pure empirical OOF Disagreement Estimation with Wilson CI
    total_samples = len(train)
    total_mismatches = int(is_mismatch.sum())
    p_overall, low_overall, high_overall = wilson_ci(total_mismatches, total_samples)

    est_rows = [{
        "slice": "overall_train_oof_cv",
        "sample_size": total_samples,
        "error_count": total_mismatches,
        "error_rate": p_overall,
        "ci_95_low": low_overall,
        "ci_95_high": high_overall,
        "method": "5-fold Stratified OOF Logistic Regression"
    }]

    for lab in labels:
        mask = (train["intent"] == lab)
        n_l = int(mask.sum())
        e_l = int((is_mismatch & mask).sum())
        p_l, low_l, high_l = wilson_ci(e_l, n_l)
        est_rows.append({
            "slice": f"label_{lab}",
            "sample_size": n_l,
            "error_count": e_l,
            "error_rate": p_l,
            "ci_95_low": low_l,
            "ci_95_high": high_l,
            "method": "5-fold Stratified OOF Logistic Regression"
        })
    pd.DataFrame(est_rows).to_csv(RESULTS_EDA / "label_noise_estimate.csv", index=False)
    print("Saved label_noise_estimate.csv")

    # Noise transition matrix based strictly on empirical OOF predictions
    matrix_counts = {l1: {l2: 0 for l2 in labels} for l1 in labels}
    for assigned, pred in zip(train["intent"], pred_labels):
        matrix_counts[assigned][pred] += 1

    trans_matrix = pd.DataFrame(matrix_counts).T
    trans_matrix.index.name = "assigned_label"
    trans_matrix.to_csv(RESULTS_EDA / "noise_transition_matrix.csv")
    print("Saved noise_transition_matrix.csv")

if __name__ == "__main__":
    run()

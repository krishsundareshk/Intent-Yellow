"""Step S2: Partition into easy / hard / noisy (Forouzesh and Thiran 2023)
1. Partition head: MLP on frozen semantic embeddings (dim -> 256 -> 12), class-weighted CE,
   SGD lr=0.01, momentum=0.9, 30 epochs on all 30,000 train rows.
2. Record per-epoch correctness: Acc_i = (1/T) * sum_t 1[pred_i(t) == y_i].
3. Find t* = first epoch where overall training accuracy >= 50% (t*=1 if reached in epoch 1).
   Save 256-d hidden feature h(x_i) at epoch t*.
4. Static centroid o_c = mean of h(x_i) over all rows assigned to silver class c.
   SCD_i = ||h(x_i) - o_{y_i}||_2.
5. Fit 3-component GaussianMixture on (Acc_i, SCD_z) per class.
   Score components by mean(Acc) - mean(SCD_z): highest=easy, middle=hard, lowest=noisy.
   Cap noisy at <= 50% of any class (excess moves to hard).
6. Sanity test: flip a random 10% of labels in a copy, rerun S2, report recall of flipped rows in noisy bucket!
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.mixture import GaussianMixture

from src.s0_guard_and_splits import DATA, RESULTS_V1, load_intents

CACHE_DIR = RESULTS_V1 / "cache"


class PartitionMLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 256, num_classes: int = 12):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x, return_hidden=False):
        h = self.act(self.fc1(x))
        logits = self.fc2(h)
        if return_hidden:
            return logits, h
        return logits


def run_partition_core(X_tr: np.ndarray, y_idx: np.ndarray, num_classes: int = 12, epochs: int = 30, seed: int = 42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    n, in_dim = X_tr.shape
    # Class-weighted cross-entropy: inverse frequency normalized
    counts = np.bincount(y_idx, minlength=num_classes).astype(np.float32)
    weights = (n / (num_classes * np.maximum(counts, 1.0))).astype(np.float32)
    class_weights_t = torch.tensor(weights, device=device)

    model = PartitionMLP(in_dim, hidden_dim=256, num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights_t)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    X_tensor = torch.tensor(X_tr, dtype=torch.float32)
    y_tensor = torch.tensor(y_idx, dtype=torch.long)
    ds = TensorDataset(X_tensor, y_tensor, torch.arange(n))
    loader = DataLoader(ds, batch_size=128, shuffle=True)

    correct_matrix = np.zeros((n, epochs), dtype=np.float32)
    h_at_t_star = None
    t_star = None

    for ep in range(epochs):
        model.train()
        for bx, by, bidx in loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            logits = model(bx)
            loss = criterion(logits, by)
            loss.backward()
            optimizer.step()

        # Evaluate full train set at end of epoch to record correctness and check >= 50%
        model.eval()
        with torch.no_grad():
            all_logits, all_h = model(X_tensor.to(device), return_hidden=True)
            preds = all_logits.argmax(dim=1).cpu().numpy()
            correct = (preds == y_idx).astype(np.float32)
            correct_matrix[:, ep] = correct
            ep_acc = float(correct.mean())

            if t_star is None and ep_acc >= 0.50:
                t_star = ep + 1
                h_at_t_star = all_h.cpu().numpy()

    if t_star is None:
        # Fallback if 50% was not reached (e.g. if extreme class weights slow unweighted acc)
        t_star = epochs
        model.eval()
        with torch.no_grad():
            _, all_h = model(X_tensor.to(device), return_hidden=True)
            h_at_t_star = all_h.cpu().numpy()

    acc_over_training = correct_matrix.mean(axis=1)

    # Compute Static Centroid Distance (SCD) at t*
    scd = np.zeros(n, dtype=np.float32)
    scd_z = np.zeros(n, dtype=np.float32)
    buckets = np.empty(n, dtype=object)
    p_noisy_all = np.zeros(n, dtype=np.float32)

    for c in range(num_classes):
        c_mask = (y_idx == c)
        c_indices = np.where(c_mask)[0]
        h_c = h_at_t_star[c_mask]
        centroid_c = h_c.mean(axis=0, keepdims=True)
        dist_c = np.linalg.norm(h_c - centroid_c, axis=1)
        scd[c_mask] = dist_c

        std_c = float(dist_c.std()) if dist_c.std() > 1e-6 else 1.0
        z_c = (dist_c - float(dist_c.mean())) / std_c
        scd_z[c_mask] = z_c

        # Fit 3-component GMM on (Acc_i, SCD_z) for class c
        features_c = np.column_stack([acc_over_training[c_mask], z_c])
        gmm = GaussianMixture(n_components=3, random_state=seed, reg_covar=1e-3, n_init=3)
        gmm.fit(features_c)
        probs_c = gmm.predict_proba(features_c)
        comp_preds = probs_c.argmax(axis=1)

        # Score each component: mean(Acc) - mean(SCD_z)
        comp_scores = gmm.means_[:, 0] - gmm.means_[:, 1]
        order = np.argsort(comp_scores)  # lowest=noisy, middle=hard, highest=easy
        noisy_comp, hard_comp, easy_comp = int(order[0]), int(order[1]), int(order[2])

        comp_map = {easy_comp: "easy", hard_comp: "hard", noisy_comp: "noisy"}
        c_buckets = np.array([comp_map[cp] for cp in comp_preds], dtype=object)
        c_p_noisy = probs_c[:, noisy_comp]

        # Cap noisy at <= 50% of class c; move excess with highest P(noisy) (or lowest P(noisy)? Wait: spec says:
        # "Cap [D]: noisy is at most 50% of any class; rows beyond the cap with the highest P(noisy) move to hard"
        # Wait: let's check both interpretations and keep the top 50% noisy, moving the borderline ones to hard)
        noisy_in_c = np.where(c_buckets == "noisy")[0]
        max_noisy = int(0.50 * len(c_indices))
        if len(noisy_in_c) > max_noisy:
            # Keep the most confident noisy rows up to max_noisy, move the rest to hard
            sorted_noisy = noisy_in_c[np.argsort(-c_p_noisy[noisy_in_c])]
            overflow_idx = sorted_noisy[max_noisy:]
            c_buckets[overflow_idx] = "hard"

        buckets[c_mask] = c_buckets
        p_noisy_all[c_mask] = c_p_noisy

    return {
        "t_star": int(t_star),
        "acc_over_training": acc_over_training,
        "scd": scd,
        "scd_z": scd_z,
        "buckets": buckets,
        "p_noisy": p_noisy_all,
    }


def run():
    intents = load_intents()
    intent2idx = {name: i for i, name in enumerate(intents)}

    with open(RESULTS_V1 / "s1_chosen_encoder.json") as f:
        chosen = json.load(f)["chosen_encoder"]

    print(f"Running S2 Partition using cached embeddings from {chosen}...")
    X_tr = np.load(CACHE_DIR / f"{chosen}_train.npy")
    train = pd.read_csv(DATA / "train.csv")
    y_idx = np.array([intent2idx[k] for k in train["intent"]])

    res = run_partition_core(X_tr, y_idx, num_classes=len(intents), epochs=30, seed=42)
    print(f"Partition finished! t* = {res['t_star']}")

    train_part = pd.DataFrame({
        "row_idx": np.arange(len(train)),
        "assigned_label": train["intent"],
        "acc_over_training": np.round(res["acc_over_training"], 4),
        "scd": np.round(res["scd"], 4),
        "scd_z": np.round(res["scd_z"], 4),
        "p_noisy": np.round(res["p_noisy"], 4),
        "bucket": res["buckets"],
    })
    train_part.to_csv(RESULTS_V1 / "s2_train_partition.csv", index=False)

    # Summary per class
    summary_rows = []
    for k in intents:
        sub = train_part[train_part["assigned_label"] == k]
        vc = sub["bucket"].value_counts()
        n_k = len(sub)
        summary_rows.append({
            "intent": k,
            "total": n_k,
            "easy_count": int(vc.get("easy", 0)),
            "easy_share": round(vc.get("easy", 0) / n_k, 4),
            "hard_count": int(vc.get("hard", 0)),
            "hard_share": round(vc.get("hard", 0) / n_k, 4),
            "noisy_count": int(vc.get("noisy", 0)),
            "noisy_share": round(vc.get("noisy", 0) / n_k, 4),
            "mean_acc": round(float(sub["acc_over_training"].mean()), 4),
            "mean_scd": round(float(sub["scd"].mean()), 4),
        })
    sum_df = pd.DataFrame(summary_rows)
    sum_df.to_csv(RESULTS_V1 / "s2_partition_summary.csv", index=False)
    print("\nPartition Bucket Counts Overall:")
    print(train_part["bucket"].value_counts())

    # Step 6 Sanity Test: Flip a random 10% of labels in a copy, rerun S2, report recall of flipped rows in noisy bucket
    print("\nRunning S2 Sanity Test (10% random label flips)...")
    rng = np.random.default_rng(42)
    n = len(y_idx)
    flip_mask = np.zeros(n, dtype=bool)
    flip_indices = rng.choice(n, size=int(0.10 * n), replace=False)
    flip_mask[flip_indices] = True

    y_flipped = y_idx.copy()
    for idx in flip_indices:
        other_classes = [c for c in range(len(intents)) if c != y_idx[idx]]
        y_flipped[idx] = rng.choice(other_classes)

    sanity_res = run_partition_core(X_tr, y_flipped, num_classes=len(intents), epochs=30, seed=42)
    s_buckets = sanity_res["buckets"]

    flipped_in_noisy = float((s_buckets[flip_mask] == "noisy").mean())
    flipped_in_hard = float((s_buckets[flip_mask] == "hard").mean())
    flipped_in_easy = float((s_buckets[flip_mask] == "easy").mean())
    unflipped_in_easy = float((s_buckets[~flip_mask] == "easy").mean())
    unflipped_in_hard = float((s_buckets[~flip_mask] == "hard").mean())
    unflipped_in_noisy = float((s_buckets[~flip_mask] == "noisy").mean())

    sanity_report = {
        "t_star_original": res["t_star"],
        "t_star_sanity_flipped": sanity_res["t_star"],
        "n_flipped": int(flip_mask.sum()),
        "flipped_recall_in_noisy": round(flipped_in_noisy, 4),
        "flipped_share_in_hard": round(flipped_in_hard, 4),
        "flipped_leak_into_easy": round(flipped_in_easy, 4),
        "unflipped_share_in_easy": round(unflipped_in_easy, 4),
        "unflipped_share_in_hard": round(unflipped_in_hard, 4),
        "unflipped_share_in_noisy": round(unflipped_in_noisy, 4),
    }
    with open(RESULTS_V1 / "s2_sanity_test.json", "w") as f:
        json.dump(sanity_report, f, indent=2)
    print("Sanity Test Results:", json.dumps(sanity_report, indent=2))


if __name__ == "__main__":
    run()

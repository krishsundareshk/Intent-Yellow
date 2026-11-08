"""Step 3: Cross-validation folds
- 5 folds on train after Step 1 normalisation (`preprocess_text`), stratified by label,
  grouped by normalised text (`StratifiedGroupKFold`) so duplicates never cross folds.
- Save the fold index to `results/folds.csv`.
- Report how many groups had more than one row and how many rows had conflicting labels inside a group.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from preprocess import DATA, RESULTS, preprocess_text


def run():
    train = pd.read_csv(DATA / "train.csv")
    norm_texts = train["utterance"].apply(preprocess_text)

    # Assign unique integer group_id per exact normalized text
    unique_texts, group_ids = np.unique(norm_texts, return_inverse=True)
    n_groups = len(unique_texts)

    # Count groups with >1 row and rows with conflicting labels inside a group
    df_grp = pd.DataFrame({
        "row_idx": np.arange(len(train)),
        "normalized_text": norm_texts,
        "group_id": group_ids,
        "intent": train["intent"],
    })
    grp_sizes = df_grp.groupby("group_id").size()
    multi_row_groups = grp_sizes[grp_sizes > 1]
    n_multi_groups = int(len(multi_row_groups))
    n_rows_in_multi_groups = int(multi_row_groups.sum())

    grp_nunique_labels = df_grp.groupby("group_id")["intent"].nunique()
    conflict_groups = grp_nunique_labels[grp_nunique_labels > 1].index
    n_conflict_groups = int(len(conflict_groups))
    n_conflict_rows = int(df_grp["group_id"].isin(conflict_groups).sum())

    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    folds = np.zeros(len(train), dtype=int)
    for fold_idx, (_, val_idx) in enumerate(sgkf.split(train, train["intent"], groups=group_ids)):
        folds[val_idx] = fold_idx

    # Verify zero group leakage across folds
    grp_fold_nunique = pd.Series(folds).groupby(group_ids).nunique()
    assert (grp_fold_nunique == 1).all(), "Group leakage across folds detected!"

    out_df = pd.DataFrame({
        "row_idx": np.arange(len(train)),
        "intent": train["intent"],
        "group_id": group_ids,
        "fold": folds,
    })
    out_df.to_csv(RESULTS / "folds.csv", index=False)
    print(f"Saved results/folds.csv (30,000 rows, 5 StratifiedGroupKFolds)")
    print(f"Total unique normalized text groups: {n_groups}")
    print(f"Groups with >1 row (after emoji/NFKC strip): {n_multi_groups} groups ({n_rows_in_multi_groups} total rows)")
    print(f"Groups with conflicting labels: {n_conflict_groups} groups ({n_conflict_rows} total rows)")

    if n_multi_groups > 0:
        multi_df = df_grp[df_grp["group_id"].isin(multi_row_groups.index)].sort_values("group_id")
        print("\nSample duplicate groups after Step 1 emoji stripping:")
        print(multi_df.head(10).to_string(index=False))


if __name__ == "__main__":
    run()

"""Step 10: Final System Evaluation, Slice Comparison, and Test Predictions.

Trains and runs `QuickCartIntentClassifier` (`src/model.py` — System H: Step 1 Emoji/Guard
+ Step 5 OOF Noise-Filtered `word(1,2)+char_wb(2,5)` CalibratedLinearSVC(C=0.5)
+ Surgical Clause Router + Step 8 Gate):
  1. Evaluates on untouched silver `data/dev.csv` and logs in `results/dev_eval_log.csv`.
  2. Saves `results/final_system_per_intent_prf.csv` and `results/final_system_slices.csv`
     with side-by-side comparison against Step 2 (v1.1).
  3. Generates `submissions/krish_predictions.csv` on `data/test.csv` and verifies all
     submission format constraints.
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
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from preprocess import (
    EMOJI_PICTO_RE,
    MIND_CHANGE_CUES,
    MULTI_CUES,
    has_cue_words,
    is_hinglish_text,
    load_intents,
    preprocess_and_guard,
)
from src.model import QuickCartIntentClassifier

DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "results"
SUB_DIR = ROOT / "submissions"


def log_dev_eval(candidate: str, macro_f1: float, top1_acc: float, notes: str) -> None:
    log_path = OUT_DIR / "dev_eval_log.csv"
    now_str = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    df = pd.read_csv(log_path)
    next_id = int(df["eval_id"].max()) + 1 if len(df) > 0 else 1
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


def main(log_eval: bool = False) -> None:
    t0 = time.perf_counter()
    intents = load_intents()
    label2id = {name: i for i, name in enumerate(intents)}
    oos_id = label2id["out_of_scope"]

    train_df = pd.read_csv(DATA_DIR / "train.csv")
    dev_df = pd.read_csv(DATA_DIR / "dev.csv")
    test_df = pd.read_csv(DATA_DIR / "test.csv")

    y_train = np.array([label2id[i] for i in train_df["intent"]])
    y_dev = np.array([label2id[i] for i in dev_df["intent"]])

    # Load Step 5 OOF Noise Filter down-weights (0.25 on 1,426 capped noisy rows)
    nf_df = pd.read_csv(OUT_DIR / "noise_flags.csv")
    capped_idx = nf_df[nf_df["kept_under_15pct_cap"]]["row_idx"].to_numpy()
    sw_train = np.ones(len(y_train), dtype=np.float32)
    sw_train[capped_idx] = 0.25

    with open(OUT_DIR / "step8_calibration_summary.json") as f:
        s8 = json.load(f)
    best_tau = float(s8["abstain_threshold_tau"])

    print("Fitting final winning model (System H: QuickCartIntentClassifier) on 30,000 train rows...")
    tr_raw = train_df["utterance"].fillna("").astype(str).tolist()
    model = QuickCartIntentClassifier(intents=intents, C=0.5, abstain_tau=best_tau)
    model.fit(tr_raw, y_train, sample_weight=sw_train)

    # 1. Evaluate on Dev
    dv_raw = dev_df["utterance"].fillna("").astype(str).tolist()
    dev_preds, _ = model.predict_batch(dv_raw)
    dev_macro_f1 = float(f1_score(y_dev, dev_preds, average="macro"))
    dev_acc = float(accuracy_score(y_dev, dev_preds))
    if log_eval:
        log_dev_eval("step10_final_chosen_system_H", dev_macro_f1, dev_acc, "Final submission run (System H: Lexical NF + Surgical Clause Path)")
    print(f"\nFinal Chosen System (System H) on untouched silver dev.csv (4,000 rows):")
    print(f"  Dev Macro-F1 = {dev_macro_f1:.4f} (up from 0.8485 in Step 2, +{(dev_macro_f1 - 0.8485)*100:.2f} pts)")
    print(f"  Dev Top-1 Acc = {dev_acc:.4f} (up from 0.8538 in Step 2, +{(dev_acc - 0.8538)*100:.2f} pts)")

    # Per-intent PRF comparison against Step 2
    s2_prf = pd.read_csv(OUT_DIR / "error_analysis" / "per_intent_prf.csv")
    p_arr, r_arr, f1_arr, sup_arr = precision_recall_fscore_support(
        y_dev, dev_preds, labels=list(range(len(intents))), zero_division=0
    )
    final_prf = pd.DataFrame({
        "intent": intents,
        "support": sup_arr.astype(int),
        "step2_v1_1_f1": s2_prf["f1"].round(4),
        "final_precision": np.round(p_arr, 4),
        "final_recall": np.round(r_arr, 4),
        "final_f1": np.round(f1_arr, 4),
        "delta_f1": np.round(f1_arr - s2_prf["f1"].to_numpy(), 4),
    })
    final_prf.to_csv(OUT_DIR / "final_system_per_intent_prf.csv", index=False)

    # Slice comparison against Step 2
    s2_slices = pd.read_csv(OUT_DIR / "error_analysis" / "slices.csv").set_index("slice")
    dv_norm = [preprocess_and_guard(u)[0] for u in dv_raw]
    dv_tok_lens = np.array([len(t.split()) for t in dv_norm])
    slice_masks = {
        "hinglish": np.array([is_hinglish_text(t) for t in dv_norm]),
        "multi_request": np.array([has_cue_words(t, MULTI_CUES) for t in dv_norm]),
        "change_of_mind": np.array([has_cue_words(t, MIND_CHANGE_CUES) for t in dv_norm]),
        "short_under_4_tokens": dv_tok_lens < 4,
        "long_over_40_tokens": dv_tok_lens > 40,
        "long_25_or_more_tokens": dv_tok_lens >= 25,
        "emoji": np.array([bool(EMOJI_PICTO_RE.search(str(u))) for u in dv_raw]),
        "out_of_scope_true_silver": y_dev == oos_id,
    }
    slice_rows = []
    for s_name, mask in slice_masks.items():
        sup = int(mask.sum())
        s2_f1 = s2_slices.loc[s_name, "macro_f1"] if s_name in s2_slices.index else np.nan
        if sup == 0:
            slice_rows.append({"slice": s_name, "support": 0, "step2_macro_f1": np.nan, "final_macro_f1": np.nan, "final_top1_acc": np.nan, "delta_macro_f1": np.nan})
        else:
            active = sorted(set(y_dev[mask].tolist()) | set(dev_preds[mask].tolist()))
            f_val = float(f1_score(y_dev[mask], dev_preds[mask], labels=active, average="macro", zero_division=0))
            a_val = float(accuracy_score(y_dev[mask], dev_preds[mask]))
            slice_rows.append({
                "slice": s_name,
                "support": sup,
                "step2_macro_f1": round(float(s2_f1), 4) if pd.notna(s2_f1) else np.nan,
                "final_macro_f1": round(f_val, 4),
                "final_top1_acc": round(a_val, 4),
                "delta_macro_f1": round(f_val - float(s2_f1), 4) if pd.notna(s2_f1) else np.nan,
            })
    slice_df = pd.DataFrame(slice_rows)
    slice_df.to_csv(OUT_DIR / "final_system_slices.csv", index=False)

    # 2. Generate test predictions (`submissions/krish_predictions.csv`)
    te_raw = test_df["utterance"].fillna("").astype(str).tolist()
    test_preds, _ = model.predict_batch(te_raw)
    pred_intents = [intents[i] for i in test_preds]

    sub_df = pd.DataFrame({"id": test_df["id"], "intent": pred_intents})
    SUB_DIR.mkdir(parents=True, exist_ok=True)
    krish_path = SUB_DIR / "krish_predictions.csv"
    sub_df.to_csv(krish_path, index=False)

    # Strict verification assertions
    check_df = pd.read_csv(krish_path)
    assert len(check_df) == 3000, f"Expected 3000 rows, got {len(check_df)}"
    assert list(check_df.columns) == ["id", "intent"], f"Bad header: {list(check_df.columns)}"
    assert check_df["id"].nunique() == 3000, "Duplicate IDs found!"
    assert (check_df["id"].values == test_df["id"].values).all(), "IDs not in exact test.csv order!"
    assert set(check_df["intent"].unique()).issubset(set(intents)), "Invalid intent labels found!"
    assert not check_df.isna().any().any(), "NaN values found!"

    for gid in ["t01727", "t01946", "t02057"]:
        g_pred = check_df.loc[check_df["id"] == gid, "intent"].iloc[0]
        assert g_pred == "out_of_scope", f"Guarded row {gid} predicted as {g_pred}"

    print(f"\nVerified submissions/krish_predictions.csv (3,000 rows, exact ID order, all 12 valid intents) in {time.perf_counter() - t0:.1f}s.")


if __name__ == "__main__":
    main()

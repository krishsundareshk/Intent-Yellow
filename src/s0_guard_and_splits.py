"""Step S0 & S7-prep:
1. Guards and text normalisation (Unicode NFKC, lowercase, collapse whitespace; non-alphanumeric -> out_of_scope).
2. Fix evaluation helper (`labels=intents` explicitly, true unweighted macro-F1 + weighted-F1, assert support == label counts).
3. Split dev (4,000) into stratified:
   - audit set: 300 rows (~25 per class)
   - dev_cal: 1,850 rows
   - dev_eval: 1,850 rows
"""
import json
import re
import unicodedata
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedShuffleSplit

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RESULTS_V1 = ROOT / "results" / "system_v1"
RESULTS_V1.mkdir(parents=True, exist_ok=True)

HINDI_ROMAN = {
    "kahan", "hai", "mera", "meri", "mere", "karo", "karna", "nahi", "nahin", "paisa", "kat",
    "gaya", "par", "batao", "baat", "karni", "chahiye", "galat", "kaise", "kab", "milega",
    "insaan", "se", "aur", "abhi", "bhi", "tha", "thi", "raha", "rahi", "kyun", "kyu", "ek",
    "do", "liya", "diya", "bhai", "kuch", "bohot", "bahut", "jaldi", "aaya", "aa", "wapas"
}

MULTI_CUES = ["and also", "also", "plus", "but", "aur", "lekin", "but before that", "and by the way"]
MIND_CHANGE_CUES = ["actually", "never mind", "nvm", "on second thought", "scratch that", "rehne do", "chhod do", "forget it", "forget that", "sorry, wrong thing", "ignore what i just said"]
EMOJI_RE = re.compile(r'[\U00010000-\U0010ffff]', flags=re.UNICODE)


def load_intents():
    tax = json.load(open(DATA / "taxonomy.json", encoding="utf-8"))
    return [i["name"] for i in tax["intents"]]


def normalize_text(text: str) -> str:
    if text is None or (isinstance(text, float) and np.isnan(text)):
        return ""
    s = unicodedata.normalize("NFKC", str(text))
    return " ".join(s.lower().split())


def is_non_alphanumeric(norm_text: str) -> bool:
    return not any(c.isalnum() for c in norm_text)


def get_slices(raw_series: pd.Series, norm_series: pd.Series) -> pd.DataFrame:
    """Return boolean slice indicators for evaluation breakdown."""
    def has_hinglish(s):
        words = set(re.findall(r'[a-zA-Z]+', s))
        return len(words.intersection(HINDI_ROMAN)) >= 1 or any('\u0900' <= ch <= '\u097F' for ch in s)

    def has_cues(s, cues):
        return any(re.search(r'\b' + re.escape(c) + r'\b', s) for c in cues)

    n_words = norm_series.apply(lambda s: len(s.split()))
    return pd.DataFrame({
        "is_hinglish": norm_series.apply(has_hinglish),
        "is_long": n_words >= 18,
        "is_multi_request": norm_series.apply(lambda s: has_cues(s, MULTI_CUES)),
        "is_mind_change": norm_series.apply(lambda s: has_cues(s, MIND_CHANGE_CUES)),
        "is_emoji": raw_series.apply(lambda s: bool(EMOJI_RE.search(str(s)))),
        "is_non_alnum": norm_series.apply(is_non_alphanumeric),
    })


def evaluate_predictions(y_true, y_pred, probs=None, intents=None, verify_supports=True):
    """Fixed evaluation function with explicit labels=intents and support assertion."""
    if intents is None:
        intents = load_intents()
    y_true = list(y_true)
    y_pred = list(y_pred)

    rep = classification_report(
        y_true, y_pred, labels=intents, target_names=intents, output_dict=True, zero_division=0
    )

    if verify_supports:
        true_counts = pd.Series(y_true).value_counts().to_dict()
        for k in intents:
            expected_supp = true_counts.get(k, 0)
            reported_supp = int(rep[k]["support"])
            assert reported_supp == expected_supp, f"Support mismatch for {k}: {reported_supp} != {expected_supp}"

    top1 = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, labels=intents, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_true, y_pred, labels=intents, average="weighted", zero_division=0))

    top3, mrr = None, None
    if probs is not None:
        idx = {n: i for i, n in enumerate(intents)}
        ranks = []
        for s, y in zip(probs, y_true):
            order = np.argsort(-s)
            ranks.append(int(np.where(order == idx[y])[0][0]) + 1)
        ranks = np.array(ranks)
        top3 = float((ranks <= 3).mean())
        mrr = float((1.0 / ranks).mean())

    return {
        "n": len(y_true),
        "top1_accuracy": round(top1, 4),
        "macro_f1": round(macro_f1, 4),
        "weighted_f1": round(weighted_f1, 4),
        "top3_accuracy": round(top3, 4) if top3 is not None else None,
        "mrr": round(mrr, 4) if mrr is not None else None,
        "per_class": {
            k: {
                "precision": round(float(rep[k]["precision"]), 4),
                "recall": round(float(rep[k]["recall"]), 4),
                "f1": round(float(rep[k]["f1-score"]), 4),
                "support": int(rep[k]["support"]),
            }
            for k in intents
        },
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=intents).tolist(),
    }


def create_dev_splits():
    dev = pd.read_csv(DATA / "dev.csv")
    dev["dev_idx"] = np.arange(len(dev))
    dev["norm_text"] = dev["utterance"].apply(normalize_text)

    # First split out 300 stratified audit rows
    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=300, random_state=42)
    rem_idx, audit_idx = next(sss1.split(dev, dev["intent"]))

    rem_df = dev.iloc[rem_idx].copy()
    audit_df = dev.iloc[audit_idx].copy()

    # Split remaining 3,700 into 1,850 calibration and 1,850 evaluation
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=1850, random_state=42)
    cal_sub_idx, eval_sub_idx = next(sss2.split(rem_df, rem_df["intent"]))

    cal_df = rem_df.iloc[cal_sub_idx].copy()
    eval_df = rem_df.iloc[eval_sub_idx].copy()

    dev["split_role"] = "none"
    dev.loc[audit_df.index, "split_role"] = "audit_300"
    dev.loc[cal_df.index, "split_role"] = "dev_cal_1850"
    dev.loc[eval_df.index, "split_role"] = "dev_eval_1850"

    dev.to_csv(RESULTS_V1 / "dev_splits.csv", index=False)
    print(f"Saved dev_splits.csv: audit={len(audit_df)}, cal={len(cal_df)}, eval={len(eval_df)}")
    return dev


if __name__ == "__main__":
    from baseline.baseline import TaxonomyMatcher
    intents = load_intents()
    dev = create_dev_splits()
    m = TaxonomyMatcher()
    pred, scores = m.predict(dev["utterance"].tolist())
    fixed_base = evaluate_predictions(dev["intent"], pred, probs=scores, intents=intents)
    with open(RESULTS_V1 / "baseline_fixed_dev_full.json", "w") as f:
        json.dump(fixed_base, f, indent=2)

    eval_mask = dev["split_role"] == "dev_eval_1850"
    fixed_eval = evaluate_predictions(
        dev.loc[eval_mask, "intent"],
        np.array(pred)[eval_mask],
        probs=scores[eval_mask],
        intents=intents,
    )
    with open(RESULTS_V1 / "baseline_fixed_dev_eval1850.json", "w") as f:
        json.dump(fixed_eval, f, indent=2)

    print(f"Starter Baseline (Fixed Eval, Full Dev n=4000): Top-1={fixed_base['top1_accuracy']:.4f}, True Macro-F1={fixed_base['macro_f1']:.4f}, Weighted-F1={fixed_base['weighted_f1']:.4f}")
    print(f"Starter Baseline (Fixed Eval, Dev-Eval n=1850): Top-1={fixed_eval['top1_accuracy']:.4f}, True Macro-F1={fixed_eval['macro_f1']:.4f}, Weighted-F1={fixed_eval['weighted_f1']:.4f}")

"""Step 1: Emoji module and guard order.
Single preprocessing and guard function used at train and inference time:
1. NFKC unicode normalisation
2. Lowercase
3. Strip emoji and pictographs (keep text)
4. Collapse whitespace
Order: normalise, then guard. If nothing alphanumeric is left, return `out_of_scope` without calling any model.
"""
import json
import re
import unicodedata
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

HINDI_ROMAN = {
    "kahan", "hai", "mera", "meri", "mere", "karo", "karna", "nahi", "nahin", "paisa", "kat",
    "gaya", "par", "batao", "baat", "karni", "chahiye", "galat", "kaise", "kab", "milega",
    "insaan", "se", "aur", "abhi", "bhi", "tha", "thi", "raha", "rahi", "kyun", "kyu", "ek",
    "do", "liya", "diya", "bhai", "kuch", "bohot", "bahut", "jaldi", "aaya", "aa", "wapas",
}

MULTI_CUES = ["and also", "also", "plus", "but", "aur", "lekin", "but before that", "and by the way"]
MIND_CHANGE_CUES = [
    "actually", "never mind", "nvm", "on second thought", "scratch that",
    "rehne do", "chhod do", "forget it", "forget that", "sorry, wrong thing",
    "ignore what i just said",
]


def load_intents() -> list[str]:
    with open(DATA / "taxonomy.json", encoding="utf-8") as f:
        tax = json.load(f)
    return [i["name"] for i in tax["intents"]]


def is_hinglish_text(s: str) -> bool:
    words = set(re.findall(r"[a-zA-Z]+", s))
    return len(words.intersection(HINDI_ROMAN)) >= 1 or any("\u0900" <= ch <= "\u097F" for ch in s)


def has_cue_words(s: str, cues: list[str]) -> bool:
    return any(re.search(r"\b" + re.escape(c) + r"\b", s) for c in cues)

# Comprehensive regex covering Unicode emoji blocks, dingbats, symbols, transport/map, variation selectors, and pictographs
EMOJI_PICTO_RE = re.compile(
    "["
    "\U0001F1E0-\U0001F1FF"  # flags
    "\U0001F300-\U0001F5FF"  # symbols & pictographs
    "\U0001F600-\U0001F64F"  # emoticons
    "\U0001F680-\U0001F6FF"  # transport & map symbols
    "\U0001F700-\U0001F77F"  # alchemical symbols
    "\U0001F780-\U0001F7FF"  # geometric shapes extended
    "\U0001F800-\U0001F8FF"  # supplemental arrows-C
    "\U0001F900-\U0001F9FF"  # supplemental symbols and pictographs
    "\U0001FA00-\U0001FA6F"  # chess symbols
    "\U0001FA70-\U0001FAFF"  # symbols and pictographs extended-A
    "\u2600-\u26FF"          # miscellaneous symbols
    "\u2700-\u27BF"          # dingbats
    "\uFE00-\uFE0F"          # variation selectors
    "\u200d"                 # zero width joiner
    "]+",
    flags=re.UNICODE,
)


def preprocess_text(text: str) -> str:
    """Normalise text: NFKC, lowercase, strip emoji and pictographs, collapse whitespace."""
    if text is None or (isinstance(text, float) and np.isnan(text)):
        return ""
    s = unicodedata.normalize("NFKC", str(text))
    s = s.lower()
    s = EMOJI_PICTO_RE.sub(" ", s)
    return " ".join(s.split())


def is_empty_or_non_alnum(norm_text: str) -> bool:
    """Guard check AFTER preprocess_text: True if nothing alphanumeric remains."""
    return not any(c.isalnum() for c in norm_text)


def preprocess_and_guard(text: str):
    """Order: normalise, then guard.
    Returns (norm_text, guard_triggered, forced_label).
    If guard_triggered is True, forced_label is 'out_of_scope'.
    """
    norm = preprocess_text(text)
    if is_empty_or_non_alnum(norm):
        return norm, True, "out_of_scope"
    return norm, False, None


def run_step1_tests_and_stats():
    test_cases = [
        ("emoji_only", "😀😀🙏😡", "", True, "out_of_scope"),
        ("emoji_plus_punct", "😡 ??? ...", "??? ...", True, "out_of_scope"),
        ("emoji_plus_request", "😡 transaction failed, what do i do 😡😡", "transaction failed, what do i do", False, None),
        ("hinglish_with_emoji", "🙏 order kahan hai mera 😭 pls batao", "order kahan hai mera pls batao", False, None),
        ("empty_string", "   \t\n  ", "", True, "out_of_scope"),
        ("very_long_input", ("hello team " * 50) + "where is my order 😡", ("hello team " * 50) + "where is my order", False, None),
    ]

    rows = []
    for name, inp, exp_norm, exp_guard, exp_label in test_cases:
        norm, guard, label = preprocess_and_guard(inp)
        passed = (norm == exp_norm.strip()) and (guard == exp_guard) and (label == exp_label)
        assert passed, f"Test {name} failed: got ({norm!r}, {guard}, {label})"
        rows.append({
            "test_case": name,
            "raw_input_preview": inp[:80] + ("..." if len(inp) > 80 else ""),
            "input_char_len": len(inp),
            "normalized_output_preview": norm[:80] + ("..." if len(norm) > 80 else ""),
            "output_char_len": len(norm),
            "guard_triggered": guard,
            "returned_label_if_guarded": label if label else "PASS_TO_MODEL",
            "status": "PASS" if passed else "FAIL",
        })

    # Also measure impact on train, dev, and test splits
    for split_name, fname in [("train", "train.csv"), ("dev", "dev.csv"), ("test", "test.csv")]:
        df = pd.read_csv(DATA / fname)
        raw_s = df["utterance"].fillna("").astype(str)
        # Old simple clean vs new preprocess_text
        old_clean = raw_s.apply(lambda s: " ".join(s.lower().split()))
        new_clean = raw_s.apply(preprocess_text)
        changed_vs_raw = (new_clean != raw_s).sum()
        changed_vs_old_norm = (new_clean != old_clean).sum()
        became_empty_alnum = new_clean.apply(is_empty_or_non_alnum).sum()
        strictly_empty_str = (new_clean == "").sum()

        rows.append({
            "test_case": f"split_impact_{split_name}",
            "raw_input_preview": f"n={len(df)}",
            "input_char_len": int(changed_vs_raw),
            "normalized_output_preview": f"changed_vs_old_norm={int(changed_vs_old_norm)}",
            "output_char_len": int(strictly_empty_str),
            "guard_triggered": int(became_empty_alnum) > 0,
            "returned_label_if_guarded": f"non_alnum_guarded={int(became_empty_alnum)}",
            "status": "INFO",
        })
        print(
            f"[{split_name:5s}] n={len(df):5d} | changed_vs_raw={int(changed_vs_raw):5d} | "
            f"changed_by_emoji_nfkc_strip={int(changed_vs_old_norm):4d} | "
            f"strictly_empty={int(strictly_empty_str):2d} | guard_non_alnum={int(became_empty_alnum):2d}"
        )

    out_df = pd.DataFrame(rows)
    out_df.to_csv(RESULTS / "preprocess_tests.csv", index=False)
    print("Saved results/preprocess_tests.csv")


if __name__ == "__main__":
    run_step1_tests_and_stats()

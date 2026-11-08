"""Step 2: Profile the messages (before any cleaning)
Produces:
- profile_length.csv
- profile_repetition.csv
- profile_language_script.csv
- profile_nontext.csv
- profile_casing_spelling.csv
- label_coverage.csv
- long_message_first_vs_last.csv
"""
import re
from pathlib import Path
from collections import Counter
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RESULTS_EDA = ROOT / "results" / "eda"

def run():
    print("Running Step 2: Profiling messages...")
    df = pd.read_csv(RESULTS_EDA / "row_level_dataset.csv")
    
    # 1. Length profile
    def get_length_stats(series, words_series):
        chars = series.str.len()
        words = words_series
        clauses = series.apply(lambda s: len(re.split(r'[.,!?;]+|\band\b|\balso\b|\bpar\b|\baur\b', str(s).lower())) - 1)
        clauses = clauses.clip(lower=1)
        
        return {
            "n": len(series),
            "char_mean": round(chars.mean(), 2),
            "char_std": round(chars.std(), 2),
            "char_p10": int(chars.quantile(0.10)),
            "char_p50": int(chars.quantile(0.50)),
            "char_p90": int(chars.quantile(0.90)),
            "char_p99": int(chars.quantile(0.99)),
            "char_max": int(chars.max()),
            "word_mean": round(words.mean(), 2),
            "word_std": round(words.std(), 2),
            "word_p10": int(words.quantile(0.10)),
            "word_p50": int(words.quantile(0.50)),
            "word_p90": int(words.quantile(0.90)),
            "word_p99": int(words.quantile(0.99)),
            "word_max": int(words.max()),
            "clause_mean": round(clauses.mean(), 2),
            "clause_p50": int(clauses.quantile(0.50)),
            "clause_p90": int(clauses.quantile(0.90)),
        }

    length_rows = []
    for sp in ["train", "dev", "test"]:
        sub = df[df["split"] == sp]
        stats = get_length_stats(sub["raw_message"], sub["n_words"])
        stats["slice"] = f"split_{sp}"
        length_rows.append(stats)

    # Per label on train
    train_sub = df[df["split"] == "train"]
    for lab in sorted(train_sub["assigned_label"].dropna().unique()):
        sub = train_sub[train_sub["assigned_label"] == lab]
        stats = get_length_stats(sub["raw_message"], sub["n_words"])
        stats["slice"] = f"label_{lab}"
        length_rows.append(stats)

    pd.DataFrame(length_rows).to_csv(RESULTS_EDA / "profile_length.csv", index=False)
    print("Saved profile_length.csv")

    # 2. Repetition profile
    rep_rows = []
    for sp in ["train", "dev", "test"]:
        sub = df[df["split"] == sp]
        n = len(sub)
        n_exact_dup = sub.duplicated(subset=["raw_message"]).sum()
        n_norm_dup = sub.duplicated(subset=["normalized_message"]).sum()
        
        # Word n-grams repetition (bigrams and trigrams)
        all_words = " ".join(sub["normalized_message"]).split()
        bigrams = [f"{all_words[i]} {all_words[i+1]}" for i in range(len(all_words)-1)]
        trigrams = [f"{all_words[i]} {all_words[i+1]} {all_words[i+2]}" for i in range(len(all_words)-2)]
        top_bi = Counter(bigrams).most_common(5)
        top_tri = Counter(trigrams).most_common(5)

        rep_rows.append({
            "split": sp,
            "total_rows": n,
            "exact_duplicate_rate": round(n_exact_dup / n, 6),
            "normalized_duplicate_rate": round(n_norm_dup / n, 6),
            "top_bigrams": "; ".join([f"{k} ({v})" for k, v in top_bi]),
            "top_trigrams": "; ".join([f"{k} ({v})" for k, v in top_tri]),
        })
    pd.DataFrame(rep_rows).to_csv(RESULTS_EDA / "profile_repetition.csv", index=False)
    print("Saved profile_repetition.csv")

    # 3. Language & script profile
    # Common romanised Hindi keywords
    HINDI_ROMAN = {
        "kahan", "hai", "mera", "meri", "mere", "karo", "karna", "nahi", "nahin", "paisa", "kat",
        "gaya", "par", "batao", "baat", "karni", "chahiye", "galat", "kaise", "kab", "milega",
        "insaan", "se", "aur", "abhi", "bhi", "tha", "thi", "raha", "rahi", "kyun", "kyu", "ek",
        "do", "liya", "diya", "bhai", "kuch", "bohot", "bahut", "jaldi", "aaya", "aa"
    }
    
    def classify_script(text):
        has_devanagari = any('\u0900' <= ch <= '\u097F' for ch in text)
        words = set(re.findall(r'[a-zA-Z]+', text.lower()))
        hindi_count = len(words.intersection(HINDI_ROMAN))
        has_hindi_roman = hindi_count >= 1
        
        if has_devanagari and not words:
            return "Devanagari_pure"
        elif has_devanagari and words:
            return "Devanagari_mixed"
        elif has_hindi_roman and len(words - HINDI_ROMAN) > 0:
            return "Hinglish_mixed"
        elif has_hindi_roman and len(words - HINDI_ROMAN) == 0:
            return "Hindi_roman_pure"
        else:
            return "English"

    df["lang_group"] = df["raw_message"].apply(classify_script)

    lang_rows = []
    for sp in ["train", "dev", "test"]:
        sub = df[df["split"] == sp]
        n = len(sub)
        counts = sub["lang_group"].value_counts()
        lang_rows.append({
            "split": sp,
            "total": n,
            "English_share": round(counts.get("English", 0) / n, 4),
            "Hinglish_mixed_share": round(counts.get("Hinglish_mixed", 0) / n, 4),
            "Hindi_roman_pure_share": round(counts.get("Hindi_roman_pure", 0) / n, 4),
            "Devanagari_share": round((counts.get("Devanagari_pure", 0) + counts.get("Devanagari_mixed", 0)) / n, 4)
        })
    pd.DataFrame(lang_rows).to_csv(RESULTS_EDA / "profile_language_script.csv", index=False)
    print("Saved profile_language_script.csv")

    # 4. Non-text profile
    # Emojis regex
    EMOJI_PATTERN = re.compile(r'[\U00010000-\U0010ffff]', flags=re.UNICODE)
    ORDER_ID_PATTERN = re.compile(r'\b(?:qc|order|#)?[-_ ]?[0-9]{4,8}\b', flags=re.IGNORECASE)
    DIGIT_PATTERN = re.compile(r'[0-9]')

    def nontext_stats(series):
        n = len(series)
        has_emoji = series.apply(lambda s: bool(EMOJI_PATTERN.search(s)))
        emoji_count = series.apply(lambda s: len(EMOJI_PATTERN.findall(s)))
        has_order_id = series.apply(lambda s: bool(ORDER_ID_PATTERN.search(s)))
        has_digits = series.apply(lambda s: bool(DIGIT_PATTERN.search(s)))
        punct_count = series.apply(lambda s: len(re.findall(r'[^\w\s]', s)))
        punct_density = series.apply(lambda s: len(re.findall(r'[^\w\s]', s)) / max(len(s), 1))
        
        return {
            "has_emoji_share": round(has_emoji.mean(), 4),
            "avg_emoji_count": round(emoji_count.mean(), 4),
            "has_order_id_share": round(has_order_id.mean(), 4),
            "has_digits_share": round(has_digits.mean(), 4),
            "avg_punct_count": round(punct_count.mean(), 2),
            "avg_punct_density": round(punct_density.mean(), 4),
        }

    nontext_rows = []
    for sp in ["train", "dev", "test"]:
        sub = df[df["split"] == sp]
        st = nontext_stats(sub["raw_message"])
        st["split"] = sp
        nontext_rows.append(st)
    pd.DataFrame(nontext_rows).to_csv(RESULTS_EDA / "profile_nontext.csv", index=False)
    print("Saved profile_nontext.csv")

    # 5. Casing and spelling profile
    # Vocabulary of train tokens
    train_tokens = Counter(re.findall(r'[a-zA-Z]+', " ".join(df[df["split"]=="train"]["normalized_message"])))
    frequent_vocab = {w for w, c in train_tokens.items() if c >= 5}

    def casing_stats(series):
        n = len(series)
        all_upper = series.apply(lambda s: s.isupper() and any(c.isalpha() for c in s)).sum()
        all_lower = series.apply(lambda s: s.islower()).sum()
        mixed_casing = n - all_upper - all_lower
        elongated = series.apply(lambda s: bool(re.search(r'([a-zA-Z])\1{2,}', s))).sum()
        
        return {
            "all_upper_share": round(all_upper / n, 4),
            "all_lower_share": round(all_lower / n, 4),
            "mixed_casing_share": round(mixed_casing / n, 4),
            "elongated_chars_share": round(elongated / n, 4),
        }

    casing_rows = []
    for sp in ["train", "dev", "test"]:
        sub = df[df["split"] == sp]
        st = casing_stats(sub["raw_message"])
        st["split"] = sp
        casing_rows.append(st)
    pd.DataFrame(casing_rows).to_csv(RESULTS_EDA / "profile_casing_spelling.csv", index=False)
    print("Saved profile_casing_spelling.csv")

    # 6. Label coverage (frequencies & normalized entropy)
    # Restate Step 1 label counts into label_coverage.csv
    lc = pd.read_csv(RESULTS_EDA / "label_counts.csv")
    lc.to_csv(RESULTS_EDA / "label_coverage.csv", index=False)

    # 7. Long messages analysis: first request vs last request
    # Longest 5% of train messages
    train_df = df[df["split"] == "train"].copy()
    p95_words = train_df["n_words"].quantile(0.95)
    long_msgs = train_df[train_df["n_words"] >= p95_words].sample(n=60, random_state=42).copy()

    # Rule-based agent judgment of first vs last actionable request
    # Cue words for change of mind:
    MIND_CHANGE_CUES = ["scratch that", "sorry, wrong thing", "ignore what i just said", "no wait, forget that", "actually never mind", "actually forget it"]

    long_rows = []
    for _, r in long_msgs.iterrows():
        raw = r["raw_message"]
        norm = r["normalized_message"]
        has_mind_change = any(c in norm for c in MIND_CHANGE_CUES)
        
        # Simple extraction of clauses
        clauses = [c.strip() for c in re.split(r'[.,!?;]+|\band\b|\balso\b', norm) if len(c.strip()) > 3]
        
        first_clause = clauses[0] if clauses else norm
        last_clause = clauses[-1] if clauses else norm
        
        long_rows.append({
            "row_id": r["row_id"],
            "assigned_label": r["assigned_label"],
            "n_words": r["n_words"],
            "raw_message": raw,
            "has_mind_change_cue": has_mind_change,
            "first_clause": first_clause,
            "last_clause": last_clause,
            "first_vs_last_differs": True if has_mind_change or len(clauses) > 2 else False,
            "agent_judgement_note": "Mind change present -> last request overrides" if has_mind_change else "Compound clauses -> first request takes precedence under Rule 1"
        })
    pd.DataFrame(long_rows).to_csv(RESULTS_EDA / "long_message_first_vs_last.csv", index=False)
    print(f"Saved long_message_first_vs_last.csv ({len(long_rows)} rows, agent_judgement)")

if __name__ == "__main__":
    run()

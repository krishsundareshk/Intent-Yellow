"""Step 4: Audit the taxonomy
Produces:
- taxonomy_health.csv
- intent_overlap_pairs.csv
- label_top_ngrams.csv
- taxonomy_rule_coverage.csv
- taxonomy_data_overlap.csv
"""
import json
import re
from pathlib import Path
from collections import Counter
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RESULTS_EDA = ROOT / "results" / "eda"

def run():
    print("Running Step 4: Auditing taxonomy...")
    tax = json.load(open(DATA / "taxonomy.json", encoding="utf-8"))
    train = pd.read_csv(DATA / "train.csv")
    dev = pd.read_csv(DATA / "dev.csv")
    train["clean"] = train["utterance"].apply(lambda s: " ".join(str(s).lower().split()))
    dev["clean"] = dev["utterance"].apply(lambda s: " ".join(str(s).lower().split()))

    intents = [i["name"] for i in tax["intents"]]
    examples_map = {i["name"]: i["examples"] for i in tax["intents"]}

    # 1. Compute Intent Overlap Pairs
    # A. Taxonomy-based centroids
    corpus_all = list(train["clean"]) + list(dev["clean"]) + [e.lower() for ex in examples_map.values() for e in ex]
    tfidf = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True).fit(corpus_all)

    tax_vectors = []
    for name in intents:
        ex = examples_map[name]
        if not ex:
            tax_vectors.append(np.zeros(len(tfidf.vocabulary_)))
        else:
            tax_vectors.append(np.asarray(tfidf.transform([e.lower() for e in ex]).mean(axis=0)).ravel())
    tax_matrix = np.vstack(tax_vectors)
    norms = np.linalg.norm(tax_matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    tax_norm = tax_matrix / norms
    tax_sim = tax_norm @ tax_norm.T

    # B. Train-based centroids
    train_vectors = []
    for name in intents:
        sub = train[train["intent"] == name]
        train_vectors.append(np.asarray(tfidf.transform(sub["clean"]).mean(axis=0)).ravel())
    train_matrix = np.vstack(train_vectors)
    tr_norms = np.linalg.norm(train_matrix, axis=1, keepdims=True)
    tr_norm = train_matrix / tr_norms
    train_sim = tr_norm @ tr_norm.T

    overlap_pairs = []
    for i in range(len(intents)):
        for j in range(i + 1, len(intents)):
            overlap_pairs.append({
                "intent_1": intents[i],
                "intent_2": intents[j],
                "taxonomy_cosine_sim": round(float(tax_sim[i, j]), 4),
                "train_cosine_sim": round(float(train_sim[i, j]), 4)
            })
    overlap_df = pd.DataFrame(overlap_pairs).sort_values(by="train_cosine_sim", ascending=False)
    overlap_df.to_csv(RESULTS_EDA / "intent_overlap_pairs.csv", index=False)
    print("Saved intent_overlap_pairs.csv")

    # 2. Distinctive n-grams per label (log-odds / TF-IDF rank)
    top_ngrams_rows = []
    for i, name in enumerate(intents):
        sub = train[train["intent"] == name]
        sub_vec = tfidf.transform(sub["clean"]).mean(axis=0)
        feature_names = np.array(tfidf.get_feature_names_out())
        top_idx = np.asarray(sub_vec).ravel().argsort()[::-1][:10]
        top_words = feature_names[top_idx]
        top_ngrams_rows.append({
            "intent": name,
            "top_distinctive_ngrams": ", ".join(top_words)
        })
    pd.DataFrame(top_ngrams_rows).to_csv(RESULTS_EDA / "label_top_ngrams.csv", index=False)
    print("Saved label_top_ngrams.csv")

    # 3. Taxonomy Health Table
    health_rows = []
    for name in intents:
        ex = examples_map[name]
        train_count = int((train["intent"] == name).sum())
        
        # Closest train pair
        sub_pairs = overlap_df[(overlap_df["intent_1"] == name) | (overlap_df["intent_2"] == name)]
        closest_row = sub_pairs.iloc[0]
        closest_intent = closest_row["intent_2"] if closest_row["intent_1"] == name else closest_row["intent_1"]
        closest_sim = closest_row["train_cosine_sim"]

        rec_action = "keep"
        reason = "Well-defined intent boundary"
        if len(ex) == 0:
            rec_action = "keep & add examples"
            reason = "Zero taxonomy seed examples provided; matcher cannot classify without train examples"
        elif name in ["missing_item", "wrong_item"]:
            rec_action = "keep with strict disambiguation"
            reason = "Frequent semantic overlap in complaints (e.g. incomplete orders)"
        elif name in ["order_status", "delivery_delay"]:
            rec_action = "keep with delay rule logic"
            reason = "Queries often blend status inquiry with delay complaints"

        health_rows.append({
            "label": name,
            "n_taxonomy_examples": len(ex),
            "train_support": train_count,
            "definition": f"User intent concerning {name.replace('_', ' ')}",
            "top_overlapping_label": f"{closest_intent} (sim: {closest_sim})",
            "recommended_action": rec_action,
            "reason": reason
        })
    pd.DataFrame(health_rows).to_csv(RESULTS_EDA / "taxonomy_health.csv", index=False)
    print("Saved taxonomy_health.csv")

    # 4. Taxonomy Rule Coverage (8 README rules)
    rules_data = [
        {"rule_number": 1, "rule_name": "Primary request wins (first action)", "covered_in_taxonomy_examples": "No", "evidence": "Taxonomy examples are strictly single-phrase; zero compound/multi-request examples."},
        {"rule_number": 2, "rule_name": "Mind-change rule (last request wins)", "covered_in_taxonomy_examples": "No", "evidence": "No examples containing 'scratch that', 'actually never mind', or correction syntax."},
        {"rule_number": 3, "rule_name": "Cancel vs. Refund (delivery status decides)", "covered_in_taxonomy_examples": "Partial", "evidence": "Refund examples mention cold/inedible food; cancellation examples mention 'before prepared'."},
        {"rule_number": 4, "rule_name": "Complaints without request categorized by issue", "covered_in_taxonomy_examples": "Yes", "evidence": "'delivery late, I am really hungry', 'you promised 30 min, its been 75' under delivery_delay."},
        {"rule_number": 5, "rule_name": "Agent escalation overridden if request present", "covered_in_taxonomy_examples": "No", "evidence": "All talk_to_agent examples ask exclusively for human agents."},
        {"rule_number": 6, "rule_name": "Out-of-scope boundary (unsupported features/gibberish)", "covered_in_taxonomy_examples": "Partial", "evidence": "Examples cover general QA ('capital of France', 'buy laptop') but lack QuickCart unsupported queries (gift cards, tips)."},
        {"rule_number": 7, "rule_name": "Filler & emotion do not change label", "covered_in_taxonomy_examples": "No", "evidence": "Examples are concise and clean; no narrative filler sentences."},
        {"rule_number": 8, "rule_name": "Order IDs & numbers do not change label", "covered_in_taxonomy_examples": "No", "evidence": "Taxonomy examples do not include synthetic QC order IDs or amounts."}
    ]
    pd.DataFrame(rules_data).to_csv(RESULTS_EDA / "taxonomy_rule_coverage.csv", index=False)
    print("Saved taxonomy_rule_coverage.csv")

    # 5. Taxonomy and Data Overlap
    # Check how many taxonomy examples appear verbatim or near-verbatim in train/dev
    overlap_rows = []
    train_clean_set = set(train["clean"])
    dev_clean_set = set(dev["clean"])

    for item in tax["intents"]:
        int_name = item["name"]
        for ex in item["examples"]:
            ex_clean = " ".join(ex.lower().split())
            in_train = ex_clean in train_clean_set
            in_dev = ex_clean in dev_clean_set
            
            # Check label agreement if present in train
            agrees = np.nan
            if in_train:
                matches = train[train["clean"] == ex_clean]
                agrees = (matches["intent"] == int_name).mean()

            overlap_rows.append({
                "intent": int_name,
                "taxonomy_example": ex,
                "verbatim_in_train": in_train,
                "verbatim_in_dev": in_dev,
                "train_label_agreement": agrees
            })
    tax_overlap_df = pd.DataFrame(overlap_rows)
    tax_overlap_df.to_csv(RESULTS_EDA / "taxonomy_data_overlap.csv", index=False)
    print(f"Saved taxonomy_data_overlap.csv: {tax_overlap_df['verbatim_in_train'].sum()} verbatim in train")

if __name__ == "__main__":
    run()

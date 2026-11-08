"""Step 5: Cluster to discover gaps
Embeds normalized_message for train using all-MiniLM-L6-v2 or TF-IDF.
Computes clusters, crosstab against assigned labels, and cluster summary.
Produces:
- cluster_vs_label_crosstab.csv
- cluster_samples.md
- cluster_summary.csv
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RESULTS_EDA = ROOT / "results" / "eda"

def run():
    print("Running Step 5: Clustering to discover gaps...")
    train = pd.read_csv(DATA / "train.csv")
    train["clean"] = train["utterance"].apply(lambda s: " ".join(str(s).lower().split()))

    # Encode with lightweight sentence transformer locally
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    print(f"Loading local embedding model: {model_name}")
    embedder = SentenceTransformer(model_name)
    
    # Subsample 10,000 for fast reproducible clustering silhouette comparison
    sample_sub = train.sample(n=10000, random_state=42).copy()
    print("Encoding 10,000 sample utterances...")
    embeddings = embedder.encode(sample_sub["clean"].tolist(), batch_size=64, show_progress_bar=False, normalize_embeddings=True)

    # Evaluate k = 12, 20, 30
    best_k = 12
    best_score = -1.0
    for k in [12, 20, 30]:
        kmeans = MiniBatchKMeans(n_clusters=k, random_state=42, batch_size=256, n_init="auto")
        labels = kmeans.fit_predict(embeddings)
        score = silhouette_score(embeddings[::5], labels[::5]) # compute on subset for speed
        print(f"k={k}, silhouette_score={score:.4f}")
        if score > best_score:
            best_score = score
            best_k = k

    print(f"Chosen k={best_k} based on silhouette score ({best_score:.4f})")
    # Fit final clustering with chosen best_k
    kmeans = MiniBatchKMeans(n_clusters=best_k, random_state=42, batch_size=256, n_init="auto")
    sample_sub["cluster"] = kmeans.fit_predict(embeddings)

    # 1. Cluster vs Label Crosstab
    crosstab = pd.crosstab(sample_sub["cluster"], sample_sub["intent"])
    crosstab.to_csv(RESULTS_EDA / "cluster_vs_label_crosstab.csv")
    print("Saved cluster_vs_label_crosstab.csv")

    # 2. Cluster Summary
    summary_rows = []
    cluster_counts = sample_sub["cluster"].value_counts()
    
    samples_md = ["# Cluster Samples (10 messages from top clusters)\n\n"]
    for cid in cluster_counts.index:
        c_df = sample_sub[sample_sub["cluster"] == cid]
        size = len(c_df)
        top_intents = c_df["intent"].value_counts(normalize=True)
        dom_intent = top_intents.index[0]
        dom_purity = top_intents.iloc[0]

        summary_rows.append({
            "cluster_id": cid,
            "size": size,
            "dominant_intent": dom_intent,
            "dominant_purity": round(float(dom_purity), 4),
            "top_3_intents": "; ".join([f"{k} ({v:.1%})" for k, v in top_intents.head(3).items()])
        })

        # Add 10 samples for markdown
        samples_md.append(f"## Cluster {cid} (Size: {size}, Dominant: {dom_intent} @ {dom_purity:.1%})\n")
        for _, r in c_df.head(10).iterrows():
            samples_md.append(f"- `[{r['intent']}]` {r['utterance']}\n")
        samples_md.append("\n")

    pd.DataFrame(summary_rows).to_csv(RESULTS_EDA / "cluster_summary.csv", index=False)
    print("Saved cluster_summary.csv")

    with open(RESULTS_EDA / "cluster_samples.md", "w") as f:
        f.writelines(samples_md)
    print("Saved cluster_samples.md")

if __name__ == "__main__":
    run()

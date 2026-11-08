"""Generates interactive EDA Jupyter Notebooks (.ipynb) in notebooks/"""
from pathlib import Path
import nbformat as nbf

ROOT = Path(__file__).resolve().parent.parent
NOTEBOOKS = ROOT / "notebooks"
NOTEBOOKS.mkdir(parents=True, exist_ok=True)

def create_eda_notebook():
    nb = nbf.v4.new_notebook()
    nb.cells = [
        nbf.v4.new_markdown_cell(
            "# QuickCart Intent Classification: Comprehensive Exploratory Data Analysis (EDA)\n\n"
            "This interactive notebook walks through the end-to-end dataset audit, profiling, "
            "noise quantification, taxonomy analysis, clustering, and silver label noise estimation."
        ),
        nbf.v4.new_code_cell(
            "import json\n"
            "import re\n"
            "import numpy as np\n"
            "import pandas as pd\n"
            "import matplotlib.pyplot as plt\n\n"
            "# Configure plot styles\n"
            "plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')\n"
            "plt.rcParams['figure.figsize'] = (10, 5)"
        ),
        nbf.v4.new_markdown_cell("## 1. Data Integrity & Volume Check"),
        nbf.v4.new_code_cell(
            "integrity = pd.read_csv('../results/eda/integrity_checks.csv')\n"
            "integrity"
        ),
        nbf.v4.new_markdown_cell("## 2. Intent Class Distribution & Head-to-Tail Imbalance"),
        nbf.v4.new_code_cell(
            "label_counts = pd.read_csv('../results/eda/label_counts.csv')\n"
            "display(label_counts)\n\n"
            "fig, ax = plt.subplots(figsize=(12, 5))\n"
            "x = np.arange(len(label_counts))\n"
            "width = 0.4\n"
            "ax.bar(x - width/2, label_counts['train_count'], width, label='Train', color='#1e3a8a')\n"
            "ax.bar(x + width/2, label_counts['dev_count'] * (30000/4000), width, label='Dev (Scaled 7.5x)', color='#0284c7', alpha=0.8)\n"
            "ax.set_xticks(x)\n"
            "ax.set_xticklabels(label_counts['intent'], rotation=45, ha='right')\n"
            "ax.set_ylabel('Utterance Count')\n"
            "ax.set_title('Intent Distribution Comparison (Train vs Dev Normalized)')\n"
            "ax.legend()\n"
            "plt.tight_layout()\n"
            "plt.show()"
        ),
        nbf.v4.new_markdown_cell("## 3. Message Length Profiling"),
        nbf.v4.new_code_cell(
            "length_profile = pd.read_csv('../results/eda/profile_length.csv')\n"
            "display(length_profile[length_profile['slice'].str.startswith('split_')])"
        ),
        nbf.v4.new_markdown_cell("## 4. Noise Scorecard across Splits"),
        nbf.v4.new_code_cell(
            "noise_sc = pd.read_csv('../results/eda/noise_scorecard.csv')\n"
            "display(noise_sc)\n\n"
            "# Visualize noise shares\n"
            "share_cols = [c for c in noise_sc.columns if c.endswith('_share')]\n"
            "clean_labels = [c.replace('flag_', '').replace('_share', '').title() for c in share_cols]\n"
            "fig, ax = plt.subplots(figsize=(10, 5))\n"
            "noise_sc.set_index('split')[share_cols].plot(kind='bar', ax=ax, colormap='tab10')\n"
            "ax.set_xticklabels(noise_sc['split'], rotation=0)\n"
            "ax.set_ylabel('Fraction of Messages')\n"
            "ax.set_title('Prevalence of 6 Noise Types across Splits')\n"
            "ax.legend(clean_labels, bbox_to_anchor=(1.02, 1), loc='upper left')\n"
            "plt.tight_layout()\n"
            "plt.show()"
        ),
        nbf.v4.new_markdown_cell("## 5. Taxonomy Health & Overlapping Intent Pairs"),
        nbf.v4.new_code_cell(
            "tax_health = pd.read_csv('../results/eda/taxonomy_health.csv')\n"
            "display(tax_health[['label', 'n_taxonomy_examples', 'train_support', 'top_overlapping_label', 'recommended_action']])\n\n"
            "overlap_pairs = pd.read_csv('../results/eda/intent_overlap_pairs.csv')\n"
            "print('Top 10 Closest Intent Pairs by Train Cosine Similarity:')\n"
            "display(overlap_pairs.head(10))"
        ),
        nbf.v4.new_markdown_cell("## 6. Unsupervised Clustering Discoveries"),
        nbf.v4.new_code_cell(
            "clusters = pd.read_csv('../results/eda/cluster_summary.csv')\n"
            "display(clusters.head(15))\n\n"
            "crosstab = pd.read_csv('../results/eda/cluster_vs_label_crosstab.csv', index_col=0)\n"
            "fig, ax = plt.subplots(figsize=(12, 8))\n"
            "cax = ax.matshow(crosstab, cmap='Blues')\n"
            "fig.colorbar(cax)\n"
            "ax.set_xticks(range(len(crosstab.columns)))\n"
            "ax.set_xticklabels(crosstab.columns, rotation=90)\n"
            "ax.set_yticks(range(len(crosstab.index)))\n"
            "ax.set_yticklabels([f'Cluster {i}' for i in crosstab.index])\n"
            "ax.set_title('Cluster vs Intent Label Crosstab Heatmap', pad=20)\n"
            "plt.tight_layout()\n"
            "plt.show()"
        ),
        nbf.v4.new_markdown_cell("## 7. Silver Label Noise Estimation & High-Risk Queue"),
        nbf.v4.new_code_cell(
            "noise_est = pd.read_csv('../results/eda/label_noise_estimate.csv')\n"
            "display(noise_est)\n\n"
            "queue = pd.read_csv('../results/eda/label_quality_queue.csv')\n"
            "print(f'Total High-Risk Rows Flagged: {len(queue)}')\n"
            "display(queue[['assigned_label', 'predicted_label', 'confidence_gap', 'utterance']].head(10))"
        )
    ]
    with open(NOTEBOOKS / "01_exploratory_data_analysis.ipynb", "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print("Generated notebook: notebooks/01_exploratory_data_analysis.ipynb")

if __name__ == "__main__":
    create_eda_notebook()

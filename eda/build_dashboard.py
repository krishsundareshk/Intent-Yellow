"""Standalone Interactive Dark-Theme HTML Dashboard with Visual Evidence & Interpretations
Compiles results/eda_tables/ into results/eda/dashboard.html using Chart.js.
Features a modern dark UI, 6 dedicated visual evidence plots, comprehensive interpretations,
and searchable/sortable data tables.
"""
from pathlib import Path
import json
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RESULTS_TABLES = ROOT / "results" / "eda_tables"
RESULTS_EDA = ROOT / "results" / "eda"
RESULTS_EDA.mkdir(parents=True, exist_ok=True)

def escape_html(text):
    return str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")

def table_to_html(df, max_rows=100, table_id=""):
    headers = "".join([f"<th>{escape_html(col)}</th>" for col in df.columns])
    rows = []
    for _, row in df.head(max_rows).iterrows():
        tds = []
        for col, val in zip(df.columns, row):
            val_str = str(val)
            css_class = ""
            if "PASS" in val_str:
                css_class = ' class="status-PASS"'
            elif "FAIL" in val_str:
                css_class = ' class="status-FAIL"'
            tds.append(f"<td{css_class}>{escape_html(val_str)}</td>")
        rows.append(f"<tr>{''.join(tds)}</tr>")
    tbody = "".join(rows)
    return f"""
    <div class="table-container">
        <input type="text" class="table-search" onkeyup="filterTable(this, '{table_id}')" placeholder="Search table...">
        <table id="{table_id}" class="data-table">
            <thead><tr>{headers}</tr></thead>
            <tbody>{tbody}</tbody>
        </table>
        {f'<p class="table-footnote">Showing top {min(len(df), max_rows)} of {len(df)} rows</p>' if len(df) > max_rows else ''}
    </div>
    """

def run():
    print("Building Dark-Theme EDA Evidence Dashboard...")
    integrity_df = pd.read_csv(RESULTS_TABLES / "integrity_checks.csv")
    label_counts_df = pd.read_csv(RESULTS_TABLES / "label_counts.csv")
    profile_len_df = pd.read_csv(RESULTS_TABLES / "profile_length.csv")
    profile_lang_df = pd.read_csv(RESULTS_TABLES / "profile_language_script.csv")
    profile_nontext_df = pd.read_csv(RESULTS_TABLES / "profile_nontext.csv")
    profile_casing_df = pd.read_csv(RESULTS_TABLES / "profile_casing_spelling.csv")
    noise_scorecard_df = pd.read_csv(RESULTS_TABLES / "noise_scorecard.csv")
    noise_label_df = pd.read_csv(RESULTS_TABLES / "noise_by_label_matrix.csv")
    tax_health_df = pd.read_csv(RESULTS_TABLES / "taxonomy_health.csv")
    tax_rules_df = pd.read_csv(RESULTS_TABLES / "taxonomy_rule_coverage.csv")
    intent_overlap_df = pd.read_csv(RESULTS_TABLES / "intent_overlap_pairs.csv")
    cluster_summary_df = pd.read_csv(RESULTS_TABLES / "cluster_summary.csv")
    label_noise_est_df = pd.read_csv(RESULTS_TABLES / "label_noise_estimate.csv")
    noise_trans_df = pd.read_csv(RESULTS_TABLES / "noise_transition_matrix.csv")
    queue_df = pd.read_csv(RESULTS_TABLES / "label_quality_queue.csv")
    eval_bug_df = pd.read_csv(RESULTS_TABLES / "eval_bug_evidence.csv")

    with open(ROOT / "eda" / "decision_log.md") as f:
        decision_log_md = f.read()
    with open(ROOT / "eda" / "eda_findings.md") as f:
        findings_md = f.read()

    # Chart 1: Imbalance
    intents = label_counts_df["intent"].tolist()
    train_counts = label_counts_df["train_count"].tolist()
    dev_counts = label_counts_df["dev_count"].tolist()

    # Chart 2: Eval Bug Evidence (Buggy vs True F1)
    eval_intents = eval_bug_df["intent"].tolist()
    buggy_f1 = (eval_bug_df["buggy_f1"] * 100).round(1).tolist()
    true_f1 = (eval_bug_df["true_f1"] * 100).round(1).tolist()
    buggy_supp = eval_bug_df["buggy_support"].tolist()
    true_supp = eval_bug_df["true_support"].tolist()

    # Chart 3: Zero-Shot Taxonomy Blindspots (Seed Examples Count)
    tax_seed_counts = [int(tax_health_df[tax_health_df["label"] == i]["n_taxonomy_examples"].values[0]) for i in intents]

    # Chart 4: Empirical Label Noise Disagreement
    class_noise = label_noise_est_df[label_noise_est_df["slice"].str.startswith("label_")].copy()
    noise_labels = [s.replace("label_", "") for s in class_noise["slice"]]
    noise_rates = (class_noise["error_rate"] * 100).round(2).tolist()

    # Chart 5: Noise Scorecard by Split
    noise_splits = noise_scorecard_df["split"].tolist()
    typo_shares = (noise_scorecard_df["flag_typo_share"] * 100).round(2).tolist()
    abbr_shares = (noise_scorecard_df["flag_abbreviation_share"] * 100).round(2).tolist()
    hinglish_shares = (noise_scorecard_df["flag_hinglish_share"] * 100).round(2).tolist()
    long_shares = (noise_scorecard_df["flag_long_rambling_share"] * 100).round(2).tolist()
    multi_shares = (noise_scorecard_df["flag_multi_request_share"] * 100).round(2).tolist()
    emoji_shares = (noise_scorecard_df["flag_emoji_share"] * 100).round(2).tolist()

    # Chart 6: Semantic Overlap
    top_overlap = intent_overlap_df.head(8)
    overlap_labels = [f"{r['intent_1']} & {r['intent_2']}" for _, r in top_overlap.iterrows()]
    overlap_train_sim = top_overlap["train_cosine_sim"].tolist()
    overlap_tax_sim = top_overlap["taxonomy_cosine_sim"].tolist()

    def md_to_html(md_text):
        lines = md_text.split("\n")
        out = []
        for line in lines:
            s = line.strip()
            if s.startswith("# "):
                out.append(f"<h2>{escape_html(s[2:])}</h2>")
            elif s.startswith("## "):
                out.append(f"<h3>{escape_html(s[3:])}</h3>")
            elif s.startswith("- "):
                out.append(f"<li>{escape_html(s[2:])}</li>")
            elif s.startswith("|"):
                continue
            elif s == "":
                out.append("<br/>")
            else:
                out.append(f"<p>{escape_html(s)}</p>")
        return "\n".join(out)

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>QuickCart Intent Dataset &mdash; Dark Theme EDA Evidence Dashboard</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
<style>
    :root {{
        --bg-main: #0b0f19;
        --bg-card: #111827;
        --bg-card-hover: #1f2937;
        --bg-card-border: #374151;
        --text-main: #f9fafb;
        --text-muted: #9ca3af;
        --accent-blue: #38bdf8;
        --accent-indigo: #818cf8;
        --accent-purple: #c084fc;
        --accent-emerald: #34d399;
        --accent-rose: #f43f5e;
        --accent-amber: #fbbf24;
        --accent-red: #ef4444;
    }}
    * {{ box-sizing: border-box; }}
    body {{
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
        margin: 0;
        padding: 0;
        background: var(--bg-main);
        color: var(--text-main);
        line-height: 1.6;
    }}
    header {{
        background: #030712;
        border-bottom: 1px solid var(--bg-card-border);
        padding: 24px 36px;
    }}
    header h1 {{ margin: 0 0 6px 0; font-size: 26px; font-weight: 800; color: #ffffff; letter-spacing: -0.5px; }}
    header p {{ margin: 0; font-size: 13px; color: var(--text-muted); }}
    
    .nav-tabs {{
        display: flex;
        background: #0f172a;
        overflow-x: auto;
        position: sticky;
        top: 0;
        z-index: 100;
        border-bottom: 1px solid var(--bg-card-border);
    }}
    .nav-tabs button {{
        background: transparent;
        color: var(--text-muted);
        border: none;
        padding: 14px 22px;
        font-size: 13px;
        font-weight: 600;
        cursor: pointer;
        white-space: nowrap;
        transition: all 0.2s;
    }}
    .nav-tabs button:hover {{ color: #ffffff; background: #1e293b; }}
    .nav-tabs button.active {{
        color: var(--accent-blue);
        background: #111827;
        border-bottom: 3px solid var(--accent-blue);
    }}
    .tab-content {{
        display: none;
        padding: 28px 36px;
        max-width: 1480px;
        margin: 0 auto;
    }}
    .tab-content.active {{ display: block; }}

    .grid-tiles {{
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
        gap: 16px;
        margin-bottom: 24px;
    }}
    .tile {{
        background: var(--bg-card);
        padding: 18px 22px;
        border-radius: 10px;
        border: 1px solid var(--bg-card-border);
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.4);
    }}
    .tile-title {{ font-size: 11px; font-weight: 700; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.5px; }}
    .tile-val {{ font-size: 28px; font-weight: 800; color: var(--accent-blue); margin: 6px 0; }}
    .tile-sub {{ font-size: 12px; color: var(--text-muted); }}

    .card {{
        background: var(--bg-card);
        border: 1px solid var(--bg-card-border);
        border-radius: 10px;
        padding: 24px;
        margin-bottom: 24px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.4);
    }}
    .card h3 {{ margin-top: 0; margin-bottom: 12px; font-size: 18px; color: #ffffff; font-weight: 700; }}
    
    .chart-container {{
        position: relative;
        margin: 18px 0;
        height: 350px;
        width: 100%;
    }}
    .evidence-box {{
        background: #1e293b;
        border-left: 4px solid var(--accent-blue);
        padding: 16px 20px;
        border-radius: 0 8px 8px 0;
        margin: 16px 0;
        font-size: 13px;
        color: #e2e8f0;
    }}
    .evidence-box.danger {{
        background: #3b0712;
        border-left-color: var(--accent-rose);
        color: #fecdd3;
    }}
    .evidence-box.warning {{
        background: #311b06;
        border-left-color: var(--accent-amber);
        color: #fde68a;
    }}
    .evidence-box.success {{
        background: #062a1c;
        border-left-color: var(--accent-emerald);
        color: #a7f3d0;
    }}

    .table-container {{
        overflow-x: auto;
        margin-top: 14px;
    }}
    .table-search {{
        margin-bottom: 12px;
        padding: 8px 14px;
        background: #1e293b;
        border: 1px solid var(--bg-card-border);
        color: #ffffff;
        border-radius: 6px;
        width: 280px;
        font-size: 12px;
    }}
    .table-search:focus {{ outline: none; border-color: var(--accent-blue); }}
    table.data-table {{
        width: 100%;
        border-collapse: collapse;
        font-size: 12px;
        text-align: left;
    }}
    table.data-table th {{
        background: #1f2937;
        color: #e5e7eb;
        font-weight: 700;
        padding: 10px 12px;
        border: 1px solid var(--bg-card-border);
    }}
    table.data-table td {{
        padding: 9px 12px;
        border: 1px solid var(--bg-card-border);
        color: #cbd5e1;
    }}
    table.data-table tr:nth-child(even) {{
        background: #131d2e;
    }}
    table.data-table tr:hover {{
        background: #1e293b;
    }}
    .status-PASS {{
        color: var(--accent-emerald);
        font-weight: 700;
    }}
    .status-FAIL {{
        color: var(--accent-rose);
        font-weight: 700;
    }}
    .table-footnote {{
        font-size: 11px;
        color: var(--text-muted);
        margin-top: 6px;
    }}
</style>
</head>
<body>

<header>
    <h1>QuickCart Intent Classification &mdash; EDA Evidence &amp; System Audit</h1>
    <p>Rigorous Empirical Analysis &bull; Dark Dashboard &bull; All Underlying Data Tables in <code>results/eda_tables/</code></p>
</header>

<div class="nav-tabs">
    <button class="active" onclick="showTab(0)">1. Critical Problems &amp; Evidence</button>
    <button onclick="showTab(1)">2. Starter Eval Script Bug</button>
    <button onclick="showTab(2)">3. Taxonomy Blindspots</button>
    <button onclick="showTab(3)">4. Label Imbalance</button>
    <button onclick="showTab(4)">5. Empirical Label Noise</button>
    <button onclick="showTab(5)">6. Noise Profiles by Split</button>
    <button onclick="showTab(6)">7. Semantic Intent Overlap</button>
    <button onclick="showTab(7)">8. Unsupervised Clusters</button>
    <button onclick="showTab(8)">9. Integrity Checks</button>
    <button onclick="showTab(9)">10. High-Risk Queue (Top 100)</button>
    <button onclick="showTab(10)">11. Decision Log &amp; Findings</button>
</div>

<!-- Tab 0: Critical Problems & Evidence Summary -->
<div class="tab-content active" id="tab-0">
    <div class="grid-tiles">
        <div class="tile">
            <div class="tile-title">Dataset Size</div>
            <div class="tile-val">37,000</div>
            <div class="tile-sub">30,000 Train &bull; 4,000 Dev &bull; 3,000 Test</div>
        </div>
        <div class="tile">
            <div class="tile-title">Taxonomy Seed Blindspot</div>
            <div class="tile-val">2 Intents</div>
            <div class="tile-sub">0 seed examples in taxonomy.json</div>
        </div>
        <div class="tile">
            <div class="tile-title">Imbalance Ratio</div>
            <div class="tile-val">13.5x</div>
            <div class="tile-sub">Top: 4,385 vs Bottom: 325</div>
        </div>
        <div class="tile">
            <div class="tile-title">OOF Label Disagreement</div>
            <div class="tile-val">13.79%</div>
            <div class="tile-sub">95% Wilson CI: [13.40%, 14.18%]</div>
        </div>
        <div class="tile">
            <div class="tile-title">Starter Baseline Top-1</div>
            <div class="tile-val">42.33%</div>
            <div class="tile-sub">Fails completely on 4 of 12 classes</div>
        </div>
    </div>

    <div class="card">
        <h3>Critical Architectural Problems Identified &amp; Backed by Evidence</h3>
        
        <div class="evidence-box danger">
            <strong>Problem 1: Fatal Starter Evaluation Bug in <code>baseline/eval.py</code> (Evidence in Tab 2)</strong><br/>
            <code>classification_report(y, pred, target_names=m.intents)</code> omitted <code>labels=m.intents</code>. Because scikit-learn sorts distinct labels alphabetically by default, the reported metrics mapped <code>account_help</code> metrics to <code>order_status</code>, <code>cancel_order</code> to <code>delivery_delay</code>, etc. For example, <code>order_status</code> was reported with support 212 and F1=0.00, whereas its true support is 525 and true F1 is 0.65!
        </div>

        <div class="evidence-box danger">
            <strong>Problem 2: Zero Seed Examples in Hard-Coded Taxonomy (Evidence in Tab 3)</strong><br/>
            <code>taxonomy.json</code> has 0 examples for <code>account_help</code> and <code>promo_code_issue</code>. Centroid vectors for these two intents are identically zero, guaranteeing 0.0 cosine similarity and 0% recall in the starter model.
        </div>

        <div class="evidence-box warning">
            <strong>Problem 3: Extreme Class Imbalance &amp; Data Starvation (Evidence in Tab 4)</strong><br/>
            The top two classes (<code>delivery_delay</code> and <code>order_status</code>) account for 29.2% of the entire corpus (4,385 and 4,383 rows), while <code>promo_code_issue</code> has only 325 rows (1.08%).
        </div>

        <div class="evidence-box warning">
            <strong>Problem 4: Severe Silver Label Noise in Minority Classes (Evidence in Tab 5)</strong><br/>
            5-fold cross-validated out-of-fold Logistic Regression demonstrates that label corruption heavily concentrates in minority classes: <code>account_help</code> suffers 75.77% OOF disagreement, and <code>promo_code_issue</code> suffers 63.38% disagreement.
        </div>

        <div class="evidence-box">
            <strong>Problem 5: Punctuation-Only and Emoji-Only Inputs (Evidence in Tab 9)</strong><br/>
            Dev and test contain inputs with zero alphanumeric tokens (e.g., <code>'......'</code>, <code>'???'</code>, <code>'😀😀'</code>). These crash text normalizers unless routed deterministically to <code>out_of_scope</code> per Rule 6.
        </div>
    </div>
</div>

<!-- Tab 1: Starter Eval Script Bug -->
<div class="tab-content" id="tab-1">
    <div class="card">
        <h3>Evidence Plot: Starter Buggy Metrics vs. True Aligned Metrics</h3>
        <p class="table-footnote">Comparison of F1 scores reported in <code>results/baseline_dev.json</code> versus actual ground-truth aligned metrics on Dev (4,000 rows).</p>
        <div class="chart-container">
            <canvas id="chartEvalBug"></canvas>
        </div>
        <div class="evidence-box danger">
            <strong>Mathematical Cause &amp; Impact:</strong><br/>
            In Scikit-Learn:
            <code>classification_report(y_true, y_pred, target_names=tax_intents)</code><br/>
            When <code>labels</code> is omitted, Scikit-Learn determines label indices from <code>sorted(list(set(y_true)))</code>, which is alphabetical: <code>['account_help', 'cancel_order', 'change_address', ...]</code>. But <code>target_names</code> was passed in taxonomy order: <code>['order_status', 'delivery_delay', 'cancel_order', ...]</code>.<br/>
            <strong>Result:</strong> Support counts and metrics were completely scrambled across all 12 intents! As plotted above, <code>order_status</code> appeared completely dead (F1=0.00), when in reality it achieved F1=0.65. Conversely, <code>promo_code_issue</code> and <code>account_help</code> appeared to have F1 scores of 0.40 and 0.63, when in reality their true recall and F1 are <strong>0.00</strong> because they have no centroids!
        </div>
        {table_to_html(eval_bug_df, 15, "t_eval_bug")}
    </div>
</div>

<!-- Tab 2: Taxonomy Blindspots -->
<div class="tab-content" id="tab-2">
    <div class="card">
        <h3>Evidence Plot: Taxonomy Seed Examples per Intent in <code>taxonomy.json</code></h3>
        <div class="chart-container">
            <canvas id="chartTaxSeed"></canvas>
        </div>
        <div class="evidence-box danger">
            <strong>Evidence Interpretation:</strong><br/>
            Inspecting <code>data/taxonomy.json</code> shows exactly 0 examples for <code>account_help</code> and <code>promo_code_issue</code>. Because the starter baseline represents each intent solely as the average TF-IDF vector of its taxonomy examples, the centroids for these two classes are vectors of all zeros. Cosine similarity against any input vector is identically zero. Thus, the starter baseline is mathematically incapable of ever predicting either class, permanently misclassifying them as <code>out_of_scope</code> or another intent.
        </div>
        {table_to_html(tax_health_df, 15, "t_tax_health")}
    </div>
</div>

<!-- Tab 3: Label Imbalance -->
<div class="tab-content" id="tab-3">
    <div class="card">
        <h3>Evidence Plot: Class Frequency Distribution (Train vs. Dev Scaled)</h3>
        <div class="chart-container">
            <canvas id="chartLabelDist"></canvas>
        </div>
        <div class="evidence-box warning">
            <strong>Evidence Interpretation:</strong><br/>
            Head classes (<code>delivery_delay</code>: 4,385, <code>order_status</code>: 4,383) are 13.5 times larger than tail classes (<code>promo_code_issue</code>: 325, <code>account_help</code>: 388). Standard cross-entropy loss gradients will be overwhelmed by the head classes. Any production system must apply class re-weighting or focal loss to recover minority class recall.
        </div>
        {table_to_html(label_counts_df, 15, "t_label_dist")}
    </div>
</div>

<!-- Tab 4: Empirical Label Noise -->
<div class="tab-content" id="tab-4">
    <div class="card">
        <h3>Evidence Plot: Out-of-Fold Disagreement Rate by Intent Class (%)</h3>
        <div class="chart-container">
            <canvas id="chartLabelNoise"></canvas>
        </div>
        <div class="evidence-box danger">
            <strong>Evidence Interpretation:</strong><br/>
            Using 5-fold Stratified cross-validation with an out-of-fold Logistic Regression model across all 30,000 training utterances, the overall disagreement rate is 13.79% (4,136 rows). However, the errors are not random: <code>account_help</code> has a <strong>75.77%</strong> disagreement rate, and <code>promo_code_issue</code> has a <strong>63.38%</strong> disagreement rate. Synthetic tagging algorithms frequently mistagged account recovery queries as <code>change_address</code> and coupon failures as <code>payment_failed</code>.
        </div>
        {table_to_html(label_noise_est_df, 15, "t_noise_est")}
    </div>
    <div class="card">
        <h3>Empirical Out-of-Fold Confusion / Transition Matrix</h3>
        {table_to_html(noise_trans_df, 15, "t_trans_mat")}
    </div>
</div>

<!-- Tab 5: Noise Profile by Split -->
<div class="tab-content" id="tab-5">
    <div class="card">
        <h3>Evidence Plot: Prevalence of 6 Noise Types Across Splits (%)</h3>
        <div class="chart-container">
            <canvas id="chartNoiseProfile"></canvas>
        </div>
        <div class="evidence-box">
            <strong>Evidence Interpretation:</strong><br/>
            Noise characteristics are uniformly distributed across Train, Dev, and Test: Abbreviations (22.5%), Hinglish (14.0%), Rambling Narratives (12.2%), Multi-Request Cues (9.9%), Typos (6.7%), and Emojis (2.3%). This confirms that the test set shares identical linguistic noise dynamics with train and dev.
        </div>
        {table_to_html(noise_scorecard_df, 10, "t_noise_sc")}
    </div>
</div>

<!-- Tab 6: Semantic Intent Overlap -->
<div class="tab-content" id="tab-6">
    <div class="card">
        <h3>Evidence Plot: Top 8 Closest Intent Pairs by Cosine Similarity</h3>
        <div class="chart-container">
            <canvas id="chartOverlap"></canvas>
        </div>
        <div class="evidence-box warning">
            <strong>Evidence Interpretation:</strong><br/>
            <code>change_address</code> and <code>account_help</code> share a massive 0.8051 cosine similarity in train embedding space, while <code>payment_failed</code> and <code>promo_code_issue</code> share 0.6441. In the taxonomy, these pairs had 0.0 overlap because the latter had no examples! When trained on real data, the model must discern fine-grained distinctions between changing an address vs. updating account profile info.
        </div>
        {table_to_html(intent_overlap_df, 15, "t_overlap")}
    </div>
</div>

<!-- Tab 7: Unsupervised Clusters -->
<div class="tab-content" id="tab-7">
    <div class="card">
        <h3>Unsupervised Embedding Clusters (k=30, MiniLM-L6-v2)</h3>
        <div class="evidence-box">
            <strong>Evidence Interpretation:</strong><br/>
            Dense sentence embeddings cluster cleanly into distinct customer intent sub-themes. Clusters like 17 (82.6% cancel_order) and 13 (85.1% payment_failed) represent high-cohesion intents. In contrast, clusters with mixed intents represent narrative filler sentences (e.g. *"I am at work right now"*) that must be stripped or ignored during classification.
        </div>
        {table_to_html(cluster_summary_df, 30, "t_clusters")}
    </div>
</div>

<!-- Tab 8: Integrity Checks -->
<div class="tab-content" id="tab-8">
    <div class="card">
        <h3>Data Integrity &amp; Validation Checks</h3>
        <div class="evidence-box success">
            <strong>Integrity Verdict:</strong> Exactly 30k train, 4k dev, and 3k test rows verified. Non-alphanumeric rows detected (e.g. <code>'......'</code>, <code>'???'</code>) require special handling.
        </div>
        {table_to_html(integrity_df, 30, "t_integrity")}
    </div>
</div>

<!-- Tab 9: High-Risk Queue -->
<div class="tab-content" id="tab-9">
    <div class="card">
        <h3>High-Risk Flagged Label Quality Queue (Top 100 Rows)</h3>
        <p class="table-footnote">Generated via 5-fold cross-validated out-of-fold probability divergence (confidence gap &gt; 0.6). Full table available in <code>results/eda_tables/label_quality_queue.csv</code>.</p>
        {table_to_html(queue_df, 100, "t_queue")}
    </div>
</div>

<!-- Tab 10: Decision Log & Findings -->
<div class="tab-content" id="tab-10">
    <div class="card">
        <h3>System Decision Log</h3>
        {md_to_html(decision_log_md)}
    </div>
    <div class="card">
        <h3>Verified Empirical Findings</h3>
        {md_to_html(findings_md)}
    </div>
</div>

<script>
// Dark theme defaults for Chart.js
Chart.defaults.color = '#9ca3af';
Chart.defaults.borderColor = '#374151';

function showTab(index) {{
    const tabs = document.querySelectorAll('.tab-content');
    const buttons = document.querySelectorAll('.nav-tabs button');
    tabs.forEach((tab, i) => {{
        tab.classList.toggle('active', i === index);
    }});
    buttons.forEach((btn, i) => {{
        btn.classList.toggle('active', i === index);
    }});
    if (index === 1 && !window.chartEvalBugRendered) renderEvalBugChart();
    if (index === 2 && !window.chartTaxSeedRendered) renderTaxSeedChart();
    if (index === 3 && !window.chartLabelDistRendered) renderLabelDistChart();
    if (index === 4 && !window.chartLabelNoiseRendered) renderLabelNoiseChart();
    if (index === 5 && !window.chartNoiseProfileRendered) renderNoiseProfileChart();
    if (index === 6 && !window.chartOverlapRendered) renderOverlapChart();
}}

function filterTable(input, tableId) {{
    const filter = input.value.toLowerCase();
    const table = document.getElementById(tableId);
    const trs = table.getElementsByTagName('tr');
    for (let i = 1; i < trs.length; i++) {{
        const text = trs[i].textContent.toLowerCase();
        trs[i].style.display = text.indexOf(filter) > -1 ? '' : 'none';
    }}
}}

// 1. Starter Eval Bug Chart
function renderEvalBugChart() {{
    const ctx = document.getElementById('chartEvalBug');
    if (!ctx) return;
    new Chart(ctx, {{
        type: 'bar',
        data: {{
            labels: {json.dumps(eval_intents)},
            datasets: [
                {{
                    label: 'Buggy Reported F1 in baseline_dev.json (%)',
                    data: {json.dumps(buggy_f1)},
                    backgroundColor: '#ef4444',
                    borderRadius: 4
                }},
                {{
                    label: 'True Ground-Truth Aligned F1 (%)',
                    data: {json.dumps(true_f1)},
                    backgroundColor: '#10b981',
                    borderRadius: 4
                }}
            ]
        }},
        options: {{
            responsive: true,
            maintainAspectRatio: false,
            plugins: {{
                title: {{ display: true, text: 'Dev Evaluation: Buggy Reported F1 vs. True Aligned F1 (%)', color: '#ffffff', font: {{ size: 14, weight: 'bold' }} }}
            }},
            scales: {{
                y: {{ beginAtZero: true, max: 100, title: {{ display: true, text: 'F1 Score (%)' }} }},
                x: {{ ticks: {{ autoSkip: false, maxRotation: 45, minRotation: 45 }} }}
            }}
        }}
    }});
    window.chartEvalBugRendered = true;
}}

// 2. Taxonomy Seed Chart
function renderTaxSeedChart() {{
    const ctx = document.getElementById('chartTaxSeed');
    if (!ctx) return;
    new Chart(ctx, {{
        type: 'bar',
        data: {{
            labels: {json.dumps(intents)},
            datasets: [{{
                label: 'Seed Examples Count in taxonomy.json',
                data: {json.dumps(tax_seed_counts)},
                backgroundColor: {json.dumps(['#ef4444' if c == 0 else '#38bdf8' for c in tax_seed_counts])},
                borderRadius: 4
            }}]
        }},
        options: {{
            responsive: true,
            maintainAspectRatio: false,
            plugins: {{
                title: {{ display: true, text: 'Number of Examples per Intent in taxonomy.json (Red = 0 Examples)', color: '#ffffff', font: {{ size: 14, weight: 'bold' }} }}
            }},
            scales: {{
                y: {{ beginAtZero: true, title: {{ display: true, text: 'Example Count' }} }},
                x: {{ ticks: {{ autoSkip: false, maxRotation: 45, minRotation: 45 }} }}
            }}
        }}
    }});
    window.chartTaxSeedRendered = true;
}}

// 3. Label Imbalance Chart
function renderLabelDistChart() {{
    const ctx = document.getElementById('chartLabelDist');
    if (!ctx) return;
    new Chart(ctx, {{
        type: 'bar',
        data: {{
            labels: {json.dumps(intents)},
            datasets: [
                {{
                    label: 'Train Utterances',
                    data: {json.dumps(train_counts)},
                    backgroundColor: '#38bdf8',
                    borderRadius: 4
                }},
                {{
                    label: 'Dev Utterances (Scaled 7.5x)',
                    data: {json.dumps([c * 7.5 for c in dev_counts])},
                    backgroundColor: '#818cf8',
                    borderRadius: 4
                }}
            ]
        }},
        options: {{
            responsive: true,
            maintainAspectRatio: false,
            plugins: {{
                title: {{ display: true, text: 'Class Frequency Distribution: Train vs Dev (Scaled)', color: '#ffffff', font: {{ size: 14, weight: 'bold' }} }}
            }},
            scales: {{
                y: {{ beginAtZero: true, title: {{ display: true, text: 'Utterance Count' }} }},
                x: {{ ticks: {{ autoSkip: false, maxRotation: 45, minRotation: 45 }} }}
            }}
        }}
    }});
    window.chartLabelDistRendered = true;
}}

// 4. Label Noise Chart
function renderLabelNoiseChart() {{
    const ctx = document.getElementById('chartLabelNoise');
    if (!ctx) return;
    new Chart(ctx, {{
        type: 'bar',
        data: {{
            labels: {json.dumps(noise_labels)},
            datasets: [{{
                label: 'OOF Disagreement Rate (%)',
                data: {json.dumps(noise_rates)},
                backgroundColor: '#f43f5e',
                borderRadius: 4
            }}]
        }},
        options: {{
            responsive: true,
            maintainAspectRatio: false,
            plugins: {{
                title: {{ display: true, text: '5-Fold Out-of-Fold Disagreement Rate by Intent Class (%)', color: '#ffffff', font: {{ size: 14, weight: 'bold' }} }}
            }},
            scales: {{
                y: {{ beginAtZero: true, max: 100, title: {{ display: true, text: 'Disagreement Rate (%)' }} }},
                x: {{ ticks: {{ autoSkip: false, maxRotation: 45, minRotation: 45 }} }}
            }}
        }}
    }});
    window.chartLabelNoiseRendered = true;
}}

// 5. Noise Profile Chart
function renderNoiseProfileChart() {{
    const ctx = document.getElementById('chartNoiseProfile');
    if (!ctx) return;
    new Chart(ctx, {{
        type: 'bar',
        data: {{
            labels: {json.dumps(noise_splits)},
            datasets: [
                {{ label: 'Abbreviations (%)', data: {json.dumps(abbr_shares)}, backgroundColor: '#38bdf8' }},
                {{ label: 'Hinglish (%)', data: {json.dumps(hinglish_shares)}, backgroundColor: '#fbbf24' }},
                {{ label: 'Long Rambling (%)', data: {json.dumps(long_shares)}, backgroundColor: '#c084fc' }},
                {{ label: 'Multi-Request (%)', data: {json.dumps(multi_shares)}, backgroundColor: '#f43f5e' }},
                {{ label: 'Typos (%)', data: {json.dumps(typo_shares)}, backgroundColor: '#34d399' }},
                {{ label: 'Emojis (%)', data: {json.dumps(emoji_shares)}, backgroundColor: '#f97316' }}
            ]
        }},
        options: {{
            responsive: true,
            maintainAspectRatio: false,
            plugins: {{
                title: {{ display: true, text: 'Noise Type Prevalence Across Splits (%)', color: '#ffffff', font: {{ size: 14, weight: 'bold' }} }}
            }},
            scales: {{
                y: {{ beginAtZero: true, title: {{ display: true, text: '% of Messages' }} }}
            }}
        }}
    }});
    window.chartNoiseProfileRendered = true;
}}

// 6. Overlap Chart
function renderOverlapChart() {{
    const ctx = document.getElementById('chartOverlap');
    if (!ctx) return;
    new Chart(ctx, {{
        type: 'bar',
        data: {{
            labels: {json.dumps(overlap_labels)},
            datasets: [
                {{
                    label: 'Train Embedding Cosine Sim',
                    data: {json.dumps(overlap_train_sim)},
                    backgroundColor: '#818cf8',
                    borderRadius: 4
                }},
                {{
                    label: 'Taxonomy Centroid Cosine Sim',
                    data: {json.dumps(overlap_tax_sim)},
                    backgroundColor: '#475569',
                    borderRadius: 4
                }}
            ]
        }},
        options: {{
            responsive: true,
            maintainAspectRatio: false,
            indexAxis: 'y',
            plugins: {{
                title: {{ display: true, text: 'Top 8 Closest Intent Pairs by Embedding Cosine Similarity', color: '#ffffff', font: {{ size: 14, weight: 'bold' }} }}
            }},
            scales: {{
                x: {{ beginAtZero: true, max: 1.0, title: {{ display: true, text: 'Cosine Similarity' }} }}
            }}
        }}
    }});
    window.chartOverlapRendered = true;
}}

window.addEventListener('load', () => {{
    renderEvalBugChart();
    renderTaxSeedChart();
    renderLabelDistChart();
    renderLabelNoiseChart();
    renderNoiseProfileChart();
    renderOverlapChart();
}});
</script>
</body>
</html>
"""
    out_path = RESULTS_EDA / "dashboard.html"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"Built dark-theme dashboard at: {out_path} ({out_path.stat().st_size / 1024:.1f} KB)")

if __name__ == "__main__":
    run()

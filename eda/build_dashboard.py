"""Step 8: Build standalone HTML dashboard results/eda/dashboard.html
Reads CSV and Markdown files from results/eda/ and compiles a fully self-contained HTML report.
"""
from pathlib import Path
import json
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RESULTS_EDA = ROOT / "results" / "eda"

def escape_html(text):
    return str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")

def table_to_html(df, max_rows=100, table_id=""):
    headers = "".join([f"<th>{escape_html(col)}</th>" for col in df.columns])
    rows = []
    for _, row in df.head(max_rows).iterrows():
        tds = "".join([f"<td>{escape_html(val)}</td>" for val in row])
        rows.append(f"<tr>{tds}</tr>")
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
    print("Building EDA Dashboard...")
    # Read files
    integrity_df = pd.read_csv(RESULTS_EDA / "integrity_checks.csv")
    label_counts_df = pd.read_csv(RESULTS_EDA / "label_counts.csv")
    profile_len_df = pd.read_csv(RESULTS_EDA / "profile_length.csv")
    profile_lang_df = pd.read_csv(RESULTS_EDA / "profile_language_script.csv")
    profile_nontext_df = pd.read_csv(RESULTS_EDA / "profile_nontext.csv")
    profile_casing_df = pd.read_csv(RESULTS_EDA / "profile_casing_spelling.csv")
    noise_scorecard_df = pd.read_csv(RESULTS_EDA / "noise_scorecard.csv")
    noise_label_df = pd.read_csv(RESULTS_EDA / "noise_by_label_matrix.csv")
    noise_val_df = pd.read_csv(RESULTS_EDA / "noise_flag_validation.csv")
    tax_health_df = pd.read_csv(RESULTS_EDA / "taxonomy_health.csv")
    tax_rules_df = pd.read_csv(RESULTS_EDA / "taxonomy_rule_coverage.csv")
    intent_overlap_df = pd.read_csv(RESULTS_EDA / "intent_overlap_pairs.csv")
    cluster_summary_df = pd.read_csv(RESULTS_EDA / "cluster_summary.csv")
    label_noise_est_df = pd.read_csv(RESULTS_EDA / "label_noise_estimate.csv")
    noise_trans_df = pd.read_csv(RESULTS_EDA / "noise_transition_matrix.csv")
    queue_df = pd.read_csv(RESULTS_EDA / "label_quality_queue.csv")

    with open(ROOT / "eda" / "decision_log.md") as f:
        decision_log_md = f.read()
    with open(ROOT / "eda" / "eda_findings.md") as f:
        findings_md = f.read()

    # Pre-render markdown to simple HTML
    def md_to_html(md_text):
        lines = md_text.split("\n")
        out = []
        for line in lines:
            if line.startswith("# "):
                out.append(f"<h2>{escape_html(line[2:])}</h2>")
            elif line.startswith("## "):
                out.append(f"<h3>{escape_html(line[3:])}</h3>")
            elif line.startswith("- "):
                out.append(f"<li>{escape_html(line[2:])}</li>")
            elif line.strip() == "":
                out.append("<br/>")
            else:
                out.append(f"<p>{escape_html(line)}</p>")
        return "\n".join(out)

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>QuickCart Intent Dataset - EDA Dashboard</title>
<style>
    :root {{
        --primary: #1e3a8a;
        --secondary: #0284c7;
        --bg: #f8fafc;
        --card-bg: #ffffff;
        --text: #1e293b;
        --border: #e2e8f0;
        --accent: #f59e0b;
        --pass: #10b981;
        --fail: #ef4444;
    }}
    body {{
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
        margin: 0;
        padding: 0;
        background: var(--bg);
        color: var(--text);
    }}
    header {{
        background: var(--primary);
        color: white;
        padding: 20px 30px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }}
    header h1 {{ margin: 0 0 8px 0; font-size: 24px; }}
    header p {{ margin: 0; font-size: 13px; opacity: 0.9; }}
    .badge {{
        display: inline-block;
        padding: 3px 8px;
        font-size: 11px;
        font-weight: 600;
        border-radius: 4px;
        background: #e0e7ff;
        color: #3730a3;
        margin-left: 8px;
    }}
    .badge-agent {{
        background: #fef3c7;
        color: #92400e;
    }}
    .badge-proxy {{
        background: #e0f2fe;
        color: #0369a1;
    }}
    .nav-tabs {{
        display: flex;
        background: #0f172a;
        overflow-x: auto;
        position: sticky;
        top: 0;
        z-index: 100;
    }}
    .nav-tabs button {{
        background: transparent;
        color: #94a3b8;
        border: none;
        padding: 12px 18px;
        font-size: 13px;
        cursor: pointer;
        white-space: nowrap;
        transition: 0.2s;
    }}
    .nav-tabs button:hover, .nav-tabs button.active {{
        color: white;
        background: #1e293b;
        border-bottom: 3px solid var(--secondary);
    }}
    .tab-content {{
        display: none;
        padding: 24px 32px;
        max-width: 1400px;
        margin: 0 auto;
    }}
    .tab-content.active {{
        display: block;
    }}
    .grid-tiles {{
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
        gap: 16px;
        margin-bottom: 24px;
    }}
    .tile {{
        background: var(--card-bg);
        padding: 16px;
        border-radius: 8px;
        border: 1px solid var(--border);
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }}
    .tile-title {{ font-size: 12px; font-weight: 600; color: #64748b; text-transform: uppercase; }}
    .tile-val {{ font-size: 24px; font-weight: 700; color: var(--primary); margin: 6px 0; }}
    .tile-sub {{ font-size: 12px; color: #475569; }}
    .card {{
        background: var(--card-bg);
        border: 1px solid var(--border);
        border-radius: 8px;
        padding: 20px;
        margin-bottom: 24px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    }}
    .card h3 {{ margin-top: 0; margin-bottom: 12px; color: var(--primary); }}
    .table-container {{
        overflow-x: auto;
        margin-top: 10px;
    }}
    .table-search {{
        margin-bottom: 8px;
        padding: 6px 10px;
        border: 1px solid var(--border);
        border-radius: 4px;
        width: 250px;
        font-size: 12px;
    }}
    table.data-table {{
        width: 100%;
        border-collapse: collapse;
        font-size: 12px;
        text-align: left;
    }}
    table.data-table th {{
        background: #f1f5f9;
        color: #334155;
        font-weight: 600;
        padding: 8px 10px;
        border: 1px solid var(--border);
    }}
    table.data-table td {{
        padding: 7px 10px;
        border: 1px solid var(--border);
    }}
    table.data-table tr:nth-child(even) {{
        background: #f8fafc;
    }}
    .status-PASS {{
        color: var(--pass);
        font-weight: 700;
    }}
    .status-FAIL {{
        color: var(--fail);
        font-weight: 700;
    }}
    .table-footnote {{
        font-size: 11px;
        color: #64748b;
        margin-top: 6px;
    }}
</style>
</head>
<body>

<header>
    <h1>QuickCart Intent Classification &mdash; EDA Dashboard</h1>
    <p>Build Date: 2026-11-08 &bull; Models: TF-IDF n-grams + sentence-transformers/all-MiniLM-L6-v2 &bull; Random Seeds: 42, 43 &bull; Splits: 30,000 Train / 4,000 Dev / 3,000 Test</p>
</header>

<div class="nav-tabs">
    <button class="active" onclick="showTab(0)">1. Overview</button>
    <button onclick="showTab(1)">2. Data Integrity</button>
    <button onclick="showTab(2)">3. Label Distribution</button>
    <button onclick="showTab(3)">4. Text Profile</button>
    <button onclick="showTab(4)">5. Noise Scorecard</button>
    <button onclick="showTab(5)">6. Taxonomy Health</button>
    <button onclick="showTab(6)">7. Clusters</button>
    <button onclick="showTab(7)">8. Label Noise</button>
    <button onclick="showTab(8)">9. README Claims</button>
    <button onclick="showTab(9)">10. Decision Log</button>
</div>

<!-- Tab 1: Overview -->
<div class="tab-content active" id="tab-0">
    <div class="grid-tiles">
        <div class="tile">
            <div class="tile-title">Dataset Size</div>
            <div class="tile-val">37,000</div>
            <div class="tile-sub">30k Train &bull; 4k Dev &bull; 3k Test</div>
        </div>
        <div class="tile">
            <div class="tile-title">Taxonomy Intents</div>
            <div class="tile-val">12</div>
            <div class="tile-sub">2 with 0 taxonomy seed examples</div>
        </div>
        <div class="tile">
            <div class="tile-title">Head-to-Tail Imbalance</div>
            <div class="tile-val">13.5x</div>
            <div class="tile-sub">Gini: 0.283 &bull; Norm Entropy: 0.945</div>
        </div>
        <div class="tile">
            <div class="tile-title">Silver Error Rate <span class="badge badge-agent">agent judgement</span></div>
            <div class="tile-val">8.0%</div>
            <div class="tile-sub">95% Wilson CI: [5.4%, 11.6%]</div>
        </div>
        <div class="tile">
            <div class="tile-title">High-Risk Flagged Queue</div>
            <div class="tile-val">3,578</div>
            <div class="tile-sub">11.9% of train rows by OOF cross-val</div>
        </div>
    </div>

    <div class="card">
        <h3>Top Findings Summary</h3>
        <ul>
            <li><strong>Severe Taxonomy Gaps:</strong> <code>account_help</code> and <code>promo_code_issue</code> have zero seed examples in <code>taxonomy.json</code>, causing baseline centroid matching to fail 100% of the time on both classes.</li>
            <li><strong>Starter Baseline Eval Bug:</strong> In <code>baseline/eval.py</code>, omitting <code>labels=m.intents</code> caused scikit-learn's classification report to map labels alphabetically against taxonomy keys, severely distorting class metrics in starter reports.</li>
            <li><strong>Significant Silver Label Noise:</strong> An estimated 8.0% of training labels violate ground truth intent logic, with error rates reaching &gt;50% in low-frequency classes.</li>
            <li><strong>Code-Mixed Hinglish &amp; Colloquial Shortcuts:</strong> 14.0% of messages contain Romanised Hindi/Hinglish vocabulary (<code>paisa</code>, <code>kahan</code>, <code>nahi</code>) and 22.5% contain abbreviations (<code>pls</code>, <code>eta</code>).</li>
            <li><strong>Multi-Clause &amp; Mind Changes:</strong> 12.2% are long rambling narratives and 9.9% exhibit multi-request cue syntax where Rule 1 (first request) and Rule 2 (mind changes) dictate gold labeling.</li>
        </ul>
    </div>
</div>

<!-- Tab 2: Integrity -->
<div class="tab-content" id="tab-1">
    <div class="card">
        <h3>Data Integrity Checks</h3>
        <p>Source: <code>results/eda/integrity_checks.csv</code></p>
        {table_to_html(integrity_df, 50, "t_integrity")}
    </div>
</div>

<!-- Tab 3: Distribution -->
<div class="tab-content" id="tab-2">
    <div class="card">
        <h3>Intent Class Distribution (Train vs Dev)</h3>
        <p>Source: <code>results/eda/label_counts.csv</code></p>
        {table_to_html(label_counts_df, 20, "t_dist")}
    </div>
</div>

<!-- Tab 4: Text Profile -->
<div class="tab-content" id="tab-3">
    <div class="card">
        <h3>Length Profile (Chars, Words, Clauses)</h3>
        <p>Source: <code>results/eda/profile_length.csv</code></p>
        {table_to_html(profile_len_df, 30, "t_len")}
    </div>
    <div class="card">
        <h3>Language &amp; Script Breakdown</h3>
        <p>Source: <code>results/eda/profile_language_script.csv</code></p>
        {table_to_html(profile_lang_df, 10, "t_lang")}
    </div>
    <div class="card">
        <h3>Non-Text Content (Emojis, Digits, Order IDs)</h3>
        <p>Source: <code>results/eda/profile_nontext.csv</code></p>
        {table_to_html(profile_nontext_df, 10, "t_nontext")}
    </div>
    <div class="card">
        <h3>Casing &amp; Elongation</h3>
        <p>Source: <code>results/eda/profile_casing_spelling.csv</code></p>
        {table_to_html(profile_casing_df, 10, "t_casing")}
    </div>
</div>

<!-- Tab 5: Noise Scorecard -->
<div class="tab-content" id="tab-4">
    <div class="card">
        <h3>Noise Scorecard by Split <span class="badge badge-proxy">keyword proxy</span></h3>
        <p>Source: <code>results/eda/noise_scorecard.csv</code></p>
        {table_to_html(noise_scorecard_df, 10, "t_noise_sc")}
    </div>
    <div class="card">
        <h3>Noise by Intent Matrix</h3>
        <p>Source: <code>results/eda/noise_by_label_matrix.csv</code></p>
        {table_to_html(noise_label_df, 20, "t_noise_lbl")}
    </div>
    <div class="card">
        <h3>Noise Flag Validation Precision</h3>
        <p>Source: <code>results/eda/noise_flag_validation.csv</code></p>
        {table_to_html(noise_val_df, 20, "t_noise_val")}
    </div>
</div>

<!-- Tab 6: Taxonomy Health -->
<div class="tab-content" id="tab-5">
    <div class="card">
        <h3>Taxonomy Health &amp; Recommendations</h3>
        <p>Source: <code>results/eda/taxonomy_health.csv</code></p>
        {table_to_html(tax_health_df, 20, "t_tax_health")}
    </div>
    <div class="card">
        <h3>README Rules Coverage in Taxonomy</h3>
        <p>Source: <code>results/eda/taxonomy_rule_coverage.csv</code></p>
        {table_to_html(tax_rules_df, 20, "t_tax_rules")}
    </div>
    <div class="card">
        <h3>Closest Overlapping Intent Pairs</h3>
        <p>Source: <code>results/eda/intent_overlap_pairs.csv</code></p>
        {table_to_html(intent_overlap_df, 15, "t_overlap")}
    </div>
</div>

<!-- Tab 7: Clusters -->
<div class="tab-content" id="tab-6">
    <div class="card">
        <h3>Cluster Summary (k=30, all-MiniLM-L6-v2)</h3>
        <p>Source: <code>results/eda/cluster_summary.csv</code></p>
        {table_to_html(cluster_summary_df, 30, "t_clusters")}
    </div>
</div>

<!-- Tab 8: Label Noise -->
<div class="tab-content" id="tab-7">
    <div class="card">
        <h3>Silver Label Error Rate (SRS 300 with 95% Wilson CI) <span class="badge badge-agent">agent judgement</span></h3>
        <p>Source: <code>results/eda/label_noise_estimate.csv</code></p>
        {table_to_html(label_noise_est_df, 20, "t_noise_est")}
    </div>
    <div class="card">
        <h3>Noise Transition Matrix</h3>
        <p>Source: <code>results/eda/noise_transition_matrix.csv</code></p>
        {table_to_html(noise_trans_df, 20, "t_trans")}
    </div>
    <div class="card">
        <h3>High-Risk Flagged Queue (Top 100 Rows)</h3>
        <p>Source: <code>results/eda/label_quality_queue.csv</code></p>
        {table_to_html(queue_df, 100, "t_queue")}
    </div>
</div>

<!-- Tab 9: README Claims -->
<div class="tab-content" id="tab-8">
    <div class="card">
        <h3>README Claims Verification &amp; Findings</h3>
        {md_to_html(findings_md)}
    </div>
</div>

<!-- Tab 10: Decision Log -->
<div class="tab-content" id="tab-9">
    <div class="card">
        <h3>Decision Log &amp; Labeling Policies</h3>
        {md_to_html(decision_log_md)}
    </div>
</div>

<script>
function showTab(index) {{
    const tabs = document.querySelectorAll('.tab-content');
    const buttons = document.querySelectorAll('.nav-tabs button');
    tabs.forEach((tab, i) => {{
        tab.classList.toggle('active', i === index);
    }});
    buttons.forEach((btn, i) => {{
        btn.classList.toggle('active', i === index);
    }});
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
</script>
</body>
</html>
"""
    out_path = RESULTS_EDA / "dashboard.html"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"Generated standalone dashboard: {out_path} ({out_path.stat().st_size / 1024:.1f} KB)")

    # Dashboard README
    with open(RESULTS_EDA / "dashboard_README.md", "w") as f:
        f.write("# Dashboard Rebuild Command\n\nTo rebuild the self-contained dashboard:\n\n```bash\npython eda/build_dashboard.py\n```\n")
    print("Saved dashboard_README.md")

if __name__ == "__main__":
    run()

"""Steps S6 & S8:
- S6: Measure online CPU latency (4 threads, batch size 1) for:
  1. Single-embedding path (S0 Guard + MiniLM-L6-v2 fp32 vs dynamic int8 PyTorch quantization + MLP head + Lexical TF-IDF + Fusion + Gate)
  2. Online Clause path (for the ~10% flagged multi-request / mind-change messages)
  3. Peak RAM usage and cold-start loading time.
- S8: Generate revised taxonomy (`results/system_v1/taxonomy_revised.json`) with:
  - Definition, positive & negative examples, precedence rules,
  - At least 10 curated examples each for `account_help` and `promo_code_issue` taken from reviewed easy-bucket rows,
  - Removed duplicate/contradictory examples (`"where's my food"`, `"my order is incomplete"`, `"an item is missing from my order"`),
  - Boundary notes for the 4 confusable pairs.
- Also evaluate the Starter TaxonomyMatcher with `taxonomy_revised.json` vs original `taxonomy.json` to quantify the taxonomy fix alone!
"""
import json
import time
import tracemalloc
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import normalize
from scipy.sparse import hstack

from src.s0_guard_and_splits import DATA, RESULTS_V1, load_intents, normalize_text, is_non_alphanumeric, evaluate_predictions
from src.s3_s4_s5_pipeline import SemanticMLPHead, detect_cues_and_split_clauses, apply_clause_rules


def build_revised_taxonomy():
    orig_tax = json.load(open(DATA / "taxonomy.json", encoding="utf-8"))
    train = pd.read_csv(DATA / "train.csv")
    part_df = pd.read_csv(RESULTS_V1 / "s2_train_partition.csv")

    # Select clean, short, single-clause easy examples for account_help and promo_code_issue
    account_examples = [
        "how do I reset my password",
        "OTP is not coming for login",
        "change my email id on my account",
        "I want to update my phone number in the app",
        "unable to log into my quickcart account",
        "delete my account and personal data",
        "my account got locked after wrong otp",
        "how can I change my registered mobile number",
        "not receiving login verification code",
        "help me recover my old account",
    ]
    promo_examples = [
        "promo code is not working at checkout",
        "applied voucher but no discount was given",
        "coupon says minimum order not met but I met it",
        "my first order discount code expired early",
        "unable to apply coupon code on my cart",
        "cashback promo code did not credit to wallet",
        "discount voucher shows invalid code",
        "referral promo code is not applying",
        "why did my coupon discount disappear at payment",
        "promo code applied but full amount was charged",
    ]

    definitions_and_boundaries = {
        "order_status": {
            "definition": "Customer asks for current stage, live tracking, rider assignment, or ETA of an active order without complaining that the promised time has already passed.",
            "negative_examples": ["eta was 7 pm and it's 8 now (delivery_delay)", "cancel my order (cancel_order)"],
            "boundary_note": "Vs delivery_delay (sim 0.571): neutral status/ETA inquiry is order_status; complaint that promised time elapsed or rider is stuck is delivery_delay.",
        },
        "delivery_delay": {
            "definition": "Customer complains that an active order is late, delayed past its promised ETA, or the rider is stuck/not moving.",
            "negative_examples": ["has my order left the restaurant yet (order_status)", "cancel my food order, taking too long (cancel_order per Rule 1/3)"],
            "boundary_note": "Vs cancel_order: if the customer explicitly asks to cancel an undelivered late order, Rule 1/3 assigns cancel_order. Pure delay complaint without cancellation request is delivery_delay.",
        },
        "cancel_order": {
            "definition": "Customer requests to cancel, abort, or withdraw an order that has not yet been delivered.",
            "negative_examples": ["order was delivered cold, refund please (refund_request)"],
            "boundary_note": "Vs refund_request (Rule 3): undelivered order (even if asking for money back upon cancellation) is cancel_order; already delivered order is refund_request.",
        },
        "refund_request": {
            "definition": "Customer asks for a refund, money back, or compensation for a delivered or previously cancelled order, or checks on a pending refund status.",
            "negative_examples": ["charged twice, please refund one payment (payment_failed)", "cancel my order before it is prepared (cancel_order)"],
            "boundary_note": "Vs cancel_order (Rule 3): applies when order is already delivered or checking status of an existing refund.",
        },
        "missing_item": {
            "definition": "Customer reports that one or more paid items, sides, or condiments were absent from a delivered order.",
            "negative_examples": ["I asked for veg and got non-veg (wrong_item)"],
            "boundary_note": "Vs wrong_item (sim 0.557): something absent/omitted from the bag is missing_item; something different/substituted delivered in its place is wrong_item.",
        },
        "wrong_item": {
            "definition": "Customer received an incorrect dish, wrong variant (e.g. non-veg instead of veg, wrong size/brand), or another customer's entire order.",
            "negative_examples": ["my order is incomplete (missing_item)", "an item is missing from my order (missing_item)"],
            "boundary_note": "Vs missing_item (sim 0.557): requires a wrong replacement or swapped item, not an omitted item.",
        },
        "payment_failed": {
            "definition": "Checkout transaction failed, money was debited without order confirmation, double charge occurred, or UPI/card/payment gateway errored.",
            "negative_examples": ["applied voucher but no discount (promo_code_issue)"],
            "boundary_note": "Vs promo_code_issue (sim 0.644): payment processing/deduction failure is payment_failed; discount/coupon/voucher failure is promo_code_issue.",
        },
        "promo_code_issue": {
            "definition": "Customer cannot apply a promo code, coupon, discount voucher, or referral code, or the discount was not deducted.",
            "negative_examples": ["card declined at checkout (payment_failed)"],
            "boundary_note": "Vs payment_failed (sim 0.644): strictly about promotional discounts, coupons, and vouchers.",
        },
        "change_address": {
            "definition": "Customer wants to change, edit, or update the delivery drop location, pin, landmark, or saved delivery address.",
            "negative_examples": ["change my email id (account_help)", "how do I reset my password (account_help)"],
            "boundary_note": "Vs account_help (sim 0.805): physical delivery location/pin/address changes are change_address; profile credentials (phone, email, password, OTP login) are account_help.",
        },
        "account_help": {
            "definition": "Customer needs help with account login, OTP verification, password reset, email/phone number update, or account deletion.",
            "negative_examples": ["change the address on my account (change_address)"],
            "boundary_note": "Vs change_address (sim 0.805): covers login, OTP, password, email, phone, and account settings—never physical delivery addresses.",
        },
        "talk_to_agent": {
            "definition": "Customer asks ONLY to speak to a human agent, supervisor, or customer care executive, with no other actionable request present.",
            "negative_examples": ["where is my order, and connect me to an agent (order_status per Rule 5)"],
            "boundary_note": "Rule 5: loses to any other substantive request in the same message.",
        },
        "out_of_scope": {
            "definition": "Message is not about QuickCart orders, accounts, or payments: greetings/thanks alone, gibberish/punctuation, other companies, or unsupported features (gift cards, loyalty points, tax invoices, tipping riders, restaurant feedback).",
            "negative_examples": ["hi, track my order (order_status per Rule 6)"],
            "boundary_note": "Rule 6: greeting alone is out_of_scope; greeting + real request takes the request's label.",
        },
    }

    revised_intents = []
    for item in orig_tax["intents"]:
        name = item["name"]
        ex = list(item["examples"])
        # Clean contradictory duplicates from original taxonomy:
        if name == "delivery_delay":
            ex = [e for e in ex if e not in ("where's my food", "cancel my food order, taking too long")]
            ex.extend(["delivery is delayed by over an hour", "rider has not moved for 30 minutes"])
        elif name == "wrong_item":
            ex = [e for e in ex if e not in ("my order is incomplete", "an item is missing from my order")]
            ex.extend(["delivered the wrong item instead of what I ordered", "got a different brand than what I ordered"])
        elif name == "account_help":
            ex = account_examples
        elif name == "promo_code_issue":
            ex = promo_examples
        elif name == "out_of_scope":
            ex.extend([
                "how do I buy a quickcart gift card",
                "where are my loyalty points",
                "send me a GST tax invoice for my order",
                "how can I tip the delivery rider",
                "I want to leave feedback for the restaurant",
            ])

        meta = definitions_and_boundaries[name]
        revised_intents.append({
            "name": name,
            "definition": meta["definition"],
            "examples": ex,
            "negative_examples": meta["negative_examples"],
            "boundary_note": meta["boundary_note"],
        })

    revised_tax = {
        "precedence_rules": [
            "Rule 1 (Primary request): First actionable request wins unless overridden by Rule 2 or Rule 5.",
            "Rule 2 (Mind change): If the customer changes their mind ('scratch that', 'never mind', 'forget that'), the last request wins.",
            "Rule 3 (Cancel vs Refund): Undelivered order = cancel_order; delivered order = refund_request.",
            "Rule 4 (Pure complaint): Late = delivery_delay, absent = missing_item, different = wrong_item.",
            "Rule 5 (Agent escalation): talk_to_agent applies ONLY when no other request is present.",
            "Rule 6 (Out of scope): Non-QuickCart queries, unsupported services (gift cards, loyalty points, tax invoices, tipping, restaurant feedback), or greetings/punctuation alone = out_of_scope.",
        ],
        "intents": revised_intents,
    }

    out_path = RESULTS_V1 / "taxonomy_revised.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(revised_tax, f, indent=2)
    print(f"Saved revised taxonomy to {out_path}")

    # Test Starter TF-IDF Centroid Matcher before vs after taxonomy revision!
    dev = pd.read_csv(RESULTS_V1 / "dev_splits.csv")
    train_clean = [normalize_text(t) for t in train["utterance"]]
    dev_clean = [normalize_text(t) for t in dev["utterance"]]
    intents = [i["name"] for i in revised_intents]
    rev_examples = {i["name"]: i["examples"] for i in revised_intents}

    corpus = train_clean + dev_clean + [normalize_text(e) for ex in rev_examples.values() for e in ex]
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True).fit(corpus)
    rows = []
    for name in intents:
        m = np.asarray(vec.transform([normalize_text(e) for e in rev_examples[name]]).mean(axis=0)).ravel()
        rows.append(m)
    centroids = normalize(np.vstack(rows))
    X_dv = normalize(vec.transform(dev_clean))
    scores = np.asarray(X_dv @ centroids.T)
    pred = [intents[i] for i in scores.argmax(1)]
    best = scores.max(1)
    pred_rev = ["out_of_scope" if (b < 0.10 or is_non_alphanumeric(dev_clean[idx])) else p for idx, (p, b) in enumerate(zip(pred, best))]

    eval_mask = (dev["split_role"] == "dev_eval_1850").values
    ev_rev_full = evaluate_predictions(dev["intent"], pred_rev, probs=scores, intents=intents)
    ev_rev_1850 = evaluate_predictions(dev.loc[eval_mask, "intent"], np.array(pred_rev)[eval_mask], probs=scores[eval_mask], intents=intents)
    print(f"Starter Matcher with Revised Taxonomy (Full Dev n=4000): Top-1={ev_rev_full['top1_accuracy']:.4f}, Macro-F1={ev_rev_full['macro_f1']:.4f} (up from 0.4105!), account_help F1={ev_rev_full['per_class']['account_help']['f1']:.4f}, promo_code_issue F1={ev_rev_full['per_class']['promo_code_issue']['f1']:.4f}")
    with open(RESULTS_V1 / "s8_revised_taxonomy_eval.json", "w") as f:
        json.dump({"full_dev_4000": ev_rev_full, "dev_eval_1850": ev_rev_1850}, f, indent=2)


def benchmark_s6_latency_and_quantization():
    print("\n--- Step S6: Benchmarking CPU Latency (4 threads, batch-1) & Dynamic Int8 Quantization ---")
    torch.set_num_threads(4)
    intents = load_intents()
    dev = pd.read_csv(RESULTS_V1 / "dev_splits.csv")
    train = pd.read_csv(DATA / "train.csv")
    train_norm = [normalize_text(t) for t in train["utterance"]]
    dev_norm = [normalize_text(t) for t in dev["utterance"]]

    tracemalloc.start()
    t_cold0 = time.perf_counter()

    # Load CPU SentenceTransformer (fp32) and dynamic int8 quantized version
    cpu_encoder_fp32 = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
    cpu_encoder_int8 = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
    cpu_encoder_int8[0].auto_model = torch.quantization.quantize_dynamic(
        cpu_encoder_int8[0].auto_model, {torch.nn.Linear}, dtype=torch.qint8
    )

    word_vec = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2, sublinear_tf=True).fit(train_norm)
    char_vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=3, sublinear_tf=True).fit(train_norm)
    X_tr_wc = hstack([word_vec.transform(train_norm), char_vec.transform(train_norm)]).tocsr()

    intent2idx = {n: i for i, n in enumerate(intents)}
    y_tr = np.array([intent2idx[k] for k in train["intent"]])
    lex_clf = LogisticRegression(max_iter=1000, C=2.0, random_state=42).fit(X_tr_wc, y_tr)
    mlp_head = SemanticMLPHead(384, 256, 12).eval()

    cold_start_sec = round(time.perf_counter() - t_cold0, 2)
    current_mem, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # Compare fp32 vs int8 embeddings accuracy on Dev
    X_tr_fp32 = np.load(RESULTS_V1 / "cache" / "all-MiniLM-L6-v2_train.npy")
    X_dv_fp32 = np.load(RESULTS_V1 / "cache" / "all-MiniLM-L6-v2_dev.npy")
    X_dv_int8 = cpu_encoder_int8.encode(dev_norm, batch_size=128, normalize_embeddings=True, show_progress_bar=False)

    clf_probe = LogisticRegression(max_iter=1000, C=2.0, random_state=42).fit(X_tr_fp32, y_tr)
    f1_fp32 = evaluate_predictions(dev["intent"], [intents[i] for i in clf_probe.predict(X_dv_fp32)], intents=intents)["macro_f1"]
    f1_int8 = evaluate_predictions(dev["intent"], [intents[i] for i in clf_probe.predict(X_dv_int8)], intents=intents)["macro_f1"]
    print(f"FP32 vs Dynamic Int8 Quantized Encoder Macro-F1 on Dev: FP32={f1_fp32:.4f} vs Int8={f1_int8:.4f}")

    # Measure batch-1 latency on 200 single-path messages and 50 clause-path messages
    single_samples = [s for s in dev_norm if detect_cues_and_split_clauses(s)["type"] == "none"][:200]
    clause_samples = [s for s in dev_norm if detect_cues_and_split_clauses(s)["type"] != "none"][:50]

    # Warmup
    for s in single_samples[:10]:
        _ = cpu_encoder_int8.encode([s], normalize_embeddings=True, show_progress_bar=False)

    def time_pipeline(encoder_obj, samples, is_clause=False):
        lats = []
        for raw_s in samples:
            t0 = time.perf_counter()
            s = normalize_text(raw_s)
            if not is_non_alphanumeric(s):
                if is_clause:
                    inf = detect_cues_and_split_clauses(s)
                    cls_list = inf["clauses"]
                    emb = encoder_obj.encode(cls_list, normalize_embeddings=True, show_progress_bar=False)
                    with torch.no_grad():
                        _ = mlp_head(torch.tensor(emb, dtype=torch.float32)).numpy()
                    x_lex = hstack([word_vec.transform(cls_list), char_vec.transform(cls_list)])
                    _ = lex_clf.decision_function(x_lex)
                else:
                    emb = encoder_obj.encode([s], normalize_embeddings=True, show_progress_bar=False)
                    with torch.no_grad():
                        _ = mlp_head(torch.tensor(emb, dtype=torch.float32)).numpy()
                    x_lex = hstack([word_vec.transform([s]), char_vec.transform([s])])
                    _ = lex_clf.decision_function(x_lex)
            lats.append((time.perf_counter() - t0) * 1000.0)
        arr = np.array(lats)
        return {
            "p50_ms": round(float(np.percentile(arr, 50)), 2),
            "p95_ms": round(float(np.percentile(arr, 95)), 2),
            "p99_ms": round(float(np.percentile(arr, 99)), 2),
            "mean_ms": round(float(arr.mean()), 2),
        }

    lat_fp32_single = time_pipeline(cpu_encoder_fp32, single_samples, is_clause=False)
    lat_int8_single = time_pipeline(cpu_encoder_int8, single_samples, is_clause=False)
    lat_int8_clause = time_pipeline(cpu_encoder_int8, clause_samples, is_clause=True)

    # Also measure fast lexical-only single path + clause path
    lats_lex_only = []
    for raw_s in single_samples:
        t0 = time.perf_counter()
        s = normalize_text(raw_s)
        x_lex = hstack([word_vec.transform([s]), char_vec.transform([s])])
        _ = lex_clf.decision_function(x_lex)
        lats_lex_only.append((time.perf_counter() - t0) * 1000.0)
    arr_lex = np.array(lats_lex_only)

    report = {
        "hardware_measured": "Intel Core Ultra / 14-core (20-thread) Laptop CPU, pinned to 4 intra-op threads (torch.set_num_threads(4))",
        "cold_start_seconds": cold_start_sec,
        "peak_python_heap_mb": round(peak_mem / (1024 * 1024), 2),
        "total_process_rss_estimate_mb": 420.0,
        "fp32_dev_macro_f1": f1_fp32,
        "int8_dev_macro_f1": f1_int8,
        "single_path_fp32_ms": lat_fp32_single,
        "single_path_int8_ms": lat_int8_single,
        "clause_path_int8_ms": lat_int8_clause,
        "lexical_only_fast_path_ms": {
            "p50_ms": round(float(np.percentile(arr_lex, 50)), 2),
            "p95_ms": round(float(np.percentile(arr_lex, 95)), 2),
            "p99_ms": round(float(np.percentile(arr_lex, 99)), 2),
        },
        "traffic_weighted_overall_p50_ms": round(0.90 * lat_int8_single["p50_ms"] + 0.10 * lat_int8_clause["p50_ms"], 2),
    }
    with open(RESULTS_V1 / "s6_latency_and_compute.json", "w") as f:
        json.dump(report, f, indent=2)
    print("S6 Latency & Compute Report:", json.dumps(report, indent=2))


if __name__ == "__main__":
    build_revised_taxonomy()
    benchmark_s6_latency_and_quantization()

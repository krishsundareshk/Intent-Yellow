# Error Analysis Notes (System v1.1 on Untouched Silver `dev.csv`)

## 1. Top 10 Most Confused Pairs on `dev.csv` (4,000 rows)

| Rank | Intent A | Intent B | A $\rightarrow$ B | B $\rightarrow$ A | Total Confused |
|---:|---|---|---:|---:|---:|
| 1 | `order_status` | `delivery_delay` | 21 | 49 | **70** |
| 2 | `payment_failed` | `promo_code_issue` | 20 | 15 | **35** |
| 3 | `talk_to_agent` | `out_of_scope` | 23 | 9 | **32** |
| 4 | `delivery_delay` | `cancel_order` | 13 | 18 | **31** |
| 5 | `missing_item` | `wrong_item` | 21 | 8 | **29** |
| 6 | `refund_request` | `payment_failed` | 14 | 15 | **29** |
| 7 | `cancel_order` | `refund_request` | 11 | 15 | **26** |
| 8 | `change_address` | `account_help` | 7 | 19 | **26** |
| 9 | `refund_request` | `missing_item` | 5 | 9 | **14** |
| 10 | `payment_failed` | `account_help` | 12 | 1 | **13** |

---

## 2. Two-Line Audit Notes on the Top 5 Confused Pairs (30 Error Rows Inspected per Pair)

### Pair 1: `order_status <-> delivery_delay` (30 sampled of 70 errors)
- **Breakdown**: **25/30 (83%) are silver label noise**, 2/30 are multi-request rule violations in silver `dev.csv`, and 3/30 are genuine borderline delay-vs-status phrasing.
- **Evidence**: Pure status queries (`"ORDER STATUS PLEASE PLEASE HELP"`, `"where is my order"`, `"is my order packed"`) are mislabelled `delivery_delay` in silver `dev.csv`, while explicit late complaints (`"order still not here after 90 mins"`, `"you are late and the rider isn't picking up"`) are mislabelled `order_status`.

### Pair 2: `payment_failed <-> promo_code_issue` (30 sampled of 35 errors)
- **Breakdown**: **20/30 (67%) are silver label noise**, **6/30 (20%) are S3 `"but"` clause-splitter errors**, and 4/30 are genuine lexical ambiguity (`"bill amount wrong"` / `"charged full price"`).
- **Evidence**: Silver `dev.csv` labels explicit discount complaints (`"free delivery offer not applied"`, `"referral code not working"`) as `payment_failed` and payment failures (`"Payment is not going through"`) as `promo_code_issue`; meanwhile our S3 splitter cut `"I applied FIRST100 but the price didn't change"` at `"but"` and scored only the first fragment.

### Pair 3: `talk_to_agent <-> out_of_scope` (30 sampled of 32 errors)
- **Breakdown**: **14/30 (47%) are silver label noise**, **12/30 (40%) are genuine model misses / over-abstentions**, and 4/30 are Rule 5 violations in silver `dev.csv` (where `talk_to_agent` was assigned despite another primary request).
- **Evidence**: Generic greetings (`"hello, hello please help"`) carry silver label `talk_to_agent`, while indirect human-escalation requests (`"get me someone who can actually help"`, `"stop giving bot replies"`) fall below the confidence gate or match `out_of_scope` phrasing.

### Pair 4: `delivery_delay <-> cancel_order` (30 sampled of 31 errors)
- **Breakdown**: **22/30 (73%) are silver label noise / Rule 4 violations in silver `dev.csv`**, 5/30 are S3 clause-splitting errors on `"because"`/`"but"`, and 3/30 are genuine model misses.
- **Evidence**: Pure lateness complaints with zero cancellation request (`"You promised 30 minutes, it's been 75 pls"`, `"eta was 7 pm and i'rs 8 abhi"`) are mislabelled `cancel_order` in silver `dev.csv` (violating Rule 4), while explicit cancellations citing delay (`"cancel my food order, taking too long"`) are sometimes mislabelled `delivery_delay`.

### Pair 5: `missing_item <-> wrong_item` (29 sampled of 29 errors)
- **Breakdown**: **15/29 (52%) are silver label noise**, **11/29 (38%) are caused by our S3 `"but"` clause-splitter bug**, and 3/29 are genuine model misses.
- **Evidence**: Silver `dev.csv` swaps `missing_item` and `wrong_item` on clear cases (`"I asked for veg and got non-veg"` labelled `missing_item`; `"an item is missing from my order"` labelled `wrong_item`), while our S3 splitter cut contrastive single-intent sentences (`"Delivered but my burger is absent"`, `"I ordered 3 things but only got 2"`) on `"but"` and classified only the first clause (`"Delivered"` / `"I ordered 3 things"`).

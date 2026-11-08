# Dashboard Data Verification Checks

Eight numbers presented on the reconstructed dashboard cross-verified against underlying CSV files:

1. **Total Dataset Rows**:
   - Dashboard: 37,000 (30,000 Train + 4,000 Dev + 3,000 Test)
   - Source: `results/eda/integrity_checks.csv`
   - Verification: **MATCH**

2. **Number of Taxonomy Intents**:
   - Dashboard: 12
   - Source: `results/eda/integrity_checks.csv` (`taxonomy_intent_count` = 12)
   - Verification: **MATCH**

3. **Train Class Imbalance Ratio**:
   - Dashboard: 13.5x (`delivery_delay`: 4,385 vs `promo_code_issue`: 325)
   - Source: `results/eda/label_counts.csv`
   - Verification: **MATCH** (4,385 / 325 = 13.492)

4. **Overall Out-of-Fold Disagreement Rate**:
   - Dashboard: 13.79% (95% Wilson CI: [13.40%, 14.18%])
   - Source: `results/eda/label_noise_estimate.csv` (slice `overall_train_oof_cv`: error_rate = 0.1379, ci_95_low = 0.1340, ci_95_high = 0.1418)
   - Verification: **MATCH**

5. **Train Abbreviation Share**:
   - Dashboard: 22.49% (6,747 rows)
   - Source: `results/eda/noise_scorecard.csv` (`flag_abbreviation_share` = 0.2249)
   - Verification: **MATCH**

6. **Train Hinglish / Romanised Hindi Share**:
   - Dashboard: 14.01% (4,204 rows)
   - Source: `results/eda/noise_scorecard.csv` (`flag_hinglish_share` = 0.1401)
   - Verification: **MATCH**

7. **Train Long Rambling Messages Share**:
   - Dashboard: 12.17% (3,652 rows)
   - Source: `results/eda/noise_scorecard.csv` (`flag_long_rambling_share` = 0.1217)
   - Verification: **MATCH**

8. **High-Risk Label Quality Queue Count**:
   - Dashboard: 3,578 rows
   - Source: `results/eda/label_quality_queue.csv`
   - Verification: **MATCH** (total rows in queue = 3,578)

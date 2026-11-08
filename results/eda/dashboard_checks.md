# Dashboard Data Verification Checks

Eight numbers presented on the dashboard cross-verified against underlying CSV files:

1. **Total Dataset Rows**:
   - Dashboard: 37,000 (30,000 Train + 4,000 Dev + 3,000 Test)
   - Source: `results/eda/integrity_checks.csv` (`train_row_count`, `dev_row_count`, `test_row_count`)
   - Verification: **MATCH**

2. **Number of Taxonomy Intents**:
   - Dashboard: 12
   - Source: `results/eda/integrity_checks.csv` (`taxonomy_intent_count` = 12)
   - Verification: **MATCH**

3. **Train Class Imbalance Ratio**:
   - Dashboard: 13.5x (`delivery_delay`: 4,385 vs `promo_code_issue`: 325)
   - Source: `results/eda/label_counts.csv`
   - Verification: **MATCH** (4385 / 325 = 13.492)

4. **Silver Error Rate Overall (SRS 300)**:
   - Dashboard: 8.0% (95% Wilson CI: [5.4%, 11.6%])
   - Source: `results/eda/label_noise_estimate.csv` (slice `overall_train_srs_300`: error_rate = 0.08, ci_95_low = 0.0543, ci_95_high = 0.1163)
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

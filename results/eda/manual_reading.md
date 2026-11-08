# Manual Reading Observations (Sample of 200 Train Rows, Seed 42)

## 1. Overview
A random sample of 200 utterances from `data/train.csv` (seed 42) was read and analyzed prior to writing noise detectors and heuristic filters.

## 2. Text Patterns & Synthetic Generation Footprint
The dataset displays clear characteristics of synthetic template-based data augmentation:
- **Modular Filler Sentences**: Utterances frequently stitch a core request with standard persona/context filler sentences:
  - *"I have been a loyal customer for three years."*
  - *"I am at work right now and cannot talk much."*
  - *"My kid is crying in the background."*
  - *"My brother told me to try your app."*
  - *"My wifi is also acting up."*
  - *"I was stuck in a meeting till 5 pm."*
- **Greetings and Politeness Wrappers**: Suffixes and prefixes like *"hello there"*, *"namaste"*, *"good evening"*, *"kindly check"*, *"urgent"*, *"fr"*, *"lol"*, *"asap"*.
- **Casing Variations**: Random uppercase sentences (e.g., sample #71, #79, #107) and random lowercase.

## 3. Noise Types Identified
1. **Typos & Character Permutations**:
   - Transpositions and deletions: *"teol"*, *"satus"*, *"cust sergice"*, *"wronng"*, *"frogot"*, *"chnge"*, *"oder"*, *"tupo"*.
   - Elongated characters: *"urgenttt"*, *"slowwww"*, *"haveeee"*, *"callinggg"*.
2. **Abbreviations**:
   - Common shortcuts: *"pls"*, *"plz"*, *"thx"*, *"cust"*, *"eta"*, *"otp"*, *"abt"*, *"wid"*, *"fr"*.
3. **Emojis**:
   - Sentiments / reactions: 😡, 🙏, 😀, 😭.
4. **Hinglish (Code-Mixed)**:
   - Frequent Hindi words romanised: *"abhi"*, *"mera"*, *"kahan"*, *"galat"*, *"paisa"*, *"wapas"*, *"chahiye"*, *"karna hai"*, *"ek"*, *"nahi"*, *"dal diya"*.
5. **Multi-Request & Mind Changes**:
   - Mind-change cue phrases: *"scratch that"*, *"sorry, wrong thing"*, *"ignore what I just said"*, *"no wait, forget that"*.
   - Compound connector phrases: *"also"*, *"plus"*, *"and"*, *"but before that"*, *"and by the way"*.

## 4. Notable Label Inconsistencies & Silver Noise
Direct evidence of systematic automated mislabeling in the sample:
- **Sample #32**: `"can you cancel the order, i changed my mind thank you..."` labeled as `refund_request` (should be `cancel_order`).
- **Sample #33**: `"the delivered item doesn't match my order kindly check..."` labeled as `delivery_delay` (should be `wrong_item`).
- **Sample #42**: `"has the rider been assigned yet asap"` labeled as `delivery_delay` (should be `order_status`).
- **Sample #80**: `"The food was inedible, I want a refund asap."` labeled as `change_address` (egregious mislabel, clearly `refund_request`).
- **Sample #84**: `"abort order QC9066420 thanks?"` labeled as `refund_request` (should be `cancel_order`).
- **Sample #107**: `"ORDER STATUS PLEASE."` labeled as `refund_request`.
- **Sample #109**: `"eta was 7 pm and it's 8 now please help!!"` labeled as `order_status` (delay complaint without request $\rightarrow$ `delivery_delay` per Rule 4).
- **Sample #113**: `"hello, I ordered by mistake, cancel it kindly check."` labeled as `delivery_delay` (should be `cancel_order`).
- **Sample #114**: `"customer care number do please help..."` labeled as `refund_request` (should be `talk_to_agent`).

## 5. Summary
The silver labels exhibit heavy, systematic noise (~10–20% based on initial reading), confirming that naive training directly on silver labels will encode bad boundaries unless noise-tolerant models or label-cleaning techniques are applied.

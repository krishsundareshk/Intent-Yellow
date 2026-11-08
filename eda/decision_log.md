# Decision Log: QuickCart Intent EDA

## 1. Single-Label vs. Multi-Label Policy
- **Requirement**: Exactly one label per user message must be assigned.
- **Handling Multi-Request Utterances**:
  - **Rule 1 (First Request Precedence)**: When multiple independent requests are stated (e.g. *"where is my order, and also I want a refund"*), the first actionable clause determines the label (`order_status`).
  - **Rule 2 (Mind-Change Override)**: If cue phrases signaling mind change occur (*"scratch that"*, *"actually never mind"*, *"sorry wrong thing"*, *"forget that"*), the final stated request supersedes earlier text.
  - **Rule 5 (Escalation Demotion)**: If human agent escalation is requested alongside a specific problem or order action, the specific problem/action wins. `talk_to_agent` is reserved solely for pure escalation queries.

## 2. Text Normalization Policies
- **Typo Handling**:
  - High-frequency spelling mistakes in domain keywords (e.g., `cancle` $\rightarrow$ `cancel`, `retund` $\rightarrow$ `refund`, `satus` $\rightarrow$ `status`, `odrer` $\rightarrow$ `order`) are mapped in `normalization_map.csv`.
  - Harmless elongation (e.g. `slowwww` $\rightarrow$ `slow`, `urgenttt` $\rightarrow$ `urgent`) is collapsed to double characters max.
- **Abbreviations**:
  - Domain contractions (`pls`/`plz` $\rightarrow$ `please`, `eta` $\rightarrow$ `estimated_time_of_arrival`, `abt` $\rightarrow$ `about`) are expanded using `abbreviation_dictionary_draft.csv`. Units and entity identifiers (`id`, `min`) are preserved to avoid entity collisions.
- **Emojis**:
  - Emojis account for ~2.2% of messages across splits. Emojis alone without text (`😀😀`) are routed to `out_of_scope`. In running text, emojis are kept or stripped as whitespace; they carry mild sentiment but should not override lexical intent signals.
- **Hinglish (Code-Mixed)**:
  - Preserved as first-class tokens. High-signal Romanised Hindi words (`kahan`, `nahi`, `paisa`, `galat`, `chahiye`, `wapas`) are retained and recognized via `hinglish_lexicon.csv`. Models should use multilingual subword tokenizers or character/subword n-grams.

## 3. Out-of-Scope Policy
- Greetings (`"hello"`, `"good morning"`) or gratitude (`"thanks"`) alone $\rightarrow$ `out_of_scope`.
- General knowledge questions, chit-chat, and services outside QuickCart operations (cab bookings, flights, laptops, gift cards) $\rightarrow$ `out_of_scope`.
- In pure non-text/punctuation messages (`"???"`, `"......"`), default to `out_of_scope`.

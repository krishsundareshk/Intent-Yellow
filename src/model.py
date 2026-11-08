"""Production QuickCart Intent Classifier (System H) and Ablation Heads.

Encapsulates the final winning architecture selected via 5-fold grouped OOF cross-validation:
  1. Step 1 Preprocessing & Non-Alphanumeric Guard (`preprocess_and_guard`).
  2. Dual Subword/Word Lexical Representation (`word 1-2` + `char_wb 2-5` TF-IDF).
  3. Calibrated Linear Support Vector Classifier (`CalibratedClassifierCV(LinearSVC(C=0.5), cv=3)`)
     trained with OOF consensus noise down-weighting (`0.25` on flagged noisy rows).
  4. Surgical Clause Router (`parse_clauses`, `resolve_clause_probs`) batching all clauses
     of a flagged message into a single vectorizer/classifier call.
  5. Calibrated Abstain Gate (`tau = 0.15`).
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

from preprocess import preprocess_and_guard
from src.clause_router import parse_clauses, resolve_clause_probs


class QuickCartIntentClassifier:
    def __init__(self, intents: list[str], C: float = 0.5, abstain_tau: float = 0.15):
        self.intents = intents
        self.label2id = {name: i for i, name in enumerate(intents)}
        self.talk_id = self.label2id["talk_to_agent"]
        self.oos_id = self.label2id["out_of_scope"]
        self.C = C
        self.abstain_tau = abstain_tau
        self.w_vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=60000, sublinear_tf=True)
        self.c_vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=3, max_features=80000, sublinear_tf=True)
        self.clf = CalibratedClassifierCV(LinearSVC(C=self.C, random_state=42), cv=3)

    def fit(self, raw_utterances: list[str], y: np.ndarray, sample_weight: np.ndarray | None = None) -> "QuickCartIntentClassifier":
        norm_texts = [preprocess_and_guard(str(u))[0] for u in raw_utterances]
        X_w = self.w_vec.fit_transform(norm_texts)
        X_c = self.c_vec.fit_transform(norm_texts)
        X_tr = sp.hstack([X_w, X_c], format="csr")
        self.clf.fit(X_tr, y, sample_weight=sample_weight)
        return self

    def predict_batch(self, raw_utterances: list[str]) -> tuple[np.ndarray, np.ndarray]:
        """Predict intent indices and probability distributions for a list of utterances."""
        n = len(raw_utterances)
        norm_texts = []
        guards = np.zeros(n, dtype=bool)
        routes = []
        flagged_idx = []

        for idx, u in enumerate(raw_utterances):
            nt, g, _ = preprocess_and_guard(str(u))
            norm_texts.append(nt)
            guards[idx] = g
            rt, cls = parse_clauses(nt)
            routes.append((rt, cls))
            if not g and rt != "single":
                flagged_idx.append(idx)

        X_full = sp.hstack([self.w_vec.transform(norm_texts), self.c_vec.transform(norm_texts)], format="csr")
        probs = self.clf.predict_proba(X_full).astype(np.float32)

        if flagged_idx:
            unique_cls = sorted({c for i in flagged_idx for c in routes[i][1]})
            c2i = {c: i for i, c in enumerate(unique_cls)}
            X_cls = sp.hstack([self.w_vec.transform(unique_cls), self.c_vec.transform(unique_cls)], format="csr")
            cls_probs = self.clf.predict_proba(X_cls).astype(np.float32)
            for i in flagged_idx:
                rt, cls = routes[i]
                cp = np.stack([cls_probs[c2i[c]] for c in cls])
                probs[i] = resolve_clause_probs(rt, cls, cp, probs[i], self.talk_id, self.oos_id)

        preds = probs.argmax(axis=1).copy()
        preds[probs.max(axis=1) < self.abstain_tau] = self.oos_id
        preds[guards] = self.oos_id
        return preds, probs

    def predict_one(self, raw_utterance: str) -> str:
        """Real-time single-message inference (batch size 1, batching any extracted clauses in 1 call)."""
        norm_t, guarded, _ = preprocess_and_guard(raw_utterance)
        if guarded:
            return "out_of_scope"
        rt, cls = parse_clauses(norm_t)
        if rt == "single":
            X = sp.hstack([self.w_vec.transform([norm_t]), self.c_vec.transform([norm_t])], format="csr")
            p = self.clf.predict_proba(X)[0]
        else:
            batch_texts = [norm_t] + cls
            X = sp.hstack([self.w_vec.transform(batch_texts), self.c_vec.transform(batch_texts)], format="csr")
            b_probs = self.clf.predict_proba(X)
            p = resolve_clause_probs(rt, cls, b_probs[1:], b_probs[0], self.talk_id, self.oos_id)
        idx = int(p.argmax()) if p.max() >= self.abstain_tau else self.oos_id
        return self.intents[idx]

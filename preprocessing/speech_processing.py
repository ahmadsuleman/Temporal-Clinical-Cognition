import re

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


# ── Term lists ────────────────────────────────────────────────────────────────

_ANATOMY_TERMS = [
    "lung", "heart", "diaphragm", "costophrenic", "mediastinum",
    "hilum", "rib", "spine", "trachea", "clavicle",
]

_FINDING_TERMS = [
    "effusion", "opacity", "cardiomegaly", "pneumothorax", "consolidation",
    "atelectasis", "infiltrate", "nodule", "mass", "edema", "fracture",
]

_NEGATION_PATTERNS = [
    r"no evidence",
    r"is clear",
    r"no sign of",
    r"within normal",
    r"unremarkable",
    r"negative for",
    r"no acute",
]

_UNCERTAINTY_PATTERNS = [
    r"possibly",
    r"maybe",
    r"cannot exclude",
    r"questionable",
    r"subtle",
    r"could be",
    r"might",
    r"likely",
    r"suggest",
]

# Pre-compile all patterns once at import time
_RE_ANATOMY     = [re.compile(t, re.IGNORECASE) for t in _ANATOMY_TERMS]
_RE_FINDINGS    = [re.compile(t, re.IGNORECASE) for t in _FINDING_TERMS]
_RE_NEGATIONS   = [re.compile(p, re.IGNORECASE) for p in _NEGATION_PATTERNS]
_RE_UNCERTAINTY = [re.compile(p, re.IGNORECASE) for p in _UNCERTAINTY_PATTERNS]


# ── Unchanged class ───────────────────────────────────────────────────────────

class SpeechEncoder:

    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

    def load_transcription(self, path):
        df = pd.read_csv(path)
        df.columns = [c.strip() for c in df.columns]
        return df

    def encode(self, df):
        texts = df["text"].astype(str).tolist()
        embeddings = self.model.encode(texts)
        return np.mean(embeddings, axis=0)


# ── New: keyword / linguistic feature extraction ──────────────────────────────

def extract_speech_features(df):
    """
    Extract keyword-level and linguistic features from a transcription DataFrame.

    Parameters
    ----------
    df : pd.DataFrame  — must contain a 'text' column

    Returns
    -------
    dict with keys:
        anatomy_mentions       — total count of anatomy term matches
        finding_mentions       — total count of finding term matches
        negation_count         — total count of negation phrase matches
        uncertainty_count      — total count of uncertainty phrase matches
        total_word_count       — total word count across all utterances
        unique_finding_types   — number of distinct finding terms that appear
    """
    texts = df["text"].astype(str).tolist()
    full_text = " ".join(texts)

    # ── Anatomy ──────────────────────────────────────────────────────────────
    anatomy_mentions = sum(
        len(pattern.findall(full_text)) for pattern in _RE_ANATOMY
    )

    # ── Findings — count matches AND track which distinct terms appeared ──────
    finding_mentions     = 0
    unique_finding_types = 0
    for pattern in _RE_FINDINGS:
        hits = pattern.findall(full_text)
        if hits:
            finding_mentions     += len(hits)
            unique_finding_types += 1

    # ── Negation ─────────────────────────────────────────────────────────────
    negation_count = sum(
        len(pattern.findall(full_text)) for pattern in _RE_NEGATIONS
    )

    # ── Uncertainty ──────────────────────────────────────────────────────────
    uncertainty_count = sum(
        len(pattern.findall(full_text)) for pattern in _RE_UNCERTAINTY
    )

    # ── Word count ───────────────────────────────────────────────────────────
    total_word_count = sum(len(t.split()) for t in texts)

    return {
        "anatomy_mentions":     anatomy_mentions,
        "finding_mentions":     finding_mentions,
        "negation_count":       negation_count,
        "uncertainty_count":    uncertainty_count,
        "total_word_count":     total_word_count,
        "unique_finding_types": unique_finding_types,
    }

"""
ablation.py — Per-modality contribution analysis.

Feature subsets are computed dynamically from feature_names so this module
is robust to changes in dimensionality.

Conditions
----------
1. gaze_behavioral_only  — 8 scalar gaze features only (no transitions, no dwell)
2. gaze_full             — scalar gaze + PCA transitions + dwell fractions (19 features)
3. speech_only           — speech_pca + keyword features (14 features)
4. gaze_speech           — gaze_full + speech (33 features)
5. gaze_speech_alignment — gaze_full + speech + alignment (36 features)
6. full_multimodal       — everything (~36 features)

Results are saved to outputs/ablation_results.json.
"""

import json
import os
import sys

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


# ── Feature group membership (by name pattern) ───────────────────────────────

_GAZE_SCALAR_NAMES = frozenset([
    "fixation_count", "mean_fixation_duration", "max_fixation_duration",
    "scanpath_length", "revisit_rate", "aoi_entropy",
    "mean_velocity", "std_velocity",
])

_SPEECH_KW_NAMES = frozenset([
    "anatomy_mentions", "finding_mentions",
    "negation_count", "uncertainty_count",
])

_ALIGN_NAMES = frozenset([
    "gaze_to_speech_lag", "revisits_before_mention",
    "mentioned_aoi_dwell_fraction",
])


def _compute_subset_indices(feature_names):
    """
    Derive index arrays for each ablation condition from feature_names.

    Returns
    -------
    subsets   : dict  condition_key → np.ndarray of column indices
    labels_map: dict  condition_key → human-readable label
    """
    fn = list(feature_names)

    gaze_scalar_idx = np.array([i for i, n in enumerate(fn) if n in _GAZE_SCALAR_NAMES])
    trans_pca_idx   = np.array([i for i, n in enumerate(fn) if n.startswith("trans_pca_")])
    dwell_idx       = np.array([i for i, n in enumerate(fn) if n.startswith("dwell_")])
    speech_kw_idx   = np.array([i for i, n in enumerate(fn) if n in _SPEECH_KW_NAMES])
    speech_pca_idx  = np.array([i for i, n in enumerate(fn) if n.startswith("speech_pca_")])
    align_idx       = np.array([i for i, n in enumerate(fn) if n in _ALIGN_NAMES])
    all_idx         = np.arange(len(fn))

    gaze_full_idx   = np.r_[trans_pca_idx, gaze_scalar_idx, dwell_idx]
    speech_idx      = np.r_[speech_kw_idx, speech_pca_idx]

    subsets = {
        "gaze_behavioral_only":  gaze_scalar_idx,
        "gaze_full":             gaze_full_idx,
        "speech_only":           speech_idx,
        "gaze_speech":           np.r_[gaze_full_idx, speech_idx],
        "gaze_speech_alignment": np.r_[gaze_full_idx, speech_idx, align_idx],
        "full_multimodal":       all_idx,
    }
    labels_map = {
        "gaze_behavioral_only":  "Gaze behavioral only (8)",
        "gaze_full":             "Gaze full (trans+scalar+dwell)",
        "speech_only":           "Speech only (PCA + keywords)",
        "gaze_speech":           "Gaze full + Speech",
        "gaze_speech_alignment": "Gaze + Speech + Alignment",
        "full_multimodal":       "Full multimodal",
    }
    return subsets, labels_map


# ── Core routine ──────────────────────────────────────────────────────────────

def _best_kmeans(X_scaled):
    best_k, best_score, best_labels = 2, -1.0, None
    for k in range(2, 8):
        km     = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_scaled)
        score  = silhouette_score(X_scaled, labels)
        if score > best_score:
            best_k, best_score, best_labels = k, score, labels
    return best_k, best_score, best_labels


def run_ablation(X, feature_names):
    """
    Run per-modality ablation on feature matrix X.

    Parameters
    ----------
    X            : np.ndarray  (n_samples, n_features)
    feature_names: list[str]

    Returns
    -------
    results : list[dict]
    """
    subsets, labels_map = _compute_subset_indices(feature_names)
    results = []

    print(f"\n{'Condition':<38} {'Features':>8}  {'Silhouette':>10}  {'Best k':>6}")
    print("─" * 68)

    for cond_key, idx in subsets.items():
        if len(idx) == 0:
            print(f"{labels_map[cond_key]:<38} {'—':>8}  {'—':>10}  {'—':>6}  (no features)")
            continue
        X_sub    = X[:, idx]
        X_scaled = StandardScaler().fit_transform(X_sub)
        best_k, best_score, _ = _best_kmeans(X_scaled)

        row = {
            "condition":  cond_key,
            "label":      labels_map[cond_key],
            "n_features": int(len(idx)),
            "best_k":     best_k,
            "silhouette": round(float(best_score), 6),
        }
        results.append(row)
        print(f"{labels_map[cond_key]:<38} {len(idx):>8}  {best_score:>10.4f}  {best_k:>6}")

    print()
    return results


def save_results(results, out_path="outputs/ablation_results.json"):
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved: {out_path}")


# ── Entrypoint ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

    import types
    if "sentence_transformers" not in sys.modules:
        st = types.ModuleType("sentence_transformers")
        class _FakeST:
            def __init__(self, *a, **kw): pass
            def encode(self, texts):
                return np.random.randn(len(texts), 384).astype("float32")
        st.SentenceTransformer = _FakeST
        sys.modules["sentence_transformers"] = st

    from preprocessing.dataset_builder import DatasetBuilder
    os.chdir(os.path.join(os.path.dirname(__file__), ".."))

    print("Building dataset …")
    X, feature_names, _ = DatasetBuilder().build_dataset()
    print(f"Dataset shape: {X.shape}")

    results = run_ablation(X, feature_names)
    save_results(results)

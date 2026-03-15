"""
dataset_builder.py — Build and normalise the multimodal feature matrix.

Feature layout of the RAW 441-dim vector (from build_case):
    0  : 36   transition_vector          (6×6 AOI transitions, flattened)
    36 : 44   gaze scalars               (fixation stats, velocity, entropy)
    44 : 50   AOI dwell fractions        (6 AOIs: 5 named + background, fixed order)
    50 : 54   speech keywords            (anatomy, findings, negation, uncertainty)
    54 : 57   alignment features
    57 : 441  speech embedding           (384-dim, raw)

build_dataset() post-processes this into a ~36-dim matrix:
    0  : 5    transition PCA             (5 dims from 36-dim transition block)
    5  : 13   gaze scalars               (8 dims)
    13 : 19   AOI dwell fractions        (6 dims)
    19 : 23   speech keywords            (4 dims)
    23 : 26   alignment features         (3 dims)
    26 : 36   speech PCA                 (10 dims)

A single StandardScaler is applied to the full assembled matrix.
No per-group weighting is applied.
"""

import os

import cv2
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from config import DATASET_DIR, IMAGE_FILE, GAZE_FILE, TRANSCRIPTION_FILE
from preprocessing.gaze_processing import load_gaze, define_aois, extract_gaze_features
from preprocessing.speech_processing import SpeechEncoder, extract_speech_features
from preprocessing.cross_modal import compute_alignment_features

# Raw vector indices (before PCA)
_TRANS_END       = 36
_GAZE_SCALAR_END = 44
_DWELL_END       = 50   # +6 dwell fractions
_SPEECH_KW_END   = 54
_ALIGN_END       = 57
_SPEECH_EMB_END  = 441  # 57 + 384

# PCA output dimensions
_TRANS_PCA_DIMS  = 5
_SPEECH_PCA_DIMS = 10

# Fixed AOI order for dwell fractions (must match gaze_processing.define_aois order + background)
_AOI_NAMES_ORDERED = [
    "left_lung", "right_lung", "heart",
    "lower_left", "lower_right", "background",
]


# ── Feature name helpers ──────────────────────────────────────────────────────

def reduced_feature_names():
    """Return the 36 feature names after PCA reduction."""
    names  = [f"trans_pca_{i}" for i in range(_TRANS_PCA_DIMS)]
    names += [
        "fixation_count", "mean_fixation_duration", "max_fixation_duration",
        "scanpath_length", "revisit_rate", "aoi_entropy",
        "mean_velocity", "std_velocity",
    ]
    names += [f"dwell_{aoi}" for aoi in _AOI_NAMES_ORDERED]
    names += ["anatomy_mentions", "finding_mentions", "negation_count", "uncertainty_count"]
    names += ["gaze_to_speech_lag", "revisits_before_mention", "mentioned_aoi_dwell_fraction"]
    names += [f"speech_pca_{i}" for i in range(_SPEECH_PCA_DIMS)]
    return names


# ── DatasetBuilder ────────────────────────────────────────────────────────────

class DatasetBuilder:

    def __init__(self):
        self.speech_encoder = SpeechEncoder()

    def build_case(self, case_dir):
        """
        Build a raw 441-dim feature vector for one case.
        NaN values are left in place; imputation happens in build_dataset().

        Returns
        -------
        vec              : np.ndarray  (441,)
        aois             : dict  name → (x1,y1,x2,y2)
        transition_matrix: np.ndarray  (6,6)  row-normalised probability matrix
        """
        image_path         = os.path.join(case_dir, IMAGE_FILE)
        gaze_path          = os.path.join(case_dir, GAZE_FILE)
        transcription_path = os.path.join(case_dir, TRANSCRIPTION_FILE)

        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not read image: {image_path}")
        h, w = img.shape[:2]
        aois = define_aois(w, h)

        gaze_df   = load_gaze(gaze_path)
        gaze_feat = extract_gaze_features(gaze_df, aois)

        speech_df        = self.speech_encoder.load_transcription(transcription_path)
        speech_embedding = self.speech_encoder.encode(speech_df)       # (384,)
        speech_feat      = extract_speech_features(speech_df)

        align_feat = compute_alignment_features(
            gaze_df, speech_df, gaze_feat["aoi_sequence"], aois
        )

        # Dwell fractions in fixed order
        dwell     = gaze_feat["dwell_time_per_aoi"]
        dwell_vec = np.array(
            [dwell.get(aoi, 0.0) for aoi in _AOI_NAMES_ORDERED], dtype=float
        )

        vec = np.concatenate([
            gaze_feat["transition_vector"],                            # 36
            [gaze_feat["fixation_count"],
             gaze_feat["mean_fixation_duration"],
             gaze_feat["max_fixation_duration"],
             gaze_feat["scanpath_length"],
             gaze_feat["revisit_rate"],
             gaze_feat["aoi_entropy"],
             gaze_feat["mean_velocity"],
             gaze_feat["std_velocity"]],                               # 8
            dwell_vec,                                                 # 6
            [speech_feat["anatomy_mentions"],
             speech_feat["finding_mentions"],
             speech_feat["negation_count"],
             speech_feat["uncertainty_count"]],                        # 4
            [align_feat["gaze_to_speech_lag"],
             align_feat["revisits_before_mention"],
             align_feat["mentioned_aoi_dwell_fraction"]],              # 3
            speech_embedding,                                          # 384
        ], dtype=float)

        return vec, aois, gaze_feat["transition_matrix"]

    def build_dataset(self):
        """
        Build the full feature matrix.

        Steps
        -----
        1. Collect raw 441-dim vectors across all cases.
        2. Impute NaN with column mean (or 0).
        3. PCA the 36-dim transition block → 5 dims.
        4. PCA the 384-dim speech embedding block → 10 dims.
        5. Assemble 36-dim matrix:
               trans_pca(5) + gaze_scalars(8) + dwell(6) + kw(4) + align(3) + speech_pca(10)
        6. Apply a SINGLE StandardScaler to the whole matrix (no per-group weighting).

        Returns
        -------
        X                   : np.ndarray  (n_cases, 36)
        feature_names       : list[str]   length 36
        transition_matrices : list of np.ndarray  (6,6), one per case (raw probabilities)
        """
        case_dirs = sorted([
            os.path.join(DATASET_DIR, d)
            for d in os.listdir(DATASET_DIR)
            if os.path.isdir(os.path.join(DATASET_DIR, d))
        ])

        raw_vecs           = []
        transition_matrices = []

        for case in case_dirs:
            vec, _aois, trans_mat = self.build_case(case)
            raw_vecs.append(vec)
            transition_matrices.append(trans_mat)

        X_raw = np.array(raw_vecs, dtype=float)

        # ── NaN imputation ────────────────────────────────────────────────────
        col_means = np.nanmean(X_raw, axis=0)
        col_means = np.where(np.isnan(col_means), 0.0, col_means)
        nan_mask  = np.isnan(X_raw)
        X_raw[nan_mask] = np.take(col_means, np.where(nan_mask)[1])

        # ── PCA: transition block 36 → 5 ─────────────────────────────────────
        trans_block   = X_raw[:, :_TRANS_END]                          # (n, 36)
        n_trans_pca   = min(_TRANS_PCA_DIMS, trans_block.shape[0] - 1, trans_block.shape[1])
        trans_pca     = PCA(n_components=n_trans_pca, random_state=42).fit_transform(trans_block)
        if trans_pca.shape[1] < _TRANS_PCA_DIMS:
            pad = np.zeros((X_raw.shape[0], _TRANS_PCA_DIMS - trans_pca.shape[1]))
            trans_pca = np.concatenate([trans_pca, pad], axis=1)

        # ── PCA: speech embedding 384 → 10 ───────────────────────────────────
        speech_block  = X_raw[:, _ALIGN_END:_SPEECH_EMB_END]          # (n, 384)
        n_speech_pca  = min(_SPEECH_PCA_DIMS, speech_block.shape[0] - 1, speech_block.shape[1])
        speech_pca    = PCA(n_components=n_speech_pca, random_state=42).fit_transform(speech_block)
        if speech_pca.shape[1] < _SPEECH_PCA_DIMS:
            pad = np.zeros((X_raw.shape[0], _SPEECH_PCA_DIMS - speech_pca.shape[1]))
            speech_pca = np.concatenate([speech_pca, pad], axis=1)

        # ── Assemble 36-dim matrix ─────────────────────────────────────────────
        # middle block: gaze_scalars(8) + dwell(6) + speech_kw(4) + align(3) = 21 dims
        middle_block = X_raw[:, _TRANS_END:_ALIGN_END]
        X = np.concatenate([trans_pca, middle_block, speech_pca], axis=1)

        # ── Single StandardScaler on the whole matrix ─────────────────────────
        X = StandardScaler().fit_transform(X)

        # ── Safety: remove any NaN/Inf ────────────────────────────────────────
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        feature_names = reduced_feature_names()
        return X, feature_names, transition_matrices

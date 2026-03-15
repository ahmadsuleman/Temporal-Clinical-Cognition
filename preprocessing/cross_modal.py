"""
cross_modal.py — Gaze-speech temporal alignment features.

Requires:
  - gaze_df         : DataFrame with columns [timestamp_sec, x, y]
  - transcription_df: DataFrame with columns [text] and optionally
                      [timestamp_start, timestamp_end]
  - aoi_sequence    : list[str] aligned row-for-row with gaze_df
  - aois            : dict of AOI name → (x1, y1, x2, y2) bounding boxes
"""

import re
import numpy as np
import pandas as pd


# ── Anatomy term → AOI name(s) mapping ───────────────────────────────────────
# Each anatomy keyword maps to the AOI region(s) it anatomically overlaps.

ANATOMY_TO_AOI = {
    "lung":          ["left_lung", "right_lung"],
    "heart":         ["heart"],
    "diaphragm":     ["lower_left", "lower_right"],
    "costophrenic":  ["lower_left", "lower_right"],
    "mediastinum":   ["heart"],
    "hilum":         ["heart"],
    "rib":           ["left_lung", "right_lung", "lower_left", "lower_right"],
    "spine":         ["heart"],
    "trachea":       ["heart"],
    "clavicle":      ["left_lung", "right_lung"],
}

_ANATOMY_RE = {term: re.compile(term, re.IGNORECASE) for term in ANATOMY_TO_AOI}


# ── Internal helpers ──────────────────────────────────────────────────────────

def _get_speech_timestamps(transcription_df, session_duration):
    """
    Return a list of (mention_time, [aoi_names]) for every anatomy term found.

    Uses transcript timestamps when available; falls back to evenly spacing
    utterances across the session duration.
    """
    df = transcription_df.copy()
    df.columns = [c.strip() for c in df.columns]
    texts = df["text"].astype(str).tolist()
    n = len(texts)

    has_timestamps = "timestamp_start" in df.columns and n > 0
    if has_timestamps:
        try:
            t_starts = df["timestamp_start"].astype(float).tolist()
        except (ValueError, KeyError):
            has_timestamps = False

    if not has_timestamps:
        # Approximate: space utterances evenly across session
        t_starts = [i * session_duration / n for i in range(n)]

    mentions = []   # list of (t_speech, [mapped_aoi_names])
    for t, text in zip(t_starts, texts):
        for term, pattern in _ANATOMY_RE.items():
            if pattern.search(text):
                aoi_names = ANATOMY_TO_AOI[term]
                mentions.append((float(t), aoi_names))

    return mentions


def _build_fixation_episodes(gaze_df, aoi_sequence):
    """
    Group consecutive gaze points with the same AOI label into episodes.

    Returns a list of dicts:
        { "aoi": str, "t_start": float, "t_end": float, "point_count": int }
    """
    has_time = "timestamp_sec" in gaze_df.columns
    if has_time:
        ts = gaze_df["timestamp_sec"].values.astype(float)
    else:
        ts = np.arange(len(aoi_sequence), dtype=float)

    if not aoi_sequence:
        return []

    episodes = []
    ep_label = aoi_sequence[0]
    ep_start_t = ts[0]
    ep_count = 1

    for i in range(1, len(aoi_sequence)):
        if aoi_sequence[i] == ep_label:
            ep_count += 1
        else:
            episodes.append({
                "aoi":         ep_label,
                "t_start":     ep_start_t,
                "t_end":       ts[i - 1],
                "point_count": ep_count,
            })
            ep_label   = aoi_sequence[i]
            ep_start_t = ts[i]
            ep_count   = 1

    episodes.append({
        "aoi":         ep_label,
        "t_start":     ep_start_t,
        "t_end":       ts[-1],
        "point_count": ep_count,
    })

    return episodes


def _dwell_per_aoi(episodes, all_aoi_names, has_time):
    """
    Return a dict of AOI → total dwell (seconds or point-counts).
    """
    dwell = {name: 0.0 for name in all_aoi_names}
    for ep in episodes:
        if ep["aoi"] in dwell:
            if has_time:
                dwell[ep["aoi"]] += ep["t_end"] - ep["t_start"]
            else:
                dwell[ep["aoi"]] += ep["point_count"]
    return dwell


# ── Public API ────────────────────────────────────────────────────────────────

def compute_alignment_features(gaze_df, transcription_df, aoi_sequence, aois):
    """
    Compute gaze-speech temporal alignment features for one reading session.

    Parameters
    ----------
    gaze_df          : pd.DataFrame  columns: timestamp_sec, x, y [, pupil_mm]
    transcription_df : pd.DataFrame  columns: text [, timestamp_start, timestamp_end]
    aoi_sequence     : list[str]     one label per gaze row (output of extract_gaze_features)
    aois             : dict          AOI name → (x1, y1, x2, y2)

    Returns
    -------
    dict with keys:
        gaze_to_speech_lag            — mean(speech_time - first_gaze_time) across anatomy mentions
        revisits_before_mention       — mean number of AOI visits before each speech mention
        mentioned_aoi_dwell_fraction  — fraction of dwell time on mentioned AOIs
        unmentioned_aoi_dwell_fraction — complement
    """
    nan_result = {
        "gaze_to_speech_lag":             float("nan"),
        "revisits_before_mention":        float("nan"),
        "mentioned_aoi_dwell_fraction":   float("nan"),
        "unmentioned_aoi_dwell_fraction": float("nan"),
    }

    # ── Guard: need at least one gaze point ──────────────────────────────────
    if gaze_df is None or len(gaze_df) == 0 or not aoi_sequence:
        return nan_result

    has_time = "timestamp_sec" in gaze_df.columns
    if has_time:
        ts_gaze = gaze_df["timestamp_sec"].values.astype(float)
        session_duration = float(ts_gaze[-1] - ts_gaze[0])
    else:
        ts_gaze = np.arange(len(aoi_sequence), dtype=float)
        session_duration = float(len(aoi_sequence))

    # ── Build fixation episodes and per-AOI dwell ────────────────────────────
    all_aoi_names = list(aois.keys()) + ["background"]
    episodes = _build_fixation_episodes(gaze_df, aoi_sequence)
    dwell    = _dwell_per_aoi(episodes, all_aoi_names, has_time)
    total_dwell = sum(dwell.values())

    # ── Resolve anatomy mentions with timestamps ─────────────────────────────
    mentions = _get_speech_timestamps(transcription_df, session_duration)

    if not mentions:
        return nan_result

    # ── a) gaze_to_speech_lag ────────────────────────────────────────────────
    # For each mention: find first gaze point in the mapped AOI(s) across the
    # whole session; lag = t_speech - t_first_gaze  (negative = gazed first)
    lags = []
    for t_speech, target_aois in mentions:
        first_gaze_t = None
        for i, label in enumerate(aoi_sequence):
            if label in target_aois:
                first_gaze_t = float(ts_gaze[i])
                break
        if first_gaze_t is not None:
            lags.append(t_speech - first_gaze_t)

    gaze_to_speech_lag = float(np.mean(lags)) if lags else float("nan")

    # ── b) revisits_before_mention ───────────────────────────────────────────
    # Count distinct fixation episodes in the target AOI(s) that START before
    # the speech mention timestamp.
    revisit_counts = []
    for t_speech, target_aois in mentions:
        count = sum(
            1 for ep in episodes
            if ep["aoi"] in target_aois and ep["t_start"] < t_speech
        )
        revisit_counts.append(count)

    revisits_before_mention = float(np.mean(revisit_counts)) if revisit_counts else float("nan")

    # ── c/d) mentioned vs unmentioned AOI dwell fractions ────────────────────
    mentioned_aois = set()
    for _, target_aois in mentions:
        mentioned_aois.update(target_aois)

    if total_dwell > 0:
        mentioned_dwell   = sum(dwell.get(a, 0.0) for a in mentioned_aois)
        mentioned_frac    = mentioned_dwell   / total_dwell
        unmentioned_frac  = 1.0 - mentioned_frac
    else:
        mentioned_frac   = float("nan")
        unmentioned_frac = float("nan")

    return {
        "gaze_to_speech_lag":             gaze_to_speech_lag,
        "revisits_before_mention":        revisits_before_mention,
        "mentioned_aoi_dwell_fraction":   mentioned_frac,
        "unmentioned_aoi_dwell_fraction": unmentioned_frac,
    }

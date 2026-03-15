import numpy as np
import pandas as pd
from scipy.stats import entropy as scipy_entropy


# ── Unchanged helpers ─────────────────────────────────────────────────────────

def load_gaze(path):
    """Load gaze CSV file."""
    gaze = pd.read_csv(path)
    gaze.columns = [c.strip() for c in gaze.columns]
    return gaze


def define_aois(width, height):
    """Define simple anatomical regions."""
    return {
        "left_lung":   (0,          0,           width // 2,       height // 2),
        "right_lung":  (width // 2, 0,           width,            height // 2),
        "heart":       (width // 3, height // 3, 2 * width // 3,   2 * height // 3),
        "lower_left":  (0,          height // 2, width // 2,       height),
        "lower_right": (width // 2, height // 2, width,            height),
    }


def map_aoi(x, y, aois):
    for name, (x1, y1, x2, y2) in aois.items():
        if x1 <= x <= x2 and y1 <= y <= y2:
            return name
    return "background"


# ── New: transition matrix ────────────────────────────────────────────────────

def compute_transition_matrix(aoi_sequence, aoi_names):
    """
    Compute AOI-to-AOI transition probabilities from a sequence of AOI labels.

    Only transitions between *different* consecutive AOIs are counted
    (self-transitions on the same point are ignored).

    Parameters
    ----------
    aoi_sequence : list[str]
    aoi_names    : list[str]  — ordered list of all possible AOI names

    Returns
    -------
    matrix : np.ndarray, shape (n, n)  — row-normalised probability matrix
    vector : np.ndarray, shape (n*n,)  — flattened version for use as features
    """
    n = len(aoi_names)
    idx = {name: i for i, name in enumerate(aoi_names)}

    counts = np.zeros((n, n), dtype=float)

    prev = None
    for label in aoi_sequence:
        if prev is not None and label != prev:
            counts[idx[prev], idx[label]] += 1
        prev = label

    # Row-normalise; rows with no outgoing transitions stay zero
    row_sums = counts.sum(axis=1, keepdims=True)
    matrix = np.divide(counts, row_sums, where=row_sums > 0)

    return matrix, matrix.flatten()


# ── New: scanpath features ────────────────────────────────────────────────────

def compute_scanpath_features(gaze_df, aoi_sequence, aois):
    """
    Derive rich scanpath statistics from a gaze DataFrame and its AOI sequence.

    Expects columns: x, y, and optionally timestamp_sec.
    If timestamp_sec is absent, each row is treated as one time unit.

    Returns a dict with:
        fixation_count          — number of distinct fixation episodes
        mean_fixation_duration  — average episode duration
        max_fixation_duration   — longest single episode duration
        scanpath_length         — total path length in pixels
        revisit_rate            — mean number of separate visits per AOI
        aoi_entropy             — Shannon entropy over dwell-time proportions
        dwell_time_per_aoi      — dict of AOI → fraction of total dwell time
    """
    has_time = "timestamp_sec" in gaze_df.columns
    xs = gaze_df["x"].values
    ys = gaze_df["y"].values
    if has_time:
        ts = gaze_df["timestamp_sec"].values
    else:
        ts = np.arange(len(xs), dtype=float)

    all_aoi_names = list(aois.keys()) + ["background"]

    # ── Fixation episodes (runs of the same AOI label) ──────────────────────
    episodes = []           # list of (aoi_name, start_idx, end_idx)
    if aoi_sequence:
        run_label = aoi_sequence[0]
        run_start = 0
        for i in range(1, len(aoi_sequence)):
            if aoi_sequence[i] != run_label:
                episodes.append((run_label, run_start, i - 1))
                run_label = aoi_sequence[i]
                run_start = i
        episodes.append((run_label, run_start, len(aoi_sequence) - 1))

    # Episode durations
    def episode_duration(ep):
        _, s, e = ep
        if has_time and e < len(ts) - 1:
            return ts[e + 1] - ts[s]
        return (e - s + 1)   # point count fallback

    durations = [episode_duration(ep) for ep in episodes]

    fixation_count         = len(episodes)
    mean_fixation_duration = float(np.mean(durations)) if durations else 0.0
    max_fixation_duration  = float(np.max(durations))  if durations else 0.0

    # ── Scanpath length ──────────────────────────────────────────────────────
    if len(xs) > 1:
        diffs = np.diff(xs) ** 2 + np.diff(ys) ** 2
        scanpath_length = float(np.sum(np.sqrt(diffs)))
    else:
        scanpath_length = 0.0

    # ── Per-AOI dwell time and revisit count ─────────────────────────────────
    dwell_counts  = {name: 0   for name in all_aoi_names}
    revisit_count = {name: 0   for name in all_aoi_names}
    last_aoi      = None

    for ep_label, ep_s, ep_e in episodes:
        dur = episode_duration((ep_label, ep_s, ep_e))
        dwell_counts[ep_label] += dur
        if ep_label != last_aoi:
            revisit_count[ep_label] += 1
        last_aoi = ep_label

    total_dwell = sum(dwell_counts.values())
    if total_dwell > 0:
        dwell_time_per_aoi = {k: v / total_dwell for k, v in dwell_counts.items()}
    else:
        dwell_time_per_aoi = {k: 0.0 for k in all_aoi_names}

    # ── Revisit rate ─────────────────────────────────────────────────────────
    revisit_rate = float(np.mean(list(revisit_count.values())))

    # ── AOI entropy (over dwell proportions) ─────────────────────────────────
    proportions = np.array([dwell_time_per_aoi[k] for k in all_aoi_names], dtype=float)
    # scipy_entropy handles zero-probability bins correctly
    aoi_entropy = float(scipy_entropy(proportions + 1e-12))

    return {
        "fixation_count":         fixation_count,
        "mean_fixation_duration": mean_fixation_duration,
        "max_fixation_duration":  max_fixation_duration,
        "scanpath_length":        scanpath_length,
        "revisit_rate":           revisit_rate,
        "aoi_entropy":            aoi_entropy,
        "dwell_time_per_aoi":     dwell_time_per_aoi,
    }


# ── Upgraded: extract_gaze_features ──────────────────────────────────────────

def extract_gaze_features(gaze, aois):
    """
    Full feature extraction for one gaze recording.

    Returns a dict containing velocity stats, AOI sequence, transition matrix
    and vector, and all scanpath features from compute_scanpath_features().
    """
    xs = gaze["x"].values
    ys = gaze["y"].values

    # Velocity
    velocities = [0.0]
    for i in range(1, len(xs)):
        velocities.append(float(np.sqrt((xs[i] - xs[i-1])**2 + (ys[i] - ys[i-1])**2)))

    # AOI sequence
    aoi_sequence = [map_aoi(x, y, aois) for x, y in zip(xs, ys)]

    # Transition matrix
    aoi_names = list(aois.keys()) + ["background"]
    transition_matrix, transition_vector = compute_transition_matrix(aoi_sequence, aoi_names)

    # Scanpath
    scanpath = compute_scanpath_features(gaze, aoi_sequence, aois)

    return {
        "mean_velocity":          float(np.mean(velocities)),
        "std_velocity":           float(np.std(velocities)),
        "aoi_sequence":           aoi_sequence,
        "transition_matrix":      transition_matrix,
        "transition_vector":      transition_vector,
        "fixation_count":         scanpath["fixation_count"],
        "mean_fixation_duration": scanpath["mean_fixation_duration"],
        "max_fixation_duration":  scanpath["max_fixation_duration"],
        "scanpath_length":        scanpath["scanpath_length"],
        "revisit_rate":           scanpath["revisit_rate"],
        "aoi_entropy":            scanpath["aoi_entropy"],
        "dwell_time_per_aoi":     scanpath["dwell_time_per_aoi"],
    }

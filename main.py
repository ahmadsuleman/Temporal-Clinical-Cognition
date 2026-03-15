import os
import warnings

from preprocessing.dataset_builder import DatasetBuilder
from modeling.clustering import BehaviorClustering
from modeling.temporal_model import create_temporal_sequences, extract_transformer_embeddings
from visualization.plots import plot_results
from experiments.ablation import run_ablation, save_results as save_ablation
from config import DATASET_DIR, IMAGE_FILE, GAZE_FILE, TRANSCRIPTION_FILE, N_CLUSTERS

OUTPUT_DIR = "outputs"


def _load_raw_cases(dataset_dir):
    """
    Load raw gaze/transcription data and image paths for all cases.

    Returns a list of dicts, each containing:
        gaze_df          — pd.DataFrame
        transcription_df — pd.DataFrame
        aoi_sequence     — list[str]
        aois             — dict  name → (x1,y1,x2,y2)
        transition_matrix— np.ndarray (6,6)
        image_path       — str  absolute path to image file
    """
    import cv2
    from preprocessing.gaze_processing import (
        load_gaze, define_aois, extract_gaze_features
    )
    from preprocessing.speech_processing import SpeechEncoder

    enc       = SpeechEncoder()
    case_dirs = sorted([
        os.path.join(dataset_dir, d)
        for d in os.listdir(dataset_dir)
        if os.path.isdir(os.path.join(dataset_dir, d))
    ])
    cases = []
    for case_dir in case_dirs:
        image_path = os.path.join(case_dir, IMAGE_FILE)
        img        = cv2.imread(image_path)
        if img is None:
            continue
        h, w = img.shape[:2]
        aois      = define_aois(w, h)
        gaze_df   = load_gaze(os.path.join(case_dir, GAZE_FILE))
        feats     = extract_gaze_features(gaze_df, aois)
        trans_df  = enc.load_transcription(os.path.join(case_dir, TRANSCRIPTION_FILE))
        cases.append({
            "gaze_df":          gaze_df,
            "transcription_df": trans_df,
            "aoi_sequence":     feats["aoi_sequence"],
            "aois":             aois,
            "transition_matrix": feats["transition_matrix"],
            "image_path":       image_path,
        })
    return cases


def _write_summary(path, X_shape, cluster_model, profiles,
                   ablation_results, saved_files):
    lines = []
    lines.append("=" * 60)
    lines.append("CXR Clinician Behaviour Pipeline — Run Summary")
    lines.append("=" * 60)
    lines.append(f"\nFeature matrix : {X_shape[0]} sessions × {X_shape[1]} features")
    lines.append(f"Clustering     : {cluster_model.method}  "
                 f"k={cluster_model.best_k}  "
                 f"silhouette={cluster_model.silhouette_score_:.4f}")

    lines.append("\nCluster Profiles")
    lines.append("-" * 40)
    for k, p in profiles.items():
        lines.append(f'  Cluster {k} — "{p["label"]}"  (n={p["n_sessions"]})')
        for fname, z in p["top_features"]:
            lines.append(f"    {fname:<42} z={z:+.3f}")

    lines.append("\nAblation Results")
    lines.append("-" * 40)
    lines.append(f"  {'Condition':<38} {'Feats':>5}  {'Sil':>8}  {'k':>4}")
    for r in ablation_results:
        lines.append(f"  {r['label']:<38} {r['n_features']:>5}  "
                     f"{r['silhouette']:>8.4f}  {r['best_k']:>4}")

    lines.append("\nSaved Files")
    lines.append("-" * 40)
    for f in saved_files:
        lines.append(f"  {f}")
    lines.append("")

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as fh:
        fh.write("\n".join(lines))
    print(f"Saved: {path}")


def main():
    import numpy as np

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── 1. Build feature matrix (36 dims, single-scaled) ─────────────────────
    print("=" * 60)
    print("Building dataset …")
    builder = DatasetBuilder()
    X, feature_names, transition_matrices = builder.build_dataset()
    print(f"Feature matrix: {X.shape[0]} sessions × {X.shape[1]} features")
    print(f"Feature names : {feature_names}")

    # ── 2. Temporal embeddings → PCA 10 dims ──────────────────────────────────
    print("\nBuilding temporal sequences …")
    raw_cases = _load_raw_cases(DATASET_DIR)
    sequences = create_temporal_sequences(raw_cases, bin_size_ms=500)
    print(f"Sequences: {sequences.shape}  (sessions × bins × features)")

    print("Extracting temporal embeddings …")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        temporal_emb, emb_source = extract_transformer_embeddings(
            sequences, n_output_dims=10
        )
        for w in caught:
            print(f"  [warn] {w.message}")
    print(f"Temporal embeddings: {temporal_emb.shape}  source={emb_source}")

    temporal_emb   = np.nan_to_num(temporal_emb, nan=0.0, posinf=0.0, neginf=0.0)
    X_aug          = np.concatenate([X, temporal_emb], axis=1)
    aug_feat_names = feature_names + [f"temporal_{i}" for i in range(temporal_emb.shape[1])]
    print(f"Augmented matrix: {X_aug.shape[0]} sessions × {X_aug.shape[1]} features")

    # ── 3. Cluster ────────────────────────────────────────────────────────────
    print("\nClustering …")
    cluster_model = BehaviorClustering(n_clusters=N_CLUSTERS)
    labels        = cluster_model.fit(X_aug)

    # ── 4. Cluster profiles ───────────────────────────────────────────────────
    profiles = cluster_model.get_cluster_profiles(X_aug, labels, aug_feat_names)
    print("\nCluster profiles:")
    for k, p in profiles.items():
        print(f'  Cluster {k} — "{p["label"]}"  (n={p["n_sessions"]})')
        for fname, z in p["top_features"]:
            print(f"    {fname:<42} z={z:+.3f}")

    # ── 5. Ablation (on base 36-dim matrix, not augmented) ────────────────────
    print("\nRunning ablation …")
    ablation_results = run_ablation(X, feature_names)
    abl_path = os.path.join(OUTPUT_DIR, "ablation_results.json")
    save_ablation(ablation_results, out_path=abl_path)

    # ── 6. Visualise ──────────────────────────────────────────────────────────
    print("\nRendering visualisations …")
    plot_results(
        X_aug, labels, cluster_model, aug_feat_names, profiles,
        output_dir=OUTPUT_DIR,
        ablation_results=ablation_results,
        raw_transition_matrices=transition_matrices,
        raw_cases=raw_cases,
    )

    # ── 7. Collect saved files list ───────────────────────────────────────────
    import glob as _glob
    saved_files = sorted(_glob.glob(os.path.join(OUTPUT_DIR, "*.png"))
                         + _glob.glob(os.path.join(OUTPUT_DIR, "*.json"))
                         + _glob.glob(os.path.join(OUTPUT_DIR, "*.txt")))
    print("\nSaved files:")
    for f in saved_files:
        print(f"  {f}")

    # ── 8. Text summary ───────────────────────────────────────────────────────
    summary_path = os.path.join(OUTPUT_DIR, "run_summary.txt")
    _write_summary(
        path=summary_path,
        X_shape=X_aug.shape,
        cluster_model=cluster_model,
        profiles=profiles,
        ablation_results=ablation_results,
        saved_files=saved_files,
    )
    print("\nDone.")


if __name__ == "__main__":
    main()

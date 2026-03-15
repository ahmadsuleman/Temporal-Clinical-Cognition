"""
plots.py — Research-grade visualisation dashboard.

Main entry-point:
    plot_results(X, labels, cluster_model, feature_names, profiles, ...)

Saved files
-----------
outputs/clustering_dashboard.png   — panels 1–5 combined
outputs/transition_matrices.png    — panel 6: averaged transition heatmap per cluster
outputs/aoi_dwell_distribution.png — panel 7: stacked bar of dwell fractions
outputs/gaze_speech_lag.png        — panel 8: boxplot of gaze-to-speech lag
outputs/ablation_comparison.png    — panel 9: silhouette per ablation condition
outputs/scanpath_cluster_N.png     — panel 10: scanpath over image per cluster
outputs/heatmap_cluster_N.png      — panel 11: gaze heatmap overlay per cluster
"""

import os

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from sklearn.decomposition import PCA

try:
    import umap as umap_module
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False

CLUSTER_COLORS  = ["#e63946", "#457b9d", "#2a9d8f", "#f4a261", "#9b59b6",
                   "#e9c46a", "#264653"]
CLUSTER_MARKERS = ["o", "s", "^", "D", "P", "X", "v"]

GAZE_PANEL_FEATURES = [
    "fixation_count", "mean_fixation_duration", "scanpath_length",
    "revisit_rate", "aoi_entropy", "mean_velocity",
]
ALIGN_PANEL_FEATURES = [
    "gaze_to_speech_lag", "revisits_before_mention",
    "mentioned_aoi_dwell_fraction",
    "anatomy_mentions", "finding_mentions", "negation_count",
]
_AOI_NAMES_ORDERED = [
    "left_lung", "right_lung", "heart",
    "lower_left", "lower_right", "background",
]
_DWELL_FEATURES = [f"dwell_{a}" for a in _AOI_NAMES_ORDERED]


# ── helpers ───────────────────────────────────────────────────────────────────

def _style_ax(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=8)


def _cluster_color(i):
    return CLUSTER_COLORS[i % len(CLUSTER_COLORS)]


def _cluster_marker(i):
    return CLUSTER_MARKERS[i % len(CLUSTER_MARKERS)]


# ── Panel 1: PCA scatter ──────────────────────────────────────────────────────

def _panel_pca(ax, X, labels, profiles):
    unique_labels = sorted(k for k in set(labels) if k != -1)
    pca  = PCA(n_components=2)
    X_2d = pca.fit_transform(X)
    var  = pca.explained_variance_ratio_ * 100

    for i, k in enumerate(unique_labels):
        mask   = labels == k
        clabel = profiles.get(k, {}).get("label", f"Cluster {k}")
        ax.scatter(
            X_2d[mask, 0], X_2d[mask, 1],
            c=_cluster_color(i), marker=_cluster_marker(i),
            s=90, alpha=0.85, edgecolors="white", linewidths=0.5,
            label=f"C{k}: {clabel} (n={mask.sum()})",
        )

    ax.set_xlabel(f"PC1 ({var[0]:.1f}% var)", fontsize=9)
    ax.set_ylabel(f"PC2 ({var[1]:.1f}% var)", fontsize=9)
    ax.set_title("Feature Space (PCA)", fontsize=10, fontweight="bold")
    ax.legend(fontsize=7, framealpha=0.7)
    ax.grid(True, linestyle="--", alpha=0.4)
    _style_ax(ax)


# ── Panel 2: UMAP scatter ─────────────────────────────────────────────────────

def _panel_umap(ax, X, labels, profiles):
    unique_labels = sorted(k for k in set(labels) if k != -1)

    if not UMAP_AVAILABLE:
        ax.text(0.5, 0.5, "umap-learn not installed\n(pip install umap-learn)",
                ha="center", va="center", transform=ax.transAxes, fontsize=9)
        ax.set_title("Feature Space (UMAP)", fontsize=10, fontweight="bold")
        return

    n_neighbors = min(15, max(2, len(X) - 1))
    reducer = umap_module.UMAP(n_components=2, n_neighbors=n_neighbors,
                                random_state=42)
    X_2d = reducer.fit_transform(X)

    for i, k in enumerate(unique_labels):
        mask   = labels == k
        clabel = profiles.get(k, {}).get("label", f"Cluster {k}")
        ax.scatter(
            X_2d[mask, 0], X_2d[mask, 1],
            c=_cluster_color(i), marker=_cluster_marker(i),
            s=90, alpha=0.85, edgecolors="white", linewidths=0.5,
            label=f"C{k}: {clabel} (n={mask.sum()})",
        )

    ax.set_xlabel("UMAP-1", fontsize=9)
    ax.set_ylabel("UMAP-2", fontsize=9)
    ax.set_title("Feature Space (UMAP)", fontsize=10, fontweight="bold")
    ax.legend(fontsize=7, framealpha=0.7)
    ax.grid(True, linestyle="--", alpha=0.4)
    _style_ax(ax)


# ── Panel 3: Z-score heatmap ──────────────────────────────────────────────────

def _panel_heatmap(ax, X, labels, feature_names, profiles):
    fn_idx        = {n: i for i, n in enumerate(feature_names)}
    unique_labels = sorted(k for k in set(labels) if k != -1)
    n_clusters    = len(unique_labels)

    top_feature_names = []
    for k in unique_labels:
        for fname, _ in profiles.get(k, {}).get("top_features", []):
            if fname not in top_feature_names:
                top_feature_names.append(fname)
    top_feature_names = top_feature_names[:12]

    if not top_feature_names or n_clusters == 0:
        return

    g_mean = X.mean(axis=0)
    g_std  = np.where(X.std(axis=0) == 0, 1.0, X.std(axis=0))

    heat = np.zeros((n_clusters, len(top_feature_names)))
    for row_i, k in enumerate(unique_labels):
        mask  = labels == k
        cmean = X[mask].mean(axis=0)
        z     = (cmean - g_mean) / g_std
        for col_j, fname in enumerate(top_feature_names):
            if fname in fn_idx:
                heat[row_i, col_j] = z[fn_idx[fname]]

    short_names = [n.replace("_", "\n") for n in top_feature_names]
    row_labels  = [f"C{k}: {profiles.get(k,{}).get('label','?')}" for k in unique_labels]

    im = ax.imshow(heat, cmap="RdBu_r", aspect="auto", vmin=-2, vmax=2)
    ax.set_xticks(range(len(short_names)))
    ax.set_xticklabels(short_names, fontsize=6, rotation=45, ha="right")
    ax.set_yticks(range(n_clusters))
    ax.set_yticklabels(row_labels, fontsize=8)
    ax.set_title("Top Features — Z-score per Cluster", fontsize=10, fontweight="bold")
    plt.colorbar(im, ax=ax, fraction=0.03, label="z-score")
    for r in range(n_clusters):
        for c in range(len(top_feature_names)):
            ax.text(c, r, f"{heat[r,c]:+.1f}", ha="center", va="center",
                    fontsize=5.5, color="black")


# ── Panel 4: Gaze scanpath bar chart ─────────────────────────────────────────

def _panel_gaze_bars(ax, X, labels, feature_names):
    fn_idx        = {n: i for i, n in enumerate(feature_names)}
    unique_labels = sorted(k for k in set(labels) if k != -1)
    n_clusters    = len(unique_labels)

    present = [f for f in GAZE_PANEL_FEATURES if f in fn_idx]
    if not present or n_clusters == 0:
        return

    x_pos = np.arange(len(present))
    bar_w = 0.8 / max(n_clusters, 1)

    for i, k in enumerate(unique_labels):
        mask   = labels == k
        means  = [X[mask, fn_idx[f]].mean() for f in present]
        offset = (i - n_clusters / 2 + 0.5) * bar_w
        ax.bar(x_pos + offset, means, bar_w * 0.9,
               color=_cluster_color(i), alpha=0.85, label=f"C{k}")

    ax.set_xticks(x_pos)
    ax.set_xticklabels([f.replace("_", "\n") for f in present], fontsize=8)
    ax.set_title("Gaze Scanpath Features per Cluster", fontsize=10, fontweight="bold")
    ax.legend(fontsize=8, framealpha=0.7)
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    _style_ax(ax)


# ── Panel 5: Alignment & speech keyword bars ──────────────────────────────────

def _panel_align_bars(ax, X, labels, feature_names):
    fn_idx        = {n: i for i, n in enumerate(feature_names)}
    unique_labels = sorted(k for k in set(labels) if k != -1)
    n_clusters    = len(unique_labels)

    present = [f for f in ALIGN_PANEL_FEATURES if f in fn_idx]
    if not present or n_clusters == 0:
        return

    x_pos = np.arange(len(present))
    bar_w = 0.8 / max(n_clusters, 1)

    for i, k in enumerate(unique_labels):
        mask   = labels == k
        means  = [X[mask, fn_idx[f]].mean() for f in present]
        offset = (i - n_clusters / 2 + 0.5) * bar_w
        ax.bar(x_pos + offset, means, bar_w * 0.9,
               color=_cluster_color(i), alpha=0.85, label=f"C{k}")

    ax.set_xticks(x_pos)
    ax.set_xticklabels([f.replace("_", "\n") for f in present], fontsize=8)
    ax.set_title("Alignment & Speech Keyword Features per Cluster",
                 fontsize=10, fontweight="bold")
    ax.legend(fontsize=8, framealpha=0.7)
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    _style_ax(ax)


# ── Panel 6: Transition matrix heatmaps ──────────────────────────────────────

def _save_transition_matrices(labels, raw_transition_matrices, profiles, output_dir):
    if raw_transition_matrices is None:
        return
    unique_labels = sorted(k for k in set(labels) if k != -1)
    n_clusters    = len(unique_labels)
    if n_clusters == 0:
        return

    fig, axes = plt.subplots(1, n_clusters, figsize=(4 * n_clusters, 4))
    if n_clusters == 1:
        axes = [axes]

    aoi_short = ["L.Lung", "R.Lung", "Heart", "Lo.L", "Lo.R", "Bg"]

    for i, k in enumerate(unique_labels):
        mask     = labels == k
        matrices = [raw_transition_matrices[j]
                    for j in range(len(labels)) if labels[j] == k]
        if not matrices:
            continue
        avg_mat  = np.mean(np.array(matrices), axis=0)
        ax       = axes[i]
        im       = ax.imshow(avg_mat, cmap="YlOrRd", vmin=0, vmax=1)
        clabel   = profiles.get(k, {}).get("label", f"Cluster {k}")
        ax.set_title(f"C{k}: {clabel}", fontsize=9, fontweight="bold")
        ax.set_xticks(range(len(aoi_short)))
        ax.set_yticks(range(len(aoi_short)))
        ax.set_xticklabels(aoi_short, fontsize=7, rotation=45, ha="right")
        ax.set_yticklabels(aoi_short, fontsize=7)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Prob.")
        for r in range(len(aoi_short)):
            for c in range(len(aoi_short)):
                ax.text(c, r, f"{avg_mat[r,c]:.2f}", ha="center", va="center",
                        fontsize=6, color="black")

    fig.suptitle("Averaged AOI Transition Matrices per Cluster",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    out = os.path.join(output_dir, "transition_matrices.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


# ── Panel 7: AOI dwell distribution ──────────────────────────────────────────

def _save_aoi_dwell(X, labels, feature_names, profiles, output_dir):
    fn_idx        = {n: i for i, n in enumerate(feature_names)}
    unique_labels = sorted(k for k in set(labels) if k != -1)
    present_dwell = [f for f in _DWELL_FEATURES if f in fn_idx]
    if not present_dwell or not unique_labels:
        return

    fig, ax = plt.subplots(figsize=(max(6, len(unique_labels) * 2), 4))
    x_pos   = np.arange(len(unique_labels))
    bottoms = np.zeros(len(unique_labels))
    aoi_colors = plt.cm.tab20(np.linspace(0, 1, len(present_dwell)))

    for j, feat in enumerate(present_dwell):
        values = np.array([X[labels == k, fn_idx[feat]].mean()
                           for k in unique_labels])
        # Clip negatives from StandardScaler shift
        values = np.clip(values, 0, None)
        ax.bar(x_pos, values, bottom=bottoms, color=aoi_colors[j],
               label=feat.replace("dwell_", ""), alpha=0.9)
        bottoms += values

    clabels = [f"C{k}\n{profiles.get(k,{}).get('label','?')}" for k in unique_labels]
    ax.set_xticks(x_pos)
    ax.set_xticklabels(clabels, fontsize=8)
    ax.set_ylabel("Mean scaled dwell fraction", fontsize=9)
    ax.set_title("AOI Dwell Time Distribution per Cluster",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=7, bbox_to_anchor=(1.01, 1), loc="upper left", framealpha=0.7)
    _style_ax(ax)
    plt.tight_layout()
    out = os.path.join(output_dir, "aoi_dwell_distribution.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


# ── Panel 8: Gaze-to-speech lag boxplot ───────────────────────────────────────

def _save_lag_boxplot(X, labels, feature_names, profiles, output_dir):
    fn_idx = {n: i for i, n in enumerate(feature_names)}
    if "gaze_to_speech_lag" not in fn_idx:
        return
    unique_labels = sorted(k for k in set(labels) if k != -1)
    if not unique_labels:
        return

    lag_idx = fn_idx["gaze_to_speech_lag"]
    data    = [X[labels == k, lag_idx] for k in unique_labels]
    clabels = [f"C{k}\n{profiles.get(k,{}).get('label','?')}" for k in unique_labels]

    fig, ax = plt.subplots(figsize=(max(5, len(unique_labels) * 1.8), 4))
    bp = ax.boxplot(data, patch_artist=True, notch=False)
    for patch, i in zip(bp["boxes"], range(len(unique_labels))):
        patch.set_facecolor(_cluster_color(i))
        patch.set_alpha(0.7)
    for median in bp["medians"]:
        median.set_color("black")
        median.set_linewidth(1.5)

    ax.set_xticks(range(1, len(unique_labels) + 1))
    ax.set_xticklabels(clabels, fontsize=8)
    ax.set_ylabel("Gaze-to-speech lag (scaled)", fontsize=9)
    ax.set_title("Gaze-to-Speech Lag per Cluster", fontsize=11, fontweight="bold")
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    _style_ax(ax)
    plt.tight_layout()
    out = os.path.join(output_dir, "gaze_speech_lag.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


# ── Panel 9: Ablation comparison bar chart ───────────────────────────────────

def _save_ablation_chart(ablation_results, output_dir):
    if not ablation_results:
        return

    labels  = [r["label"] for r in ablation_results]
    scores  = [r["silhouette"] for r in ablation_results]
    n_feats = [r["n_features"] for r in ablation_results]
    colors  = [_cluster_color(i) for i in range(len(labels))]

    fig, ax = plt.subplots(figsize=(max(7, len(labels) * 1.5), 4))
    bars = ax.bar(range(len(labels)), scores, color=colors, alpha=0.85, edgecolor="white")
    for bar, nf in zip(bars, n_feats):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.003,
                f"n={nf}", ha="center", va="bottom", fontsize=7)

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels([l.replace(" ", "\n") for l in labels], fontsize=8)
    ax.set_ylabel("Best silhouette score", fontsize=9)
    ax.set_title("Ablation: Silhouette Score per Feature Condition",
                 fontsize=11, fontweight="bold")
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    _style_ax(ax)
    plt.tight_layout()
    out = os.path.join(output_dir, "ablation_comparison.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


# ── Panel 10: Scanpath diagram ────────────────────────────────────────────────

def _aoi_centroid(bbox):
    """Return (cx, cy) of an AOI bounding box (x1,y1,x2,y2)."""
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def _save_scanpath_diagrams(labels, raw_cases, profiles, output_dir):
    """
    For each cluster, pick the representative case (first session) and draw
    the gaze scanpath over the X-ray image as arrows between AOI centroids.
    Arrow thickness is proportional to transition frequency.
    """
    if raw_cases is None:
        return
    unique_labels = sorted(k for k in set(labels) if k != -1)

    for k in unique_labels:
        mask    = np.where(labels == k)[0]
        if len(mask) == 0:
            continue
        rep_idx = int(mask[0])
        case    = raw_cases[rep_idx]

        image_path = case.get("image_path")
        aois       = case.get("aois")
        gaze_df    = case.get("gaze_df")
        if image_path is None or aois is None or gaze_df is None:
            continue

        img = cv2.imread(image_path)
        if img is None:
            continue
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Build average transition matrix for this cluster
        cluster_trans = [raw_cases[j].get("transition_matrix")
                         for j in mask if raw_cases[j].get("transition_matrix") is not None]
        if not cluster_trans:
            continue
        avg_trans = np.mean(np.array(cluster_trans), axis=0)

        all_aoi_names = list(aois.keys()) + ["background"]
        centroids = {}
        for aoi_name, bbox in aois.items():
            cx, cy = _aoi_centroid(bbox)
            centroids[aoi_name] = (cx, cy)
        # Background centroid: center of image
        h, w = img_rgb.shape[:2]
        centroids["background"] = (w / 2.0, h / 2.0)

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.imshow(img_rgb, alpha=0.7)

        # Draw transitions as arrows; line width proportional to probability
        max_prob = avg_trans.max()
        for i, src in enumerate(all_aoi_names):
            for j, dst in enumerate(all_aoi_names):
                if i == j:
                    continue
                prob = avg_trans[i, j]
                if prob < 0.05:
                    continue
                lw = max(0.5, (prob / max(max_prob, 1e-9)) * 6)
                sx, sy = centroids[src]
                dx, dy = centroids[dst]
                ax.annotate(
                    "", xy=(dx, dy), xytext=(sx, sy),
                    arrowprops=dict(arrowstyle="->", lw=lw,
                                   color=_cluster_color(k), alpha=0.7),
                )

        # Draw AOI centroid markers
        for aoi_name, (cx, cy) in centroids.items():
            ax.plot(cx, cy, "o", ms=8, color="white",
                    markeredgecolor="black", markeredgewidth=1)
            ax.text(cx, cy - 18, aoi_name.replace("_", "\n"),
                    ha="center", fontsize=6, color="white",
                    bbox=dict(boxstyle="round,pad=0.2", fc="black", alpha=0.5))

        clabel = profiles.get(k, {}).get("label", f"Cluster {k}")
        ax.set_title(f"C{k}: {clabel} — Scanpath (representative case)",
                     fontsize=10, fontweight="bold")
        ax.axis("off")
        plt.tight_layout()
        out = os.path.join(output_dir, f"scanpath_cluster_{k}.png")
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {out}")


# ── Panel 11: Gaze heatmap overlay ───────────────────────────────────────────

def _save_heatmap_overlays(labels, raw_cases, profiles, output_dir):
    """
    For each cluster, accumulate a gaze density heatmap from all sessions
    in that cluster and overlay it on the representative image.
    """
    if raw_cases is None:
        return
    unique_labels = sorted(k for k in set(labels) if k != -1)

    for k in unique_labels:
        mask    = np.where(labels == k)[0]
        if len(mask) == 0:
            continue

        # Load representative image to get size
        rep_case   = raw_cases[int(mask[0])]
        image_path = rep_case.get("image_path")
        if image_path is None:
            continue
        img = cv2.imread(image_path)
        if img is None:
            continue
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        H, W    = img_rgb.shape[:2]

        # Accumulate heatmap from all sessions in this cluster
        heatmap = np.zeros((H, W), dtype=np.float32)
        for j in mask:
            gaze_df = raw_cases[j].get("gaze_df")
            if gaze_df is None:
                continue
            for _, row in gaze_df.iterrows():
                x = int(np.clip(row["x"], 0, W - 1))
                y = int(np.clip(row["y"], 0, H - 1))
                cv2.circle(heatmap, (x, y), radius=30, color=1, thickness=-1)

        if heatmap.max() > 0:
            heatmap = heatmap / heatmap.max()
        heatmap = cv2.GaussianBlur(heatmap, (51, 51), 0)

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.imshow(img_rgb)
        ax.imshow(heatmap, cmap="jet", alpha=0.5, vmin=0, vmax=1)
        clabel = profiles.get(k, {}).get("label", f"Cluster {k}")
        ax.set_title(f"C{k}: {clabel} — Gaze Heatmap (n={len(mask)} sessions)",
                     fontsize=10, fontweight="bold")
        ax.axis("off")
        plt.tight_layout()
        out = os.path.join(output_dir, f"heatmap_cluster_{k}.png")
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {out}")


# ── Main entry-point ──────────────────────────────────────────────────────────

def plot_results(X, labels, cluster_model, feature_names, profiles,
                 output_dir="outputs",
                 ablation_results=None,
                 raw_transition_matrices=None,
                 raw_cases=None):
    """
    Generate and save all visualisation outputs.

    Parameters
    ----------
    X                      : np.ndarray  (n_sessions, n_features)
    labels                 : np.ndarray  (n_sessions,)
    cluster_model          : BehaviorClustering
    feature_names          : list[str]
    profiles               : dict  cluster_id → profile dict
    output_dir             : str
    ablation_results       : list[dict] from run_ablation(), optional
    raw_transition_matrices: list of np.ndarray (6,6), one per session, optional
    raw_cases              : list of dicts with keys:
                               image_path, gaze_df, aoi_sequence, aois,
                               transition_matrix   (all optional but enable more plots)
    """
    os.makedirs(output_dir, exist_ok=True)

    title = (f"CXR Clinician Behaviour — {cluster_model.method}  "
             f"k={cluster_model.best_k}  "
             f"silhouette={cluster_model.silhouette_score_:.3f}")

    # ── Dashboard: panels 1–5 ─────────────────────────────────────────────────
    fig = plt.figure(figsize=(22, 14))
    fig.suptitle(title, fontsize=13, fontweight="bold", y=1.01)
    gs  = gridspec.GridSpec(3, 2, figure=fig, hspace=0.52, wspace=0.38)

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])
    ax5 = fig.add_subplot(gs[2, :])

    _panel_pca(ax1, X, labels, profiles)
    _panel_umap(ax2, X, labels, profiles)
    _panel_heatmap(ax3, X, labels, feature_names, profiles)
    _panel_gaze_bars(ax4, X, labels, feature_names)
    _panel_align_bars(ax5, X, labels, feature_names)

    plt.tight_layout()
    dash_path = os.path.join(output_dir, "clustering_dashboard.png")
    plt.savefig(dash_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {dash_path}")

    # ── Individual files: panels 6–11 ─────────────────────────────────────────
    _save_transition_matrices(labels, raw_transition_matrices, profiles, output_dir)
    _save_aoi_dwell(X, labels, feature_names, profiles, output_dir)
    _save_lag_boxplot(X, labels, feature_names, profiles, output_dir)
    _save_ablation_chart(ablation_results, output_dir)
    _save_scanpath_diagrams(labels, raw_cases, profiles, output_dir)
    _save_heatmap_overlays(labels, raw_cases, profiles, output_dir)

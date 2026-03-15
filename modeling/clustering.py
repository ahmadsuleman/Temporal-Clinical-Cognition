import numpy as np
from sklearn.cluster import KMeans, HDBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


# ── Embedding dimension prefixes to exclude from behavioral profiling ─────────
_EMBEDDING_PREFIXES = ("speech_emb_", "speech_pca_", "temporal_emb_", "temporal_")


def _is_behavioral(name):
    return not any(name.startswith(p) for p in _EMBEDDING_PREFIXES)


# ── Cluster naming ────────────────────────────────────────────────────────────
# Rules are scored from z-scores of behavioral features.
# A name is only assigned if at least one defining feature has |z| > 1.0.
# Otherwise the cluster is labelled "Mixed Strategy (n=X)".

_RULES = [
    # (label, defining_feature_names, score_fn)
    ("Systematic Scanner",   ["fixation_count", "aoi_entropy"],
     lambda z: z("fixation_count") + z("aoi_entropy")),
    ("Focused Inspector",    ["mean_fixation_duration", "aoi_entropy"],
     lambda z: z("mean_fixation_duration") - z("aoi_entropy")),
    ("Uncertain Revisitor",  ["revisit_rate", "uncertainty_count"],
     lambda z: z("revisit_rate") + z("uncertainty_count")),
    ("Rule-Out Strategist",  ["anatomy_mentions", "negation_count"],
     lambda z: z("anatomy_mentions") + z("negation_count")),
    ("Hypothesis-Driven",    ["finding_mentions", "gaze_to_speech_lag"],
     lambda z: -z("gaze_to_speech_lag") + z("finding_mentions")),
    ("Rapid Scanner",        ["scanpath_length", "mean_velocity"],
     lambda z: z("scanpath_length") + z("mean_velocity")),
    ("Speech-Guided Viewer", ["mentioned_aoi_dwell_fraction"],
     lambda z: z("mentioned_aoi_dwell_fraction")),
]


def _score_rules(z_all, feature_names, n_sessions):
    idx = {n: i for i, n in enumerate(feature_names)}

    def z(name):
        return float(z_all[idx[name]]) if name in idx else 0.0

    best_label = None
    best_score = 0.0

    for label, defining_feats, score_fn in _RULES:
        # Only fire if at least one defining feature clears the threshold
        max_z = max(abs(z(f)) for f in defining_feats)
        if max_z <= 1.0:
            continue
        score = score_fn(z)
        if score > best_score:
            best_score = score
            best_label = label

    if best_label is None:
        return f"Mixed Strategy (n={n_sessions})"
    return best_label


# ── BehaviorClustering ────────────────────────────────────────────────────────

class BehaviorClustering:
    """
    Selects the best clustering method and k automatically via silhouette score.

    Methods tried:
      KMeans:               k ∈ {2, 3, 4, 5, 6, 7}
      HDBSCAN:              min_cluster_size ∈ {2,3,4,5} × min_samples ∈ {1,2,3}
                            — only configs with noise < 25% are accepted
      AgglomerativeClustering (ward): k ∈ {2, 3, 4, 5, 6, 7}

    All candidates are printed. The overall winner is chosen by silhouette score.

    Post-fit attributes:
        method            : str
        best_k            : int
        silhouette_score_ : float
        X_scaled_         : np.ndarray
    """

    def __init__(self, n_clusters=3):
        self.n_clusters        = n_clusters
        self.scaler            = StandardScaler()
        self.model             = None
        self.method            = None
        self.best_k            = None
        self.silhouette_score_ = None
        self.X_scaled_         = None

    # ── fit ──────────────────────────────────────────────────────────────────

    def fit(self, X):
        X_scaled       = self.scaler.fit_transform(X)
        self.X_scaled_ = X_scaled

        print(f"\n{'Method':<30} {'k':>4}  {'Silhouette':>10}  {'Notes'}")
        print("─" * 65)

        # ── KMeans sweep k=2..7 ──────────────────────────────────────────────
        best_km = {"score": -1.0, "labels": None, "model": None, "k": self.n_clusters}

        for k in range(2, 8):
            km     = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = km.fit_predict(X_scaled)
            score  = silhouette_score(X_scaled, labels)
            print(f"  KMeans                       {k:>4}  {score:>10.4f}")
            if score > best_km["score"]:
                best_km = {"score": score, "labels": labels, "model": km, "k": k}

        # ── AgglomerativeClustering sweep k=2..7 ─────────────────────────────
        best_agg = {"score": -1.0, "labels": None, "k": self.n_clusters}

        for k in range(2, 8):
            agg    = AgglomerativeClustering(n_clusters=k, linkage="ward")
            labels = agg.fit_predict(X_scaled)
            score  = silhouette_score(X_scaled, labels)
            print(f"  Agglomerative (ward)         {k:>4}  {score:>10.4f}")
            if score > best_agg["score"]:
                best_agg = {"score": score, "labels": labels, "k": k}

        # ── HDBSCAN grid search (noise < 25%) ─────────────────────────────────
        best_hdb = {"score": -1.0, "labels": None, "k": 0,
                    "noise_pct": 1.0, "config": None}

        for min_cs in [2, 3, 4, 5]:
            for min_s in [1, 2, 3]:
                hdb        = HDBSCAN(min_cluster_size=min_cs, min_samples=min_s)
                hdb_labels = hdb.fit_predict(X_scaled)
                unique_k   = set(hdb_labels) - {-1}
                noise_pct  = float((hdb_labels == -1).mean())
                non_noise  = hdb_labels != -1

                if len(unique_k) < 2 or non_noise.sum() < 2:
                    print(f"  HDBSCAN mcs={min_cs} ms={min_s}          —      —  <2 clusters")
                    continue
                if noise_pct >= 0.25:
                    print(f"  HDBSCAN mcs={min_cs} ms={min_s}          "
                          f"{len(unique_k):>4}       —  noise={noise_pct:.0%} ≥25% (skipped)")
                    continue

                score = silhouette_score(X_scaled[non_noise], hdb_labels[non_noise])
                note  = f"noise={noise_pct:.0%}  k={len(unique_k)}"
                print(f"  HDBSCAN mcs={min_cs} ms={min_s}          "
                      f"{len(unique_k):>4}  {score:>10.4f}  {note}")
                if score > best_hdb["score"]:
                    best_hdb = {
                        "score":     score,
                        "labels":    hdb_labels,
                        "k":         len(unique_k),
                        "noise_pct": noise_pct,
                        "config":    (min_cs, min_s),
                    }

        # ── Select overall winner ─────────────────────────────────────────────
        candidates = [
            ("KMeans",              best_km["score"],  best_km),
            ("Agglomerative(ward)", best_agg["score"], best_agg),
        ]
        if best_hdb["config"] is not None:
            candidates.append(("HDBSCAN", best_hdb["score"], best_hdb))

        winner_name, _, winner = max(candidates, key=lambda t: t[1])

        if winner_name == "KMeans":
            self.method            = "KMeans"
            self.silhouette_score_ = winner["score"]
            self.best_k            = winner["k"]
            self.model             = winner["model"]
            labels                 = winner["labels"]
        elif winner_name == "Agglomerative(ward)":
            self.method            = "Agglomerative(ward)"
            self.silhouette_score_ = winner["score"]
            self.best_k            = winner["k"]
            labels                 = winner["labels"]
        else:  # HDBSCAN
            self.method            = "HDBSCAN"
            self.silhouette_score_ = winner["score"]
            self.best_k            = winner["k"]
            labels                 = winner["labels"]

        print(f"\n  → Selected: {self.method}  k={self.best_k}"
              f"  silhouette={self.silhouette_score_:.4f}")
        return labels

    # ── cluster profiles ─────────────────────────────────────────────────────

    def get_cluster_profiles(self, X, labels, feature_names):
        """
        Characterise each cluster using ONLY behavioral feature dimensions.

        Only non-embedding features are used for ranking and naming.
        Shows top 5 features per cluster.

        Returns
        -------
        profiles : dict  cluster_id → {
            label, n_sessions,
            top_features  : [(name, z_score), ...]   top 5 behavioral
            mean_features : {name: value, ...}        top 5 behavioral
        }
        """
        valid  = labels != -1
        X_v    = X[valid]
        labs_v = labels[valid]

        g_mean = X_v.mean(axis=0)
        g_std  = np.where(X_v.std(axis=0) == 0, 1.0, X_v.std(axis=0))

        behavioral_idx = [i for i, n in enumerate(feature_names) if _is_behavioral(n)]

        profiles = {}
        for k in sorted(set(labs_v)):
            mask         = labs_v == k
            n_sessions   = int(mask.sum())
            cluster_mean = X_v[mask].mean(axis=0)
            z_all        = (cluster_mean - g_mean) / g_std

            # Rank by |z| within behavioral features, take top 5
            z_behavioral = [(i, z_all[i]) for i in behavioral_idx]
            top5         = sorted(z_behavioral, key=lambda t: abs(t[1]), reverse=True)[:5]
            top5_named   = [(feature_names[i], float(z)) for i, z in top5]

            label = _score_rules(z_all, feature_names, n_sessions)

            profiles[int(k)] = {
                "label":         label,
                "n_sessions":    n_sessions,
                "top_features":  top5_named,
                "mean_features": {feature_names[i]: float(cluster_mean[i])
                                  for i, _ in top5},
            }

        return profiles

# Modeling Clinical Reasoning from Multimodal Interaction Signals

<p align="center">
  <strong>Unsupervised Discovery of Radiological Reading Strategies via Gaze–Speech Behavioral Representations</strong>
</p>

<p align="center">
  <a href="#abstract">Abstract</a> •
  <a href="#key-contributions">Contributions</a> •
  <a href="#method">Method</a> •
  <a href="#data">Data</a> •
  <a href="#results">Results</a> •
  <a href="#reproduction">Reproduction</a> •
  <a href="#citation">Citation</a>
</p>

---

## Abstract

Understanding how clinicians visually and verbally reason during medical image interpretation is a key challenge for developing interpretable clinical AI systems. We present a computational framework for analyzing clinician behavior during chest X-ray (CXR) interpretation using multimodal interaction signals. The pipeline integrates eye-tracking data, spoken diagnostic reasoning, and cross-modal temporal alignment to characterize latent radiological reading strategies through unsupervised representation learning.

A dataset of 50 clinician interpretation sessions is processed, each containing synchronized gaze recordings and speech transcripts. Eye-tracking data are mapped to anatomically defined regions of interest (ROIs), from which gaze descriptors are extracted — including scanpath statistics, revisit rates, velocity measures, and AOI dwell distributions. Speech transcripts are encoded using Sentence-Transformers (`all-MiniLM-L6-v2`) combined with interpretable linguistic features (anatomical mentions, radiological findings, negation markers, uncertainty markers).

The resulting **46-dimensional multimodal behavioral representation** is analyzed via unsupervised clustering, revealing **three clinician reading strategies**: *Mixed Strategy*, *Rapid Scanner*, and *Focused Inspector*. Feature ablation analysis demonstrates that gaze dynamics alone provide the strongest signal for differentiating clinician strategies (silhouette = 0.369), while speech features capture complementary but weaker reasoning signals (silhouette = 0.211).

## Key Contributions

1. **Multimodal behavioral representation** combining gaze dynamics (scanpath statistics, velocity profiles, AOI dwell distributions), speech embeddings, and cross-modal temporal features into a unified 46-dimensional feature space.
2. **Unsupervised discovery of radiological reading strategies** from behavioral interaction signals, identifying three interpretable clinician archetypes without supervision.
3. **Quantitative analysis of gaze–speech coordination**, revealing distinct temporal coupling patterns across diagnostic workflows.
4. **Systematic feature ablation study** demonstrating that visual attention dynamics provide the dominant signal for distinguishing clinician strategies, with implications for modality selection in clinical AI systems.

## Method

### Pipeline Architecture

The system follows a modular multimodal processing architecture with five sequential stages:

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Data Ingestion                               │
│   CXR Image  ·  Eye-Tracking Scanpaths  ·  Speech Transcripts      │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│                 Modality-Specific Feature Extraction                 │
│                                                                     │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐  │
│  │  Gaze Features    │  │ Speech Features   │  │ Cross-Modal      │  │
│  │  (19 dims)        │  │ (14 dims)         │  │ Alignment        │  │
│  │                   │  │                   │  │ (13 dims)        │  │
│  │ · Scanpath stats  │  │ · Sentence embeds │  │ · Temporal lag   │  │
│  │ · Fixation dists  │  │   (MiniLM-L6-v2) │  │ · Coordination   │  │
│  │ · Velocity prof.  │  │ · Anatomical refs │  │   metrics        │  │
│  │ · Revisit rates   │  │ · Negation count  │  │                  │  │
│  │ · AOI dwells      │  │ · Uncertainty idx │  │                  │  │
│  └──────────────────┘  └──────────────────┘  └──────────────────┘  │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│          Behavioral Representation (46 features)                    │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│          Unsupervised Clustering (KMeans, k=3)                      │
│          + Dimensionality Reduction (PCA, UMAP)                     │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│          Clinician Strategy Discovery & Ablation Analysis           │
└─────────────────────────────────────────────────────────────────────┘
```

### Feature Extraction Details

**Gaze behavioral features (19 dims):** Scanpath length, fixation count, mean/max fixation duration, mean/variance of saccadic velocity, revisit rate per AOI, and dwell-time distributions across anatomically defined regions (left lung, right lung, mediastinum, cardiac silhouette, costophrenic angles).

**Speech features (14 dims):** Dense sentence embeddings via `all-MiniLM-L6-v2` (384-d, projected to 8-d via PCA), augmented with interpretable linguistic features — counts of anatomical mentions, radiological findings, negation markers, uncertainty hedges, report length, and lexical diversity.

**Cross-modal alignment features (13 dims):** Gaze–speech temporal lag (mean, variance), fixation-to-utterance synchrony, AOI-conditioned speech onset latency, and coordination entropy.

### Clustering and Evaluation

Behavioral representations are standardized (z-score) and clustered via KMeans (k=3, selected by silhouette analysis). Cluster validity is assessed using silhouette score, and cluster interpretability is evaluated through feature-level z-score profiling, spatial attention heatmaps, scanpath visualization, and AOI transition analysis.

## Data

### Modalities

The dataset provides five synchronized behavioral modalities per session:

| Modality | File | Description | Sampling |
|---|---|---|---|
| Visual stimulus | `image.jpeg` | Chest X-ray (PA view) | — |
| Visual attention | `gaze.csv` | Simulated eye-tracking scanpath | ~60 Hz |
| Verbal reasoning | `transcription.csv` | Time-stamped diagnostic dictation segments | Utterance-level |
| Audio signal | `audio.wav` | TTS-generated dictation recording | 16 kHz |
| Structured metadata | `metadata.json` | Diagnostic findings and session parameters | Per-session |

### Data Generation

The multimodal dataset is generated using a simulation framework that produces clinically plausible gaze and speech patterns conditioned on CXR pathology labels.

**Simulator:** [Multi-Modal Clinical Dataset Generator](https://github.com/ahmadsuleman/Multi_Model-Clinical-Dataset-Generator.git)

> **Note:** The simulated data is designed for method development and behavioral modeling research. It does not replace real clinician recordings for clinical validation.

## Results

### Behavioral Strategy Discovery

KMeans clustering (k=3) over the 46-dimensional behavioral space identifies three clinician reading strategies:

| Strategy | n | Defining Behavioral Signature | Key Features (z-score) |
|---|---|---|---|
| **Mixed Strategy** | 18 | Systematic exploration with iterative hypothesis testing | scanpath_length (+0.964), fixation_count (+0.935), revisit_rate (+0.935), negation_markers (+0.778), anatomical_mentions (+0.674) |
| **Rapid Scanner** | 17 | Fast pattern-recognition-driven visual search | mean_velocity (+1.299), velocity_variance (+1.047), anatomical_refs (−0.568), negation_markers (−0.703) |
| **Focused Inspector** | 15 | Concentrated inspection of limited anatomical regions | mean_fixation_duration (+1.001), max_fixation_duration (+0.904), mean_velocity (−0.899) |

Overall silhouette score: **0.104** — modest but consistent with the continuous nature of human diagnostic behavior, where strategies form overlapping behavioral distributions rather than discrete categories.

### Spatial Attention Analysis

<p align="center">
  <img src="task_1_results/heatmap_cluster_0.png" width="30%" alt="Mixed Strategy Heatmap"/>
  <img src="task_1_results/heatmap_cluster_1.png" width="30%" alt="Rapid Scanner Heatmap"/>
  <img src="task_1_results/heatmap_cluster_2.png" width="30%" alt="Focused Inspector Heatmap"/>
</p>
<p align="center">
  <em>Figure 1 — Gaze heatmaps per cluster. Left: Mixed Strategy (broad bilateral coverage). Center: Rapid Scanner (diffuse distribution). Right: Focused Inspector (concentrated regions).</em>
</p>

### Representative Scanpaths

<p align="center">
  <img src="task_1_results/scanpath_cluster_0.png" width="30%" alt="Mixed Strategy Scanpath"/>
  <img src="task_1_results/scanpath_cluster_1.png" width="30%" alt="Rapid Scanner Scanpath"/>
  <img src="task_1_results/scanpath_cluster_2.png" width="30%" alt="Focused Inspector Scanpath"/>
</p>
<p align="center">
  <em>Figure 2 — Representative scanpaths. Mixed Strategy: long paths with cross-lung transitions. Rapid Scanner: fast inter-region saccades. Focused Inspector: short paths with prolonged fixations.</em>
</p>

### AOI Transition Analysis

<p align="center">
  <img src="task_1_results/transition_matrices.png" width="70%" alt="AOI Transition Matrices"/>
</p>
<p align="center">
  <em>Figure 3 — Average transition probabilities between anatomical regions of interest. Mixed Strategy clinicians exhibit strong bilateral lung comparisons (L→R ≈ 0.95). Rapid Scanners show distributed lung–mediastinum transitions. Focused Inspectors maintain lung-to-lung transitions with fewer exploratory movements.</em>
</p>

### AOI Dwell Time Distribution

<p align="center">
  <img src="task_1_results/aoi_dwell_distribution.png" width="70%" alt="AOI Dwell Distribution"/>
</p>
<p align="center">
  <em>Figure 4 — Dwell time distribution across anatomical regions. Focused Inspectors allocate prolonged dwell to specific zones (cardiac silhouette, lower lung). Mixed Strategy clinicians distribute viewing time more uniformly.</em>
</p>

### Gaze–Speech Temporal Coordination

<p align="center">
  <img src="task_1_results/gaze_speech_lag.png" width="60%" alt="Gaze-Speech Temporal Lag"/>
</p>
<p align="center">
  <em>Figure 5 — Temporal lag between visual fixation onset and verbal reasoning onset. Mixed Strategy: small positive lag (gaze leads speech). Rapid Scanner: near-zero or negative lag (concurrent processing). Focused Inspector: high variability (flexible coordination).</em>
</p>

### Feature Ablation Study

<p align="center">
  <img src="task_1_results/ablation_comparison.png" width="60%" alt="Feature Ablation"/>
</p>

| Feature Set | Dimensionality | Silhouette Score |
|---|---|---|
| Gaze behavioral only | 8 | **0.369** |
| Speech only | 14 | 0.211 |
| Gaze full (behavioral + AOI) | 19 | 0.170 |
| Multimodal (all) | 36 | 0.104 |

<p align="center">
  <em>Table — Clustering performance under feature ablation. Gaze behavioral features alone yield the highest cluster separability, indicating that temporal visual attention dynamics are the dominant signal for distinguishing clinician strategies. The reduction in silhouette under multimodal fusion reflects the curse of dimensionality and the weaker discriminative contribution of speech and cross-modal features.</em>
</p>

### Clustering Dashboard

<p align="center">
  <img src="task_1_results/clustering_dashboard.png" width="80%" alt="Clustering Dashboard"/>
</p>
<p align="center">
  <em>Figure 6 — Full clustering dashboard. PCA and UMAP projections of the 46-dimensional behavioral space with cluster-level feature profiles.</em>
</p>

## Key Findings

- **Three interpretable clinician reading strategies** emerge from unsupervised clustering of multimodal behavioral data, consistent with established models of radiological reasoning (systematic search, pattern recognition, focal analysis).
- **Gaze dynamics are the dominant modality** for behavioral clustering (silhouette = 0.369 vs. 0.211 for speech, 0.104 for full multimodal).
- **Speech features capture complementary reasoning signals** (negation patterns, anatomical references) but reduce cluster separability when fused naïvely with gaze features, suggesting that more sophisticated fusion strategies (e.g., attention-based or late fusion) may be warranted.
- **Gaze–speech temporal coordination patterns** differ systematically across strategies, providing evidence that visual–verbal coupling is an informative dimension of clinical reasoning behavior.
- These findings support the use of **gaze-based behavioral signatures as interpretable features** for modeling diagnostic reasoning and developing cognitively informed clinical AI systems.


---



## Reproduction

### Requirements

```
Python >= 3.9
numpy
pandas
scikit-learn
scipy
sentence-transformers
umap-learn
matplotlib
seaborn
```

### Installation

```bash
git clone https://github.com/ahmadsuleman/Multi_Model-Clinical-Dataset-Generator.git
cd Multi_Model-Clinical-Dataset-Generator
pip install -r requirements.txt
```

### Data Generation

```bash
# Generate the multimodal clinician interaction dataset
python generate_dataset.py --n_sessions 50 --output_dir data/
```

### Running the Pipeline

```bash
# Feature extraction and clustering
python run_pipeline.py --data_dir data/ --output_dir task_1_results/
```

## Project Structure

```
.
├── README.md
├── requirements.txt
├── generate_dataset.py          # Multimodal data generation
├── run_pipeline.py              # Main analysis pipeline
├── data/
│   └── session_XXX/
│       ├── image.jpeg
│       ├── gaze.csv
│       ├── transcription.csv
│       ├── audio.wav
│       └── metadata.json
└── task_1_results/
    ├── clustering_dashboard.png
    ├── heatmap_cluster_0.png
    ├── heatmap_cluster_1.png
    ├── heatmap_cluster_2.png
    ├── scanpath_cluster_0.png
    ├── scanpath_cluster_1.png
    ├── scanpath_cluster_2.png
    ├── transition_matrices.png
    ├── aoi_dwell_distribution.png
    ├── gaze_speech_lag.png
    └── ablation_comparison.png
```

## License

This project is released for academic research purposes. Please see `LICENSE` for details.

## Citation

If you use this work in your research, please cite:

```bibtex
@misc{suleman2025multimodal,
  title   = {Modeling Clinical Reasoning from Multimodal Interaction Signals},
  author  = {Suleman, Ahmad},
  year    = {2025},
  url     = {https://github.com/ahmadsuleman/Multi_Model-Clinical-Dataset-Generator}
}
```

## Acknowledgments

This work builds on tools and methods from the open-source scientific computing ecosystem, including [Sentence-Transformers](https://www.sbert.net/), [UMAP](https://umap-learn.readthedocs.io/), and [scikit-learn](https://scikit-learn.org/).

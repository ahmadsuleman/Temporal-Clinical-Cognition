
## **Abstract**

This work presents a computational framework for analyzing clinician behavior during chest X-ray (CXR) interpretation using multimodal interaction signals. The pipeline integrates **eye-tracking data, spoken diagnostic reasoning, and cross-modal temporal alignment** to characterize latent radiological reading strategies through unsupervised representation learning.

A dataset of **50 clinician interpretation sessions** was processed, each containing synchronized gaze recordings and speech transcripts. Eye-tracking data were mapped to anatomically defined **regions of interest (ROIs)** within the radiograph, from which gaze descriptors were extracted, including scanpath statistics, revisit rates, velocity measures, and **AOI dwell distributions**. Spatial attention dynamics were further modeled using **AOI transition matrices**, which were reduced via **Principal Component Analysis (PCA)** to obtain compact representations of visual search behavior.

Speech transcripts were encoded using a hybrid natural language processing approach. Semantic representations were generated using the **Sentence-Transformers model (all-MiniLM-L6-v2)** and compressed using PCA, while interpretable linguistic indicators—such as anatomical mentions, radiological findings, negations, and uncertainty markers—were extracted through rule-based clinical term parsing. To capture cognitive coordination between modalities, **cross-modal features** were computed, including gaze-to-speech temporal lag and gaze dwell within verbally referenced anatomical regions.

All modalities were standardized and combined into a **46-dimensional multimodal behavioral feature representation** per session. Latent clinician interaction patterns were identified using **density-based clustering (HDBSCAN)**, yielding **two behavioral groups (k = 2)** with a silhouette score of **0.1129**. A systematic **feature ablation analysis** demonstrated that **gaze behavioral features alone achieved the highest cluster separability (silhouette = 0.3047)**, followed by speech features (0.2843), indicating that visual search behavior provides the strongest signal for differentiating clinician strategies.

The proposed framework integrates **eye-tracking analytics, transformer-based language embeddings, dimensionality reduction, cross-modal temporal analysis, and density-based unsupervised clustering** to model clinician diagnostic behavior from multimodal interaction data.


## Stage 1: 
A synthetic reading-session simulator generates multimodal clinician interaction data by producing gaze trajectories, dictation transcripts, and structured diagnostic findings from chest X-ray images, thereby enabling controlled experiments on multimodal temporal modeling of clinician behavior.


## 📊 Data Modalities

The dataset provides five synchronized behavioural modalities per session:

| # | Modality | File | Description |
|---|----------|------|-------------|
| 1 | **Image** | `image.jpeg` | Chest X-ray used as the visual stimulus |
| 2 | **Visual Attention** | `gaze.csv` | Simulated eye-tracking scanpath (~60 Hz) |
| 3 | **Verbal Reasoning** | `transcription.csv` | Time-stamped diagnostic dictation segments |
| 4 | **Audio** | `audio.wav` | Text-to-speech generated dictation recording |
| 5 | **Documentation** | `metadata.json` | Structured diagnostic findings (report-level) |

### Data Generation Simulator Link: 
```
https://github.com/ahmadsuleman/Multi_Model-Clinical-Dataset-Generator.git
```
---

## Methodology Summary

The proposed system follows a modular multimodal processing architecture designed to transform raw clinician interaction data into structured behavioral representations for unsupervised pattern discovery. The architecture consists of five sequential layers: data ingestion, modality-specific feature extraction, multimodal representation construction, unsupervised clustering, and behavioral analysis.

---

## Implementation Summary

The overall architecture integrates computer vision interaction analytics, natural language processing, dimensionality reduction, and density-based unsupervised learning within a unified analytical pipeline. The system is implemented entirely in Python, combining scientific computing tools with modern NLP and machine learning frameworks to enable reproducible analysis of multimodal clinician interaction data.

---

## Results Interpertation

The clustering analysis of 50 clinician sessions (46 features) identified three distinct behavioral strategies, although the overall cluster separation was modest (silhouette = 0.104), consistent with the continuous nature of human diagnostic behavior. The Mixed Strategy cluster (n=18) exhibited high scanpath length (z = +0.964), fixation count (z = +0.935), and revisit rate (z = +0.935), alongside elevated negation (z = +0.778) and anatomical mentions (z = +0.674), indicating systematic exploration coupled with verbal hypothesis testing. In contrast, the Rapid Scanner cluster (n=17) showed markedly higher gaze velocity (z = +1.299) and velocity variability (z = +1.047) but reduced anatomical references (z = −0.568) and negation markers (z = −0.703), suggesting a pattern-recognition strategy driven by rapid visual search. The Focused Inspector cluster (n=15) was characterized by prolonged fixation durations (mean z = +1.001, max z = +0.904) and reduced gaze velocity (z = −0.899), indicating concentrated inspection of limited image regions.

The ablation study further shows that gaze behavioral features alone produced the strongest cluster separation (silhouette = 0.369), whereas multimodal feature combinations reduced separability (≈0.104–0.107), suggesting that visual attention dynamics provide the dominant signal for distinguishing clinician strategies. Speech features alone produced intermediate structure (silhouette = 0.211), indicating that verbal reasoning markers capture complementary but weaker signals. Together, these findings suggest that temporal gaze patterns encode distinct diagnostic workflows, providing interpretable behavioral signatures of clinical reasoning during image interpretation.


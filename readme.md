
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


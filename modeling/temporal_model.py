"""
temporal_model.py — Self-supervised Transformer for radiologist behaviour embedding.

Requires PyTorch (torch >= 2.0).  If PyTorch is not available, or if training
does not converge (< 10% improvement over initial loss), the module falls back
to a PCA-based surrogate.

All embedding outputs are PCA-reduced to n_output_dims (default 10) before
returning, preventing the temporal block from dominating the feature matrix.

Public API
----------
create_temporal_sequences(all_cases_data, bin_size_ms=500)
    → np.ndarray  (n_sessions, max_seq_len, FEATURE_DIM)

extract_transformer_embeddings(sequences, n_epochs=150, lr=5e-4,
                               device="cpu", n_output_dims=10)
    → np.ndarray  (n_sessions, n_output_dims),  source: str

TORCH_AVAILABLE : bool
"""

import warnings
import math
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    TORCH_AVAILABLE = True
except ModuleNotFoundError:
    TORCH_AVAILABLE = False

AOI_NAMES   = ["left_lung", "right_lung", "heart", "lower_left", "lower_right", "background"]
N_AOI       = len(AOI_NAMES)
FEATURE_DIM = N_AOI + 3      # one-hot(6) + fix_dur(1) + saccade_vel(1) + speech_active(1) = 9
D_MODEL     = 64
MASK_RATIO  = 0.15
MIN_CONVERGENCE_IMPROVEMENT = 0.10   # 10% improvement required


# ── Sequence builder ──────────────────────────────────────────────────────────

def create_temporal_sequences(all_cases_data, bin_size_ms=500):
    """
    Discretise each session into fixed-width time bins.

    Parameters
    ----------
    all_cases_data : list[dict]  — keys: gaze_df, transcription_df, aoi_sequence
    bin_size_ms    : int

    Returns
    -------
    np.ndarray  (n_sessions, max_seq_len, FEATURE_DIM=9)
    """
    aoi_idx = {name: i for i, name in enumerate(AOI_NAMES)}
    bin_sec = bin_size_ms / 1000.0

    all_seqs = []
    for case in all_cases_data:
        gaze_df  = case["gaze_df"]
        trans_df = case["transcription_df"]
        aoi_seq  = case["aoi_sequence"]

        ts = gaze_df["timestamp_sec"].values.astype(float)
        xs = gaze_df["x"].values.astype(float)
        ys = gaze_df["y"].values.astype(float)

        t_start = ts[0]
        t_end   = ts[-1]
        n_bins  = max(1, int(np.ceil((t_end - t_start) / bin_sec)))

        velocities = np.zeros(len(ts))
        if len(ts) > 1:
            velocities[1:] = np.sqrt(np.diff(xs)**2 + np.diff(ys)**2)

        speech_intervals = []
        if trans_df is not None and "timestamp_start" in trans_df.columns:
            try:
                t_starts = trans_df["timestamp_start"].astype(float).values
                t_ends   = (trans_df["timestamp_end"].astype(float).values
                            if "timestamp_end" in trans_df.columns
                            else t_starts + 1.0)
                speech_intervals = list(zip(t_starts, t_ends))
            except (ValueError, KeyError):
                pass

        seq = np.zeros((n_bins, FEATURE_DIM), dtype=np.float32)
        for b in range(n_bins):
            bin_lo = t_start + b * bin_sec
            bin_hi = bin_lo  + bin_sec
            mask   = (ts >= bin_lo) & (ts < bin_hi)
            if mask.sum() == 0:
                continue

            bin_aois = [aoi_seq[i] for i in np.where(mask)[0]]
            counts   = {a: bin_aois.count(a) for a in set(bin_aois)}
            dominant = max(counts, key=counts.get)
            seq[b, aoi_idx.get(dominant, N_AOI - 1)] = 1.0
            seq[b, N_AOI]     = float(mask.sum()) * bin_sec / max(1, len(set(bin_aois)))
            seq[b, N_AOI + 1] = float(velocities[mask].mean())
            for t0, t1 in speech_intervals:
                if t0 < bin_hi and t1 > bin_lo:
                    seq[b, N_AOI + 2] = 1.0
                    break

        all_seqs.append(seq)

    max_len = max(s.shape[0] for s in all_seqs)
    padded  = np.zeros((len(all_seqs), max_len, FEATURE_DIM), dtype=np.float32)
    for i, s in enumerate(all_seqs):
        padded[i, :s.shape[0]] = s
    return padded


# ── PyTorch model ─────────────────────────────────────────────────────────────

if TORCH_AVAILABLE:

    class _SinusoidalPE(nn.Module):
        def __init__(self, d_model, max_len=2000):
            super().__init__()
            pe  = torch.zeros(max_len, d_model)
            pos = torch.arange(max_len).unsqueeze(1).float()
            div = torch.exp(torch.arange(0, d_model, 2).float()
                            * (-math.log(10000.0) / d_model))
            pe[:, 0::2] = torch.sin(pos * div)
            pe[:, 1::2] = torch.cos(pos * div)
            self.register_buffer("pe", pe.unsqueeze(0))

        def forward(self, x):
            return x + self.pe[:, :x.size(1)]

    class BehaviorTransformer(nn.Module):
        def __init__(self, input_dim=FEATURE_DIM, d_model=D_MODEL,
                     nhead=4, num_layers=2, dim_feedforward=128, dropout=0.1):
            super().__init__()
            self.input_proj = nn.Linear(input_dim, d_model)
            self.pos_enc    = _SinusoidalPE(d_model)
            encoder_layer   = nn.TransformerEncoderLayer(
                d_model=d_model, nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout, batch_first=True,
            )
            self.encoder  = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
            self.aoi_head = nn.Linear(d_model, N_AOI)

        def forward(self, x, src_key_padding_mask=None):
            h   = self.pos_enc(self.input_proj(x))
            h   = self.encoder(h, src_key_padding_mask=src_key_padding_mask)
            emb = h.mean(dim=1)
            return emb, self.aoi_head(h)


def _train_torch(sequences, n_epochs, lr, device):
    X        = torch.tensor(sequences, dtype=torch.float32, device=device)
    pad_mask = (X.abs().sum(dim=-1) == 0)
    model    = BehaviorTransformer().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn   = nn.CrossEntropyLoss()

    loss_history = []
    for epoch in range(n_epochs):
        model.train()
        B, T, _ = X.shape
        mask        = (torch.rand(B, T, device=device) < MASK_RATIO) & ~pad_mask
        aoi_targets = X[:, :, :N_AOI].argmax(dim=-1)
        x_masked    = X.clone()
        x_masked[mask] = 0.0

        _, logits = model(x_masked, src_key_padding_mask=pad_mask)
        if mask.sum() == 0:
            continue
        loss = loss_fn(logits[mask], aoi_targets[mask])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_history.append(float(loss))

        if (epoch + 1) % 50 == 0:
            print(f"  [Transformer] epoch {epoch+1:3d}/{n_epochs}  loss={loss:.4f}")

    return model, loss_history


def _pca_fallback(sequences, n_components):
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    B, T, F = sequences.shape
    flat    = sequences.reshape(B, T * F)
    flat_s  = StandardScaler().fit_transform(flat)
    n_comp  = min(n_components, flat_s.shape[0] - 1, flat_s.shape[1])
    emb     = PCA(n_components=n_comp).fit_transform(flat_s)
    if emb.shape[1] < n_components:
        emb = np.concatenate([emb,
               np.zeros((B, n_components - emb.shape[1]), dtype=np.float32)], axis=1)
    return emb.astype(np.float32)


def _apply_output_pca(embeddings, n_output_dims):
    """Reduce high-dim embeddings to exactly n_output_dims via PCA."""
    from sklearn.decomposition import PCA
    n = min(n_output_dims, embeddings.shape[0] - 1, embeddings.shape[1])
    emb = PCA(n_components=n).fit_transform(embeddings)
    if emb.shape[1] < n_output_dims:
        emb = np.concatenate([emb,
               np.zeros((emb.shape[0], n_output_dims - emb.shape[1]),
                        dtype=np.float32)], axis=1)
    return emb.astype(np.float32)


# ── Public API ────────────────────────────────────────────────────────────────

def extract_transformer_embeddings(sequences, n_epochs=150, lr=5e-4,
                                   device="cpu", n_output_dims=10):
    """
    Train or approximate temporal behaviour embeddings.

    Returns embeddings of shape (n_sessions, n_output_dims) and a source string.
    Falls back to PCA if:
      - PyTorch is not installed
      - Training raises an exception
      - Loss improved < MIN_CONVERGENCE_IMPROVEMENT (10%) from epoch 1 to end
    """
    if not TORCH_AVAILABLE:
        warnings.warn(
            "PyTorch not found — using PCA fallback for temporal embeddings.",
            RuntimeWarning, stacklevel=2,
        )
        return _pca_fallback(sequences, n_output_dims), "pca_fallback"

    try:
        model, loss_history = _train_torch(sequences, n_epochs, lr, device)

        # ── Convergence check ────────────────────────────────────────────────
        if len(loss_history) >= 2:
            initial = loss_history[0]
            final   = loss_history[-1]
            if initial > 0:
                improvement = (initial - final) / initial
                if improvement < MIN_CONVERGENCE_IMPROVEMENT:
                    warnings.warn(
                        f"Transformer did not converge "
                        f"(improvement={improvement:.1%} < {MIN_CONVERGENCE_IMPROVEMENT:.0%}). "
                        "Falling back to PCA temporal embeddings.",
                        RuntimeWarning, stacklevel=2,
                    )
                    return _pca_fallback(sequences, n_output_dims), "pca_fallback"

        if loss_history and math.isnan(loss_history[-1]):
            raise ValueError("Training diverged (NaN loss).")

        model.eval()
        X_t = torch.tensor(sequences, dtype=torch.float32, device=device)
        with torch.no_grad():
            raw_emb, _ = model(X_t)
        raw_np = raw_emb.cpu().numpy().astype(np.float32)

        return _apply_output_pca(raw_np, n_output_dims), "transformer"

    except Exception as exc:
        warnings.warn(
            f"Transformer training failed ({exc}). "
            "Falling back to PCA temporal embeddings.",
            RuntimeWarning, stacklevel=2,
        )
        return _pca_fallback(sequences, n_output_dims), "pca_fallback"

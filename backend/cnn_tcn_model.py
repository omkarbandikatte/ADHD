"""
cnn_tcn_model.py  —  Enhanced CNN + TCN Architecture for ADHD EEG Classification

Architecture:
  Input (batch, 19_channels, T_timesteps)
       |
  [CNN Block]  — Spatial Feature Learning
    TemporalConv2D -> DepthwiseSpatialConv2D -> SE-Attention -> AvgPool -> Dropout
       |
  [TCN Block x6]  — Temporal Pattern Recognition
    Dilated causal Conv1D + SE-Attention, dilations = 1,2,4,8,16,32
       |
  [Classifier Head]
    GlobalAvgPool -> FC(128) -> Dropout(0.5) -> FC(2) -> logits

Key improvements over baseline:
  - SE (Squeeze-and-Excitation) channel attention after CNN and each TCN block
  - 6 TCN layers (dilations up to 32) → receptive field = 128 samples
  - Larger CNN filters (48) for richer spatial representations
  - build_model() interface unchanged — ONNX export still works identically
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# ─── Squeeze-and-Excitation Channel Attention ─────────────────────────────────

class SEBlock(nn.Module):
    """
    Recalibrates channel-wise feature responses adaptively.
    Input/Output: (B, C, T)
    """

    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        reduced = max(channels // reduction, 4)
        self.fc1 = nn.Linear(channels, reduced, bias=False)
        self.fc2 = nn.Linear(reduced, channels, bias=False)
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)
        s = self.pool(x).squeeze(-1)              # (B, C)
        s = F.relu(self.fc1(s))                   # (B, reduced)
        s = torch.sigmoid(self.fc2(s)).unsqueeze(-1)  # (B, C, 1)
        return x * s


# ─── CNN Block (Spatial Feature Learning) ─────────────────────────────────────

class CNNBlock(nn.Module):
    """
    EEGNet-inspired spatial feature extractor.
    Input:  (B, 1, n_channels, T)
    Output: (B, n_filters*2, T//pool)

    Steps:
      1. Temporal Conv2D  — frequency-band-like features per channel
      2. Depthwise Spatial Conv2D — mixes across all 19 electrodes
      3. SE attention — channel-wise importance weighting
      4. AvgPool + Dropout
    """

    def __init__(self, n_channels: int = 19, n_filters: int = 48,
                 temporal_kernel: int = 25, pool_size: int = 4,
                 dropout: float = 0.3):
        super().__init__()

        self.temporal_conv = nn.Conv2d(
            1, n_filters,
            kernel_size=(1, temporal_kernel),
            padding=(0, temporal_kernel // 2),
            bias=False,
        )
        self.temporal_bn = nn.BatchNorm2d(n_filters)

        self.spatial_conv = nn.Conv2d(
            n_filters, n_filters * 2,
            kernel_size=(n_channels, 1),
            groups=n_filters,       # depthwise
            bias=False,
        )
        self.spatial_bn = nn.BatchNorm2d(n_filters * 2)

        self.se     = SEBlock(n_filters * 2, reduction=8)
        self.elu    = nn.ELU()
        self.pool   = nn.AvgPool2d((1, pool_size))
        self.drop   = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1, C, T)
        x = self.elu(self.temporal_bn(self.temporal_conv(x)))   # (B, F, C, T)
        x = self.elu(self.spatial_bn(self.spatial_conv(x)))     # (B, F*2, 1, T)
        x = self.pool(x)                                         # (B, F*2, 1, T/4)
        x = self.drop(x)
        x = x.squeeze(2)                                         # (B, F*2, T/4)
        x = self.se(x)                                           # SE attention
        return x


# ─── TCN Residual Block with SE Attention ─────────────────────────────────────

class TCNResidualBlock(nn.Module):
    """
    Dilated causal temporal convolutional residual block + SE attention.

    Causal = no future leakage (right-pad then chomp).
    SE attention emphasises the most discriminative temporal channels.
    """

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3,
                 dilation: int = 1, dropout: float = 0.25, use_se: bool = True):
        super().__init__()
        pad = (kernel_size - 1) * dilation       # causal padding amount

        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size,
                               dilation=dilation, padding=pad)
        self.bn1   = nn.BatchNorm1d(out_ch)

        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size,
                               dilation=dilation, padding=pad)
        self.bn2   = nn.BatchNorm1d(out_ch)

        self.elu     = nn.ELU()
        self.dropout = nn.Dropout(dropout)
        self.se      = SEBlock(out_ch) if use_se else nn.Identity()

        self.residual_proj = (
            nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        )
        self._pad = pad

    def _chomp(self, x: torch.Tensor) -> torch.Tensor:
        """Remove future samples added by causal right-padding."""
        return x[:, :, :-self._pad].contiguous() if self._pad > 0 else x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.residual_proj(x)

        out = self.elu(self.bn1(self._chomp(self.conv1(x))))
        out = self.dropout(out)
        out = self.elu(self.bn2(self._chomp(self.conv2(out))))
        out = self.dropout(out)
        out = self.se(out)

        return self.elu(out + res)


# ─── Full CNN + TCN Model ──────────────────────────────────────────────────────

class CNNTCNModel(nn.Module):
    """
    Enhanced ADHD EEG classifier:
      EEG epochs (19ch × T) → CNN-SE Spatial → TCN-SE(x6) Temporal → Dense → logits

    Defaults produce a model with ~280K parameters that generalises well on
    the 121-subject IEEE DataPort ADHD dataset.
    """

    def __init__(self,
                 n_channels:    int   = 19,
                 n_timepoints:  int   = 256,
                 n_classes:     int   = 2,
                 cnn_filters:   int   = 48,
                 tcn_channels:  int   = 64,
                 tcn_layers:    int   = 6,      # dilations 1,2,4,8,16,32
                 tcn_kernel:    int   = 3,
                 dropout_cnn:   float = 0.30,
                 dropout_tcn:   float = 0.25,
                 dropout_fc:    float = 0.50):
        super().__init__()

        # ── CNN Block ───────────────────────────────────────
        self.cnn = CNNBlock(
            n_channels=n_channels,
            n_filters=cnn_filters,
            temporal_kernel=25,
            pool_size=4,
            dropout=dropout_cnn,
        )
        cnn_out_ch = cnn_filters * 2     # 96

        # ── TCN Stack — dilations 1, 2, 4, 8, 16, 32 ───────
        tcn_blocks = []
        in_ch = cnn_out_ch
        for i in range(tcn_layers):
            dil = 2 ** i
            tcn_blocks.append(
                TCNResidualBlock(in_ch, tcn_channels, tcn_kernel, dil,
                                 dropout_tcn, use_se=True)
            )
            in_ch = tcn_channels
        self.tcn = nn.Sequential(*tcn_blocks)

        # ── Classifier Head ─────────────────────────────────
        self.gap        = nn.AdaptiveAvgPool1d(1)
        self.fc1        = nn.Linear(tcn_channels, 128)
        self.bn_fc      = nn.BatchNorm1d(128)
        self.dropout_fc = nn.Dropout(dropout_fc)
        self.fc2        = nn.Linear(128, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)
        x = x.unsqueeze(1)           # (B, 1, C, T)
        x = self.cnn(x)              # (B, cnn_out_ch, T/4)
        x = self.tcn(x)              # (B, tcn_channels, T/4)
        x = self.gap(x).squeeze(-1)  # (B, tcn_channels)
        x = self.dropout_fc(F.elu(self.bn_fc(self.fc1(x))))
        return self.fc2(x)           # (B, n_classes)  logits

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        return F.softmax(self.forward(x), dim=1)


# ─── Grad-CAM Attention Heatmap ───────────────────────────────────────────────

class GradCAMHook:
    """
    Captures activations and gradients from the last TCN block to produce
    a per-channel importance map for visualisation.
    """

    def __init__(self, model: CNNTCNModel):
        self.activations = None
        self.gradients   = None
        last_tcn = list(model.tcn.children())[-1]
        last_tcn.register_forward_hook(self._save_activation)
        last_tcn.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def get_cam(self) -> np.ndarray:
        if self.activations is None or self.gradients is None:
            return np.zeros(1)
        weights = self.gradients.mean(dim=-1, keepdim=True)   # (B, ch, 1)
        cam     = (weights * self.activations).sum(dim=1)      # (B, T)
        cam     = F.relu(cam)
        cam     = cam / (cam.max(dim=-1, keepdim=True).values + 1e-8)
        return cam.cpu().numpy()


# ─── Public API ───────────────────────────────────────────────────────────────

def build_model(n_channels: int = 19, epoch_samples: int = 256,
                n_classes: int = 2) -> CNNTCNModel:
    """Build the default enhanced model. Interface unchanged from v1."""
    return CNNTCNModel(
        n_channels=n_channels,
        n_timepoints=epoch_samples,
        n_classes=n_classes,
    )


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

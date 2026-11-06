"""Neural network models.

The primary model is a faithful reconstruction of *Configuration 5* from the
original paper's Table IV:

    Conv1D(1 -> 4, k=4, s=2) + BN + ReLU
    Conv1D(4 -> 8, k=4, s=1) + BN + ReLU
    Conv1D(8 -> 16, k=4, s=1) + BN + ReLU
    AvgPool(16)
    Conv1D(16 -> 8, k=1) + BN + ReLU
    Flatten
    Fully connected -> number of classes

With a 540-sample input (1.5 s at 1440 Hz after a 2-level Haar DWT) the
flattened feature vector has 8 * 16 = 128 values and the binary classifier has
**1150** parameters.  That matches the paper's reported 1537 parameters for
the five-class version plus one extra output unit.

The ECG branch and the fusion model are small additions for the optional E3
experiment.  They are deliberately *not* a reproduction of LGC2-Net and contain
no attention modules.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Shape helpers
# ---------------------------------------------------------------------------
def conv1d_output_length(
    n: int,
    kernel: int,
    stride: int = 1,
    padding: int = 0,
    dilation: int = 1,
) -> int:
    """Output length of a 1-D convolution (no padding by default)."""
    return (int(n) + 2 * int(padding) - int(dilation) * (int(kernel) - 1) - 1) // int(
        stride
    ) + 1


def _pool_plan(
    post_conv_length: int,
    pool_kernel: int,
    pool_output_length: int,
) -> tuple[str, int]:
    """Choose non-overlapping pooling or adaptive pooling.

    The paper specifies ``AvgPool(16)`` for a 540-sample input, where a kernel
    of 16 produces exactly 16 frames.  For other input lengths (for example
    the 2000 Hz sensitivity profile) a kernel of 16 would produce a different
    feature dimension.  In that case we use adaptive pooling to keep the same
    ``pool_output_length`` frames, so the classifier head keeps the same shape.
    The choice is deterministic and recorded in the model card.
    """
    if post_conv_length > 0 and post_conv_length // int(pool_kernel) == int(
        pool_output_length
    ):
        return "avg_kernel", int(pool_kernel)
    return "adaptive", int(pool_output_length)


@dataclass
class ModelInfo:
    """Static description of a model, useful for reports."""

    name: str
    n_parameters: int
    n_trainable_parameters: int
    input_length: int
    output_dim: int
    flatten_dim: int
    pool_mode: str
    pool_kernel: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "n_parameters": int(self.n_parameters),
            "n_trainable_parameters": int(self.n_trainable_parameters),
            "input_length": int(self.input_length),
            "output_dim": int(self.output_dim),
            "flatten_dim": int(self.flatten_dim),
            "pool_mode": self.pool_mode,
            "pool_kernel": int(self.pool_kernel),
        }


# ---------------------------------------------------------------------------
# PCG branch (Configuration 5)
# ---------------------------------------------------------------------------
class PCGCNN(nn.Module):
    """Lightweight 1-D CNN used identically for E1 and E2.

    Parameters
    ----------
    input_length:
        Number of samples in one window *after* the profile's DWT.
    conv_channels, kernel_sizes, strides:
        Three convolution stages.  Defaults reproduce the paper.
    pointwise_channels:
        Channels of the final 1x1 convolution.
    pool_kernel, pool_output_length:
        Pooling configuration.  ``pool_kernel=16`` is the paper setting.
    num_classes:
        2 for binary HFrEF classification.
    dropout:
        Applied after the flatten layer; 0.0 matches the paper.
    """

    def __init__(
        self,
        input_length: int = 540,
        conv_channels: Sequence[int] = (4, 8, 16),
        kernel_sizes: Sequence[int] = (4, 4, 4),
        strides: Sequence[int] = (2, 1, 1),
        pointwise_channels: int = 8,
        pool_kernel: int = 16,
        pool_output_length: int = 16,
        num_classes: int = 2,
        dropout: float = 0.0,
        in_channels: int = 1,
    ) -> None:
        super().__init__()
        if not (len(conv_channels) == len(kernel_sizes) == len(strides)):
            raise ValueError("conv_channels, kernel_sizes and strides must have equal length")
        if len(conv_channels) != 3:
            raise ValueError(
                "Configuration 5 expects exactly three convolution stages; "
                f"got {len(conv_channels)}."
            )
        self.input_length = int(input_length)
        self.num_classes = int(num_classes)
        self.pool_kernel = int(pool_kernel)
        self.pool_output_length = int(pool_output_length)

        layers: list[nn.Module] = []
        channels_in = int(in_channels)
        length = int(input_length)
        for channels_out, kernel, stride in zip(conv_channels, kernel_sizes, strides):
            layers.append(
                nn.Conv1d(
                    channels_in,
                    int(channels_out),
                    kernel_size=int(kernel),
                    stride=int(stride),
                    bias=True,
                )
            )
            layers.append(nn.BatchNorm1d(int(channels_out)))
            layers.append(nn.ReLU(inplace=True))
            length = conv1d_output_length(length, int(kernel), int(stride))
            channels_in = int(channels_out)
        if length <= 0:
            raise ValueError(
                f"Input length {input_length} is too short for the convolution stack."
            )

        self.pool_mode, self.pool_value = _pool_plan(
            length, self.pool_kernel, self.pool_output_length
        )
        if self.pool_mode == "avg_kernel":
            layers.append(nn.AvgPool1d(kernel_size=self.pool_value, stride=self.pool_value))
            pooled_length = length // self.pool_value
        else:
            layers.append(nn.AdaptiveAvgPool1d(self.pool_value))
            pooled_length = self.pool_value

        layers.append(
            nn.Conv1d(
                channels_in,
                int(pointwise_channels),
                kernel_size=1,
                stride=1,
                bias=True,
            )
        )
        layers.append(nn.BatchNorm1d(int(pointwise_channels)))
        layers.append(nn.ReLU(inplace=True))
        self.features = nn.Sequential(*layers)

        self.flatten_dim = int(pointwise_channels) * int(pooled_length)
        self.dropout = nn.Dropout(float(dropout)) if dropout and dropout > 0 else nn.Identity()
        self.classifier = nn.Linear(self.flatten_dim, self.num_classes)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return the flattened 128-dimensional feature vector."""
        x = x.unsqueeze(1) if x.dim() == 2 else x
        x = self.features(x)
        return torch.flatten(x, start_dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward_features(x)
        x = self.dropout(x)
        return self.classifier(x)

    def info(self) -> ModelInfo:
        return ModelInfo(
            name="pcg_cnn_config5",
            n_parameters=count_parameters(self, trainable_only=False),
            n_trainable_parameters=count_parameters(self, trainable_only=True),
            input_length=self.input_length,
            output_dim=self.num_classes,
            flatten_dim=self.flatten_dim,
            pool_mode=self.pool_mode,
            pool_kernel=self.pool_value,
        )


# ---------------------------------------------------------------------------
# ECG branch and fusion (optional E3)
# ---------------------------------------------------------------------------
class ECGCNN(nn.Module):
    """Small ECG feature extractor for the optional fusion experiment.

    It is intentionally much smaller than any branch of LGC2-Net.  It learns a
    compact embedding from the time-aligned ECG window; the PCG branch remains
    the Configuration-5 network.
    """

    def __init__(
        self,
        input_length: int = 750,
        conv_channels: Sequence[int] = (8, 16, 16),
        kernel_sizes: Sequence[int] = (7, 5, 3),
        strides: Sequence[int] = (2, 2, 2),
        pool_output_length: int = 8,
        feature_dim: int = 32,
        dropout: float = 0.1,
        in_channels: int = 1,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        channels_in = int(in_channels)
        length = int(input_length)
        for channels_out, kernel, stride in zip(conv_channels, kernel_sizes, strides):
            layers.append(
                nn.Conv1d(
                    channels_in,
                    int(channels_out),
                    kernel_size=int(kernel),
                    stride=int(stride),
                    bias=True,
                )
            )
            layers.append(nn.BatchNorm1d(int(channels_out)))
            layers.append(nn.ReLU(inplace=True))
            length = conv1d_output_length(length, int(kernel), int(stride))
            channels_in = int(channels_out)
            if length < 2:
                raise ValueError("ECG input is too short for the ECG branch.")
        layers.append(nn.AdaptiveAvgPool1d(int(pool_output_length)))
        self.features = nn.Sequential(*layers)
        self.flatten_dim = int(conv_channels[-1]) * int(pool_output_length)
        self.projection = nn.Sequential(
            nn.Linear(self.flatten_dim, int(feature_dim)),
            nn.ReLU(inplace=True),
            nn.Dropout(float(dropout)) if dropout and dropout > 0 else nn.Identity(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1) if x.dim() == 2 else x
        x = self.features(x)
        x = torch.flatten(x, start_dim=1)
        return self.projection(x)


class FusionCNN(nn.Module):
    """E3: Configuration-5 PCG branch concatenated with a small ECG branch."""

    def __init__(
        self,
        pcg_input_length: int = 540,
        ecg_input_length: int = 750,
        pcg_kwargs: Mapping[str, Any] | None = None,
        ecg_kwargs: Mapping[str, Any] | None = None,
        hidden_dim: int = 32,
        num_classes: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        pcg_kwargs = dict(pcg_kwargs or {})
        ecg_kwargs = dict(ecg_kwargs or {})
        self.pcg = PCGCNN(
            input_length=pcg_input_length,
            num_classes=num_classes,
            **pcg_kwargs,
        )
        ecg_feature_dim = int(ecg_kwargs.pop("feature_dim", 32))
        self.ecg = ECGCNN(
            input_length=ecg_input_length,
            feature_dim=ecg_feature_dim,
            **ecg_kwargs,
        )
        # The ECG branch's output dimension is fixed by its constructor above.
        self.hidden_dim = int(hidden_dim)
        fusion_dim = int(self.pcg.flatten_dim + ecg_feature_dim)
        self.head = nn.Sequential(
            nn.Linear(fusion_dim, self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(float(dropout)) if dropout and dropout > 0 else nn.Identity(),
            nn.Linear(self.hidden_dim, int(num_classes)),
        )

    def forward(self, pcg: torch.Tensor, ecg: torch.Tensor) -> torch.Tensor:
        pcg_features = self.pcg.forward_features(pcg)
        ecg_features = self.ecg(ecg)
        return self.head(torch.cat([pcg_features, ecg_features], dim=1))


# ---------------------------------------------------------------------------
# Construction / reporting
# ---------------------------------------------------------------------------
def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    """Count model parameters."""
    params = model.parameters()
    if trainable_only:
        return int(sum(p.numel() for p in params if p.requires_grad))
    return int(sum(p.numel() for p in params))


def build_model(
    cfg: Mapping[str, Any],
    input_length: int,
    model_type: str = "pcg",
    ecg_input_length: int | None = None,
) -> nn.Module:
    """Build the requested model from the configuration.

    Parameters
    ----------
    input_length:
        PCG samples per window after the profile's DWT.
    model_type:
        ``pcg`` (E1/E2), ``ecg``, or ``fusion`` (E3).
    ecg_input_length:
        ECG samples per window; required for ``fusion``.
    """
    model_cfg = dict(cfg.get("model", {}))
    fusion_cfg = dict(cfg.get("fusion", {}))
    num_classes = int(model_cfg.get("num_classes", 2))
    pcg_kwargs = dict(
        conv_channels=model_cfg.get("conv_channels", [4, 8, 16]),
        kernel_sizes=model_cfg.get("kernel_sizes", [4, 4, 4]),
        strides=model_cfg.get("strides", [2, 1, 1]),
        pointwise_channels=int(model_cfg.get("pointwise_channels", 8)),
        pool_kernel=int(model_cfg.get("pool_kernel", 16)),
        pool_output_length=int(model_cfg.get("pool_output_length", 16)),
        dropout=float(model_cfg.get("dropout", 0.0)),
    )
    if model_type == "pcg":
        return PCGCNN(input_length=int(input_length), num_classes=num_classes, **pcg_kwargs)
    if model_type == "ecg":
        return ECGCNN(input_length=int(input_length))
    if model_type == "fusion":
        if ecg_input_length is None:
            raise ValueError("ecg_input_length is required for the fusion model")
        ecg_kwargs = dict(
            conv_channels=fusion_cfg.get("ecg_conv_channels", [8, 16, 16]),
            kernel_sizes=fusion_cfg.get("ecg_kernel_sizes", [7, 5, 3]),
            strides=fusion_cfg.get("ecg_strides", [2, 2, 2]),
            pool_output_length=int(fusion_cfg.get("ecg_pool_output_length", 8)),
            feature_dim=int(fusion_cfg.get("ecg_feature_dim", 32)),
            dropout=float(fusion_cfg.get("dropout", 0.1)),
        )
        return FusionCNN(
            pcg_input_length=int(input_length),
            ecg_input_length=int(ecg_input_length),
            pcg_kwargs=pcg_kwargs,
            ecg_kwargs=ecg_kwargs,
            hidden_dim=int(fusion_cfg.get("head_hidden_dim", 32)),
            num_classes=num_classes,
            dropout=float(fusion_cfg.get("dropout", 0.1)),
        )
    raise ValueError(f"Unknown model_type {model_type!r}; use 'pcg', 'ecg' or 'fusion'.")


def model_summary(model: nn.Module) -> str:
    """Return a compact per-layer parameter table."""
    lines = []
    total = 0
    trainable = 0
    for name, parameter in model.named_parameters():
        n = int(parameter.numel())
        total += n
        if parameter.requires_grad:
            trainable += n
        lines.append(f"  {name:<48s} {str(tuple(parameter.shape)):>20s} {n:>10,d}")
    lines.append("-" * 82)
    lines.append(f"  {'TOTAL':<48s} {'':>20s} {total:>10,d}")
    lines.append(f"  {'TRAINABLE':<48s} {'':>20s} {trainable:>10,d}")
    return "\n".join(lines)


def forward_dummy(
    model: nn.Module,
    input_length: int,
    device: torch.device | str = "cpu",
    ecg_input_length: int | None = None,
) -> torch.Tensor:
    """Run one dummy forward pass and return the output (used for checks)."""
    device = torch.device(device)
    model = model.to(device).eval()
    x = torch.zeros(1, 1, int(input_length), device=device)
    with torch.no_grad():
        if isinstance(model, FusionCNN):
            if ecg_input_length is None:
                raise ValueError("ecg_input_length is required for FusionCNN")
            ecg = torch.zeros(1, 1, int(ecg_input_length), device=device)
            return model(x, ecg)
        return model(x)

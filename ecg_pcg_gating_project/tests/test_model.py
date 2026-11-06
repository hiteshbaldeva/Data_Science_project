"""Model architecture, parameter count and forward-pass checks."""

from __future__ import annotations

import torch

from src.model import (
    ECGCNN,
    FusionCNN,
    PCGCNN,
    build_model,
    count_parameters,
    forward_dummy,
    model_summary,
)


def test_configuration5_binary_parameter_count():
    model = PCGCNN(input_length=540, num_classes=2)
    assert count_parameters(model, trainable_only=False) == 1150
    assert model.flatten_dim == 128
    assert model.pool_mode == "avg_kernel"
    assert model.pool_value == 16


def test_configuration5_five_class_parameter_count():
    model = PCGCNN(input_length=540, num_classes=5)
    assert count_parameters(model, trainable_only=False) == 1537


def test_forward_pass_shape():
    model = PCGCNN(input_length=540, num_classes=2)
    x = torch.randn(4, 1, 540)
    output = model(x)
    assert output.shape == (4, 2)
    assert torch.isfinite(output).all()


def test_forward_accepts_2d_input():
    model = PCGCNN(input_length=540)
    output = model(torch.randn(3, 540))
    assert output.shape == (3, 2)


def test_other_input_length_uses_adaptive_pool():
    model = PCGCNN(input_length=2560, num_classes=2)
    assert model.pool_mode == "adaptive"
    assert model.flatten_dim == 128
    assert model(torch.randn(2, 1, 2560)).shape == (2, 2)


def test_ecg_branch_forward():
    model = ECGCNN(input_length=750)
    assert model(torch.randn(2, 1, 750)).shape == (2, 32)


def test_fusion_model_forward():
    model = FusionCNN(pcg_input_length=540, ecg_input_length=750)
    output = model(torch.randn(2, 1, 540), torch.randn(2, 1, 750))
    assert output.shape == (2, 2)


def test_build_model_from_config(cfg):
    model = build_model(cfg, input_length=540, model_type="pcg")
    assert isinstance(model, PCGCNN)
    assert count_parameters(model, trainable_only=False) == 1150
    fusion = build_model(cfg, input_length=540, model_type="fusion", ecg_input_length=750)
    assert isinstance(fusion, FusionCNN)
    assert forward_dummy(fusion, 540, ecg_input_length=750).shape == (1, 2)


def test_model_summary_mentions_total():
    text = model_summary(PCGCNN(input_length=540))
    assert "TOTAL" in text
    assert "1,150" in text

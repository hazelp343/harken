import pytest
import torch

from harken.projectors import build_projector, list_projectors

ALL = ["linear", "mlp", "stack", "query"]


def test_all_projectors_registered():
    for name in ALL:
        assert name in list_projectors()


@pytest.mark.parametrize("name", ALL)
def test_projector_output_shape(name):
    proj = build_projector(name, in_dim=24, out_dim=32, num_tokens=6)
    frames = torch.randn(3, 20, 24)
    out = proj(frames)
    assert out.shape == (3, 6, 32)


@pytest.mark.parametrize("name", ALL)
def test_projector_handles_short_input(name):
    proj = build_projector(name, in_dim=16, out_dim=16, num_tokens=8)
    frames = torch.randn(1, 2, 16)
    out = proj(frames)
    assert out.shape == (1, 8, 16)


def test_stack_factor_passthrough():
    proj = build_projector("stack", in_dim=8, out_dim=16, num_tokens=4, stack_factor=2)
    assert proj.stack_factor == 2
    out = proj(torch.randn(2, 9, 8))
    assert out.shape == (2, 4, 16)


def test_query_handles_indivisible_dim():
    # out_dim not divisible by 8 should still build (heads auto-reduced).
    proj = build_projector("query", in_dim=10, out_dim=12, num_tokens=3)
    out = proj(torch.randn(2, 7, 10))
    assert out.shape == (2, 3, 12)


def test_projectors_are_trainable():
    proj = build_projector("mlp", in_dim=8, out_dim=8, num_tokens=2)
    out = proj(torch.randn(1, 5, 8)).sum()
    out.backward()
    assert any(p.grad is not None for p in proj.parameters())

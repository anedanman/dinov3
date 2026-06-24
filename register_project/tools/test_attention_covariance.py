#!/usr/bin/env python3

"""Focused forward and gradient checks for covariance-aware attention."""

import torch
import torch.nn.functional as F

from dinov3.layers.attention import SelfAttention
from dinov3.models.vision_transformer import DinoVisionTransformer


def explicit_reference(attn: SelfAttention, qkv: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    batch, tokens, _ = qkv.shape
    dim = attn.qkv.in_features
    q, k, v = torch.unbind(
        qkv.reshape(batch, tokens, 3, attn.num_heads, dim // attn.num_heads), 2
    )
    q, k, v = [tensor.transpose(1, 2) for tensor in (q, k, v)]
    weights = torch.softmax(q @ k.transpose(-2, -1) * attn.scale, dim=-1)
    value_mean = weights @ v
    covariance_source = k if attn.covariance_space == "key" else v
    covariance_mean = weights @ covariance_source
    second_moment = weights @ covariance_source.square()
    covariance = (second_moment - covariance_mean.square()).clamp_min(0.0)
    covariance = (covariance / second_moment.clamp_min(attn.covariance_eps)).clamp(0.0, 1.0)
    if attn.covariance_mode == "gate":
        output = value_mean * torch.exp(-attn.covariance_gate_strength * covariance)
        return output.transpose(1, 2).reshape(batch, tokens, dim), covariance
    value_mean = value_mean.transpose(1, 2).reshape(batch, tokens, dim)
    covariance_flat = covariance.transpose(1, 2).reshape(batch, tokens, dim)
    return value_mean + attn.covariance_proj(covariance_flat), covariance


def main() -> None:
    torch.manual_seed(0)
    batch, tokens, dim, heads = 2, 7, 24, 3
    for mode in ("add", "gate"):
        for space in ("key", "value"):
            attn = SelfAttention(
                dim,
                num_heads=heads,
                qkv_bias=True,
                covariance_mode=mode,
                covariance_space=space,
            )
            if attn.covariance_proj is not None:
                torch.nn.init.normal_(attn.covariance_proj.weight, std=0.1)
            qkv = torch.randn(batch, tokens, 3 * dim, requires_grad=True)
            actual = attn.compute_attention(qkv)
            expected, covariance = explicit_reference(attn, qkv)
            torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-6)
            assert covariance.min() >= 0 and covariance.max() <= 1

            actual.square().mean().backward()
            assert qkv.grad is not None and torch.isfinite(qkv.grad).all()
            if attn.covariance_proj is not None:
                assert attn.covariance_proj.weight.grad is not None
                assert torch.isfinite(attn.covariance_proj.weight.grad).all()

    model = DinoVisionTransformer(
        img_size=32,
        patch_size=16,
        embed_dim=48,
        depth=2,
        num_heads=3,
        covariance_mode="add",
        covariance_space="key",
    )
    model.init_weights()
    covariance_weights = [block.attn.covariance_proj.weight for block in model.blocks]
    assert all(torch.count_nonzero(weight) == 0 for weight in covariance_weights)

    baseline = DinoVisionTransformer(
        img_size=32,
        patch_size=16,
        embed_dim=48,
        depth=2,
        num_heads=3,
    )
    baseline.load_state_dict(
        {name: value for name, value in model.state_dict().items() if "covariance_proj" not in name},
        strict=True,
    )
    images = torch.randn(2, 3, 32, 32)
    torch.testing.assert_close(model(images), baseline(images), atol=1e-6, rtol=1e-6)
    print("[OK] key/value add/gate covariance forward, gradients, bounds, and zero-init equivalence")


if __name__ == "__main__":
    main()

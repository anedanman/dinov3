# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

"""Cross-crop register consistency loss.

Pulls each student register token toward the matching teacher register token of
a *different* crop of the same image, so register/slot identities become stable
across views ("the same slot binds the same content in every crop"). Matching is
per-image Hungarian assignment on cosine similarity by default, since slot roles
may permute between crops (especially with sampled `register_init=gaussian`).
"""

from typing import Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor


@torch.no_grad()
def hungarian_match(sim: Tensor) -> Tensor:
    """Batched Hungarian matching on a similarity matrix.

    Args:
        sim: [B, R, R] similarity between student registers (rows) and teacher
            registers (columns).

    Returns:
        [B, R] int64 indices: teacher register matched to each student register.
    """
    from scipy.optimize import linear_sum_assignment

    sim_np = sim.detach().float().cpu().numpy()
    out = torch.empty(sim.shape[:2], dtype=torch.int64)
    for b in range(sim_np.shape[0]):
        _, cols = linear_sum_assignment(-sim_np[b])
        out[b] = torch.from_numpy(cols)
    return out.to(sim.device)


def matched_cosine_loss(
    student_reg: Tensor,
    teacher_reg: Tensor,
    matching: str = "hungarian",
    subtract_mean: bool = False,
) -> Tensor:
    """Mean (1 - cos) between matched student/teacher registers of one crop pair.

    Args:
        student_reg: [B, R, D] student register tokens (one crop).
        teacher_reg: [B, R, D] teacher register tokens (another crop), no grad.
        matching: "hungarian" (per-image optimal permutation) or "fixed"
            (index-aligned registers).
        subtract_mean: subtract the per-image mean register before the cosine.
            Registers tend to share one dominant direction (reg_pairwise_cos
            ~0.99+), which makes raw cosines saturate and the loss vacuous;
            matching the residuals keeps the loss acting on what actually
            differentiates the slots.
    """
    student_reg = student_reg.float()
    teacher_reg = teacher_reg.float()
    if subtract_mean:
        student_reg = student_reg - student_reg.mean(dim=1, keepdim=True)
        teacher_reg = teacher_reg - teacher_reg.mean(dim=1, keepdim=True)
    s = F.normalize(student_reg, dim=-1)
    t = F.normalize(teacher_reg, dim=-1)
    sim = torch.einsum("brd,bsd->brs", s, t)  # [B, R, R]
    if matching == "hungarian":
        cols = hungarian_match(sim)  # [B, R]
        matched = sim.gather(2, cols.unsqueeze(-1)).squeeze(-1)  # [B, R]
    elif matching == "fixed":
        matched = sim.diagonal(dim1=-2, dim2=-1)  # [B, R]
    else:
        raise ValueError(f"unknown matching={matching}")
    return (1.0 - matched).mean()


def register_consistency_loss(
    student_reg: Tensor,
    teacher_reg: Tensor,
    matching: str = "hungarian",
    pairs: Sequence[Tuple[int, int]] | None = None,
    subtract_mean: bool = False,
) -> Tensor:
    """Cross-crop register consistency over crop pairs.

    Args:
        student_reg: [n_student_crops, B, R, D].
        teacher_reg: [n_teacher_crops, B, R, D] (detached teacher features).
        pairs: (student_crop, teacher_crop) index pairs. Default: every student
            crop against every *other* teacher crop (cross-view only), mirroring
            the DINO global loss with the diagonal ignored.
        subtract_mean: match mean-subtracted register residuals (see
            `matched_cosine_loss`).
    """
    n_student = student_reg.shape[0]
    n_teacher = teacher_reg.shape[0]
    if pairs is None:
        pairs = [(i, j) for i in range(n_student) for j in range(n_teacher) if i != j]
    assert len(pairs) > 0, "register_consistency_loss needs at least one crop pair"
    loss = student_reg.new_zeros((), dtype=torch.float32)
    for i, j in pairs:
        loss = loss + matched_cosine_loss(
            student_reg[i], teacher_reg[j].detach(), matching=matching, subtract_mean=subtract_mean
        )
    return loss / len(pairs)

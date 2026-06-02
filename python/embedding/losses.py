"""
Loss functions for voxel embedding training.

All functions operate on 3-class logits [B, 3, D, H, W] where class 0 = empty,
class 1 = sparse, class 2 = filled.  The `make_recon_loss` / `make_aux_loss`
factories return plain callables so they can be stored in loss-weight dicts
without subclassing nn.Module.
"""

import torch
import torch.nn.functional as F
from monai.losses import DiceLoss, TverskyLoss, FocalLoss


# ===================== reconstruction loss factory =====================

def make_recon_loss(name: str, **kwargs):
    """
    Returns ``fn(logits, target, extra=None) -> scalar loss``.

    Supported names:
      - ``"ce"`` — cross-entropy with optional focal modulation and per-call
        class weights via ``extra["class_weights"]``.
    """
    name = name.lower()

    if name == "ce":
        label_smoothing = float(kwargs.get("label_smoothing", 0.05))
        default_gamma   = kwargs.get("focal_gamma", None)
        ignore_index    = kwargs.get("ignore_index", -100)

        def f(logits, target, extra=None):
            extra = extra or {}
            w     = extra.get("class_weights", None)
            mask  = extra.get("mask", None)
            gamma = extra.get("focal_gamma", default_gamma)

            if w is not None:
                w = w.to(logits.device, dtype=logits.dtype)

            ce = F.cross_entropy(
                logits, target,
                weight=w,
                label_smoothing=label_smoothing,
                ignore_index=ignore_index,
                reduction="none",
            )

            if gamma is not None and gamma > 0:
                with torch.no_grad():
                    pt = logits.softmax(dim=1).gather(
                        1, target.unsqueeze(1)
                    ).squeeze(1).clamp_min(1e-6)
                ce = ((1.0 - pt) ** float(gamma)) * ce

            if mask is not None:
                m = mask.to(logits.device, dtype=ce.dtype)
                return (ce * m).sum() / m.sum().clamp_min(1.0)
            return ce.mean()

        return f

    raise ValueError(f"Unknown recon loss name: {name}")


# ===================== shared helpers =====================

def _soft_occupancy_from_logits(logits: torch.Tensor) -> torch.Tensor:
    """Soft occupancy in [0,1]: filled=1.0, sparse=0.5, empty=0.0 (differentiable)."""
    p = logits.softmax(dim=1)
    return (p[:, 2] + 0.5 * p[:, 1]).clamp(0, 1)


def _avg_downsample_3d(x: torch.Tensor, scale: int) -> torch.Tensor:
    if scale == 1:
        return x
    return F.avg_pool3d(x.unsqueeze(1), kernel_size=scale, stride=scale).squeeze(1)


def _pairwise_dist2(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return ((a[:, None, :] - b[None, :, :]) ** 2).sum(-1)


# ===================== shape-aware losses =====================

def soft_iou_loss(logits: torch.Tensor, target_occ: torch.Tensor, *, eps: float = 1e-6) -> torch.Tensor:
    """Soft IoU on occupancy. ``target_occ``: [B,D,H,W] float in [0,1]."""
    pred  = _soft_occupancy_from_logits(logits)
    inter = (pred * target_occ).sum(dim=(1, 2, 3))
    union = (pred + target_occ - pred * target_occ).sum(dim=(1, 2, 3)).clamp_min(eps)
    return (1.0 - inter / union).mean()


def tv_smoothness_loss(logits: torch.Tensor) -> torch.Tensor:
    """3D total variation on soft occupancy — encourages smooth surfaces."""
    occ = _soft_occupancy_from_logits(logits)
    dx  = occ[:, 1:, :, :] - occ[:, :-1, :, :]
    dy  = occ[:, :, 1:, :] - occ[:, :, :-1, :]
    dz  = occ[:, :, :, 1:] - occ[:, :, :, :-1]
    return dx.abs().mean() + dy.abs().mean() + dz.abs().mean()


def boundary_loss(logits: torch.Tensor, *, labels: torch.Tensor, dt: torch.Tensor) -> torch.Tensor:
    """
    Distance-transform-weighted CE — emphasises voxels near occupancy boundaries.

    ``labels``: [B,D,H,W] Long {0/1/2}.
    ``dt``:     [B,D,H,W] Float distance-to-occupied (0 at boundary/occupied).
    """
    if labels.max() > 1:
        labels = ((labels == 1) | (labels == 2)).long()

    occ_logit = logits[:, 1:].logsumexp(dim=1)
    logits2   = torch.stack([logits[:, 0], occ_logit], dim=1)

    w  = 1.0 / (1.0 + dt)
    ce = F.cross_entropy(logits2, labels, reduction="none")
    return (ce * w).mean()


def center_of_mass_loss(logits: torch.Tensor, target_com: torch.Tensor) -> torch.Tensor:
    """Align predicted centre-of-mass with target. ``target_com``: [B,3] in voxel coords."""
    occ = _soft_occupancy_from_logits(logits)
    B, D, H, W = occ.shape
    z = torch.linspace(0, D - 1, D, device=occ.device)
    y = torch.linspace(0, H - 1, H, device=occ.device)
    x = torch.linspace(0, W - 1, W, device=occ.device)

    s = occ.sum(dim=(1, 2, 3)).clamp_min(1e-6)
    zc = (occ.sum(dim=(2, 3)) * z).sum(dim=1) / s
    yc = (occ.sum(dim=(1, 3)) * y).sum(dim=1) / s
    xc = (occ.sum(dim=(1, 2)) * x).sum(dim=1) / s
    return F.smooth_l1_loss(torch.stack([xc, yc, zc], dim=1), target_com)


def class_balance_loss(logits: torch.Tensor, prior: torch.Tensor = None) -> torch.Tensor:
    """Encourage reasonable class proportions. If ``prior`` is None, maximise entropy."""
    p    = logits.softmax(dim=1)
    freq = p.mean(dim=(0, 2, 3, 4))
    if prior is None:
        return (freq * freq.clamp_min(1e-6).log()).sum()  # negative entropy
    prior = prior / prior.sum()
    return F.kl_div(freq.log(), prior, reduction="batchmean")


def multiscale_occ_loss(logits: torch.Tensor, target_low: torch.Tensor, *, scale: int = 4) -> torch.Tensor:
    """Match average-pooled occupancy at lower resolution."""
    pred     = _soft_occupancy_from_logits(logits)
    pred_low = _avg_downsample_3d(pred, scale)
    return F.binary_cross_entropy(pred_low, target_low)


def chamfer_points_loss(
    logits: torch.Tensor,
    target_points: list,
    *,
    topk: int = 2048,
) -> torch.Tensor:
    """
    Chamfer distance between top-K predicted occupied voxels and GT points.

    ``target_points``: list of length B, each a [Ni, 3] tensor in voxel coords.
    Gradient flows through the chosen voxels' occupancy scores.
    """
    occ  = _soft_occupancy_from_logits(logits)
    B, D, H, W = occ.shape
    flat = occ.view(B, -1)
    k    = min(topk, flat.shape[1])
    _, idx = flat.topk(k, dim=1)

    z = idx // (H * W)
    y = (idx % (H * W)) // W
    x = idx % W
    pred_pts_list = [torch.stack([x[i], y[i], z[i]], dim=1).float() for i in range(B)]

    per_sample = []
    for pred_pts, tgt_pts in zip(pred_pts_list, target_points):
        if pred_pts.numel() == 0 or tgt_pts.numel() == 0:
            per_sample.append(torch.tensor(0.0, device=logits.device))
            continue
        tgt_pts = tgt_pts.to(logits.device).float()
        d2 = _pairwise_dist2(pred_pts, tgt_pts)
        per_sample.append(d2.min(dim=1).values.mean() + d2.min(dim=0).values.mean())
    return torch.stack(per_sample).mean()


# ===================== auxiliary loss factory =====================

def make_aux_loss(name: str, **kwargs):
    """
    Returns ``fn(logits, targets, extra=None) -> scalar loss``.

    Built-ins and their required ``targets`` keys:

    ============== ===============================
    ``"soft_iou"`` ``targets["occ_dense"]``
    ``"tv"``       *(none)*
    ``"boundary"`` ``targets["labels"]``, ``targets["dt"]``
    ``"com"``      ``targets["com"]``
    ``"class_balance"`` *(none; optional prior via extra)*
    ``"ms_occ"``   ``targets["occ_low"]``; kwarg ``scale`` (default 4)
    ``"chamfer"``  ``targets["points"]`` (list of [Ni,3]); kwarg ``topk`` (default 2048)
    ============== ===============================
    """
    name = name.lower()

    if name == "soft_iou":
        return lambda logits, targets, extra=None: soft_iou_loss(logits, targets["occ_dense"])

    if name == "tv":
        return lambda logits, targets=None, extra=None: tv_smoothness_loss(logits)

    if name == "boundary":
        return lambda logits, targets, extra=None: boundary_loss(
            logits, labels=targets["labels"], dt=targets["dt"]
        )

    if name == "com":
        return lambda logits, targets, extra=None: center_of_mass_loss(logits, targets["com"])

    if name == "class_balance":
        return lambda logits, targets=None, extra=None: class_balance_loss(
            logits, prior=None if extra is None else extra.get("prior")
        )

    if name == "ms_occ":
        scale = int(kwargs.get("scale", 4))
        return lambda logits, targets, extra=None: multiscale_occ_loss(
            logits, targets["occ_low"], scale=scale
        )

    if name == "chamfer":
        topk = int(kwargs.get("topk", 2048))
        return lambda logits, targets, extra=None: chamfer_points_loss(
            logits, targets["points"], topk=topk
        )

    raise ValueError(f"Unknown aux loss name: {name}")

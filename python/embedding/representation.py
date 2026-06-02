"""
Voxel embedding testbed for drone navigation.

Trains an encoder-decoder pair to compress a sparse voxel occupancy map
(produced by the voxelsim terrain generator) into a fixed-size latent vector,
then reconstruct the 3-class occupancy volume (empty / sparse / filled) from
that vector.  The modular design lets you swap encoders, decoders, and
auxiliary loss heads independently.

Encoders
--------
SpConvCNNEncoder        Sparse 3D convolutions via spconv — processes only
                        occupied voxels, memory-efficient for large worlds.
SimpleCNNEncoder        Dense 3D CNN; 6 strided conv layers → 5-D feature map.
UNet3DEncoder           Residual blocks + global average pool → latent.
ResNet3DEncoder         ResNet-style with GroupNorm; no skip connections.
PointMLPEncoder         Operates directly on sparse (xyz, value) points;
                        mean+max aggregation; optional Fourier features.
PointNetPPLiteFPEncoder Hierarchical PointNet++ with feature propagation.
CrossAttnTokensEncoder  Conv stem → /4 and /8 tokens + 3-D sinusoidal PE
                        → cross-attention with learnable queries → latent.

Decoders
--------
SimpleCNNDecoder        Transposed-conv upsampling from 5-D feature map.
MeinDecoder             Compact transposed-conv stack (FC → 3 deconvs).
UNet3DDecoder           Residual transposed-conv stack.
ResNet3DDecoder         Symmetric to ResNet3DEncoder.
ImplicitMLPDecoder      MLP queried at every voxel coordinate; chunked.
Factorised3DDecoder     Separable 3-D transposed convs — fewer parameters.
ImplicitFourierDecoder  MLP with multi-scale Fourier position encoding.

Quick start
-----------
    python representation.py          # runs sweep() with default config

Dependencies: torch, spconv, monai, scipy, numpy, tensorboard
"""

import os
import csv
import datetime
import time
import collections
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import spconv.pytorch as spconv
from spconv.pytorch import SparseConvTensor
from scipy.ndimage import distance_transform_edt as edt
from torch.utils.data import DataLoader, IterableDataset
from torch.utils.tensorboard import SummaryWriter

import voxelsim
from losses import make_recon_loss, make_aux_loss


# ============================================================
#  Data container
# ============================================================

@dataclass
class VoxelData:
    """Sparse voxel occupancy map for one sample."""
    occupied_coords: torch.Tensor  # [N, 3] integer voxel coordinates
    values: torch.Tensor           # [N]   0.5 = sparse, 1.0 = filled
    bounds: torch.Tensor           # [3]   world dimensions (x, y, z)
    drone_pos: torch.Tensor        # [3]   agent position in voxel coords

    def to_device(self, device):
        return VoxelData(
            occupied_coords=self.occupied_coords.to(device),
            values=self.values.to(device),
            bounds=self.bounds.to(device),
            drone_pos=self.drone_pos.to(device),
        )


# ============================================================
#  Abstract base classes
# ============================================================

class EmbeddingEncoder(ABC):
    @abstractmethod
    def encode(self, voxel_data: VoxelData) -> torch.Tensor:
        pass

    @abstractmethod
    def get_embedding_dim(self) -> int:
        pass


class EmbeddingDecoder(ABC):
    @abstractmethod
    def decode(self, embedding: torch.Tensor, query_points: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        pass


class LossHead(ABC, nn.Module):
    variable_length_target: bool = False

    def __init__(self, logits_fn: Optional[Callable] = None):
        super().__init__()
        self.logits_fn = logits_fn

    def bind_logits(self, fn: Callable):
        self.logits_fn = fn
        return self

    @abstractmethod
    def forward(self, embedding: torch.Tensor, logits: Optional[torch.Tensor] = None) -> torch.Tensor:
        pass

    @abstractmethod
    def compute_loss(self, prediction: torch.Tensor, target) -> torch.Tensor:
        pass


# ============================================================
#  Encoders
# ============================================================

class SpConvCNNEncoder(EmbeddingEncoder, nn.Module):
    """
    Sparse 3D CNN encoder using spconv-2.3+.

    Processes only occupied voxels — memory cost scales with scene density
    rather than world volume.  Three strided sparse convolutions downsample
    48³ → 24³ → 12³ → 6³, then global average pool → projection head.
    """

    def __init__(self, voxel_size: int = 48, embedding_dim: int = 128):
        super().__init__()
        self.voxel_size    = voxel_size
        self.embedding_dim = embedding_dim

        self.backbone = spconv.SparseSequential(
            spconv.SparseConv3d(1, 32, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm1d(32), nn.ReLU(True),
            spconv.SparseConv3d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(64), nn.ReLU(True),
            spconv.SparseConv3d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(128), nn.ReLU(True),
        )
        self.proj = nn.Linear(128, embedding_dim)

    def encode(self, voxel_batch: List[VoxelData]) -> torch.Tensor:
        sct  = self._batch_to_spconv(voxel_batch)
        x    = self.backbone(sct)
        feats, idx = x.features, x.indices[:, 0]
        B    = len(voxel_batch)
        sums   = torch.zeros(B, 128, device=feats.device).index_add_(0, idx, feats)
        counts = torch.bincount(idx, minlength=B).clamp_(min=1).float().unsqueeze(1)
        return self.proj(sums / counts)

    def get_embedding_dim(self) -> int:
        return self.embedding_dim

    def _batch_to_spconv(self, voxels: List[VoxelData]) -> SparseConvTensor:
        coords_all, feats_all = [], []
        for b, vd in enumerate(voxels):
            if vd.occupied_coords.numel() == 0:
                continue
            xyz = (vd.occupied_coords.float()
                   / vd.bounds.float() * (self.voxel_size - 1)).long()
            zyx = torch.stack([xyz[:, 2], xyz[:, 1], xyz[:, 0]], dim=1)
            batch_col = torch.full((zyx.size(0), 1), b, dtype=torch.long, device=xyz.device)
            coords_all.append(torch.cat([batch_col, zyx], dim=1))
            feats_all.append(vd.values.unsqueeze(1).float())
        coords = torch.cat(coords_all, 0).int()
        feats  = torch.cat(feats_all,  0)
        return SparseConvTensor(
            features      = feats,
            indices       = coords,
            spatial_shape = [self.voxel_size] * 3,
            batch_size    = len(voxels),
        )


class SimpleCNNEncoder(EmbeddingEncoder, nn.Module):
    """Dense 3D CNN — 6 strided conv layers, returns a 5-D feature map [B,1024,1,1,1]."""

    def __init__(self, voxel_size=48, embedding_dim=128):
        super().__init__()
        self.voxel_size    = voxel_size
        self.embedding_dim = embedding_dim
        self.conv1 = nn.Conv3d(1,    32,   kernel_size=5, stride=2, padding=2)
        self.conv2 = nn.Conv3d(32,   64,   kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv3d(64,   128,  kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv3d(128,  256,  kernel_size=3, stride=2, padding=1)
        self.conv5 = nn.Conv3d(256,  512,  kernel_size=3, stride=2, padding=1)
        self.conv6 = nn.Conv3d(512,  1024, kernel_size=3, stride=2, padding=1)

    def encode(self, voxel_batch: List[VoxelData]) -> torch.Tensor:
        x = torch.cat([self._sparse_to_dense(vd) for vd in voxel_batch], 0)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        return F.relu(self.conv6(x))

    def get_embedding_dim(self) -> int:
        return self.embedding_dim

    def _sparse_to_dense(self, vd: VoxelData) -> torch.Tensor:
        src = int(vd.bounds[0].item())
        g   = torch.zeros((1, 1, src, src, src), device=vd.occupied_coords.device)
        if vd.occupied_coords.shape[0] > 0:
            c = vd.occupied_coords.clone()
            c[:, 2] = -c[:, 2]
            c = c.long()
            c[:, 0].clamp_(0, src - 1)
            c[:, 1].clamp_(0, src - 1)
            c[:, 2].clamp_(0, src - 1)
            g[0, 0, c[:, 0], c[:, 1], c[:, 2]] = vd.values
        return self._center_crop_or_pad_to(g, self.voxel_size)

    @staticmethod
    def _center_crop_or_pad_to(x: torch.Tensor, target: int) -> torch.Tensor:
        B, C, D, H, W = x.shape
        td, th, tw = target - D, target - H, target - W
        if td > 0 or th > 0 or tw > 0:
            x = F.pad(x, (tw//2, tw-tw//2, th//2, th-th//2, td//2, td-td//2))
            D, H, W = x.shape[-3:]
        if D > target or H > target or W > target:
            sd, sh, sw = (D-target)//2, (H-target)//2, (W-target)//2
            x = x[..., sd:sd+target, sh:sh+target, sw:sw+target]
        return x


class SimpleCNNDecoder(EmbeddingDecoder, nn.Module):
    """Transposed-conv decoder mirroring SimpleCNNEncoder."""

    def __init__(self, embedding_dim=128, voxel_size=48):
        super().__init__()
        self.voxel_size    = voxel_size
        self.embedding_dim = embedding_dim
        self.init_size     = voxel_size // 64  # after 6 stride-2 layers
        self.deconv1 = nn.ConvTranspose3d(1024, 512, 3, stride=2, padding=1, output_padding=1)
        self.deconv2 = nn.ConvTranspose3d(512,  256, 3, stride=2, padding=1, output_padding=1)
        self.deconv3 = nn.ConvTranspose3d(256,  128, 3, stride=2, padding=1, output_padding=1)
        self.deconv4 = nn.ConvTranspose3d(128,  64,  3, stride=2, padding=1, output_padding=1)
        self.deconv5 = nn.ConvTranspose3d(64,   32,  3, stride=2, padding=1, output_padding=1)
        self.deconv6 = nn.ConvTranspose3d(32,   3,   5, stride=2, padding=2, output_padding=1)

    def decode(self, embedding: torch.Tensor, query_points=None, skips=None):
        x = embedding
        x = F.relu(self.deconv1(x))
        x = F.relu(self.deconv2(x))
        x = F.relu(self.deconv3(x))
        x = F.relu(self.deconv4(x))
        x = F.relu(self.deconv5(x))
        logits = self.deconv6(x)
        logits = torch.flip(logits, dims=[4])  # undo the z-flip applied in the encoder
        return {"logits": logits}


# ---- Compact MeinEncoder / MeinDecoder (latent vector, smaller world sizes) ----

class MeinEncoder(EmbeddingEncoder, nn.Module):
    def __init__(self, voxel_size=48, embedding_dim=128):
        super().__init__()
        self.voxel_size    = voxel_size
        self.embedding_dim = embedding_dim
        self.conv1 = nn.Conv3d(1,  32,  kernel_size=5, stride=2, padding=2)
        self.conv2 = nn.Conv3d(32, 64,  kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv3d(64, 128, kernel_size=3, stride=1, padding=1)
        self.flat_size = 128 * (voxel_size // 4) ** 3
        self.fc = nn.Linear(self.flat_size, embedding_dim)

    def encode(self, voxel_batch: List[VoxelData]) -> torch.Tensor:
        x = torch.cat([self._sparse_to_dense(vd) for vd in voxel_batch], 0)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        return self.fc(x.flatten(1))

    def get_embedding_dim(self) -> int:
        return self.embedding_dim

    def _sparse_to_dense(self, vd: VoxelData) -> torch.Tensor:
        g = torch.zeros((1, 1, self.voxel_size, self.voxel_size, self.voxel_size),
                        device=vd.occupied_coords.device, dtype=torch.float32)
        if vd.occupied_coords.shape[0] > 0:
            c = (vd.occupied_coords.float() / vd.bounds.float() * (self.voxel_size - 1)
                 ).long().clamp(0, self.voxel_size - 1)
            g[0, 0, c[:, 0], c[:, 1], c[:, 2]] = vd.values
        return g


class MeinDecoder(EmbeddingDecoder, nn.Module):
    def __init__(self, embedding_dim=128, voxel_size=48):
        super().__init__()
        self.voxel_size = voxel_size
        self.init_size  = voxel_size // 8
        self.fc      = nn.Linear(embedding_dim, 128 * self.init_size ** 3)
        self.deconv1 = nn.ConvTranspose3d(128, 64, 3, stride=2, padding=1, output_padding=1)
        self.deconv2 = nn.ConvTranspose3d(64,  32, 3, stride=2, padding=1, output_padding=1)
        self.deconv3 = nn.ConvTranspose3d(32,  3,  5, stride=2, padding=2, output_padding=1)

    def decode(self, embedding: torch.Tensor, query_points=None, skips=None):
        x = self.fc(embedding).view(-1, 128, self.init_size, self.init_size, self.init_size)
        x = F.relu(self.deconv1(x))
        x = F.relu(self.deconv2(x))
        return {"logits": self.deconv3(x)}


# ---- ResBlock helpers ----

class _ResBlock3D_BN(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(c, c, 3, padding=1, bias=False), nn.BatchNorm3d(c), nn.ReLU(True),
            nn.Conv3d(c, c, 3, padding=1, bias=False), nn.BatchNorm3d(c),
        )
        self.act = nn.ReLU(True)

    def forward(self, x):
        return self.act(x + self.net(x))


class _GN(torch.nn.GroupNorm):
    def __init__(self, c, groups=32):
        super().__init__(min(groups, c), c)


class _ResBlock3D_GN(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.conv1 = nn.Conv3d(c, c, 3, padding=1, bias=False)
        self.gn1   = _GN(c)
        self.conv2 = nn.Conv3d(c, c, 3, padding=1, bias=False)
        self.gn2   = _GN(c)
        self.act   = nn.ReLU(True)

    def forward(self, x):
        return self.act(x + self.gn2(self.conv2(self.act(self.gn1(self.conv1(x))))))


# ---- UNet3D encoder/decoder (BatchNorm residual blocks) ----

class UNet3DEncoder(EmbeddingEncoder, nn.Module):
    def __init__(self, voxel_size=48, embedding_dim=512):
        super().__init__()
        self.voxel_size    = voxel_size
        self.embedding_dim = embedding_dim
        self.stem = nn.Sequential(
            nn.Conv3d(1, 32, 3, padding=1, bias=False), nn.BatchNorm3d(32), nn.ReLU(True),
            _ResBlock3D_BN(32),
            nn.Conv3d(32, 64, 3, stride=2, padding=1, bias=False), nn.BatchNorm3d(64), nn.ReLU(True),
            _ResBlock3D_BN(64),
        )
        self.down = nn.Sequential(
            nn.Conv3d(64, 128, 3, stride=2, padding=1, bias=False), nn.BatchNorm3d(128), nn.ReLU(True),
            _ResBlock3D_BN(128),
        )
        self.fc = nn.Linear(128, embedding_dim)

    def encode(self, voxel_batch: List[VoxelData]) -> torch.Tensor:
        x = torch.cat([self._sparse_to_dense(v) for v in voxel_batch], 0)
        x = self.stem(x)
        x = self.down(x)
        return self.fc(x.mean(dim=[2, 3, 4]))

    def get_embedding_dim(self): return self.embedding_dim

    def _sparse_to_dense(self, vd: VoxelData) -> torch.Tensor:
        g = torch.zeros((1, 1, self.voxel_size, self.voxel_size, self.voxel_size),
                        device=vd.occupied_coords.device)
        if vd.occupied_coords.numel():
            xyz = (vd.occupied_coords.float() / vd.bounds.float() * (self.voxel_size - 1)
                   ).long().clamp(0, self.voxel_size - 1)
            g[0, 0, xyz[:, 0], xyz[:, 1], xyz[:, 2]] = vd.values
        return g


class UNet3DDecoder(EmbeddingDecoder, nn.Module):
    def __init__(self, embedding_dim=512, voxel_size=48):
        super().__init__()
        self.voxel_size = voxel_size
        self.init_size  = voxel_size // 4
        self.fc  = nn.Linear(embedding_dim, 128 * self.init_size ** 3)
        self.up1 = nn.Sequential(
            nn.ConvTranspose3d(128, 64, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm3d(64), nn.ReLU(True), _ResBlock3D_BN(64),
        )
        self.up0 = nn.Sequential(
            nn.ConvTranspose3d(64, 32, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm3d(32), nn.ReLU(True), _ResBlock3D_BN(32),
        )
        self.out = nn.Conv3d(32, 3, 3, padding=1)

    def decode(self, embedding, skips=None):
        x = self.fc(embedding).view(-1, 128, self.init_size, self.init_size, self.init_size)
        x = self.up1(x)
        x = self.up0(x)
        return {"logits": self.out(x)}


# ---- ResNet3D encoder/decoder (GroupNorm) ----

class ResNet3DEncoder(EmbeddingEncoder, nn.Module):
    """Strided GroupNorm ResNet; 48→24→12→6, global average pool → latent."""

    def __init__(self, voxel_size=48, embedding_dim=1024, widths=(32, 64, 128)):
        super().__init__()
        assert voxel_size % 8 == 0
        self.voxel_size    = voxel_size
        self.embedding_dim = embedding_dim
        c0, c1, c2        = widths
        self.stem   = nn.Sequential(nn.Conv3d(1, c0, 5, stride=2, padding=2, bias=False), _GN(c0), nn.ReLU(True), _ResBlock3D_GN(c0))
        self.stage1 = nn.Sequential(nn.Conv3d(c0, c1, 3, stride=2, padding=1, bias=False), _GN(c1), nn.ReLU(True), _ResBlock3D_GN(c1))
        self.stage2 = nn.Sequential(nn.Conv3d(c1, c2, 3, stride=2, padding=1, bias=False), _GN(c2), nn.ReLU(True), _ResBlock3D_GN(c2))
        self.proj   = nn.Linear(c2, embedding_dim)

    def encode(self, voxel_batch: List[VoxelData]) -> torch.Tensor:
        x = torch.cat([self._sparse_to_dense(v) for v in voxel_batch], 0)
        x = self.stem(x); x = self.stage1(x); x = self.stage2(x)
        return self.proj(x.mean(dim=(2, 3, 4)))

    def get_embedding_dim(self): return self.embedding_dim

    def _sparse_to_dense(self, vd: VoxelData) -> torch.Tensor:
        g = torch.zeros((1, 1, self.voxel_size, self.voxel_size, self.voxel_size),
                        device=vd.occupied_coords.device, dtype=torch.float32)
        if vd.occupied_coords.numel():
            xyz = (vd.occupied_coords.float() / vd.bounds.float() * (self.voxel_size - 1)
                   ).long().clamp(0, self.voxel_size - 1)
            g[0, 0, xyz[:, 0], xyz[:, 1], xyz[:, 2]] = vd.values
        return g


class ResNet3DDecoder(EmbeddingDecoder, nn.Module):
    def __init__(self, embedding_dim=1024, voxel_size=48, widths=(128, 64, 32)):
        super().__init__()
        assert voxel_size % 8 == 0
        self.voxel_size = voxel_size
        self.init_size  = voxel_size // 8
        c2, c1, c0     = widths
        self.fc  = nn.Linear(embedding_dim, c2 * self.init_size ** 3)
        self.up1 = nn.Sequential(nn.ConvTranspose3d(c2, c1, 4, stride=2, padding=1, bias=False), _GN(c1), nn.ReLU(True), _ResBlock3D_GN(c1))
        self.up2 = nn.Sequential(nn.ConvTranspose3d(c1, c0, 4, stride=2, padding=1, bias=False), _GN(c0), nn.ReLU(True), _ResBlock3D_GN(c0))
        self.up3 = nn.Sequential(nn.ConvTranspose3d(c0, 32,  4, stride=2, padding=1, bias=False), _GN(32),  nn.ReLU(True))
        self.out = nn.Conv3d(32, 3, 3, padding=1)

    def decode(self, embedding, query_points=None, skips=None):
        x = self.fc(embedding).view(-1, self.up1[0].in_channels, self.init_size, self.init_size, self.init_size)
        x = self.up1(x); x = self.up2(x); x = self.up3(x)
        return {"logits": self.out(x)}


# ---- Implicit decoders (MLP-based) ----

class ImplicitMLPDecoder(EmbeddingDecoder, nn.Module):
    """MLP queried at every voxel coordinate; chunked for memory efficiency."""

    def __init__(self, embedding_dim=1024, voxel_size=48, hidden=256, depth=4, chunk=65536):
        super().__init__()
        self.voxel_size = voxel_size
        self.chunk      = chunk
        layers, d = [], embedding_dim + 3
        for i in range(depth):
            out = hidden if i < depth - 1 else 3
            layers.append(nn.Linear(d, out))
            if i < depth - 1:
                layers.append(nn.ReLU(True))
                d = hidden
        self.mlp = nn.Sequential(*layers)
        coords = torch.stack(torch.meshgrid(
            torch.linspace(-1, 1, voxel_size),
            torch.linspace(-1, 1, voxel_size),
            torch.linspace(-1, 1, voxel_size),
            indexing='ij',
        ), dim=-1).view(-1, 3)
        self.register_buffer("grid", coords, persistent=False)

    def decode(self, embedding, query_points=None, skips=None):
        B, E = embedding.shape
        V    = self.grid.shape[0]
        outs = []
        for b in range(B):
            z     = embedding[b].unsqueeze(0).expand(V, -1)
            feats = torch.cat([z, self.grid.to(z.dtype)], dim=1)
            pred  = torch.cat([self.mlp(feats[i:i+self.chunk]) for i in range(0, V, self.chunk)])
            outs.append(pred.view(self.voxel_size, self.voxel_size, self.voxel_size, 3).permute(3, 0, 1, 2))
        return {"logits": torch.stack(outs, 0)}


class ImplicitFourierDecoder(EmbeddingDecoder, nn.Module):
    """Implicit field decoder with multi-scale Fourier position encoding."""

    def __init__(self, embedding_dim=512, voxel_size=48, fourier_bands=8, hidden=256, depth=4, chunk=65536):
        super().__init__()
        self.voxel_size = voxel_size
        self.chunk      = chunk
        coords = torch.stack(torch.meshgrid(
            torch.linspace(-1, 1, voxel_size),
            torch.linspace(-1, 1, voxel_size),
            torch.linspace(-1, 1, voxel_size),
            indexing='ij',
        ), dim=-1).view(-1, 3)
        self.register_buffer("xyz", coords, persistent=False)
        freqs = 2.0 ** torch.arange(fourier_bands) * np.pi
        self.register_buffer("freqs", freqs, persistent=False)
        in_dim = embedding_dim + 3 + 6 * fourier_bands
        layers, d = [], in_dim
        for i in range(depth - 1):
            layers += [nn.Linear(d, hidden), nn.ReLU(True)]
            d = hidden
        layers += [nn.Linear(d, 3)]
        self.mlp = nn.Sequential(*layers)

    def _encode_xyz(self, xyz):
        ang = xyz.unsqueeze(-1) * self.freqs
        pe  = torch.cat([torch.sin(ang), torch.cos(ang)], -1).view(xyz.size(0), -1)
        return torch.cat([xyz, pe], dim=1)

    def decode(self, embedding, query_points=None, skips=None):
        import torch.cuda.amp as amp
        with amp.autocast(enabled=False):
            embedding = embedding.float()
            B, E      = embedding.shape
            V         = self.xyz.size(0)
            xyz_pe    = self._encode_xyz(self.xyz).float()
            outs = []
            for b in range(B):
                z     = embedding[b].unsqueeze(0).expand(V, -1)
                feats = torch.cat([z, xyz_pe], dim=1)
                logits = torch.cat([self.mlp(feats[i:i+self.chunk]) for i in range(0, V, self.chunk)])
                outs.append(logits.view(self.voxel_size, self.voxel_size, self.voxel_size, 3).permute(3, 0, 1, 2))
            return {"logits": torch.stack(outs, 0)}


class _SepConv3D(nn.Module):
    def __init__(self, c_in, c_out):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(c_in, c_in, (3, 1, 1), padding=(1, 0, 0), groups=c_in, bias=False),
            nn.Conv3d(c_in, c_in, (1, 3, 1), padding=(0, 1, 0), groups=c_in, bias=False),
            nn.Conv3d(c_in, c_in, (1, 1, 3), padding=(0, 0, 1), groups=c_in, bias=False),
            nn.Conv3d(c_in, c_out, 1, bias=False),
            nn.ReLU(True),
        )
    def forward(self, x): return self.net(x)


class Factorised3DDecoder(EmbeddingDecoder, nn.Module):
    """Separable 3-D transposed convs — fewer parameters than full ConvTranspose3d."""

    def __init__(self, embedding_dim=512, voxel_size=48, base=96):
        super().__init__()
        assert voxel_size % 8 == 0
        self.voxel_size = voxel_size
        s       = voxel_size // 8
        self.fc = nn.Linear(embedding_dim, base * s * s * s)
        self.up1 = nn.Sequential(nn.ConvTranspose3d(base,      base//2, 4, stride=2, padding=1, bias=False), _SepConv3D(base//2, base//2))
        self.up2 = nn.Sequential(nn.ConvTranspose3d(base//2,   base//4, 4, stride=2, padding=1, bias=False), _SepConv3D(base//4, base//4))
        self.up3 = nn.Sequential(nn.ConvTranspose3d(base//4,   base//8, 4, stride=2, padding=1, bias=False), _SepConv3D(base//8, base//8))
        self.out = nn.Conv3d(base//8, 3, 1)

    def decode(self, embedding, query_points=None, skips=None):
        B, s = embedding.size(0), self.voxel_size // 8
        x = self.fc(embedding).view(B, -1, s, s, s)
        x = self.up1(x); x = self.up2(x); x = self.up3(x)
        return {"logits": self.out(x)}


# ---- Point-cloud encoders ----

class PointMLPEncoder(EmbeddingEncoder, nn.Module):
    """
    Operates directly on sparse (xyz, value) points — no dense voxel grid.

    Mean + max aggregation over per-point MLP features gives a permutation-
    invariant embedding.  Optional Fourier position encoding improves
    high-frequency detail.
    """

    def __init__(self, voxel_size=48, embedding_dim=512, fourier_feats: int = 0, max_points: int = 8192):
        super().__init__()
        self.voxel_size    = voxel_size
        self.embedding_dim = embedding_dim
        self.max_points    = max_points
        self.k             = fourier_feats
        in_dim = 4 + (6 * fourier_feats if fourier_feats > 0 else 0)
        self.mlp  = nn.Sequential(nn.Linear(in_dim, 128), nn.ReLU(True), nn.Linear(128, 256), nn.ReLU(True), nn.Linear(256, 256), nn.ReLU(True))
        self.head = nn.Linear(512, embedding_dim)
        if fourier_feats > 0:
            bands = 2.0 ** torch.arange(fourier_feats).float() * np.pi
            self.register_buffer("bands", bands, persistent=False)

    def get_embedding_dim(self): return self.embedding_dim

    def _featify(self, vd: VoxelData) -> torch.Tensor:
        if vd.occupied_coords.numel() == 0:
            return torch.zeros((1, 4 + (6*self.k if self.k > 0 else 0)), device=self.head.weight.device)
        xyz = vd.occupied_coords.float() / vd.bounds.float()
        val = vd.values.float().unsqueeze(1)
        x   = torch.cat([xyz, val], dim=1)
        if self.k > 0:
            uvw    = xyz * 2 - 1
            ang    = uvw.unsqueeze(-1) * self.bands
            fourier = torch.cat([torch.sin(ang), torch.cos(ang)], -1).reshape(uvw.size(0), -1)
            x = torch.cat([x, fourier], dim=1)
        if x.size(0) > self.max_points:
            x = x[torch.randperm(x.size(0), device=x.device)[:self.max_points]]
        return x

    def encode(self, voxel_batch: List[VoxelData]) -> torch.Tensor:
        embs = []
        for vd in voxel_batch:
            f    = self.mlp(self._featify(vd))
            mean = f.mean(0)
            mx,_ = f.max(0)
            embs.append(self.head(torch.cat([mean, mx])))
        return torch.stack(embs, 0)


class PointNetPPLiteFPEncoder(EmbeddingEncoder, nn.Module):
    """
    Hierarchical PointNet++ encoder with feature propagation.

    Stages sample progressively fewer centres (voxel-grid or FPS) and
    aggregate neighbour features via a local MLP + mean/max pooling.
    Final global mean+max over the last stage's tokens feeds the projection head.
    """

    def __init__(self, voxel_size=48, embedding_dim=512, max_points=8192,
                 stages=((2048, 2.5), (512, 5.0), (128, 9.0)),
                 nbrs_cap=64, fourier_feats=0, width=128, use_fps=False):
        super().__init__()
        self.voxel_size    = voxel_size
        self.embedding_dim = embedding_dim
        self.max_points    = max_points
        self.stages        = list(stages)
        self.nbrs_cap      = int(nbrs_cap)
        self.k_fourier     = int(fourier_feats)
        self.width         = int(width)
        self.use_fps       = bool(use_fps)

        in0 = 3 + 1 + 1 + (6 * self.k_fourier if self.k_fourier > 0 else 0)
        inL = 3 + 1 + width + (6 * self.k_fourier if self.k_fourier > 0 else 0)
        self.stage_mlps = nn.ModuleList()
        self.post_mlps  = nn.ModuleList()
        for si in range(len(self.stages)):
            ind = in0 if si == 0 else inL
            self.stage_mlps.append(nn.Sequential(nn.Linear(ind, width), nn.ReLU(True), nn.Linear(width, width), nn.ReLU(True)))
            self.post_mlps.append(nn.Sequential(nn.Linear(2 * width, width), nn.ReLU(True)))
        self.head = nn.Linear(2 * width, embedding_dim)
        if self.k_fourier > 0:
            self.register_buffer("bands", (2.0 ** torch.arange(self.k_fourier).float()) * np.pi, persistent=False)

    def get_embedding_dim(self): return self.embedding_dim

    @staticmethod
    def _pairwise_dist2(a, b):
        a2 = (a**2).sum(1, keepdim=True)
        b2 = (b**2).sum(1).unsqueeze(0)
        return (a2 + b2 - 2.0 * (a @ b.t())).clamp_min_(0.0)

    @staticmethod
    def _fps(xyz, K):
        N = xyz.size(0); K = min(K, N)
        if K == N:
            return torch.arange(N, device=xyz.device, dtype=torch.long)
        sel  = torch.empty(K, dtype=torch.long, device=xyz.device)
        sel[0] = torch.randint(0, N, (1,), device=xyz.device)
        dists  = torch.full((N,), float('inf'), device=xyz.device)
        for i in range(1, K):
            d2 = ((xyz - xyz[sel[i-1]].unsqueeze(0))**2).sum(1)
            dists = torch.minimum(dists, d2)
            sel[i] = torch.argmax(dists)
        return sel

    def _voxel_grid_subsample(self, xyz_int, target_k):
        device = xyz_int.device
        N = int(xyz_int.size(0))
        if N == 0: return torch.empty(0, dtype=torch.long, device=device)
        if N <= target_k: return torch.arange(N, device=device, dtype=torch.long)
        cell = max(1, int(np.ceil(self.voxel_size / (target_k ** (1/3)))))
        cells = (xyz_int // cell).to(torch.int64)
        h = (cells[:, 0] + cells[:, 1] * 73856093 + cells[:, 2] * 19349663).to(torch.int64)
        h_sorted, perm = torch.sort(h)
        keep_mask = torch.ones_like(h_sorted, dtype=torch.bool)
        keep_mask[1:] = h_sorted[1:] != h_sorted[:-1]
        keep = perm[keep_mask]
        if keep.numel() > target_k:
            keep = keep[torch.randperm(keep.numel(), device=device)[:target_k]]
        return keep

    def _fourier(self, x):
        ang = x.unsqueeze(-1) * self.bands
        return torch.cat([torch.sin(ang), torch.cos(ang)], -1).reshape(*x.shape[:-1], -1)

    def _stage(self, src_xyz, src_feat, base_xyz, K, radius, si):
        device = src_xyz.device
        idx    = self._fps(base_xyz, K) if self.use_fps else self._voxel_grid_subsample(base_xyz.long(), K)
        centers_xyz = base_xyz[idx]
        d2     = self._pairwise_dist2(centers_xyz, src_xyz)
        r2     = float(radius * radius)
        d2_m   = d2.clone(); d2_m[d2_m > r2] = float('inf')
        k_take = min(self.nbrs_cap, src_xyz.size(0))
        vals, nn_idx = torch.topk(d2_m, k=k_take, dim=1, largest=False)
        none_mask = torch.isinf(vals[:, 0])
        if none_mask.any():
            fb = d2.argmin(dim=1)
            nn_idx[none_mask, 0] = fb[none_mask]
            vals[none_mask, 0]   = d2[none_mask, fb[none_mask]]
        valid   = torch.isfinite(vals)
        valid_e = valid.unsqueeze(-1)
        nbr_xyz  = src_xyz[nn_idx]
        delta    = (nbr_xyz - centers_xyz.unsqueeze(1)) / max(radius, 1e-6)
        distn    = torch.sqrt(torch.clamp(vals, 0.0, r2)).unsqueeze(-1) / max(radius, 1e-6)
        nbr_feat = src_feat[nn_idx]
        delta = delta * valid_e; distn = distn * valid_e; nbr_feat = nbr_feat * valid_e
        parts = [delta, distn, nbr_feat]
        if self.k_fourier > 0:
            parts.append(self._fourier(delta))
        x    = torch.cat(parts, -1)
        Kk   = x.shape[0] * x.shape[1]
        x    = self.stage_mlps[si](x.view(Kk, -1)).view(x.shape[0], x.shape[1], -1)
        counts = valid.float().sum(1).clamp_min_(1.0).unsqueeze(-1)
        mean   = x.sum(1) / counts
        x_m    = torch.where(valid.unsqueeze(-1), x, x.new_full((), torch.finfo(x.dtype).min))
        mx, _  = x_m.max(1)
        return centers_xyz, self.post_mlps[si](torch.cat([mean, mx], -1))

    def _encode_one(self, vd: VoxelData) -> torch.Tensor:
        dev = self.head.weight.device
        if vd.occupied_coords.numel() == 0:
            return torch.zeros(self.embedding_dim, device=dev)
        xyz = vd.occupied_coords.to(dev, dtype=torch.float32)
        val = vd.values.to(dev, dtype=torch.float32).unsqueeze(1)
        if xyz.size(0) > self.max_points:
            keep = self._voxel_grid_subsample(xyz.long(), self.max_points)
            xyz, val = xyz[keep], val[keep]
        K0, r0 = self.stages[0]
        c_xyz, c_feat = self._stage(xyz, val, xyz, K0, r0, 0)
        for si in range(1, len(self.stages)):
            Ki, ri = self.stages[si]
            c_xyz, c_feat = self._stage(c_xyz, c_feat, c_xyz, Ki, ri, si)
        mean = c_feat.mean(0)
        mx,_ = c_feat.max(0)
        return self.head(torch.cat([mean, mx]))

    def encode(self, voxel_batch: List[VoxelData]) -> torch.Tensor:
        return torch.stack([self._encode_one(vd) for vd in voxel_batch], 0)


class CrossAttnTokensEncoder(EmbeddingEncoder, nn.Module):
    """
    Cross-attention encoder with 3-D sinusoidal positional encoding.

    Tokens are read from /4 and /8 conv feature maps, then a set of
    learnable query vectors attends to those tokens over multiple layers.
    The output queries are projected to ``embedding_dim``.
    """

    def __init__(self, voxel_size=48, embedding_dim=1024,
                 token_channels=256, num_queries=64, heads=4, layers=2, use_global_g4=True):
        super().__init__()
        assert voxel_size % 8 == 0
        self.voxel_size    = voxel_size
        self.embedding_dim = embedding_dim
        self.use_global_g4 = use_global_g4
        self.conv1   = nn.Sequential(nn.Conv3d(1,  32, 5, stride=2, padding=2, bias=False), nn.ReLU(True))
        self.conv2   = nn.Sequential(nn.Conv3d(32, 64, 3, stride=2, padding=1, bias=False), nn.ReLU(True))
        self.conv3   = nn.Sequential(nn.Conv3d(64, token_channels, 3, stride=2, padding=1, bias=False), nn.ReLU(True))
        self.reduce4 = nn.Conv3d(64, token_channels, 1, bias=False)
        self.queries = nn.Parameter(torch.randn(num_queries, token_channels))
        self.mhas    = nn.ModuleList([nn.MultiheadAttention(token_channels, heads, batch_first=True) for _ in range(layers)])
        self.ffns    = nn.ModuleList([nn.Sequential(nn.Linear(token_channels, 4*token_channels), nn.ReLU(True), nn.Linear(4*token_channels, token_channels)) for _ in range(layers)])
        self.norm_q  = nn.LayerNorm(token_channels)
        self.norm_t  = nn.LayerNorm(token_channels)
        self.proj    = nn.Linear(num_queries * token_channels, embedding_dim, bias=False)
        self.register_buffer("pe_cached", None, persistent=False)

    def _sparse_to_dense(self, vd):
        g = torch.zeros((1, 1, self.voxel_size, self.voxel_size, self.voxel_size), device=vd.occupied_coords.device)
        if vd.occupied_coords.numel():
            xyz = (vd.occupied_coords.float() / vd.bounds.float() * (self.voxel_size - 1)).long().clamp(0, self.voxel_size - 1)
            g[0, 0, xyz[:, 0], xyz[:, 1], xyz[:, 2]] = vd.values
        return g

    def _posenc_3d(self, D, H, W, C, device):
        if (self.pe_cached is not None and self.pe_cached.shape == (1, D, H, W, C) and self.pe_cached.device == device):
            return self.pe_cached
        z, y, x = [torch.linspace(-1, 1, s, device=device) for s in (D, H, W)]
        zz, yy, xx = torch.meshgrid(z, y, x, indexing='ij')
        coords = torch.stack([xx, yy, zz], -1)
        bands  = torch.arange(max(1, C // 6), device=device).float()
        ang    = coords.unsqueeze(-1) * ((2.0 ** bands) * np.pi)
        pe     = torch.cat([torch.sin(ang), torch.cos(ang)], -1).flatten(-2)
        if pe.shape[-1] < C:
            pe = F.pad(pe, (0, C - pe.shape[-1]))
        else:
            pe = pe[..., :C]
        self.pe_cached = pe.unsqueeze(0)
        return self.pe_cached

    def _flatten_tokens(self, x):
        B, C, D, H, W = x.shape
        return x.permute(0, 2, 3, 4, 1).reshape(B, D*H*W, C)

    def encode(self, voxel_batch):
        x  = torch.cat([self._sparse_to_dense(v) for v in voxel_batch], 0)
        B  = x.size(0)
        x2 = self.conv1(x); x4 = self.conv2(x2); x8 = self.conv3(x4)
        pe8    = self._posenc_3d(*x8.shape[2:], x8.shape[1], x8.device)
        t_list = [self._flatten_tokens(x8 + pe8.permute(0, 4, 1, 2, 3))]
        if self.use_global_g4:
            x4r = self.reduce4(x4)
            pe4 = self._posenc_3d(*x4r.shape[2:], x4r.shape[1], x4r.device)
            t_list.append(self._flatten_tokens(x4r + pe4.permute(0, 4, 1, 2, 3)))
        tokens = self.norm_t(torch.cat(t_list, 1))
        q = self.queries.unsqueeze(0).expand(B, -1, -1)
        for mha, ffn in zip(self.mhas, self.ffns):
            q2, _ = mha(self.norm_q(q), tokens, tokens)
            q = q + q2 + ffn(q)
        return self.proj(q.reshape(B, -1))

    def forward(self, voxel_batch): return self.encode(voxel_batch)
    def get_embedding_dim(self): return self.embedding_dim


# ============================================================
#  Loss heads (auxiliary supervision)
# ============================================================

class AuxFnHead(LossHead):
    """Wraps a ``make_aux_loss`` function as a ``LossHead``."""
    variable_length_target: bool = False

    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, embedding, logits=None):
        if logits is None and self.logits_fn is not None:
            logits = self.logits_fn(embedding)
        return logits

    def compute_loss(self, prediction, targets):
        return self.fn(prediction, targets)


class TopDownHeightHead(LossHead, nn.Module):
    """Predicts the 2-D max-height map (x, z) normalised to [0, 1]."""

    def __init__(self, embedding_dim=128, map_size=32):
        super().__init__()
        self.map_size = map_size
        self.net = nn.Sequential(nn.Linear(embedding_dim, 256), nn.ReLU(True), nn.Linear(256, map_size * map_size))

    def forward(self, embedding, logits=None):
        return self.net(embedding).view(-1, self.map_size, self.map_size)

    def compute_loss(self, prediction, target):
        return F.l1_loss(prediction, target)


class DistanceTransformHead(LossHead, nn.Module):
    """Predicts a coarse 3-D distance transform to nearest occupancy."""

    def __init__(self, embedding_dim=128, grid=24):
        super().__init__()
        self.grid = grid
        self.net  = nn.Sequential(nn.Linear(embedding_dim, 512), nn.ReLU(True), nn.Linear(512, grid * grid * grid))

    def forward(self, embedding, logits=None):
        return self.net(embedding).view(-1, self.grid, self.grid, self.grid)

    def compute_loss(self, prediction, target_dt):
        return F.smooth_l1_loss(prediction, target_dt)


class OccupancyProjectionHead(LossHead, nn.Module):
    """Predicts 3-D binary occupancy at reduced resolution."""

    def __init__(self, embedding_dim=128, grid=24):
        super().__init__()
        self.grid = grid
        self.fc   = nn.Sequential(nn.Linear(embedding_dim, 1024), nn.ReLU(True), nn.Linear(1024, grid * grid * grid))

    def forward(self, embedding, logits=None):
        return torch.sigmoid(self.fc(embedding).view(-1, 1, self.grid, self.grid, self.grid))

    def compute_loss(self, prediction, target_occ):
        return F.binary_cross_entropy(prediction, target_occ)


class RelativeOffsetHead(LossHead, nn.Module):
    """Predicts 3-D offsets to the K nearest obstacles from the drone position."""

    def __init__(self, embedding_dim=128, k_nearest=5):
        super().__init__()
        self.k_nearest = k_nearest
        self.net = nn.Sequential(nn.Linear(embedding_dim, 128), nn.ReLU(True), nn.Linear(128, k_nearest * 3))

    def forward(self, embedding, logits=None):
        return self.net(embedding).view(-1, self.k_nearest, 3)

    def compute_loss(self, prediction, target):
        return F.mse_loss(prediction, target)


# ============================================================
#  Training framework
# ============================================================

class EmbeddingTrainer:
    def __init__(self, encoder: EmbeddingEncoder, decoder: EmbeddingDecoder,
                 loss_heads: Dict[str, LossHead], device='cuda', lr=1e-3, recon_loss=None):
        self.encoder   = encoder
        self.decoder   = decoder
        self.device    = torch.device(device if torch.cuda.is_available() else 'cpu')

        if isinstance(encoder, nn.Module):
            encoder.to(self.device)
        if isinstance(decoder, nn.Module):
            decoder.to(self.device)

        self.loss_heads = nn.ModuleDict(loss_heads)
        self.loss_heads.to(self.device)

        all_params = []
        if isinstance(encoder, nn.Module):
            all_params += list(encoder.parameters())
        if isinstance(decoder, nn.Module):
            all_params += list(decoder.parameters())
        all_params += list(self.loss_heads.parameters())

        self.recon_loss_fn = recon_loss or (lambda l, t, extra=None: F.cross_entropy(l, t))
        self.optimizer     = torch.optim.Adam(all_params, lr=lr)

    def train_step(self, voxel_batch: List[VoxelData],
                   target_batch: List[Dict[str, torch.Tensor]],
                   loss_weights: Dict[str, float]) -> Dict[str, float]:
        self.optimizer.zero_grad()

        out = self.encoder.encode(voxel_batch)
        embedding, skips = (out if isinstance(out, tuple) else (out, None))

        reconstruction = self.decoder.decode(embedding, skips=skips)
        logits = reconstruction["logits"]

        losses = {}

        if "reconstruction" in loss_weights:
            recon_targets = torch.cat([self._create_reconstruction_target(vd) for vd in voxel_batch], 0)

            # Dynamic class weighting (EMA) — compensates for the heavy class imbalance
            # between empty voxels and occupied ones.
            num_classes = 3
            counts = torch.bincount(recon_targets.view(-1), minlength=num_classes).float()
            freq   = counts / counts.sum().clamp_min(1)
            med    = freq[freq > 0].median()
            w      = (med / freq.clamp_min(1e-6)).clamp(0.25, 4.0)

            if not hasattr(self, "_class_weight_ema"):
                self._class_weight_ema = w.to(logits.device)
            else:
                self._class_weight_ema = (0.9 * self._class_weight_ema
                                          + 0.1 * w.to(logits.device))

            losses["reconstruction"] = self.recon_loss_fn(
                logits, recon_targets, extra={"class_weights": self._class_weight_ema}
            )

        # Merge per-sample target dicts for auxiliary heads.
        merged_targets = {}
        if target_batch:
            keys = set().union(*[t.keys() for t in target_batch])
            for k in keys:
                if k == "points":
                    merged_targets[k] = [t[k] for t in target_batch if k in t]
                else:
                    merged_targets[k] = torch.stack(
                        [t[k] for t in target_batch if k in t], 0
                    ).to(self.device)

        for name, head in self.loss_heads.items():
            if name not in loss_weights:
                continue
            if isinstance(head, AuxFnHead):
                preds = head(embedding, logits=logits)
                losses[name] = head.compute_loss(preds, merged_targets)
                continue
            if not all(name in tb for tb in target_batch):
                continue
            preds = head(embedding, logits=logits)
            t_for_head = ([tb[name] for tb in target_batch] if getattr(head, "variable_length_target", False)
                          else torch.stack([tb[name] for tb in target_batch], 0))
            losses[name] = head.compute_loss(preds, t_for_head)

        total_loss    = sum(loss_weights.get(n, 0) * v for n, v in losses.items())
        losses["total"] = total_loss
        total_loss.backward()
        self.optimizer.step()
        return {k: v.item() for k, v in losses.items()}

    def _create_reconstruction_target(self, vd: VoxelData) -> torch.Tensor:
        side   = self.encoder.voxel_size
        target = torch.zeros((1, side, side, side), device=self.device, dtype=torch.long)
        if vd.occupied_coords.shape[0] > 0:
            c = vd.occupied_coords.clone()
            c[:, 2] = -c[:, 2]
            c = c.long()
            c[:, 0].clamp_(0, side - 1)
            c[:, 1].clamp_(0, side - 1)
            c[:, 2].clamp_(0, side - 1)
            filled_mask = vd.values >= 0.6
            sparse_mask = (~filled_mask) & (vd.values >= 0.3)
            target[0, c[filled_mask, 0], c[filled_mask, 1], c[filled_mask, 2]] = 2
            target[0, c[sparse_mask, 0], c[sparse_mask, 1], c[sparse_mask, 2]] = 1
        return target


# ============================================================
#  Dataset
# ============================================================

class TerrainBatch(IterableDataset):
    """
    Infinite iterable dataset of procedurally generated voxel worlds.

    Each iteration calls the Rust terrain generator with a fresh random seed,
    converts the resulting ``VoxelGrid`` to a ``VoxelData`` tensor pair, and
    optionally builds supervision targets (distance transform, low-res
    occupancy, centre-of-mass, point cloud).
    """

    def __init__(self, world_size: int = 120,
                 build_dt: bool = True,
                 build_low: bool = True,
                 low_scale: int = 4,
                 build_com: bool = True,
                 build_points: bool = True):
        self.world_size = int(world_size)
        self.build_dt   = build_dt
        self.build_low  = build_low
        self.low_scale  = int(low_scale)
        self.build_com  = build_com
        self.build_pts  = build_points

    @staticmethod
    def world_to_voxeldata(world: voxelsim.VoxelGrid, side: int) -> VoxelData:
        coords_np, vals_np = world.as_numpy()
        return VoxelData(
            occupied_coords=torch.from_numpy(coords_np),
            values=torch.from_numpy(vals_np),
            bounds=torch.tensor([side] * 3, dtype=torch.float32),
            drone_pos=torch.tensor([side // 2] * 3, dtype=torch.float32),
        )

    def __iter__(self):
        side = self.world_size
        while True:
            g   = voxelsim.TerrainGenerator()
            cfg = voxelsim.TerrainConfig.default_py()
            cfg.set_seed_py(int(np.random.randint(0, 2**31)))
            cfg.set_world_size_py(side)
            g.generate_terrain_py(cfg)
            world      = g.generate_world_py()
            voxel_data = self.world_to_voxeldata(world, side)
            targets    = self._generate_targets(voxel_data)
            yield voxel_data, targets

    @staticmethod
    def _block_mean_3d(x: np.ndarray, scale: int) -> np.ndarray:
        if scale <= 1:
            return x
        D, H, W = x.shape
        return x.reshape(D//scale, scale, H//scale, scale, W//scale, scale).mean(axis=(1, 3, 5))

    def _generate_targets(self, voxel_data: VoxelData) -> Dict[str, torch.Tensor]:
        side = int(voxel_data.bounds[0].item())
        labels_np = np.zeros((side, side, side), dtype=np.uint8)

        if voxel_data.occupied_coords.numel() > 0:
            coords = voxel_data.occupied_coords.cpu().numpy().astype(np.int64)
            coords[:, 2] = -coords[:, 2]
            np.clip(coords, 0, side - 1, out=coords)
            vals = voxel_data.values.cpu().numpy().astype(np.float32)
            filled = vals >= 0.6
            sparse = (~filled) & (vals >= 0.3)
            if filled.any():
                labels_np[coords[filled, 0], coords[filled, 1], coords[filled, 2]] = 2
            if sparse.any():
                labels_np[coords[sparse, 0], coords[sparse, 1], coords[sparse, 2]] = 1

        occ_np = (labels_np == 2).astype(np.float32) + 0.5 * (labels_np == 1).astype(np.float32)

        targets: Dict[str, torch.Tensor] = {
            "labels":    torch.from_numpy(labels_np.astype(np.int64)),
            "occ_dense": torch.from_numpy(occ_np),
        }

        if self.build_low:
            targets["occ_low"] = torch.from_numpy(self._block_mean_3d(occ_np, self.low_scale))

        if self.build_dt:
            occ_bool   = occ_np > 0
            dt_to_empty = edt(occ_bool.astype(np.uint8))
            dt_to_occ   = edt((~occ_bool).astype(np.uint8))
            dt_bound    = np.minimum(dt_to_empty, dt_to_occ)
            mx = float(dt_bound.max()) if dt_bound.size else 1.0
            targets["dt"] = torch.from_numpy((dt_bound / (mx + 1e-6)).astype(np.float32))

        if self.build_com:
            mass = occ_np.sum()
            if mass > 1e-6:
                xs = np.arange(side, dtype=np.float32)
                com_np = np.array([
                    (occ_np.sum(axis=(0, 1)) * xs).sum() / mass,
                    (occ_np.sum(axis=(0, 2)) * xs).sum() / mass,
                    (occ_np.sum(axis=(1, 2)) * xs).sum() / mass,
                ], dtype=np.float32)
            else:
                com_np = np.full(3, side / 2.0, dtype=np.float32)
            targets["com"] = torch.from_numpy(com_np)

        if self.build_pts and voxel_data.occupied_coords.numel() > 0:
            targets["points"] = voxel_data.occupied_coords.clone().detach().float()
        else:
            targets["points"] = torch.zeros((0, 3), dtype=torch.float32)

        return targets


# ============================================================
#  Utilities
# ============================================================

class Timer:
    def __enter__(self):
        self.t0 = time.perf_counter(); return self
    def __exit__(self, *_):
        self.dt = time.perf_counter() - self.t0


def collate_fn(batch):
    voxel_list, target_list = zip(*batch)
    return list(voxel_list), list(target_list)


def show_voxels(sample, client):
    """Send either a VoxelData (GT) or logit tensor [B,3,D,H,W] to the renderer."""
    cell_dict = {}
    if isinstance(sample, VoxelData):
        coords = sample.occupied_coords.long().cpu().numpy()
        vals   = sample.values.cpu().numpy()
        for (x, y, z), v in zip(coords, vals):
            cell_dict[(int(x), int(y), int(z))] = (
                voxelsim.Cell.filled() if v > 0.6 else voxelsim.Cell.sparse()
            )
    else:
        logits = sample[0] if sample.dim() == 5 else sample
        pred   = logits.argmax(0).cpu().numpy()
        for x, y, z in np.argwhere(pred == 2):
            cell_dict[(int(x), int(y), int(z))] = voxelsim.Cell.filled()
        for x, y, z in np.argwhere(pred == 1):
            cell_dict[(int(x), int(y), int(z))] = voxelsim.Cell.sparse()
    client.send_world_py(voxelsim.VoxelGrid.from_dict_py(cell_dict))


class RunLogger:
    """Logs losses and timing to CSV and TensorBoard; saves periodic checkpoints."""

    def __init__(self, enc_name, dec_name, *, loss_keys, root="runs", ckpt_every=50):
        stamp    = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
        self.dir = os.path.join(root, f"{stamp}-{enc_name}_{dec_name}")
        os.makedirs(os.path.join(self.dir, "checkpoints"), exist_ok=True)
        self.loss_keys  = list(loss_keys)
        self.ckpt_every = ckpt_every
        header = ["epoch"] + [f"loss_{k}" for k in self.loss_keys] + ["t_fetch_ms", "t_move_ms", "t_model_ms"]
        self.csv_f = open(os.path.join(self.dir, "losses.csv"), "w", newline="")
        self.csv_w = csv.writer(self.csv_f)
        self.csv_w.writerow(header)
        self.tb = SummaryWriter(self.dir)

    def log_epoch(self, epoch: int, losses: dict, t: dict):
        row = [epoch] + [losses.get(k, 0.0) for k in self.loss_keys] + [
            t.get("fetch", 0) * 1e3, t.get("move", 0) * 1e3, t.get("model", 0) * 1e3
        ]
        self.csv_w.writerow(row); self.csv_f.flush()
        for k in self.loss_keys:
            self.tb.add_scalar(f"loss/{k}", losses.get(k, 0.0), epoch)
        for k, v in t.items():
            self.tb.add_scalar(f"time/{k}", v, epoch)

    def maybe_ckpt(self, epoch: int, enc, dec, opt):
        if (epoch + 1) % self.ckpt_every == 0:
            torch.save({
                "epoch": epoch + 1,
                "encoder": enc.state_dict(),
                "decoder": dec.state_dict(),
                "optim":   opt.state_dict(),
            }, os.path.join(self.dir, "checkpoints", f"epoch-{epoch+1:04d}.pt"))

    def close(self):
        self.csv_f.close(); self.tb.close()


# ============================================================
#  Experiment runner
# ============================================================

def make_aux_heads(embedding_dim):
    return {
        # Uncomment heads to add auxiliary supervision signals:
        # "soft_iou":      AuxFnHead(make_aux_loss("soft_iou")),
        # "tv":            AuxFnHead(make_aux_loss("tv")),
        # "boundary":      AuxFnHead(make_aux_loss("boundary")),
        # "com":           AuxFnHead(make_aux_loss("com")),
        # "class_balance": AuxFnHead(make_aux_loss("class_balance")),
        # "ms_occ":        AuxFnHead(make_aux_loss("ms_occ", scale=4)),
        # "chamfer":       AuxFnHead(make_aux_loss("chamfer", topk=2048)),
        # "topdown":       TopDownHeightHead(embedding_dim=embedding_dim),
        # "dt":            DistanceTransformHead(embedding_dim=embedding_dim, grid=24),
        # "rel_offset":    RelativeOffsetHead(embedding_dim=embedding_dim, k_nearest=5),
    }


def run_experiment(encoder_class, decoder_class, loss_heads, recon_loss,
                   embedding_dim=128, num_epochs=100, batch_size=1,
                   visualize_every=10, size=48, ckpt_every=100):
    encoder = encoder_class(voxel_size=size, embedding_dim=embedding_dim)
    decoder = decoder_class(voxel_size=size, embedding_dim=embedding_dim)

    logits_fn = lambda z: decoder.decode(z)["logits"]
    for h in loss_heads.values():
        if hasattr(h, "bind_logits"):
            h.bind_logits(logits_fn)

    loss_keys = {"total", "reconstruction", *loss_heads.keys()}
    logger    = RunLogger(encoder_class.__name__, decoder_class.__name__,
                          loss_keys=loss_keys, ckpt_every=ckpt_every)
    print("logging to:", logger.dir)

    trainer    = EmbeddingTrainer(encoder, decoder, loss_heads, recon_loss=recon_loss)
    dataset    = TerrainBatch(world_size=size, build_dt=False, build_low=False,
                              build_points=False, build_com=False)
    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn,
                            num_workers=os.cpu_count(), pin_memory=True,
                            persistent_workers=True, prefetch_factor=1)

    steps_per_epoch  = 10
    loss_history     = collections.defaultdict(list)
    stats            = collections.defaultdict(list)
    viz_sample       = None
    last_viz_wall_t  = time.perf_counter()
    dataloader_iter  = iter(dataloader)

    for epoch in range(num_epochs):
        epoch_losses = collections.defaultdict(float)

        for _ in range(steps_per_epoch):
            with Timer() as t_fetch:
                voxel_batch, target_batch = next(dataloader_iter)
            if viz_sample is None:
                viz_sample = (voxel_batch[0], target_batch[0])

            with Timer() as t_move:
                voxel_batch  = [vd.to_device(trainer.device) for vd in voxel_batch]
                target_batch = [{k: v.to(trainer.device) for k, v in t.items()} for t in target_batch]

            loss_weights = {
                "reconstruction": 1.0,
                "soft_iou":       0.5,
                "tv":             0.05,
                "boundary":       0.5,
                "com":            0.2,
                "class_balance":  0.05,
                "ms_occ":         0.3,
                "chamfer":        0.2,
            }
            with Timer() as t_model:
                losses = trainer.train_step(voxel_batch, target_batch, loss_weights)

            stats["fetch"].append(t_fetch.dt)
            stats["move"].append(t_move.dt)
            stats["model"].append(t_model.dt)
            for name, value in losses.items():
                epoch_losses[name] += value

        for name in epoch_losses:
            epoch_losses[name] /= steps_per_epoch
            loss_history[name].append(epoch_losses[name])

        f = np.mean(stats["fetch"][-steps_per_epoch:])
        m = np.mean(stats["move"][-steps_per_epoch:])
        g = np.mean(stats["model"][-steps_per_epoch:])

        if epoch % visualize_every == 0:
            now = time.perf_counter()
            print(f"Epoch {epoch:04d}  losses={dict(epoch_losses)}")
            print(f"  fetch {f*1e3:.1f} ms  move {m*1e3:.1f} ms  model {g*1e3:.1f} ms  "
                  f"wall {now - last_viz_wall_t:.2f} s")
            last_viz_wall_t = now

        logger.log_epoch(epoch, epoch_losses, {"fetch": f, "move": m, "model": g})
        logger.maybe_ckpt(epoch, encoder, decoder, trainer.optimizer)

    logger.close()
    return loss_history


def sweep():
    recon_loss = make_recon_loss("ce")

    encoder_decoder_pairs = [
        (SimpleCNNEncoder, SimpleCNNDecoder),
        # (SpConvCNNEncoder,          SimpleCNNDecoder),
        # (ResNet3DEncoder,           ResNet3DDecoder),
        # (CrossAttnTokensEncoder,    ImplicitFourierDecoder),
        # (PointMLPEncoder,           ImplicitFourierDecoder),
        # (PointNetPPLiteFPEncoder,   ImplicitFourierDecoder),
    ]

    emb_dims = [1000]
    size     = 128

    regimes = [
        ("recon_only",     lambda dim: {}),
        ("recon_plus_aux", make_aux_heads),
    ]

    for regime_name, heads_factory in regimes:
        for Enc, Dec in encoder_decoder_pairs:
            for d in emb_dims:
                loss_heads = heads_factory(d)
                print(f"\n=== {regime_name} | {Enc.__name__} → {Dec.__name__} | dim={d} ===")
                run_experiment(
                    encoder_class=Enc, decoder_class=Dec,
                    loss_heads=loss_heads, recon_loss=recon_loss,
                    embedding_dim=d, num_epochs=5000, batch_size=1,
                    visualize_every=500, size=size, ckpt_every=500,
                )


if __name__ == "__main__":
    print("CUDA available:", torch.cuda.is_available())
    torch.cuda.empty_cache()
    sweep()

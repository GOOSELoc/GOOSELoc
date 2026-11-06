"""GOOSELocNet: our multimodal cross-view localization baseline (paper, Sec. 4).

Frozen DINOv3 ViT-L/16 encodes the 1024 px aerial tile (64 x 64 patch tokens)
and the four surround cameras ([CLS] tokens); frozen Utonia with a LoRA
adapter encodes the LiDAR sweep (global token + 8 azimuth-sector tokens).
Everything after the encoders is small and trainable:

* **Position.** Each modality correlates its own query with the projected
  tile patches. RGB+L fuses the camera and LiDAR score maps with an evidence
  gate (entropy, peak probability, top-1/top-2 margin of each map); the
  single-modality variants pool their tokens into one query. The heat map is
  decoded by soft-argmax.
* **Heading.** Per-camera and per-LiDAR-sector match maps against the tile are
  reduced by small CNNs, concatenated with the raw camera [CLS] tokens and
  sat-conditioned cross-attention read-outs, and classified into 72 circular
  bins (5 degrees).

Three variants are provided: ``rgb``, ``lidar`` and ``rgbl``. Parameter
names are those of the paper checkpoints, so released weights load as is.

Inputs are RGB in ``[0, 1]`` without mean/std normalization, exactly as the
paper models were trained.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from gooseloc.models.gooselocnet.encoders import (
    DEFAULT_UTONIA,
    DEFAULT_VIT,
    apply_lora,
    build_utonia,
    build_utonia_transform,
    build_vit,
    encode_point_cloud,
)

VARIANTS = ("rgb", "lidar", "rgbl")
N_CAMS = 4


@dataclass
class GOOSELocNetConfig:
    variant: str = "rgbl"
    vit_name: str = DEFAULT_VIT
    vit_dim: int = 1024
    tile_size: int = 1024
    grid: int = 64                    # tile_size / ViT patch size
    d_match: int = 512
    lidar_dim: int = 1224             # Utonia multi-level descriptor
    utonia_name: str = DEFAULT_UTONIA
    utonia_scale: float = 0.2
    utonia_enc_patch_size: int = 1024
    lora_rank: int = 8
    lora_alpha: int = 16
    lidar_sectors: int = 8
    heading_bins: int = 72
    # Per-sample modality dropout of the RGB+L fusion: P(drop cameras), P(drop LiDAR).
    modality_dropout: tuple[float, float] = (0.3, 0.15)
    tau_init: float = 0.07

    def __post_init__(self) -> None:
        if self.variant not in VARIANTS:
            raise ValueError(f"variant must be one of {VARIANTS}, got {self.variant!r}")
        self.modality_dropout = tuple(self.modality_dropout)

    @property
    def use_cameras(self) -> bool:
        return self.variant in ("rgb", "rgbl")

    @property
    def use_lidar(self) -> bool:
        return self.variant in ("lidar", "rgbl")

    def to_dict(self) -> dict:
        return asdict(self)


class AttentionPool(nn.Module):
    """Single-head attention of one query over a token set."""

    def __init__(self, d: int) -> None:
        super().__init__()
        self.q_proj, self.k_proj = nn.Linear(d, d), nn.Linear(d, d)
        self.v_proj, self.out_proj = nn.Linear(d, d), nn.Linear(d, d)
        self.scale = 1.0 / math.sqrt(d)

    def forward(self, query: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
        q = self.q_proj(query).unsqueeze(1)                              # (B, 1, D)
        attn = torch.einsum("bqd,bkd->bqk", q, self.k_proj(tokens)) * self.scale
        out = torch.einsum("bqk,bkd->bqd", attn.softmax(dim=-1), self.v_proj(tokens))
        return self.out_proj(out).squeeze(1)


def _score_map_stats(scores: torch.Tensor) -> torch.Tensor:
    """``(B, N)`` logits -> ``(B, 3)``: normalized entropy, max probability, top-1/top-2 gap."""
    p = F.softmax(scores.detach().float(), dim=1)
    ent = -(p * (p + 1e-9).log()).sum(dim=1) / math.log(p.shape[1])
    top2 = p.topk(2, dim=1).values
    return torch.stack([ent, top2[:, 0], top2[:, 0] - top2[:, 1]], dim=1)


def _sincos_pe_2d(grid: int, dim: int) -> torch.Tensor:
    """Fixed 2D sin-cos positional encoding ``(grid * grid, dim)``, row-major."""
    d4 = dim // 4
    omega = 1.0 / (10000.0 ** (torch.arange(d4, dtype=torch.float32) / d4))
    out = torch.einsum("p,d->pd", torch.arange(grid, dtype=torch.float32), omega)
    row = torch.cat([out.sin(), out.cos()], dim=1)
    pe = torch.cat([row.unsqueeze(1).expand(grid, grid, d4 * 2),
                    row.unsqueeze(0).expand(grid, grid, d4 * 2)], dim=2)
    return pe.reshape(grid * grid, dim)


def _embedding(*shape: int) -> nn.Parameter:
    p = nn.Parameter(torch.zeros(*shape))
    nn.init.normal_(p, std=0.02)
    return p


def _match_cnn(in_ch: int) -> nn.Sequential:
    """Two strided convolutions: 64 x 64 match maps -> 32 x 4 x 4 (keeps where peaks are)."""
    return nn.Sequential(nn.Conv2d(in_ch, 16, 5, stride=4, padding=2), nn.ReLU(inplace=True),
                         nn.Conv2d(16, 32, 3, stride=4, padding=1), nn.ReLU(inplace=True))


class GOOSELocNet(nn.Module):
    def __init__(self, config: GOOSELocNetConfig | None = None, *, pretrained: bool = True) -> None:
        super().__init__()
        cfg = self.config = config or GOOSELocNetConfig()
        D, V, L, G = cfg.d_match, cfg.vit_dim, cfg.lidar_dim, cfg.grid

        # Frozen encoders (kept in eval mode, see train()).
        self.vit = build_vit(cfg.vit_name, cfg.tile_size, pretrained=pretrained)
        self.vit_num_prefix = int(getattr(self.vit, "num_prefix_tokens", 1))
        if cfg.use_lidar:
            self.utonia = build_utonia(cfg.utonia_name, enc_patch_size=cfg.utonia_enc_patch_size)
            self._utonia_transform = build_utonia_transform(cfg.utonia_scale)
        else:
            self.utonia = None
        for enc in (self.vit, self.utonia):
            if enc is not None:
                enc.requires_grad_(False)
                enc.eval()
        if cfg.use_lidar and cfg.lora_rank > 0:
            apply_lora(self.utonia, cfg.lora_rank, cfg.lora_alpha)

        # Position branch.
        self.proj_sat = nn.Linear(V, D)
        self.proj_cam = nn.Linear(V, D)
        self.proj_lidar = nn.Linear(L, D) if cfg.use_lidar else None
        self.cam_pe = _embedding(N_CAMS, D)       # azimuth embedding
        self.fusion_query = _embedding(D)
        self.fusion_attn = AttentionPool(D)
        self.fusion_scoremap = cfg.variant == "rgbl"
        if self.fusion_scoremap:
            self.fusion_token_norm = nn.LayerNorm(D)
            self.lidar_pe = _embedding(D)
            self.fusion_gate = nn.Sequential(nn.Linear(6, 32), nn.ReLU(), nn.Linear(32, 2))
            self.fusion_map_gain = nn.Parameter(torch.zeros(2))           # log gains
        else:
            self.fusion_token_norm = self.lidar_pe = self.fusion_gate = self.fusion_map_gain = None

        # Heading branch.
        heading_in = N_CAMS * V                                          # raw camera [CLS]
        self.yaw_cam_az = _embedding(N_CAMS, V)
        if cfg.use_cameras:
            self.yaw_cam_q, self.yaw_sat_k = nn.Linear(V, D), nn.Linear(V, D)
            self.yaw_score_cnn = _match_cnn(N_CAMS)
            heading_in += N_CAMS * D + 32 * (G // 16) ** 2
        else:
            self.yaw_cam_q = self.yaw_sat_k = self.yaw_score_cnn = None
        K = cfg.lidar_sectors
        if cfg.use_lidar:
            self.yaw_lidar_proj = nn.Linear(L, D)
            self.yaw_lidar_az = _embedding(K, L)
            self.yaw_lidar_score_cnn = _match_cnn(K)
            self.yaw_lidar_q, self.yaw_lidar_satcond_k = nn.Linear(L, D), nn.Linear(V, D)
            heading_in += 32 * (G // 16) ** 2 + K * D
        else:
            self.yaw_lidar_proj = self.yaw_lidar_az = self.yaw_lidar_score_cnn = None
            self.yaw_lidar_q = self.yaw_lidar_satcond_k = None
        self.register_buffer("yaw_pe_2d", _sincos_pe_2d(G, D), persistent=False)
        self.yaw_coarse_head = nn.Linear(heading_in, cfg.heading_bins)
        self.log_tau = nn.Parameter(torch.tensor(math.log(cfg.tau_init)))

    # -- housekeeping ---------------------------------------------------------

    def train(self, mode: bool = True):
        super().train(mode)
        self.vit.eval()
        if self.utonia is not None:
            self.utonia.eval()
        return self

    @property
    def device(self) -> torch.device:
        return self.proj_sat.weight.device

    @property
    def tau(self) -> torch.Tensor:
        return self.log_tau.exp().clamp_min(1e-3)

    def trainable_state_dict(self) -> dict[str, torch.Tensor]:
        """Only the trained surface (heads, projections, LoRA): what checkpoints store."""
        return {n: p.detach().cpu() for n, p in self.named_parameters() if p.requires_grad}

    def load_trainable_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        missing, unexpected = self.load_state_dict(state, strict=False)
        trainable = {n for n, p in self.named_parameters() if p.requires_grad}
        if unexpected or [k for k in missing if k in trainable]:
            raise RuntimeError(f"checkpoint does not fit the {self.config.variant!r} variant: "
                               f"unexpected {unexpected[:5]}, missing {[k for k in missing if k in trainable][:5]}")

    # -- encoders -------------------------------------------------------------

    @torch.no_grad()
    def _vit(self, x: torch.Tensor) -> torch.Tensor:
        return self.vit.forward_features(x)

    def encode_lidar(self, points: Sequence[torch.Tensor]):
        out = [encode_point_cloud(self.utonia, self._utonia_transform, p[:, :3], self.device,
                                  self.config.lidar_sectors) for p in points]
        return torch.stack([o[0] for o in out]), torch.stack([o[1] for o in out])

    # -- forward --------------------------------------------------------------

    def forward(self, aerial: torch.Tensor, cameras: torch.Tensor | None = None,
                lidar: Sequence[torch.Tensor] | None = None) -> dict:
        """``aerial (B, 3, 1024, 1024)``, ``cameras (B, 4, 3, H, W)``, ``lidar`` list of ``(N, >=3)``."""
        cfg = self.config
        B = aerial.shape[0]
        sat_tokens = self._vit(aerial)[:, self.vit_num_prefix:]
        if cfg.use_cameras:
            b, c, _, h, w = cameras.shape
            cam_cls = self._vit(cameras.reshape(b * c, 3, h, w))[:, 0].reshape(b, c, -1)
        else:   # LiDAR-only: the camera slots carry no information
            cam_cls = sat_tokens.new_zeros(B, N_CAMS, cfg.vit_dim)
        lidar_feat = lidar_sectors = None
        if cfg.use_lidar:
            lidar_feat, lidar_sectors = self.encode_lidar(lidar)
        return self.heads(sat_tokens, cam_cls, lidar_feat, lidar_sectors)

    def heads(self, sat_tokens, cam_cls, lidar_feat=None, lidar_sectors=None) -> dict:
        """Everything after the frozen encoders."""
        cfg = self.config
        B, G = sat_tokens.shape[0], cfg.grid
        sat_proj = self.proj_sat(sat_tokens.float())                     # (B, G*G, D)
        cam_proj = self.proj_cam(cam_cls.float())                        # (B, 4, D)
        tokens = cam_proj + self.cam_pe.unsqueeze(0)
        if cfg.use_lidar:
            lid = self.proj_lidar(lidar_feat.float())
            if self.lidar_pe is not None:
                lid = lid + self.lidar_pe
            tokens = torch.cat([tokens, lid.unsqueeze(1)], dim=1)
        if self.fusion_token_norm is not None:
            tokens = self.fusion_token_norm(tokens)
        query = self.fusion_query.unsqueeze(0).expand(B, -1)

        out: dict = {}
        if self.fusion_scoremap:
            cam_q = F.normalize(self.fusion_attn(query, tokens[:, :N_CAMS]), dim=-1)
            lidar_q = F.normalize(tokens[:, N_CAMS], dim=-1)
            gain = torch.exp(self.fusion_map_gain)
            s_cam = torch.einsum("bd,bnd->bn", cam_q, sat_proj) * gain[0] / self.tau
            s_lid = torch.einsum("bd,bnd->bn", lidar_q, sat_proj) * gain[1] / self.tau
            w = self.fusion_gate(torch.cat([_score_map_stats(s_cam), _score_map_stats(s_lid)], 1).to(s_cam.dtype))
            n_active = 2.0
            if self.training and sum(cfg.modality_dropout) > 0:
                # One draw per sample: drop cameras, drop LiDAR, or keep both.
                u = torch.rand(B, device=w.device)
                p_cam, p_lid = cfg.modality_dropout
                mask = torch.stack([u < p_cam, (u >= p_cam) & (u < p_cam + p_lid)], dim=1)
                w = w.masked_fill(mask, float("-inf"))
                n_active = (~mask).sum(dim=1, keepdim=True).to(s_cam.dtype)
            # Gates sum to the number of active modalities, so a map keeps its
            # logit scale whether or not its partner is dropped.
            gates = F.softmax(w, dim=1) * n_active
            scores = gates[:, :1] * s_cam + gates[:, 1:] * s_lid
            out["fusion_gates"] = gates
        else:
            scores = torch.einsum("bd,bnd->bn", self.fusion_attn(query, tokens), sat_proj) / self.tau
        scores = scores.reshape(B, G, G)
        heatmap = F.softmax(scores.reshape(B, -1), dim=1).reshape(B, G, G)
        idx = torch.arange(G, device=scores.device, dtype=scores.dtype)
        pred_patch = torch.stack([(heatmap * idx.view(1, G, 1)).sum((1, 2)),
                                  (heatmap * idx.view(1, 1, G)).sum((1, 2))], dim=1)   # (row, col)

        cls_yaw = cam_cls.float() + self.yaw_cam_az.unsqueeze(0)
        feats = [cls_yaw.reshape(B, -1)]
        if cfg.use_cameras:
            kv = self.yaw_sat_k(sat_tokens.float()) + self.yaw_pe_2d.unsqueeze(0)
            feats.append(F.scaled_dot_product_attention(self.yaw_cam_q(cls_yaw), kv, kv).reshape(B, -1))
            sm = torch.einsum("bcd,bnd->bcn", cam_proj.detach().float(), sat_proj.detach().float())
            sm = F.softmax(sm / self.tau.detach(), dim=2).reshape(B, N_CAMS, G, G)
            feats.append(self.yaw_score_cnn(sm).reshape(B, -1))
        if cfg.use_lidar:
            K = cfg.lidar_sectors
            sec = lidar_sectors.float() + self.yaw_lidar_az.unsqueeze(0)
            lsm = torch.einsum("bkd,bnd->bkn", self.yaw_lidar_proj(sec), sat_proj.detach()) / self.tau.detach()
            lsm = F.softmax(lsm, dim=2).reshape(B, K, G, G)
            feats.append(self.yaw_lidar_score_cnn(lsm).reshape(B, -1))
            lkv = self.yaw_lidar_satcond_k(sat_tokens.float()) + self.yaw_pe_2d.unsqueeze(0)
            feats.append(F.scaled_dot_product_attention(self.yaw_lidar_q(sec), lkv, lkv).reshape(B, -1))

        out.update(scores=scores, heatmap=heatmap, pred_patch=pred_patch,
                   heading_logits=self.yaw_coarse_head(torch.cat(feats, dim=1)))
        return out


# --- decoding -------------------------------------------------------------------

def patch_to_xy(pred_patch: torch.Tensor, tile_size: int = 1024, grid: int = 64) -> torch.Tensor:
    """Soft-argmax ``(row, col)`` in patch units -> ``(x, y)`` pixels of the input tile."""
    return pred_patch.flip(-1) * (tile_size / grid)


def heading_from_logits(logits: torch.Tensor, local_window: int = 1) -> torch.Tensor:
    """Circular classifier -> degrees in ``[0, 360)`` (relative to the tile's up-axis).

    Takes the arg-max bin (never the expectation, which would average the two
    modes of a front/back ambiguity) and refines it with the probability-weighted
    circular centroid of its +-``local_window`` neighbours. Bin k is centred at
    ``k * 360 / K``.
    """
    B, K = logits.shape
    w = 360.0 / K
    probs = logits.float().softmax(dim=1)
    top = probs.argmax(dim=1)
    if local_window <= 0:
        return (top.float() * w) % 360.0
    offsets = torch.arange(-local_window, local_window + 1, device=logits.device)
    wts = torch.gather(probs, 1, (top.view(B, 1) + offsets.view(1, -1)) % K)
    ang = (top.view(B, 1).float() + offsets.view(1, -1).float()) * w
    return ((wts * ang).sum(dim=1) / wts.sum(dim=1).clamp_min(1e-9)) % 360.0

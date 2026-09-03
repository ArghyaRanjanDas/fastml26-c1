"""Teacher architectures (no FPGA constraint).

Both models emit `n_out` logits: n_out=1 is the binary HH-vs-background head (squeezed to
(B,)), n_out=4 is the softmax head over the process classes in team/data.py GROUP_ID order
(0 QCD, 1 HH_4b, 2 tt, 3 Wjets), returned as (B, 4).

BigDeepSet : phi 128-64-32 per candidate, mean+max pooling, concat event feats, rho 256-128-64.
ParTLite   : Particle-Transformer-lite -- per-candidate embedding d=128, N particle-attention
             blocks (8 heads) whose attention logits get a learned bias from the pairwise
             (ln dR, ln kT, ln z, ln m^2) features, then class-attention pooling, concat the
             mean-pooled tokens and an event-feature MLP, MLP head.  One logit out.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from common import particle_features, pair_features, N_RAW_PART, N_RICH_PART, N_PAIR


def _init_linear(m: nn.Module):
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=0.02)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


class BigDeepSet(nn.Module):
    def __init__(self, n_event: int = 11, rich: bool = True, phi=(128, 64, 32),
                 rho=(256, 128, 64), dropout: float = 0.1, n_out: int = 1):
        super().__init__()
        self.n_out = n_out
        self.rich = rich
        d_in = N_RICH_PART if rich else N_RAW_PART
        self.in_bn = nn.BatchNorm1d(d_in)
        layers, d = [], d_in
        for h in phi:
            layers += [nn.Linear(d, h), nn.GELU()]
            d = h
        self.phi = nn.Sequential(*layers)
        self.pool_bn = nn.BatchNorm1d(2 * d + n_event)
        layers, d = [], 2 * d + n_event
        for h in rho:
            layers += [nn.Linear(d, h), nn.GELU(), nn.Dropout(dropout)]
            d = h
        self.rho = nn.Sequential(*layers)
        self.out = nn.Linear(d, n_out)
        self.apply(_init_linear)

    def forward(self, x: torch.Tensor, f: torch.Tensor) -> torch.Tensor:
        feats = particle_features(x, self.rich)                # (B,P,d_in) fp32
        mask = (x[..., 0] > 0)
        B, P, D = feats.shape
        h = self.in_bn(feats.reshape(B * P, D)).reshape(B, P, D) * mask[..., None]
        h = self.phi(h)                                        # (B,P,32)
        m = mask[..., None].to(h.dtype)
        n = m.sum(1).clamp_min(1.0)
        mean = (h * m).sum(1) / n
        mx = h.masked_fill(~mask[..., None], float("-inf")).amax(1)
        mx = torch.where(torch.isfinite(mx), mx, torch.zeros_like(mx))
        z = torch.cat([mean, mx, f.to(h.dtype)], dim=1)
        z = self.pool_bn(z)
        out = self.out(self.rho(z))
        return out.squeeze(-1) if self.n_out == 1 else out


# --------------------------------------------------------------- ParT-lite

class Attention(nn.Module):
    """Multi-head self-attention with an additive (B,H,P,P) bias on the logits."""

    def __init__(self, d: int, n_heads: int, dropout: float):
        super().__init__()
        assert d % n_heads == 0
        self.h, self.dh, self.p = n_heads, d // n_heads, dropout
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)

    def forward(self, x: torch.Tensor, bias: torch.Tensor | None) -> torch.Tensor:
        B, P, _ = x.shape
        q, k, v = self.qkv(x).view(B, P, 3, self.h, self.dh).permute(2, 0, 3, 1, 4)
        if bias is not None:
            bias = bias.to(q.dtype)
        o = F.scaled_dot_product_attention(q, k, v, attn_mask=bias,
                                           dropout_p=self.p if self.training else 0.0)
        return self.proj(o.transpose(1, 2).reshape(B, P, -1))


class ClassAttention(nn.Module):
    """The class token queries [cls; particles] (CaiT / ParT class-attention block)."""

    def __init__(self, d: int, n_heads: int, dropout: float):
        super().__init__()
        self.h, self.dh, self.p = n_heads, d // n_heads, dropout
        self.q = nn.Linear(d, d)
        self.kv = nn.Linear(d, 2 * d)
        self.proj = nn.Linear(d, d)

    def forward(self, cls: torch.Tensor, x: torch.Tensor, key_mask: torch.Tensor | None):
        B, P, _ = x.shape
        q = self.q(cls).view(B, 1, self.h, self.dh).transpose(1, 2)              # (B,H,1,dh)
        kv = self.kv(torch.cat([cls, x], 1)).view(B, P + 1, 2, self.h, self.dh)
        k, v = kv.permute(2, 0, 3, 1, 4)                                           # (B,H,P+1,dh)
        am = None
        if key_mask is not None:
            full = torch.cat([torch.ones_like(key_mask[:, :1]), key_mask], 1)     # cls always visible
            am = torch.zeros(B, 1, 1, P + 1, dtype=q.dtype, device=q.device)
            am = am.masked_fill(~full[:, None, None, :], float("-inf"))
        o = F.scaled_dot_product_attention(q, k, v, attn_mask=am,
                                           dropout_p=self.p if self.training else 0.0)
        return self.proj(o.transpose(1, 2).reshape(B, 1, -1))


class Block(nn.Module):
    def __init__(self, d: int, n_heads: int, mlp_ratio: int, dropout: float, class_attn: bool = False):
        super().__init__()
        self.class_attn = class_attn
        self.ln1 = nn.LayerNorm(d)
        self.attn = ClassAttention(d, n_heads, dropout) if class_attn else Attention(d, n_heads, dropout)
        self.ln2 = nn.LayerNorm(d)
        self.mlp = nn.Sequential(nn.Linear(d, mlp_ratio * d), nn.GELU(), nn.Dropout(dropout),
                                 nn.Linear(mlp_ratio * d, d))
        self.drop = nn.Dropout(dropout)

    def forward(self, x, bias=None, cls=None, key_mask=None):
        if self.class_attn:
            cls = cls + self.drop(self.attn(self.ln1(cls), self.ln1(x), key_mask))
            cls = cls + self.drop(self.mlp(self.ln2(cls)))
            return cls
        x = x + self.drop(self.attn(self.ln1(x), bias))
        x = x + self.drop(self.mlp(self.ln2(x)))
        return x


class ParTLite(nn.Module):
    def __init__(self, n_event: int = 11, rich: bool = True, d: int = 128, n_heads: int = 8,
                 n_blocks: int = 4, n_cls_blocks: int = 2, mlp_ratio: int = 4,
                 dropout: float = 0.1, pair_hidden: int = 64, head_dims=(256, 64),
                 event_hidden: int = 64, n_out: int = 1):
        super().__init__()
        self.rich = rich
        self.n_out = n_out
        d_in = N_RICH_PART if rich else N_RAW_PART
        # particle embedding (ParT: BN on inputs, then 3x [LN, Linear, GELU])
        self.in_bn = nn.BatchNorm1d(d_in)
        self.embed = nn.Sequential(
            nn.LayerNorm(d_in), nn.Linear(d_in, d), nn.GELU(),
            nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(),
            nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(),
        )
        # pairwise interaction -> per-head attention bias (ParT: BN, 3x [Linear, GELU], Linear->H)
        self.pair_bn = nn.BatchNorm1d(N_PAIR)
        self.pair_mlp = nn.Sequential(
            nn.Linear(N_PAIR, pair_hidden), nn.GELU(),
            nn.Linear(pair_hidden, pair_hidden), nn.GELU(),
            nn.Linear(pair_hidden, pair_hidden), nn.GELU(),
            nn.Linear(pair_hidden, n_heads),
        )
        self.blocks = nn.ModuleList([Block(d, n_heads, mlp_ratio, dropout) for _ in range(n_blocks)])
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.cls_blocks = nn.ModuleList([Block(d, n_heads, mlp_ratio, dropout, class_attn=True)
                                         for _ in range(n_cls_blocks)])
        self.norm_cls = nn.LayerNorm(d)
        self.norm_tok = nn.LayerNorm(d)
        self.event_mlp = nn.Sequential(nn.Linear(n_event, event_hidden), nn.GELU())
        layers, dd = [], 2 * d + event_hidden
        for h in head_dims:
            layers += [nn.Linear(dd, h), nn.GELU(), nn.Dropout(dropout)]
            dd = h
        self.head = nn.Sequential(*layers, nn.Linear(dd, n_out))
        self.apply(_init_linear)

    def forward(self, x: torch.Tensor, f: torch.Tensor) -> torch.Tensor:
        mask = x[..., 0] > 0                                            # (B,P)
        feats = particle_features(x, self.rich)                         # fp32
        pf, pm = pair_features(x)                                       # (B,P,P,4), (B,P,P)
        B, P, D = feats.shape
        h = self.in_bn(feats.reshape(B * P, D)).reshape(B, P, D) * mask[..., None]
        h = self.embed(h)                                               # (B,P,d)
        u = self.pair_bn(pf.reshape(B * P * P, N_PAIR)) * pm.reshape(-1, 1)
        u = self.pair_mlp(u).reshape(B, P, P, -1).permute(0, 3, 1, 2)   # (B,H,P,P)
        # padded keys are unreachable; the diagonal keeps its learned bias
        u = u.masked_fill(~mask[:, None, None, :], float("-inf"))
        for blk in self.blocks:
            h = blk(h, bias=u)
        cls = self.cls_token.expand(B, -1, -1)
        for blk in self.cls_blocks:
            cls = blk(h, cls=cls, key_mask=mask)
        cls = self.norm_cls(cls[:, 0])
        m = mask[..., None].to(h.dtype)
        mean = self.norm_tok((h * m).sum(1) / m.sum(1).clamp_min(1.0))
        z = torch.cat([cls, mean, self.event_mlp(f.to(cls.dtype))], dim=1)
        out = self.head(z)
        return out.squeeze(-1) if self.n_out == 1 else out


MODELS = {"deepset": BigDeepSet, "part": ParTLite}


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

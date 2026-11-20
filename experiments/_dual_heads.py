"""modules/dual_heads.py — dual output heads for final-odds vs nonfinal-sequence.

See docs/DUAL_HEAD_DESIGN.md for the design rationale. Short version:

- The backbone is trained on ALL historical race data; produces a per-horse
  context vector ctx_h.
- Head F consumes (ctx_h, log(final_odds)) → p(win). Trained on all races.
- Head N consumes (ctx_h, odds_seq_embed(subfinal_sequence)) → p(win).
  Trained on the smaller Azure-mirror subset.
- Both heads are **zero-init on output**, so adding them to a trained
  backbone is a no-op at step 0.

This module does NOT include training loops; see tools/finetune_nonfinal_head.py
(to be written) for Stage-2 fine-tuning of Head N.
"""
from __future__ import annotations
from dataclasses import dataclass, field

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Final-odds head (Stage 1 target) ────────────────────────────────────

class FinalOddsHead(nn.Module):
    """Per-horse p(win) head conditioned on final odds.

    Input:
        ctx_h:         [B, N, d_model]  per-horse context from backbone
        log_final_odds: [B, N]          natural log of the decimal final odds
        mask:          [B, N] bool      valid horse (for softmax masking)

    Output:
        logits: [B, N]  pre-softmax per-horse logit
        probs:  [B, N]  softmax(logits) with mask applied

    Zero-init on output so forward returns probs = 1/N (uniform) at step 0.
    """

    def __init__(self, d_model: int, hidden: int = 32):
        super().__init__()
        self.d_model = int(d_model)
        self.fc1 = nn.Linear(d_model + 1, hidden)   # +1 = log_final_odds
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, 1)
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(
        self,
        ctx_h: torch.Tensor,
        log_final_odds: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if ctx_h.shape[-1] != self.d_model:
            raise ValueError(
                f'FinalOddsHead: expected ctx_h[..., {self.d_model}], '
                f'got {tuple(ctx_h.shape)}')
        if log_final_odds.shape != ctx_h.shape[:-1]:
            raise ValueError(
                f'log_final_odds shape {tuple(log_final_odds.shape)} must '
                f'match ctx_h[:-1] = {tuple(ctx_h.shape[:-1])}')
        x = torch.cat([ctx_h, log_final_odds.unsqueeze(-1)], dim=-1)
        logits = self.fc2(self.act(self.fc1(x))).squeeze(-1)
        if mask is None:
            return logits, F.softmax(logits, dim=-1)
        very_neg = torch.full_like(logits, -1e9)
        masked = torch.where(mask, logits, very_neg)
        return logits, F.softmax(masked, dim=-1)


# ── Odds-sequence encoder (lightweight DeepSet) ─────────────────────────

@dataclass
class OddsSeqEmbedConfig:
    d_tok: int = 16
    d_out: int = 16
    dropout: float = 0.1


class OddsSeqEmbed(nn.Module):
    """DeepSet over per-horse (log_odds, time_to_post_minutes) snapshots.

    Input:
        odds_series: [B, N, T, 2]  (log_odds, t_to_post_min); -1e9 for pad
        series_mask: [B, N, T] bool — True = valid snapshot
    Output:
        embed: [B, N, d_out]  pooled embedding per horse

    Pool = mean + max over valid snapshots along T axis.
    Horses with zero valid snapshots get a zero embedding.
    """

    def __init__(self, cfg: OddsSeqEmbedConfig | None = None):
        super().__init__()
        cfg = cfg or OddsSeqEmbedConfig()
        self.cfg = cfg
        # Per-token MLP: (log_odds, t_to_post, Δlog_odds, is_recent) → d_tok
        self.tok_mlp = nn.Sequential(
            nn.Linear(4, cfg.d_tok),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.d_tok, cfg.d_tok),
        )
        # After mean+max pool: 2·d_tok → d_out
        self.fuse = nn.Linear(2 * cfg.d_tok, cfg.d_out)

    def forward(
        self,
        odds_series: torch.Tensor,
        series_mask: torch.Tensor,
    ) -> torch.Tensor:
        B, N, T, _ = odds_series.shape
        log_odds = odds_series[..., 0]                           # [B,N,T]
        t_to_post = odds_series[..., 1]                          # [B,N,T]
        # Per-token features: (log_odds, t_to_post/60, Δlog_odds vs prev, 1[t<=10m])
        # Δlog_odds: pairwise with previous VALID snapshot per horse. Approximate
        # by broadcasting from each horse's earliest snapshot.
        # Simpler: Δlog_odds = log_odds - mean(valid log_odds). Lossy but cheap.
        valid = series_mask.float()
        denom = valid.sum(dim=-1, keepdim=True).clamp(min=1)     # [B,N,1]
        mean_log = (log_odds * valid).sum(dim=-1, keepdim=True) / denom
        delta_log = (log_odds - mean_log) * valid
        is_recent = (t_to_post <= 10).float() * valid
        features = torch.stack([log_odds, t_to_post / 60.0, delta_log, is_recent], dim=-1)  # [B,N,T,4]

        tok_h = self.tok_mlp(features)                            # [B,N,T,d_tok]
        tm = valid.unsqueeze(-1)                                   # [B,N,T,1]
        denom2 = tm.sum(dim=2).clamp(min=1.0)                     # [B,N,1]
        mean_pool = (tok_h * tm).sum(dim=2) / denom2              # [B,N,d_tok]
        very_neg = torch.full_like(tok_h, -1e9)
        max_pool = torch.where(tm.bool(), tok_h, very_neg).max(dim=2).values  # [B,N,d_tok]

        # Zero out horses with no valid snapshots
        has_any = (series_mask.any(dim=2, keepdim=True)).float()  # [B,N,1]
        mean_pool = mean_pool * has_any
        max_pool  = torch.where(has_any.bool().expand_as(max_pool),
                                max_pool, torch.zeros_like(max_pool))

        pooled = torch.cat([mean_pool, max_pool], dim=-1)          # [B,N,2*d_tok]
        return self.fuse(pooled)                                    # [B,N,d_out]


# ── Nonfinal-odds head (Stage 2 target) ─────────────────────────────────

class NonfinalOddsHead(nn.Module):
    """Per-horse p(win) head conditioned on an odds-snapshot sequence.

    Combines ctx_h (from frozen backbone) with an odds-sequence embedding
    (from OddsSeqEmbed) to produce p(win).

    Zero-init on output so this head's forward returns uniform at step 0
    regardless of backbone state.

    Input:
        ctx_h:       [B, N, d_model]
        odds_series: [B, N, T, 2]
        series_mask: [B, N, T] bool
        mask:        [B, N]    bool valid horse
    Output:
        logits: [B, N]
        probs:  [B, N]
    """

    def __init__(
        self,
        d_model: int,
        hidden: int = 32,
        seq_cfg: OddsSeqEmbedConfig | None = None,
        dropout: float = 0.2,
        layernorm: bool = False,
    ):
        super().__init__()
        self.d_model = int(d_model)
        self.seq_embed = OddsSeqEmbed(seq_cfg)
        d_in = d_model + self.seq_embed.cfg.d_out
        self.norm = nn.LayerNorm(d_in) if layernorm else nn.Identity()
        self.fc1 = nn.Linear(d_in, hidden)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden, 1)
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(
        self,
        ctx_h: torch.Tensor,
        odds_series: torch.Tensor,
        series_mask: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if ctx_h.shape[-1] != self.d_model:
            raise ValueError(
                f'NonfinalOddsHead: expected ctx_h[..., {self.d_model}], '
                f'got {tuple(ctx_h.shape)}')
        seq = self.seq_embed(odds_series, series_mask)             # [B,N,d_out]
        x = self.norm(torch.cat([ctx_h, seq], dim=-1))
        h = self.drop(self.act(self.fc1(x)))
        logits = self.fc2(h).squeeze(-1)
        if mask is None:
            return logits, F.softmax(logits, dim=-1)
        very_neg = torch.full_like(logits, -1e9)
        masked = torch.where(mask, logits, very_neg)
        return logits, F.softmax(masked, dim=-1)


# ── Disagreement-based confidence ───────────────────────────────────────

def head_disagreement(probs_f: torch.Tensor, probs_n: torch.Tensor) -> torch.Tensor:
    """Per-race disagreement = L1 distance between F and N probability vectors.

    Input:
        probs_f: [B, N]  softmax from FinalOddsHead
        probs_n: [B, N]  softmax from NonfinalOddsHead

    Output:
        disagreement: [B]  L1 distance per race; range [0, 2]

    When disagreement is small → F and N agree → high confidence.
    When disagreement is large → low confidence; abstain from betting.
    """
    return (probs_f - probs_n).abs().sum(dim=-1)

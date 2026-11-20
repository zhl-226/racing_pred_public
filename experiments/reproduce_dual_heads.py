"""demo/dual_heads_reproduction.py — final-odds + nonfinal-odds head smoke.

Demonstrates the dual-head architecture from `docs/plans/DUAL_HEAD_DESIGN.md`:

  - FinalOddsHead: p(win | ctx_h, log_final_odds). Trained on ALL
    historical races (large dataset).
  - NonfinalOddsHead: p(win | ctx_h, odds_seq_embed(subfinal_snapshots)).
    Trained on the smaller Azure-mirror subset.
  - head_disagreement(F, N) → L1 distance per race = free UQ signal.

Both heads are zero-init on output so adding them to a trained backbone
is a no-op at step 0 — important for staged training (freeze backbone +
Head F during Stage 2 Head N fine-tune).

Needs torch. Runs ~10 seconds. Prints shapes + a single-race overfit
sanity check (Head N drives CE to near-zero on one race).
"""
from __future__ import annotations
import sys
from pathlib import Path

# Import the local copy
HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

try:
    import torch
    import torch.nn.functional as F
except ImportError:
    print('ERROR: torch not installed. pip install torch')
    sys.exit(1)

from _dual_heads import (
    FinalOddsHead, NonfinalOddsHead, OddsSeqEmbed, OddsSeqEmbedConfig,
    head_disagreement,
)


def main():
    print('=== Dual-head reproduction ===')
    print()
    torch.manual_seed(0)

    # Simulated race: B=2 batches, N=10 horses, backbone d_model=16
    B, N = 2, 10
    ctx = torch.randn(B, N, 16)
    log_final = torch.log(torch.rand(B, N) * 20 + 2)   # odds in [2, 22]
    mask = torch.ones(B, N).bool()

    print(f'input: batch_size={B}, horses={N}, d_model=16')
    print()

    # Head F — final-odds-conditioned
    head_f = FinalOddsHead(d_model=16)
    head_f.eval()
    with torch.no_grad():
        _, probs_f = head_f(ctx, log_final, mask=mask)
    print('--- FinalOddsHead ---')
    print(f'  probs shape: {probs_f.shape}')
    print(f'  probs sum per race (should be 1): {probs_f.sum(-1).tolist()}')
    print(f'  Zero-init check: all uniform? '
          f'{torch.allclose(probs_f, torch.full_like(probs_f, 1/N), atol=1e-5)}')
    print()

    # Head N — nonfinal-odds-sequence
    head_n = NonfinalOddsHead(d_model=16)
    head_n.eval()
    # Fake odds sequence: T=5 snapshots per horse
    T = 5
    series = torch.stack([log_final.unsqueeze(-1).repeat(1, 1, T),       # log_odds
                           torch.arange(60, 0, -12.0).expand(B, N, T)],  # t_to_post
                          dim=-1)
    series_mask = torch.ones(B, N, T).bool()
    with torch.no_grad():
        _, probs_n = head_n(ctx, series, series_mask, mask=mask)
    print('--- NonfinalOddsHead (odds-sequence conditioned) ---')
    print(f'  probs shape: {probs_n.shape}')
    print(f'  Zero-init check: all uniform? '
          f'{torch.allclose(probs_n, torch.full_like(probs_n, 1/N), atol=1e-5)}')
    print()

    # Disagreement signal
    disag = head_disagreement(probs_f, probs_n)
    print('--- Head disagreement (UQ signal) ---')
    print(f'  disagreement per race: {disag.tolist()}')
    print(f'  (bounded in [0, 2]; 0 = full agreement = high confidence)')
    print()

    # Overfit sanity: can Head N learn a single race?
    print('--- Overfit single race (sanity check) ---')
    head_train = NonfinalOddsHead(d_model=16, dropout=0.0)
    target_winner = torch.tensor([3])       # race 0: horse 3 wins
    ctx_s = ctx[:1]; series_s = series[:1]; sm_s = series_mask[:1]
    opt = torch.optim.Adam(head_train.parameters(), lr=1e-2)
    for step in range(100):
        opt.zero_grad()
        logits, _ = head_train(ctx_s, series_s, sm_s)
        loss = F.cross_entropy(logits, target_winner)
        loss.backward(); opt.step()
    _, probs = head_train(ctx_s, series_s, sm_s)
    print(f'  after 100 steps, prob on true winner (horse 3): {float(probs[0, 3]):.3f}')
    assert float(probs[0, 3]) > 0.3, 'overfit check failed'
    print('  ✅ Head N converges toward the correct winner')


if __name__ == '__main__':
    main()

# Dual-head architecture: final-odds + nonfinal-sequence + disagreement UQ

A two-head design for prediction tasks where training data has richer
labels than deployment time. Specifically:

- **Head F (final-odds head)** consumes `(ctx_h, log(final_odds)) → p(win)`.
  Trained on every historical race (every race has a final-odds label).
- **Head N (nonfinal-odds head)** consumes
  `(ctx_h, odds_sequence_embedding) → p(win)`. Trained on a much
  smaller subset of races where we have the complete subfinal-odds
  trajectory. This is the deployment-time head: when live-betting, we
  only have pre-race snapshots.
- **Head disagreement** `|F(x) − N(x)|` (L1 per race) gives a free
  uncertainty estimate. Disagreement ≈ 0 → both heads agree on the
  distribution → high-confidence bet. Disagreement large → abstain.

## Zero-init property

Both heads are constructed with zero-init on the output projection.
This means:

- Adding them to a trained backbone is a no-op at step 0.
- Head N can be grafted onto a backbone pre-trained with Head F, and
  fine-tuning Head N doesn't collapse the existing predictions.

This is the standard staged-training recipe (Stage 1: train backbone +
Head F end-to-end; Stage 2: freeze trunk + Head F, train Head N on the
nonfinal subset).

## Architecture

```
ctx_h ───┬─► FinalOddsHead  ──── p_F ─┐
         │                             ├─► disagreement (UQ)
odds_seq─┴─► NonfinalOddsHead ── p_N ─┘
```

`OddsSeqEmbed` is a small DeepSet over per-horse (log_odds, time_to_post)
snapshots with per-token features: `[log_odds, t_to_post/60, Δlog_odds,
is_recent]`. Mean + max pool across the time axis, fused into a `d_out`
embedding.

## Reproducing the UQ signal + overfit check

```bash
python experiments/reproduce_dual_heads.py
```

~10 seconds on CPU. Verifies:
1. Zero-init: both heads return uniform softmax before training.
2. Softmax masking: invalid horses get prob 0.
3. Head disagreement bounded in [0, 2] per race.
4. Overfit sanity: on one race, 100 Adam steps converge Head N's
   softmax to put >30% mass on the correct winner.

## Empirical validation on a pari-mutuel task

As a proof-of-research check, the two-head design was exercised on a
small time-sensitive prediction task where training data has richer
labels (final-pool prices) than deployment (pre-event pool snapshots
at T−2 min). On a held-out 395-event subset, Head N reached
calibrated softmax but the pre-event signal was too sparse for a clean
positive outcome (negative ROI point estimate, CI straddling zero).
The experiment validates the zero-init + freeze-trunk staged recipe;
the empirical outcome is a known failure mode on small training
subsets. Full training pipeline is in the private sibling repository.

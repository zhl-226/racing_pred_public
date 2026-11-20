# Dual-head design evaluation — final-odds vs nonfinal-sequence

**Written 2026-11-15.** User's proposal: two output heads sharing the
same backbone.

- **Head F (final):** p(win) conditioned on final odds (large dataset,
  all historical races)
- **Head N (nonfinal):** p(win) conditioned on nonfinal-odds sequence up
  to a specific time T (small dataset, only the Azure mirror window)

This doc evaluates the proposal against what we already have planned
(TRACKS_PROFIT_RESEARCH.md P1: wire drift predictor as input feature).

## Why this is a good idea

Separation of concerns:
- Head F exploits the ENTIRE training set (every race ever has final
  odds). It learns a strong representation of "given final odds,
  which horse wins" — essentially a calibrated win probability
  conditioned on market consensus.
- Head N exploits the MIRROR SUBSET (~400 races, ~5K pairs). It learns
  the actual deployment-time task: "given the sequence of odds
  observed up to T, which horse wins at the final pool."

The shared backbone lets Head N reuse the horse/race representation
learned by Head F, so Head N doesn't have to learn everything from
scratch on a small dataset.

## Comparison to current plan (P1 drift-as-feature)

| aspect | Current P1 | Dual-head |
|---|---|---|
| data used | model-pred features from one trained model | all race data (F) + mirror subset (N) |
| production head | model's existing win-head + drift multiplier | Head N only |
| training stages | 1 (train model) + fit sklearn GBR | 2 (pretrain F, then fine-tune N) |
| final probability | reweight p by predicted final odds | directly output by Head N |
| implementation cost | ~0.5 day (sklearn stack already built) | ~3-5 days (new torch heads + loss schedule) |
| long-term flexibility | sklearn on top ⇒ limited | NN end-to-end ⇒ composable with other heads |
| principled UQ | requires conformal wrapper | compare F vs N → built-in disagreement signal |

**Verdict: dual-head is better long-term.** Both leverage odds-sequence
info, but dual-head's architectural separation is cleaner AND it gives
us free uncertainty (when F and N agree → confident; disagree → low
confidence).

## Design

### Backbone
Keep the existing `jpx_champion_decomp_no_odds_head` backbone (= trunk
without OddsHead, which was the +12% ROI baseline). This is our trained
feature extractor → produces `ctx_h ∈ R^{B×N×d_model}` per horse per race.

### Head F — final odds
```python
class FinalOddsHead(nn.Module):
    def __init__(self, d_model, hidden=32):
        super().__init__()
        self.lin1 = nn.Linear(d_model + 1, hidden)  # +1 = log(final_odds)
        self.lin2 = nn.Linear(hidden, 1)
        nn.init.zeros_(self.lin2.weight); nn.init.zeros_(self.lin2.bias)

    def forward(self, ctx_h, log_final_odds):
        # ctx_h: [B, N, d_model], log_final_odds: [B, N]
        x = torch.cat([ctx_h, log_final_odds.unsqueeze(-1)], dim=-1)
        return self.lin2(F.gelu(self.lin1(x))).squeeze(-1)  # [B, N]
```
- Trained on the **full historical training set**. Every row has final
  odds in `races_scaled.parquet`.
- Produces per-horse logit; softmax across race for win-probability.
- Zero-init output so adding to a trained backbone doesn't disturb it.

### Head N — nonfinal sequence
```python
class NonfinalOddsHead(nn.Module):
    def __init__(self, d_model, odds_seq_dim=16, hidden=32):
        super().__init__()
        self.odds_encoder = odds_seq_net.OddsSeqEncoder(d_out=odds_seq_dim)
        self.lin1 = nn.Linear(d_model + odds_seq_dim, hidden)
        self.lin2 = nn.Linear(hidden, 1)
        nn.init.zeros_(self.lin2.weight); nn.init.zeros_(self.lin2.bias)

    def forward(self, ctx_h, odds_series, odds_mask):
        # odds_series: [B, N, T, 2]  (per-horse T timestamps of (log_odds, time_to_post))
        enc = self.odds_encoder(odds_series, odds_mask)      # [B, N, odds_seq_dim]
        x = torch.cat([ctx_h, enc], dim=-1)
        return self.lin2(F.gelu(self.lin1(x))).squeeze(-1)
```
- `odds_seq_net.py` already exists — a DeepSet-style pooler over
  variable-length (log_odds, time_to_post) snapshots. Reuse it.
- Trained on **the Azure mirror subset** (races where we have snapshot
  sequences, not just final odds).

### Training recipe

Stage 1 — pretrain backbone + Head F on all historical data:
- Loss = CE on win-head (softmax over field) OR negative log-likelihood
  of the winner's final-odds-conditioned probability
- This is essentially a re-train of a `use_odds_head=True` version of
  the existing model, but with the `FinalOddsHead` architecture instead
  of MP-13's decomposed OddsHead. ~3 hours CPU per 30 epochs.

Stage 2 — freeze backbone + Head F, train Head N on mirror data:
- Freeze all backbone + Head F params
- Zero-init Head N, train with CE loss on mirror races only
- ~5K training pairs → small dataset; use heavy regularization + early
  stopping to avoid overfitting. 20 epochs max.
- Compare Head N's predictions on held-out mirror races to:
  - Head F (with final odds)
  - Current `no_odds_head` + drift-GBR

### Inference time

**Deployment uses Head N only.** But keep Head F accessible for:
- Historical analysis / backfill (if we have final odds)
- Disagreement signal: `|F(x) - N(x)|` as a confidence proxy
  → when |disagreement| is small, trust the bet
  → when large, abstain

## Risks

1. **Small mirror dataset.** 395 races × ~11 horses = ~4,400 training rows.
   For a 2-layer MLP head this is borderline-sufficient. Mitigations:
   - Heavy dropout (0.3-0.5) in Head N's hidden layer
   - Early stopping with 20% held-out mirror races
   - L2 regularization on Head N weights
2. **Backbone trained without odds.** Our current backbone was trained
   without OddsHead. Head F may underperform the full retrained-with-
   odds version. Fix: either accept the limitation OR retrain backbone
   with Head F active from epoch 1. The latter is safer.
3. **Mirror distribution drift.** Mirror window is narrow (~3 months).
   Over time it should grow; Head N can be re-trained monthly.
4. **Odds-sequence sparsity.** Some races have only 1-2 snapshots; some
   have 10+. `odds_seq_net` handles variable length but very-few-snapshot
   races may not inform Head N well.

## Comparison summary

Dual-head is **more work but strictly better architecture**. Specifically:

- Preserves the large-data advantage (backbone + Head F see all races)
- Specializes the small-data head (Head N only sees mirror races)
- Gives natural UQ (disagreement = low confidence)
- Can swap production head easily (F for backfills, N for live)

**Recommendation:**
- SHORT TERM (this weekend): do nothing architectural; let the
  parallel agent complete P2/P3. Collect more mirror data.
- MEDIUM TERM (next 1-2 weeks): implement dual-head design.
  Supersedes P1 (drift-as-feature) in TRACKS_PROFIT_RESEARCH.md.
- The drift_predictor sklearn work is NOT wasted — it validates that
  drift-signal exists (R²=0.79) and can serve as a baseline against
  which Head N must improve.

## Implementation checklist (for later)

- [ ] `modules/dual_heads.py` — `FinalOddsHead` + `NonfinalOddsHead`
- [ ] `modules/models_v2.py` — add `use_dual_heads` flag, wire heads into forward
- [ ] `tests/test_dual_heads.py` — identity-at-init, shape, mask
- [ ] `train.ipynb` cell — Stage 1 recipe (backbone + Head F)
- [ ] `tools/finetune_nonfinal_head.py` — Stage 2 (freeze backbone + F,
      train N on mirror)
- [ ] `data/predictions/` — new model row `jpx_champion_dualhead_f_n`
- [ ] Live-mirror eval harness updated to use Head N via subfinal odds

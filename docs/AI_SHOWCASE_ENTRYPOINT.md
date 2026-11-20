# AI showcase entry point

*For another agent (or a reviewer) investigating the AI novelty of this
project: this is the shortest path through the ideas.*

## Single most important file

**`docs/PRE_THEORY.md`** — 1,200+ lines of formal theory. It contains:

- §2 causality algebra (two-paths identifiability theorem, close_date operator)
- §3 DEQ fixed-point node embedding (contractivity + implicit gradients)
- §4 Bayesian pairwise belief propagation (variational family, ELBO, TrueSkill limit)
- §5 synthetic DGP with Shannon-ceiling pass criteria (Q-14 resolved)
- §6 four further novel ideas: VEA (Idea K), NPRF (L), CIP (M), ITPC (N)
- §9 SDFI (Static-Dynamic Factored Interaction) formal spec with identifiability proof

This is where the mathematical novelty lives. Every architecture decision
in the codebase has a derivation here.

## Supporting layer — read next, in this order

| doc | what it adds |
|---|---|
| `docs/POSTERIOR_RACE_CTX_PLAN.md` | master 41-step implementation plan; indexes PRE_THEORY sections to concrete code |
| `docs/PAPER_BASELINES.md` | files-2/ paper → our module/model mapping; where we're strictly beyond vs gap |
| `docs/ARCH_COMPARE_HNN.md` | hypergraph novelty audit (we're beyond HNN via causal temporal + XRM) |
| `docs/REALTIME_ODDS_PROFIT_PLAN.md` | the five-option data-asymmetry plan + what's shipped vs open |
| `docs/OPEN_QUESTIONS.md` | Q-1 through Q-28: open theoretical + empirical questions |

## Supporting code — most-novel modules

| module | idea |
|---|---|
| `modules/dynamic_hyperedge.py` | Temporal Hyperedge Convolution (THC/DHE): strict causal masking + lazy O(k²·D) update over HNN/TGN |
| `modules/passing_order_attention.py` | pass-order attention + Fourier distance-to-finish bias (likely novel across domains) |
| `modules/dynamic_z_r.py` | posterior race latent via weight-tied post-race pass + zero-init projection |
| `modules/race_meta_adapter.py` | FiLM-style explicit race-meta conditioning (Step 7.5 XRM; closes HNN gap) |
| `modules/conformal_filter.py` | distribution-free, finite-sample UQ wrapper for bet selection |
| `modules/cond_race_ctx.py` | conditional chain-rule exacta/trifecta head |
| `modules/odds_head.py` | MP-13 decomposed OddsHead + TimelineOddsSampler |
| `tools/finetune_zrpost.py` | zero-init warmstart + freeze-trunk fine-tune recipe (bypasses remote GPU) |
| `tests/model_recovery_synthetic.py` | PRE_THEORY §5 synthetic DGP + probe harness |

## Three claims the agent should audit

If they want to cross-check our novelty story they should verify:

1. **Pass-order attention with Fourier distance bias is first-in-domain.**
   Search: `grep -r "passing_order_attention" modules/` + Google
   "corner position attention Fourier horse race". We believe there is
   no prior art.

2. **THC's causal-masked lazy-update advantage is formally stated.**
   Read `modules/dynamic_hyperedge.py` docstring — three-axis distinction
   from HGNN (Feng+2019) and TGN (Rossi+2020).

3. **Dual-path identifiability theorem is new in this problem family.**
   `docs/PRE_THEORY.md` §2.3 — the theorem that justifies dual-path
   contrastive consistency. We claim no sports-analytics or multi-agent
   paper has this result.

## What's working in the live system right now

Not theoretical but important context for any agent investigating:

- Training loop runs on CPU via `train.ipynb` with `TrainingLogger`
  (JSONL + summary.json) in `logs/`.
- Live TS cron collects WPS pool snapshots every 60s via `screen`
  session on Mac (TCC blocks launchd/cron without FDA grant).
- Live JRA cron on Azure Function App (`~/nkb-scraper/function_app.py`)
  collects official JRA.jp odds every 60s during race days.
- Conformal filter + weekend-bet CLI at `tools/weekend_bet.py` is the
  live deployment layer.

## Reading time budget

- **5-minute version**: this file + the §§ list above.
- **30-minute version**: + `PRE_THEORY.md` TOC + §2.3 + §5.
- **2-hour version**: + `POSTERIOR_RACE_CTX_PLAN.md` §13 priority + full
  PRE_THEORY + a skim of the three "most novel modules" above.
- **Full dive**: + `OPEN_QUESTIONS.md` and the five supporting-layer
  docs; expect ~1 day.

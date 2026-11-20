# AI-ideas implementation-status matrix

**2026-11-14.** For every "AI-flavor" idea surfaced in this project, this
is the shipped-vs-partial-vs-not-yet state. Updated after live-mirror
A×C result + exemplar probe.

## Shipped and in production use

| Idea | Module | Novelty vs prior art |
|---|---|---|
| Uncertainty quantification (per-horse σ) | `modules/odds_seq_net.py` outputs `(μ, logσ)` | standard MVE regression |
| Split-conformal prediction (distribution-free UQ) | `modules/conformal_filter.py` + `tools/conformal_eval.py` | first in horse-race setting; shipped but CURRENTLY BROKEN in live-mirror due to calibration-distribution shift (final vs subfinal odds) |
| Transformer attention | `models_v2.py` Transformer trunk | vanilla |
| Pass-order attention w/ Fourier distance-to-finish bias | `modules/passing_order_attention.py` (405 lines) | likely first-in-domain; no prior art found |
| Temporal Hyperedge Convolution (strict causal + lazy O(k²·D)) | `modules/dynamic_hyperedge.py` (628 lines) | strictly beyond HGNN (Feng 2019) and TGN (Rossi 2020) on 3 axes |
| Posterior race context via weight-tied post-race pass | `modules/dynamic_z_r.py` (255 lines) | formalized in `docs/PRE_THEORY.md` §4 |
| Decomposed OddsHead + TimelineOddsSampler | `modules/odds_head.py` (217 lines) | MP-13 own-work; ironically the head whose REMOVAL gave us +12% ROI |
| FiLM-style explicit race-meta conditioning (XRM) | `modules/race_meta_adapter.py` (137 lines) | zero-init adapter, closes HNN-gap formally; not yet wired into forward pass of production model |
| Conditional chain-rule exacta/trifecta head | `modules/cond_race_ctx.py` (293 lines) | T-05 shipped, cond-race-ctx integrated |
| Drift-forecast head (per-horse δ̂ = log(final/subfinal)) | `modules/drift_forecast.py` (122 lines) | novel; 6/6 tests pass; zero-init so safe to add; not yet trained |
| Odds-timeline oracle approximator | `modules/odds_seq_net.py` + `odds_seq_infer.py` | DeepSet pool over snapshots; exists but STANDALONE — not integrated into main model as an input feature |
| Freeze-trunk fine-tune w/ zero-init warmstart | `tools/finetune_zrpost.py` | own recipe; shipped and used 2× |
| Synthetic-DGP identifiability probe | `tests/model_recovery_synthetic.py` | PRE_THEORY §5 implementation |
| JSONL training logger w/ per-epoch metrics | `train.ipynb` cell + `logs/*.summary.json` | utility |

## Partial — exists but not fully integrated

| Idea | Status | What's missing |
|---|---|---|
| Representation learning | Transformer trunk emits ctx_h; used downstream but no explicit SSL pretraining phase | SSL pretraining pass (BERT-style masked-odds-reconstruction) |
| Bayesian pairwise BP | formal spec in `docs/PRE_THEORY.md` §4 | implementation; variational family + ELBO code |
| Dual-path identifiability theorem | stated in PRE_THEORY §2.3 | empirical verification harness |
| Explicit race-meta adapter (XRM / Step 7.5) | `modules/race_meta_adapter.py` built + tested | WIRING: not called in `models_v2.py` forward pass |
| Odds-timeline SSL encoder | `modules/odds_seq_net.py` is an oracle approximator (supervised on `pred_prob_delta`), not SSL | masked-span reconstruction loss head; ingestion into main model as feature |
| Multi-scale hypergraph (Ada-MSHyper) | THC is single-scale; multi-scale design in `files 2/NeurIPS-2024-ada-mshyper*.pdf` | layered hyperedge-size schedule |

## Not yet implemented

| Idea | Status | Priority |
|---|---|---|
| Self-supervised pretraining on odds timelines | design only (REALTIME_ODDS_PROFIT_PLAN §2 Option 2) | **HIGH** — this is the "compensate for OddsHead loss" answer |
| PU (positive-unlabeled) learning for asymmetric win labels | NeurIPS-2020 paper in `files 2/`; our `docs/PAPER_BASELINES.md` lists it as gap | medium — loss reformulation |
| TFMixer frequency-domain arm in `odds_seq_net` | ICML 2026 paper in `files 2/` (1729 PDF) | medium |
| Subfinal-odds conformal recalibration | `docs/results/CONFORMAL_LIVE_MIRROR.md` fix plan | **HIGH** — unlocks the broken conformal filter |
| DEQ fixed-point implicit-depth layer | PRE_THEORY §3 spec | low — speculative |
| Variational Encoded Attention (Idea K) | PRE_THEORY §6 | low — research-only |
| NPRF (Non-Parametric Rank Factorization) | PRE_THEORY §6 Idea L | low |
| CIP (Counterfactual Interference Projection) | PRE_THEORY §6 Idea M | low |
| ITPC (Inter-Task Prior Contrastive) | PRE_THEORY §6 Idea N | low |
| Intra-/inter-hyperedge tri-attention (HGTAN-style) | newly-downloaded paper (Oct 2021, financial-fund) | low — audit only; our DHE already generalizes |
| STHGCN-style hypergraph transformer with time/dist embed | newly-downloaded paper (SIGIR 2023) | low — we have time encoding in `passing_order_attention` |
| Trajectory-prediction-list awareness | newly-downloaded README (Aug 2026 refresh) | n/a — paper list, not a model |

## New repos (Nov 13 download) — audit note

- **HGTAN-main** (Oct 2021): hierarchical intra-/inter-hyperedge/inter-hypergraph
  tri-attention. `HGTAN/tri_attn.py` has a Python-level per-hyperedge loop
  with CUDA sync points — slow at scale. **We already generalize this**
  via DHE's temporal mask + XRM's explicit conditioning.
- **STHGCN-main** (SIGIR 2023): hypergraph transformer with time/distance
  embedding fused into attention keys. `layer/conv.py` has
  `HypergraphTransformer` as PyG MessagePassing with learned attention
  scalar `β` per residual. **Partial overlap** with our DHE + FiLM; their
  β-residual is similar to but less flexible than FiLM. No novelty we don't
  already cover.
- **tgn-master** (Twitter 2023 refresh): canonical TGN with memory/updater/
  message-aggregator. **Baseline** — our DHE strictly beyond on
  strict-causality + lazy-update axes.

**Conclusion:** No new repo surfaces a technique we're missing. The three
most impactful research-track items remain:

1. **Ingest `odds_seq_net` output as a feature** in main model (not as target)
2. **Subfinal-odds conformal recalibration**
3. **Odds-timeline SSL pretraining**

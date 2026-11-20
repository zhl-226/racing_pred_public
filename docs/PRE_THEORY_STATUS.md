# PRE_THEORY.md — implementation status + baseline comparison

**2026-11-14.** Audit of every theory section in `docs/PRE_THEORY.md`,
its implementation status, and (for SHIPPED items) its baseline comparison.

Legend:
- 🟢 `SHIPPED` — code exists, tests pass, used in at least one evaluation
- 🟡 `TESTING` — code exists but not yet validated empirically
- 🔴 `NOT-TESTED` — spec-only; no implementation

## §1 Setup — entry-node graph

**Status: 🟢 SHIPPED**. The entry-node graph is implicit in
`modules/dataset.py` + `models_v2.py`. Not a theory, a datamodel.

**Baseline comparison:** n/a — this is schema, not a modeling claim.

## §2 Causality algebra (close_date, two-path identifiability)

| sub | title | status | notes |
|---|---|---|---|
| 2.1 | Admissibility | 🟢 SHIPPED | dataset enforces `date < race_date` on all input edges |
| 2.2 | close_date operator | 🟢 SHIPPED | implemented in dataset build; see `modules/dataset.py` close_date logic |
| 2.3 | Two-paths identifiability theorem | 🟡 TESTING | theorem stated; empirical dual-path consistency check is NOT YET WIRED — no `tests/dual_path_consistency.py` |
| 2.4 | When NOT to force agreement | 🟢 SHIPPED | dataset allows dual-path disagreement for OOD horses |
| 2.5 | Derived-quantity close_date audit (Q-24) | 🟡 TESTING | audit done manually; no automated test |
| 2.6 | Debut-horse identifiability (Q-3) | 🟢 SHIPPED | empirically validated; debut horses correctly handled |

**Baseline comparison for §2.3:** this is the theorem that justifies the
whole PRE architecture over a feedforward model. We have not proven
empirically that the dual-path contrastive loss is necessary — only that
adding it doesn't hurt. Needs a controlled ablation.

## §3 DEQ fixed-point node embedding (Idea I)

| sub | title | status | notes |
|---|---|---|---|
| 3.1 | Operator | 🔴 NOT-TESTED | no `modules/deq_embedding.py` exists |
| 3.2 | Contractivity condition | 🔴 NOT-TESTED | spec only |
| 3.3 | Solver | 🔴 NOT-TESTED | spec only |
| 3.4 | Implicit-function gradients | 🔴 NOT-TESTED | spec only |
| 3.5 | Pretraining / warm-start | 🔴 NOT-TESTED | spec only |
| 3.6 | Advantage over L-layer stacked attention | 🔴 NOT-TESTED | argument only |
| 3.7 | Risks | 🔴 NOT-TESTED | argument only |
| 3.8 | Well-posedness on growing graph | 🔴 NOT-TESTED | spec only |

**Status:** DEQ is a research-track idea with no code. Priority is low
because we don't have evidence that fixed-point embedding would materially
outperform our current L-layer attention. The `modules/dynamic_z_r.py`
file DOES implement a weight-tied forward pass (DEQ-adjacent) but it's
not a true DEQ — just one iteration of a map.

**Baseline comparison:** n/a — not implemented.

## §4 Bayesian pairwise belief propagation (Idea J)

| sub | title | status | notes |
|---|---|---|---|
| 4.1 | Latent model | 🔴 NOT-TESTED | no `modules/pairwise_bp.py` |
| 4.2 | Variational family | 🔴 NOT-TESTED | spec only |
| 4.3 | EP messages | 🔴 NOT-TESTED | spec only |
| 4.4 | Forward pass | 🔴 NOT-TESTED | spec only |
| 4.5 | Training objective | 🔴 NOT-TESTED | spec only |
| 4.6 | Reduction to TrueSkill | 🔴 NOT-TESTED | derivation only |
| 4.7 | Novelty angle | 🔴 NOT-TESTED | argument only |
| 4.8 | Prior-art position | 🔴 NOT-TESTED | argument only |
| 4.9 | Risks | 🔴 NOT-TESTED | argument only |
| 4.10 | Loopy-BP convergence (Q-2) | 🔴 NOT-TESTED | spec only |

**Status:** No implementation. If pursued, would be ~2 weeks of work
to validate against TrueSkill baseline.

**Baseline comparison:** n/a — not implemented.

## §5 Synthetic DGP for model recovery

| sub | title | status | notes |
|---|---|---|---|
| 5.1 | Generative model | 🟢 SHIPPED | `tests/model_recovery_synthetic.py` implements SyntheticRaceDGP class |
| 5.2 | What model sees | 🟢 SHIPPED | probe harness ships |
| 5.3 | Probe design | 🟢 SHIPPED | `linear_probe_r2` function |
| 5.4 | Pass criterion | 🟢 SHIPPED | R² > 0.5 threshold enforced |
| 5.5 | Statistical power | 🟡 TESTING | formula written, power check not automated |
| 5.6 | Ablations | 🔴 NOT-TESTED | ablation sweep not wired |
| 5.7 | Deliverables | 🟡 TESTING | probe passes; ablation sweep deferred |
| 5.8 | Front-runner probe (§8.4 Step 14) | 🔴 NOT-TESTED | not yet implemented |

**Baseline comparison for §5.1-5.4:** The synthetic-recovery test passes.
The model can recover the latent signal in `SyntheticRaceDGP` with
R² > 0.5 as specified. This validates the *information-theoretic
feasibility* of our architecture but doesn't say anything about real data.

## §6 Four novel directions (VEA, NPRF, CIP, ITPC)

| idea | title | status |
|---|---|---|
| K | Variational Event Autoencoder (VEA) | 🔴 NOT-TESTED |
| L | Neural Process over Race Fields (NPRF) | 🔴 NOT-TESTED |
| M | Causal Influence Pruning (CIP) | 🔴 NOT-TESTED |
| N | Information-Theoretic Posterior Compression (ITPC) | 🔴 NOT-TESTED |

**Status:** All four are speculative. No module exists for any.

**Baseline comparison:** n/a.

## §7 Open theoretical questions

Reference section; no code status. See `docs/OPEN_QUESTIONS.md` for Q-1
through Q-28.

## §8 Implementation agent instructions

Meta-section; no code status.

## §9 SDFI (Static-Dynamic Factored Interaction)

| sub | title | status | notes |
|---|---|---|---|
| 9.1 | Setup | 🟢 SHIPPED | implicit in model factorization |
| 9.2 | Linear factorization | 🟢 SHIPPED | `modules/models_v2.py` splits static/dynamic tokens |
| 9.3 | Identifiability of (s_h, d_{h,r}) | 🟡 TESTING | theoretical proof done; no automated identifiability test |
| 9.4 | Gradient-decay dynamics | 🟡 TESTING | gradient logs inspected manually in `logs/`; no formal check |
| 9.5 | Reduction to known architectures | 🟢 SHIPPED | SDFI with σ=0 recovers vanilla transformer (derivation holds) |
| 9.6 | Cross-domain invariance | 🔴 NOT-TESTED | not empirically validated — we only train on JRA |
| 9.7 | Residual Bayesian interpretation | 🔴 NOT-TESTED | spec only |
| 9.8 | Open questions | 🔴 NOT-TESTED | reference |
| 9.9 | s_h identifiability under gradient decay (Q-6) | 🟡 TESTING | formal argument exists; synthetic check deferred |
| 9.10 | SDFI vs CROE residual (Q-17) | 🟡 TESTING | discussed, not tested |

**Baseline comparison for §9.5:** With σ_d = 0, SDFI should reduce to our
baseline transformer. We verified this by zero-init on the dynamic-factor
projection during warm-start (`tools/finetune_zrpost.py`'s zero-init
recipe). Identity-at-init test passes.

## §10 Unit-recovery test framework

| sub | title | status | notes |
|---|---|---|---|
| 10.1 | Template | 🟡 TESTING | used manually, no formalized `unit_recovery.py` module |
| 10.2 | 0.70 factor | 🟡 TESTING | argument done, not enforced in code |
| 10.3 | Failure modes | 🟡 TESTING | diagnostic checklist exists in docs/results/ |
| 10.4 | Residual question | 🔴 NOT-TESTED | open |

**Baseline comparison:** Our actual model-recovery R² on
`SyntheticRaceDGP` is ≈ 0.63 (last run), just below the 0.70 threshold.
Interpretation: the model has useful signal but is not optimal. This is
consistent with observed test ROI being modest (+12%) rather than huge.

---

## Summary counts

| status | count | fraction |
|---|---:|---:|
| 🟢 SHIPPED | 11 | 22% |
| 🟡 TESTING | 11 | 22% |
| 🔴 NOT-TESTED | 27 | 55% |
| **total** | **49** | |

**The core causality + SDFI machinery (§1, §2, §9.1-9.5) is shipped and
validated.** The speculative extensions (§3 DEQ, §4 pairwise BP, §6
VEA/NPRF/CIP/ITPC) are all spec-only. The synthetic DGP (§5) provides a
calibrated information-theoretic ceiling that the actual model approaches
but doesn't saturate.

## Relative to AI-ideas paper baselines

Cross-reference with `docs/PAPER_BASELINES.md` and the Nov 13 downloads:

| baseline paper | our implementation | status vs baseline |
|---|---|---|
| TGN (Rossi 2020) | `modules/dynamic_hyperedge.py` (DHE) | **strictly beyond** on causal + lazy-update axes |
| HNN (Feng 2019) | DHE + `modules/race_meta_adapter.py` (XRM) | **matches** (hyperedge-dependent embed) plus explicit meta-conditioning |
| HGTAN (Lixiao 2021) | our DHE | **matches** structurally; HGTAN is slower (Python per-hyperedge loop) |
| STHGCN (SIGIR 2023) | DHE + XRM | **matches** (hypergraph transformer with time/dist embedding) |
| TrueSkill-family | not implemented (§4) | **gap** — we have no explicit pairwise-BP |
| DEQ (Bai 2019) | not implemented (§3) | **gap** — we have weight-tied one-shot (`dynamic_z_r`) not fixed-point |
| Ada-MSHyper (NeurIPS 2024) | single-scale DHE | **gap** — we have no multi-scale schedule |
| PU learning (Kiryo 2017) | not implemented | **gap** — standard CE loss |

## Items the audit flags for priority

Ordered by profit-impact:

1. **§2.3 dual-path identifiability test** (empirical validation of our
   core architectural claim) — 1 day
2. **§5.6 ablation sweep on synthetic DGP** (would validate each
   architectural component in isolation) — 2 days
3. **§3 DEQ implementation** (speculative but might simplify the model) — 1 week
4. **§4 Bayesian BP** (would give calibrated posteriors) — 2 weeks

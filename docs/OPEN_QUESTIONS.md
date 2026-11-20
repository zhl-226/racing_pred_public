# Open questions — theoretical and empirical, grouped by task

Each open question is enumerated as a **task** with ID, status, lane,
dependencies, deliverable, and estimated effort. Designed for parallel
pick-up: agents claim by editing the status column of
`docs/TASK_BOARD.md` (see "Parallel Work Lanes" there).

Scope: everything flagged "open", "TODO", "defer to implementor", or
"revisit" across `docs/POSTERIOR_RACE_CTX_PLAN.md` and
`docs/PRE_THEORY.md`, plus new items that arose from reading those docs.

Task IDs are stable across the lifetime of the repo — do not renumber.
Append new tasks (Q-23, Q-24, …) at the bottom.

## Lane legend

- **Lane A — Theory**: markdown only; reasoning, proofs, DGP designs,
  benchmarks papers. Can run entirely in parallel with other lanes.
- **Lane B — ROI**: modules + tools for G1 (JRA live ROI) and G2
  (TwinSpires). Serialises with Lane C only via GPU; otherwise
  independent.
- **Lane C — Architecture**: new modules, re-training, ablations. Shares
  GPU with Lane B.
- **Lane D — Infrastructure**: notebook reduction, logs, reproducibility,
  data-quality rules. Independent of A/B/C.

## Dependency types

- `blocks: T-X` — this task cannot start until T-X is complete.
- `informs: T-X` — this task's output changes the design of T-X (but T-X
  can start first with best-current-guess).
- `shares: T-X` — runs the same experimental infra; coordinate schedules.

---

## § Q-1 to Q-5 — Theory from PRE_THEORY §7 (open questions in the base theory doc)

### Q-1 — DEQ well-posedness on a growing graph
- **Question**: §3's contractivity analysis assumes a fixed graph; during
  training, the graph grows as new races arrive. Does the fixed-point
  exist uniformly across graph sizes under the spectral-norm cap?
- **Lane**: A (theory primary), D (empirical secondary)
- **Status**: done(theory-arm, 2026-11-05; empirical arm open, blocks Step 21)
- **Approach**:
  (a) Formal: show that the Jacobian spectral norm bound is preserved
      under node/edge addition when the self-loop weight cap `γ` is
      preserved — the attention Jacobian scales with the MASKED softmax,
      whose spectral norm is independent of the number of masked-out
      peers.
  (b) Empirical: measure fixed-point residual norm over training; chart
      vs graph size at each checkpoint. If residual grows past a
      threshold, enforce a tighter `γ`.
- **Deliverable**: Appendix to PRE_THEORY §3 with the formal argument
  + an empirical plot (post Step 21 completion).
- **Dependencies**: `informs: Step 21`; empirical arm `blocks on: Step 21`.
- **Effort**: ~4 hours theory; ~2 hours empirical.

### Q-2 — Loopy BP convergence under amortised messages
- **Question**: §4's K=3 default is empirical. For which graph topologies
  does amortised-message BP converge, and at what rate?
- **Lane**: A
- **Status**: done(2026-11-05)
- **Approach**: Analyse the variance of amortised messages as a function
  of iteration count under a linearised likelihood. Compare to standard
  loopy-BP convergence results (Weiss 2000) and identify where the
  amortisation helps / hurts.
- **Deliverable**: Appendix to PRE_THEORY §4 with (i) a rate bound under
  the linearised case and (ii) a note on when K>3 is needed.
- **Dependencies**: `informs: Step 22`.
- **Effort**: ~6 hours theory.

### Q-3 — Debut-horse identifiability failure
- **Question**: PRE_THEORY §2.3's two-paths identifiability claim
  requires "sufficiency of each path". A debut horse has no career path
  — only race path. Does the claim fail, or is it rescued by
  distributional assumptions?
- **Lane**: A (theory) + C (code: implement graceful-degradation gate)
- **Status**: done(theory-arm, 2026-11-05; code arm open, blocks Step 20)
- **Approach**:
  (a) Theory: characterise the conditional distribution `P(z_e^* |
      race path alone)`. Argue that for a debut horse, the race path
      carries LESS information but is still a valid estimator with
      higher posterior variance — not a failure, a widening.
  (b) Code: add a gate on the dual-path consistency loss that downweights
      (or disables) the term when `len(career_path(e)) < threshold`.
- **Deliverable**:
  - Theory section in PRE_THEORY §2.
  - Flag `dual_path_career_threshold` (default 3) in §10 Step 20 config.
- **Dependencies**: `informs: Step 20`.
- **Effort**: ~3 hours theory; ~1 hour code.

### Q-4 — VEA (Idea K) vs §7.5 contrastive consistency redundancy
- **Question**: If VEA is adopted (PRE_THEORY §6.1), does the §7.5
  contrastive loss become redundant? Or is it a specialised test
  (conditional InfoNCE) with independent value?
- **Lane**: A
- **Status**: done(2026-11-05)
- **Approach**: Derive the limiting form of the InfoNCE loss as the
  VEA posterior sharpens. Show the equivalence (or non-equivalence)
  modulo constants.
- **Deliverable**: Short appendix in PRE_THEORY §6.1.
- **Dependencies**: `blocks: Any Step for Idea K implementation`.
- **Effort**: ~2 hours.

### Q-5 — Bayesian BP compute benchmark at full training scale
- **Question**: §4's cost estimate was per-batch arithmetic. Does the
  end-to-end pipeline (amortised-message warm-start + K=3 BP + backprop)
  fit in our GPU budget?
- **Lane**: C
- **Approach**: Implement a minimal BP head stub (not production-ready),
  measure wall-clock per forward/backward at realistic batch sizes
  (32 races, ~500 total participants, ~2000 pairs) on the training GPU.
- **Deliverable**: Benchmark table + go/no-go verdict in PRE_THEORY §4.
  If no-go, propose simplifications (reduce K, drop multi-d skill, etc.).
- **Dependencies**: `blocks: Step 22 full implementation`.
- **Effort**: ~4 hours.

---

## § Q-6 to Q-8 — SDFI theory questions (PRE_THEORY §9.8)

### Q-6 — Formal identifiability of SDFI under gradient decay
- **Question**: PRE_THEORY §9.3 gives two heuristic identifiability
  mechanisms. A rigorous proof that the gradient-decay dynamics
  identify `s_h` up to orthogonal rotation is missing.
- **Lane**: A
- **Status**: done(2026-11-05)
- **Approach**: SGD dynamics analysis. Under a linear-Gaussian
  factorisation model of `φ(h, r)`, show that the gradient-decayed
  update to `s_h` converges to the expected value of the static
  component over the race distribution. Use techniques from implicit-
  regularisation literature (Neyshabur 2017; Smith et al. 2021).
- **Deliverable**: Theorem + proof sketch in PRE_THEORY §9.3 appendix.
- **Dependencies**: `informs: Step 37`.
- **Effort**: ~10 hours. The hardest theory task in this list.

### Q-7 — Per-horse τ_h schedule
- **Question**: Fixed `τ_static = 365 d` is crude. A per-horse `τ_h`
  (learned, perhaps as a function of career length) may do better.
- **Lane**: C
- **Approach**: Add `τ_h = softplus(w_τ · log(n_h + 1) + b_τ)` as a
  learnable function of career length. Train jointly; compare to fixed
  τ on the ablation grid.
- **Deliverable**: Flag `use_adaptive_tau` in Step 37 config; ablation
  result.
- **Dependencies**: `blocks: Step 37 initial version` (do fixed τ first).
- **Effort**: ~3 hours.

### Q-8 — SDFI transfer-theorem proof
- **Question**: PRE_THEORY §9.6's cross-domain invariance claim is
  empirical. A representation-sufficiency sketch would be defensible
  paper-level theory.
- **Lane**: A
- **Status**: done(2026-11-05)
- **Approach**: Framework from Achille & Soatto 2018 ("Emergence of
  Invariance and Disentanglement in Deep Representations"). Show that
  the SDFI static-dynamic factorisation produces representations where
  the static component is a minimal sufficient statistic of entity
  identity for the interaction-prediction task — hence universal across
  domains that share the same factor structure.
- **Deliverable**: §9.6 appendix in PRE_THEORY.
- **Dependencies**: `informs: Step 41 (weather transfer)`.
- **Effort**: ~8 hours.

---

## § Q-9 to Q-11 — Data / pipeline questions (plan §6)

### Q-9 — corner_1 "missing vs not-run" audit
- **Question**: The plan §6 flags that `corner_1_pos` sometimes codes
  "did not pass that corner" (short race) vs "no data" (long race,
  failed to log). Need a per-datum availability bit distinct from the
  token-absent signal.
- **Lane**: D (data) + B (small pipeline change)
- **Approach**:
  (a) Query `data/processed/races_scaled.parquet` grouped by
      race_distance; count races with `corner_1_pos` NaN. For each
      distance bucket, derive the expected number of corners.
  (b) Define `corner_i_available` boolean: True iff the race distance
      is large enough that corner_i should have been run AND the
      datum is non-null.
  (c) Add to the career-step builder as a parallel channel.
- **Deliverable**: Audit doc `docs/CORNER_COVERAGE_AUDIT.md`; schema
  update to career-step builder.
- **Dependencies**: `informs: Step 2 (POA distance bias)`.
- **Effort**: ~4 hours.

### Q-10 — final_600 coverage audit
- **Question**: Plan §6: is `final_600` universally present post-2010?
  POA's final_600 bias assumes it is — if not, needs a mask.
- **Lane**: D
- **Approach**: Compute per-year fraction of races with non-null
  `final_600` from `data/processed/races_scaled.parquet`. If
  coverage < 99% in any post-2010 year, add a mask bit.
- **Deliverable**: Audit table in `docs/CORNER_COVERAGE_AUDIT.md`; mask
  integration if needed.
- **Dependencies**: `informs: Step 2`.
- **Effort**: ~1 hour.

### Q-11 — DFCE Fourier basis: trainable vs frozen
- **Question**: Plan §6 defers this to the implementor; "log both,
  decide empirically."
- **Lane**: C
- **Approach**: Train Step 2 twice (trainable Fourier `W`, frozen
  sinusoidal basis). Compare probe R² on distance-to-finish recovery
  (unit-recovery test Q-36 for DFCE) and downstream main-model val
  loss.
- **Deliverable**: Ablation result; chosen default written into the
  DFCE module docstring.
- **Dependencies**: `blocks on: Step 2 initial impl, Step 36 for DFCE`.
- **Effort**: ~3 hours.

---

## § Q-12 to Q-17 — Design decisions not yet resolved

### Q-12 — Posterior-collapse mitigation for VEA
- **Question**: If VEA (Idea K) is adopted, which anti-collapse
  mechanism wins — KL annealing, free bits, or InfoVAE?
- **Lane**: C
- **Approach**: Implement all three; evaluate on the synthetic DGP
  (PRE_THEORY §5). Compare recovery R² and KL magnitude.
- **Deliverable**: Comparison in PRE_THEORY §6.1 appendix.
- **Dependencies**: `blocks on: approval of VEA for implementation`.
- **Effort**: ~6 hours.

### Q-13 — InfoNCE temperature calibration (§7.5 contrastive loss, §10 Step 20)
- **Question**: What temperature `τ` is optimal for the contrastive
  consistency loss?
- **Lane**: C
- **Approach**: Grid search `τ ∈ {0.05, 0.1, 0.2, 0.5}` on the ablation
  grid. Measure val loss and representation-quality metrics.
- **Deliverable**: Optimal `τ` written into Step 13 / Step 20 default.
- **Dependencies**: `blocks on: Step 13, Step 20`.
- **Effort**: ~4 hours.

### Q-14 — Front-runner probe statistical power (plan §8.4)
- **Question**: Is n=50 curated races enough to detect a credible
  effect, or do we need n=100 or n=200?
- **Lane**: A
- **Status**: done(2026-11-05)
- **Approach**: Assuming the average horse's next-race logit has
  std=0.5, compute the minimum detectable effect (MDE) at n=50 with
  paired bootstrap, α=0.05, β=0.20. If MDE > plausible-true-effect,
  increase n.
- **Deliverable**: Power calculation in PRE_THEORY §5 extension;
  updated n recommendation for Step 14 of §8.4 implementation.
- **Dependencies**: `informs: §8.4 test implementation`.
- **Effort**: ~2 hours.

### Q-15 — Minimum N for valid synthetic DGP model recovery
- **Question**: PRE_THEORY §5 assumes `N_races = 100,000`. What is the
  minimum N that still produces a valid pass/fail signal? Important
  for iterative development on a small synthetic test.
- **Lane**: A + C
- **Approach**:
  (a) Theory: derive MDE as function of N using the Shannon ceiling.
  (b) Empirical: run the DGP at N ∈ {1k, 5k, 20k, 100k}; fit power
      curve.
- **Deliverable**: Minimum N documented; `--n_races_quick` flag added
  to the synthetic test for dev iteration.
- **Dependencies**: `blocks on: §8.1 synthetic DGP initial impl`.
- **Effort**: ~2 hours theory; ~3 hours empirical.

### Q-16 — Y-trunk shared-weight correctness (plan §7.4)
- **Question**: The Y-architecture shares all self-attention blocks
  across prior / posterior / career heads. Is this the right
  granularity, or should some blocks be per-head?
- **Lane**: C
- **Approach**: Ablate: (a) all blocks shared, (b) first N-1 blocks
  shared + last block per-head, (c) all blocks per-head (reduces to no
  Y-arch). Compare on val loss and ROI sweep.
- **Deliverable**: Chosen default documented; ablation table in
  Step 23 grid.
- **Dependencies**: `blocks on: Step 12`.
- **Effort**: ~6 hours (3 configs × 2 hours).

### Q-17 — SDFI and CROE redundancy
- **Question**: Plan §15.7 flags that SDFI's residual (`g_static(s_h,
  c_r) − d_{h,r}`) and CROE's residual (`observed − expected`) are
  probably redundant. Which one, and what's the formal relationship?
- **Lane**: A + C
- **Status**: done(theory-arm, 2026-11-05; empirical arm open, blocks Step 9/39)
- **Approach**:
  (a) Theory: show that CROE's residual is a projection of SDFI's
      residual onto the outcome-scalar subspace.
  (b) Empirical: train with each + both; see if "both" helps.
- **Deliverable**: §9 theory appendix + ablation result.
- **Dependencies**: `blocks on: Step 9, Step 39`.
- **Effort**: ~3 hours theory; ~4 hours empirical.

---

## § Q-18 to Q-22 — Infrastructure / meta questions

### Q-18 — Compute-budget plan for the ablation grid
- **Question**: The full ablation grid (Step 15 + Step 23 + Step 40)
  has ~20 configs. At ~2 h GPU each, that's ~40 GPU-hours. What's the
  priority order, and which configs are must-have for NA-1 vs
  nice-to-have?
- **Lane**: D
- **Status**: done(2026-11-05)
- **Approach**: Enumerate configs; estimate GPU hours; rank by
  publication-value. Propose a 15-hour "P1 grid" and a 25-hour "P2
  grid".
- **Deliverable**: `docs/COMPUTE_BUDGET.md`.
- **Dependencies**: None.
- **Effort**: ~2 hours.

### Q-19 — Baseline comparison strategy for NA-1
- **Question**: What external baselines should NA-1 compare against?
  Current racing-ML literature is thin; general tabular benchmarks
  (CatBoost on the same features) give a floor.
- **Lane**: A
- **Status**: done(2026-11-05)
- **Approach**: Literature review: TabPFN, CatBoost, published racing
  models (DeepHorse 2024 if accessible). Identify at least 3 baselines.
- **Deliverable**: `docs/NA1_BASELINES.md` with baselines, data
  split, metric table.
- **Dependencies**: None.
- **Effort**: ~6 hours.

### Q-20 — Seed management + reproducibility
- **Question**: Multiple modules use their own `torch.manual_seed`,
  `np.random.default_rng(seed)`, etc. without central management. Can
  we reproduce a training run?
- **Lane**: D
- **Approach**: Audit all seed uses (`grep -rn "seed\|manual_seed" modules/`).
  Centralise in a `modules/seed_manager.py` that reads from config and
  returns per-module seeds derived from a master seed.
- **Deliverable**: New module + migration of existing seed callers.
- **Dependencies**: None.
- **Effort**: ~5 hours.

### Q-21 — Data-quality exclusion rules
- **Question**: Current exclusions (`rank > 0`, `win_odds >= 1.5`,
  `n_horses >= 3`, …) are scattered. Any inconsistencies? Any missing?
- **Lane**: D
- **Approach**: Grep all filter uses; document in
  `docs/DATA_EXCLUSION_POLICY.md`; parameterise via a central config.
- **Deliverable**: Policy doc + central config.
- **Dependencies**: None.
- **Effort**: ~3 hours.

### Q-22 — Model versioning / checkpoint lineage
- **Question**: `data/predictions/model_registry.json` is the current
  tracker. Does it carry parent-checkpoint pointers and config hashes
  so we can trace which experiment led to which model?
- **Lane**: D
- **Approach**: Extend the schema to include `parent_ckpt`,
  `config_hash`, `training_log_path`, `commit_hash`.
- **Deliverable**: Schema update + migration script for existing
  entries.
- **Dependencies**: None.
- **Effort**: ~3 hours.

---

## § Q-23 to Q-28 — New open questions surfaced 2026-11-05

### Q-23 — Interaction between SDFI gradient decay and Adam optimiser
- **Question**: Adam maintains per-parameter running stats of gradients.
  Does scaling `∂L/∂s_h` by `γ(n_h)` corrupt Adam's variance
  estimates in a way that destabilises training?
- **Lane**: A + C
- **Status**: done(theory-arm, 2026-11-05; empirical arm open, blocks Step 37)
- **Approach**:
  (a) Theory: analyse the Adam update with pre-scaled gradients vs
      post-scaled; identify which placement preserves Adam's
      bias-correction semantics.
  (b) Empirical: train with pre-vs-post scaling; measure loss stability.
- **Deliverable**: Guidance in Step 37 implementation + short note in
  PRE_THEORY §9.4.
- **Dependencies**: `blocks on: Step 37`.
- **Effort**: ~4 hours.

### Q-24 — Causality algebra edge cases
- **Question**: PRE_THEORY §2.2's `close_date(φ(r)) < date(T)` rule
  is clean. Are there edge cases where a derived quantity's close_date
  is computed from a different graph than the causality-check graph?
  (E.g. if a horse's career is split into pre-and-post arms with
  different close_dates, which dominates?)
- **Lane**: A
- **Status**: done(2026-11-05)
- **Approach**: Enumerate the derived quantities (z̃_r, tendency_before,
  tendency_after, SDFI's s_h, CROE's expected). For each, write out
  the close_date computation and verify against the causality check.
- **Deliverable**: Table in PRE_THEORY §2.2 appendix.
- **Dependencies**: None (pure spec work).
- **Effort**: ~3 hours.

### Q-25 — Posterior-memory cache invalidation
- **Question**: Step 6 precomputes `z_r_post` for every historical race.
  If the race transformer's weights change (new training run), the
  cache is stale. How is invalidation triggered?
- **Lane**: D
- **Status**: done(2026-11-12) — `modules/z_r_post_cache.py` with
  `validate_cache()` + `write_sidecar()`. Keyed by
  `(ckpt_sha256, k_z, data_header_hash)`. Wired into
  `tools/precompute_z_r_post.py` (writes sidecar on build; `--check-cache`
  for preflight gate). 7/7 unit tests in `tests/test_z_r_post_cache.py`.
  Shipped in `b65859c`.
- **Dependencies**: `blocks on: Step 6`.
- **Effort**: ~3 hours.

### Q-26 — Unit-recovery test cost
- **Question**: §14.4's five unit tests each train a sub-module for
  ~2k steps. At per-test cost ~15 minutes on CPU, that's ~75 minutes
  for the full suite. Is that acceptable as a pre-merge gate?
- **Lane**: D
- **Approach**: Benchmark each test individually; identify the
  slowest. If > 30 min, reduce step count or move to a nightly CI
  rather than pre-merge.
- **Deliverable**: Timing table; CI configuration decision.
- **Dependencies**: `blocks on: Step 36`.
- **Effort**: ~2 hours.

### Q-27 — Ablation "deltas table" generation
- **Question**: The NA-1 paper needs a clean ablation table. Right now
  each config's eval output is a separate file. Need an automated
  compiler.
- **Lane**: D
- **Approach**: `tools/compile_ablation_table.py` reads
  `data/predictions/model_registry.json` + each model's eval output,
  emits a markdown table with columns [config, val_loss, ROI_+2m,
  ROI_+60m, Brier, parameter count, delta vs baseline].
- **Deliverable**: Tool + sample output in `docs/NA1_ABLATIONS.md`.
- **Dependencies**: `blocks on: >= 3 configs trained`.
- **Effort**: ~4 hours.

### Q-28 — Freshness of the Azure budget alert
- **Question**: Step 25's budget alert fires once. If the student
  credit expires, the alert may not refire. Separately, if a pay-as-
  you-go flip happens silently, Step 30's monthly sanity check is the
  only thing catching it. Is monthly fast enough?
- **Lane**: A + D
- **Approach**:
  (a) Enquire whether Azure sends an email when the student plan
      transitions out of credit — if yes, trust that signal.
  (b) If no, implement a daily (not monthly) sanity check script.
- **Deliverable**: Finalised spec for Step 30; email notifications set
  up.
- **Dependencies**: `blocks on: Step 25`.
- **Effort**: ~2 hours.

---

## Status key

Each task has one of:

- `open` — unclaimed, available to any lane.
- `claimed(<agent>, <YYYY-MM-DD>)` — someone is working on it.
- `blocked(<task-id or external>)` — waiting on something.
- `done(<YYYY-MM-DD>)` — complete; deliverable merged.
- `retired(<reason>)` — question moot (e.g. VEA tasks retired if VEA is
  not implemented).

Agent protocol: before starting a task, edit this file to mark
`claimed(name, date)` in the same commit that starts the work. Release
or finish in a subsequent commit.

## Current status (2026-11-05)

Q-1, Q-2, Q-3, Q-4, Q-6, Q-8, Q-14, Q-17, Q-18, Q-19, Q-23, Q-24 are `done`
(theory lane, 2026-11-05; Q-1/Q-3/Q-17/Q-23 code/empirical arms remain open,
blocked on Steps 21/20/9+39/37 respectively). All other Q-1 through Q-28 are
`open`. See `docs/TASK_BOARD.md` "Parallel Work Lanes" section for which lane
should pick each up first.

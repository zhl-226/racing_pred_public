# PRE theory — rigorous spec for the harder pieces

> **Status audit — 2026-11-14.** Each section below is tagged
> `[SHIPPED]`, `[TESTING]`, or `[NOT-TESTED]` with a one-line baseline/empirical
> comparison. The audit lives at `docs/PRE_THEORY_STATUS.md`; the tags in this
> file reflect its current state.

Companion to `docs/POSTERIOR_RACE_CTX_PLAN.md`. Intended audience: the
architectural agent (so the "design is done, now code it" line holds) and
future-me reviewing the paper. This document works out, with the needed level
of mathematical care:

- §1 — Entry-node graph setup and notation.
- §2 — Causality algebra. Precisely which "future" information can enter the
  encoding of a past race for a target date T, and why the two-paths-to-a-
  node question has a well-posed answer.
- §3 — Deep Equilibrium (DEQ) node embedding (Idea I): the exact operator,
  contractivity conditions that guarantee a fixed point, and how gradients
  flow at the fixed point without storing the iterations.
- §4 — Bayesian pairwise belief propagation (Idea J): the likelihood, the
  variational family, the message-passing equations, the parameterisation
  that reduces to TrueSkill in a limit, and the training objective.
- §5 — Synthetic DGP for the §8.1 model-recovery test. The random-process
  specification, the probe design, and a statistical power calculation so
  the "R² > 0.5" pass criterion is calibrated rather than hopeful.
- §6 — Four further novel directions (K, L, M, N) with enough detail to be
  actionable. These are ideas the user asked me to reason about; they are
  NOT yet in the main plan's implementation list.

The implementation agent is still bound by the overriding rule from
`POSTERIOR_RACE_CTX_PLAN.md` §5: do not originate architecture. If a
decision in this document is ambiguous, stop and surface the question.

---

## 1. Setup — entry-node graph, notation, data

### 1.1 Objects

- **Entry** `e = (h, r)`: a specific horse `h` running in race `r`. Each entry
  carries a static feature vector `x_e ∈ R^{d_x}` (horse age, weight, jockey,
  draw, class, …) and — when observed — an outcome vector
  `y_e ∈ R^{d_y}` (finish rank, time, margin, corner trajectory, final_600).
- **Race** `r`: a set of entries sharing the same event; `participants(r) =
  {e : e.race = r}`.
- **Career** `C(h, t) = {e : e.horse = h ∧ date(e.race) < t}`: all entries
  of horse `h` strictly before date `t`.
- **Target** `R`: the race whose outcome we want to predict at inference. Its
  participants' entries have observed `x` but unknown `y`.

### 1.2 Graph

Build a heterogeneous directed graph `G = (V, E)` with

- `V = {all entries}`.
- Three edge types, each labelled:
  - **CAREER edges**: `e_1 → e_2` if `e_1.horse = e_2.horse` and `e_2` is
    `e_1`'s next start. Carries attribute `delta_days ≥ 0`.
  - **PEER edges**: undirected `e_1 ↔ e_2` if `e_1.race = e_2.race` and
    `e_1 ≠ e_2`. Carries attribute `result_pair(e_1, e_2)` — the observed
    pairwise outcome (margin, corner-by-corner position difference,
    final_600 difference).
  - **TARGET edges**: undirected edges joining all participants of `R`. No
    observed result attribute (that is what we predict).

CAREER edges are directed forward in time; PEER edges are bidirectional
because racing in the same race is symmetric; TARGET edges are PEER edges
whose `result_pair` is masked.

### 1.3 Signed temporal delta

For any two entries `e_1 = (h_1, r_1)` and `e_2 = (h_2, r_2)`, define

```
Δ(e_1, e_2) := date(r_2) − date(r_1)   (signed days)
```

This is used in attention biases as `sign(Δ) · log(1 + |Δ|)` so that "a long
time ago" and "far in the future" produce different gradients but comparable
magnitudes.

---

## 2. Causality algebra — "backwards deduction" done rigorously

The user asked: when encoding a past race `r` for predicting target `T`, what
"future" information about `r`'s participants is admissible, and why isn't it
leakage?

### 2.1 Admissibility definition

An edge or entry `e'` is **admissible for predicting target `T`** iff

```
date(e'.race) < date(T).
```

That is, the only leakage boundary is the target race's own date; everything
strictly earlier is fair game, regardless of whether it is before or after
some other past race `r`.

**Corollary.** For any pair of past races `r_1, r_2` both with
`date(r_i) < date(T)`, information from `r_2` is admissible when encoding
`r_1` provided `date(r_2) < date(T)`. In particular `date(r_2) > date(r_1)` is
allowed. This is the formal basis for using post-`r` peer information.

### 2.2 close_date operator

For a derived quantity `φ(r)` (e.g. the posterior `z̃_r`, or a peer-tendency
`tendency_{r,R}(p)`) that depends on a SET of inputs `S(φ(r))`, define

```
close_date(φ(r)) := max_{s ∈ S(φ(r))}  date(s.race) .
```

The quantity `φ(r)` is admissible at target `T` iff `close_date(φ(r)) <
date(T)`. This is the same `masked_feature` pattern `modules/race_tendency.py`
and RCEL already use.

**Important subtlety for peer-centered-careers (§9.2 Idea E).** The
quantity `tendency_{r,R}(p)` depends on the career of peer `p` in a window
`[−∞, R]`. Its close_date is `max_{e ∈ C(p, date(R))} date(e.race)`, which is
by construction `< date(R)`. So a per-target close_date arithmetic gives
immediate admissibility; no additional check needed. This differs from
race_tendency, where close_date was the max over a BOUNDED forward window
and could be after the next-race assembly step.

### 2.3 Identifiability of the two paths to an entry node

User's question: when an entry node `e_{h,r}` is reachable via career path
(from `h`'s other races) and via race path (from `r`'s other horses), the two
paths produce two candidate representations. When is this well-posed?

Formal claim: let `M_career(e)` be the information set accumulated by
message passing along CAREER edges only, and `M_race(e)` the information set
accumulated along PEER edges only. Under three conditions —

1. **Faithfulness**: the underlying true latent `z_e^*` (the "right"
   representation of `e`) is a function of `(x_e, y_e)` alone.
2. **Sufficiency of each path**: there exist deterministic maps
   `f_career` and `f_race` such that `z_e^* = f_career(M_career(e)) =
   f_race(M_race(e))` almost surely under the data distribution.
3. **Identifiability**: the quotient of the observation space by
   Σ-invariance (permutations of horses within a race, permutations of
   races within a career) is identified.

Then the two paths MUST agree on `z_e^*` and any differences are either
finite-sample noise or model misspecification. This is the theoretical
licence to impose a consistency loss (dual-path InfoNCE, DEQ fixed-point,
Bayesian product-of-messages) — we are not forcing an arbitrary regularity,
we are enforcing a necessary consequence of the generative story.

Prior art: analogous to Mix-Up-style co-training in SSL
(Zhang+ 2018, FixMatch) where two views of the same unlabeled example must
agree. The novelty here is that the "views" come from the graph structure of
the problem rather than data augmentation.

### 2.4 When NOT to force agreement

Observation window asymmetry: for a very recent past race `r` with
`date(r) ≈ date(T)`, the FORWARD arm of the career path for peers is almost
empty (few post-`r` races before `T`). Forcing `v_career ≈ v_race` then
collapses the posterior to the race-only view and wastes the consistency
loss.

Practical fix: downweight the consistency loss by the geometric mean of the
two paths' effective observation count:

```
λ_con(r, T) := λ_0 · sqrt( n_future_peer_races(r, T) · n_peer_horses(r) ) /
               normaliser
```

where the normaliser is the batch-mean of the same quantity.

### 2.5 Derived-quantity close_date audit (Q-24)

*Resolves Q-24. §2.2 defines `close_date(φ) = max_{s ∈ S(φ)} date(s.race)` and
declares admissibility iff `close_date(φ) < date(T)`. The edge cases arise
when a derived quantity's **computation graph** differs from the graph used
for its causality check. Enumerating each quantity settles which graph the
max must be taken over.*

| φ | inputs S(φ) | close_date | admissible at T iff | edge case |
|---|---|---|---|---|
| ẑ_r (prior) | pre-race entry feats of r | date(r) | date(r) < date(T) | none |
| z̃_r (posterior) | entry feats + outcomes of r | date(r) | date(r) < date(T) | none — all inputs at r's own finish |
| z_r_post cache (Step 4) | same as z̃_r | date(r) | date(r) < date(T) | cache key must bind ckpt hash (Q-25) |
| CROE `expected_h(r)` | career_summary_h (cutoff r), conditions_r | date(r) | date(r) < date(T) | career cutoff must be `< date(r)`; including r's outcome leaks |
| CROE `residual_h(r)` | observed_h(r) − expected_h(r) | date(r) | date(r) < date(T) | same as expected |
| `tendency_before_{r,R}(p)` | career of p with date < date(r) | date(r)−ε | < date(T) | none |
| `tendency_after_{r,R}(p)` | career of p in [r, T) | max date in [r, T) | < date(T) | depends on T → recompute per target |
| `tendency_{r,R}(p)` fused | before + after arms | max(before, after) | < date(T) | **MAX dominates** (answers Q1) |
| SDFI `s_h` | full career of h | max over career | only if computed with cutoff date(T) | static ≠ timeless; reuse-at-all-targets leaks |
| SDFI `d_{h,r}` | conditions_r + recent form < r | date(r) | date(r) < date(T) | recent-form cutoff `< date(r)` |
| SDFI `r_{h,r}` residual | s_h (cutoff T), d_{h,r} | max(close_date(s_h), date(r)) | close_date(s_h) < T ∧ date(r) < T | if s_h uses full career → leakage |
| CIP `gate(r,T)` (Idea M) | r_summary, T_summary | date(T) | only valid at T | gate conditions on T; computed at target time, no leak |

**Q1 — pre/post arms: which close_date dominates?** Per §2.2, close_date is
the MAX over all inputs, so the **post arm (later dates) dominates**. A
quantity fusing a pre-arm (cutoff r) and a post-arm (cutoff T) has close_date
= the post arm's last pre-T date, which is still `< date(T)` → admissible, but
less causally tight than a pre-only quantity. This is exactly the asymmetry
§2.4 already downweights: the effective observation window is `[r, T)`, so the
consistency term should scale with the post-arm's richness.

**Q2 — SDFI `s_h` is the real trap.** `s_h` is a "static" embedding, but it is
trained/EMA'd over the horse's career. At inference for target T, an `s_h`
computed over the full career (including races after T) leaks future form.
Two safe resolutions: (a) per-target recompute with a `date < date(T)` cutoff
(costly); or (b) a strict temporal split where all training races precede all
eval races, so a single global `s_h` is automatically causal. The repo already
trains on a time-ordered split, so (b) is the default; flag (a) as a hard
requirement if an out-of-order split is ever introduced.

**Q3 — CROE expected-outcome self-leak.** `expected_h(r) = f_career(
career_summary_h, conditions_r)` is causal only if `career_summary_h` uses
races strictly before r. Including r's own outcome turns the residual into a
self-prediction of a known outcome. Same `close_date < date(r)` discipline as
`race_tendency`'s `masked_feature`, but the cutoff is r (not the next-race
assembly step).

**Q4 — per-target quantities.** `tendency_after` and `gate(r,T)` condition on
the target date T, so their close_date is a function of T: they must be
recomputed per target and cached keyed by (r, T), never globally precomputed.
Only `tendency_before` is safe to precompute globally.

**Rule of thumb (appends to §2.2).** Every derived quantity should state its
`S(φ)` set and close_date formula in its module docstring. When `S(φ)` spans
two graphs (career arms, static-vs-dynamic, pre-vs-post), take the max over
*both* graphs' dates. Any quantity whose `S(φ)` includes the target's own date
or any post-T race is inadmissible and must be recomputed with a cutoff.

### 2.6 Debut-horse identifiability (Q-3)

*Resolves Q-3 (theory arm). The graceful-degradation gate
(`dual_path_career_threshold`) is a code task for Lane C, flagged for Step 20.*

**Question.** §2.3's two-paths claim requires sufficiency of each path. A
debut horse has an empty career path — only the race path. Does the claim
fail, or is it rescued by distributional assumptions?

**Answer: the claim does not fail — it degenerates into a shrinkage-to-prior
regulariser.** The career path's "sufficiency" holds vacuously: an empty career
maps to the prior, which is a valid estimator with maximal (prior) variance.
What needs correcting is not the identification but the *consistency loss*,
which must be downweighted when the career arm is degenerate.

**Characterisation.** Restrict to the posterior pass (encoding a past race `r`
whose outcomes are observed). For `e = (h, r)` where `r` is `h`'s debut:

```
M_career(e) = ∅  ⇒  v_career(e) = prior p(z_e) ,
```

while the race path stays informative:

```
P(z_e* | race path) = P(z_e* | x_e, {x_{e'} : peers}, {y_{e'}}, y_e) .
```

By faithfulness (§2.3 condition 1), `z_e* = f(x_e, y_e)`, and the race path
observes `y_e` at encoding time, so the race path is **alone sufficient** for a
debut horse. The information decomposition gives

```
I(z_e* ; M_career, M_race) = I(z_e* ; M_race) + I(z_e* ; M_career | M_race)
                           = I(z_e* ; M_race) + 0     (empty career).
```

The debut horse loses only the career term: its posterior has **higher
variance but is unbiased** — a *widening*, not a failure. In this case the
claim's content is simply that `v_race(e)` is a valid point estimate of
`z_e*`.

**Why forcing agreement is wrong here.** The §2.3 consistency loss requires
`v_career ≈ v_race`. For a debut horse `v_career = prior`, so the loss
penalises any race-path deviation from the prior — it *shrinks* the debut
horse's (correct, informative) representation toward the prior and wastes
capacity. This is the extreme case of §2.4's observation-window asymmetry,
with the career/forward arm empty rather than merely thin.

**Consequence for Step 20 (flagged).** Gate the dual-path consistency loss by
career length:

```
λ_con(e) = λ_0 · 1[ len(career_path(e)) ≥ dual_path_career_threshold ],
```

default `dual_path_career_threshold = 3`. Below threshold, **disable** the term
rather than downweighting smoothly — for `len = 0` there is no career
information to reconcile and any nonzero weight is pure shrinkage bias. A
smooth ramp `min(1, len/threshold)` is acceptable for `1 ≤ len < threshold` if
a nonzero weight is wanted for second-start horses; flag either choice in the
Step 20 config review.

**Restatement of §2.3.** Amend condition 2 ("sufficiency of each path") to
"sufficiency of the race path, and sufficiency of the career path *whenever it
is non-empty*". With this amendment the two-paths claim holds uniformly,
including the debut long-tail.

---

## 3. DEQ fixed-point node embedding — exact spec (Idea I)

### 3.1 Operator

Per entry node `e`, let `X_e ∈ R^d` be the node embedding. Define the
pre-equilibrium operator

```
F_θ(X)_e := LN( σ( W_h x_e
                 + α_c · AttnCareer_θ({X_{e'} : e' ∈ N_career(e)}, Δ)
                 + α_p · AttnPeer_θ({X_{e'} : e' ∈ N_peer(e)}, result_pair)
                 + W_o X_e )) .
```

- `AttnCareer` is a dot-product attention block keyed by signed `Δ(e, e')`.
- `AttnPeer` is dot-product attention keyed by the observed pairwise outcome.
- `W_h x_e` is the static-feature injection; it ensures `X_e` has an
  identifiable base irrespective of neighbourhood richness.
- `σ` is GELU; `LN` is LayerNorm.
- `W_o X_e` is a self-loop so sparse-neighbourhood nodes don't collapse to
  the base `W_h x_e`.

The **fixed point** is the solution of

```
X* = F_θ(X*) .
```

### 3.2 Contractivity (Lipschitz) condition

For the fixed point to be unique and iteration to converge, `F_θ` must be a
contraction on the training distribution — i.e. there exists `L < 1` such
that `||F_θ(X_a) − F_θ(X_b)|| ≤ L · ||X_a − X_b||` for all `X_a, X_b` in a
compact set.

**Sufficient conditions in practice:**

1. **Spectral norm cap on `W_o`**: enforce `σ_max(W_o) ≤ γ < 1 − Σ (bounds
   on the attention Jacobians)` via spectral normalisation (Miyato+ 2018).
2. **Softmax attention is bounded**: `||AttnCareer(X)||_2 ≤ ||V_c||_2`
   where `V_c` are the values. Bounding the value projection suffices to
   bound the attention block's Jacobian.
3. **LayerNorm has bounded Jacobian** under the standard parameterisation.

The implementation must EXPOSE the spectral-norm cap `γ` as a config knob,
default `γ = 0.9`. This is what gives the contractivity margin without
introducing a dedicated penalty term.

### 3.3 Solver

Three practical choices, ordered by ease:

1. **Unrolled iterations (K steps, K=8)**. Dead simple, works when `L ≪ 1`.
2. **Anderson acceleration** (Walker & Ni 2011): a quasi-Newton extrapolation
   over a window of `m` past iterates; converges in `O(log(1/ε))` under
   contractivity. m=5 is a safe default.
3. **Broyden's method**: full-rank Newton approximation. Overkill at `d=32`.

Pick (2) for production. The DEQ literature (Bai, Kolter, Koltun 2019)
provides numerical recipes.

### 3.4 Implicit-function gradients

At convergence, `X* − F_θ(X*) = 0`. Differentiating implicitly with respect
to `θ`:

```
dX*/dθ = (I − ∂F_θ/∂X |_{X*})^{-1} · ∂F_θ/∂θ |_{X*} .
```

The `(I − J_F)^{-1}` solve can be done with another fixed-point iteration
(conjugate gradient on `(I − J_F)^T (I − J_F)`), which is `O(d · n_iter)` per
batch — tractable at our scale.

**Memory win**: only `X*` and `θ` need to be stored for the backward pass.
The unrolled-K-step approach would store K copies of `X_k` at every batch;
the implicit approach avoids it.

### 3.5 Pretraining / warm-start

Zero-init `W_o` so the initial iteration map is just `LN(σ(W_h x_e +
contributions))`, which trivially satisfies `L < 1` since `W_o = 0`. The
spectral-norm cap kicks in as `W_o` grows during training. This is a
cheap way to ensure the initial training steps are well-behaved even before
the cap bites.

### 3.6 What this buys over L-layer stacked attention

- **Infinite effective depth at constant parameter count.** An L-layer
  stacked message passer has L × (attention + FFN) parameters. DEQ has 1.
- **Principled convergence.** At convergence, `X_e` is literally the
  solution to the two-paths-consistency equation (§2.3). No layer-depth
  hyperparameter to tune.
- **Memory tractable at graph scale.** For a career-centered graph of
  ~100 entries, L=4 stacked attention stores 400 intermediate tensors per
  batch; DEQ stores 1.

### 3.7 Risks

- **Non-convergence under adversarial inputs.** If a batch contains an entry
  with exploding feature norm, `F_θ` may violate the contraction. Clip
  feature norms at ingestion (`x_e ← x_e · min(1, c/||x_e||)`), default
  `c = 10`.
- **Fixed-point instability during early training.** Mitigated by zero-init
  `W_o` (§3.5).
- **Numerical Jacobian solves.** CG on `(I − J_F)` can diverge if contraction
  margin is tight. Regularise with `(I − J_F + εI)` with `ε = 1e-4`.

### 3.8 Well-posedness on a growing graph (Q-1)

*Resolves Q-1 (theory arm). The empirical arm — residual-norm vs graph size
per checkpoint — remains, blocks on Step 21.*

**Question.** §3.2's contractivity analysis assumes a FIXED graph. During
training the graph grows as new races arrive. Does the fixed point exist
uniformly across graph sizes under the spectral-norm cap?

**Answer: yes.** The DEQ operator is applied *node-wise* over a locally
bounded neighbourhood, so its global contraction constant does not scale with
the number of nodes or edges.

**Argument.**

1. **Local update.** `F_θ(X)_e` depends only on `X_e` and `{X_{e'} : e' ∈ N(e)}`.
   The neighbourhood size is bounded by the *data*, not the graph: a race has
   at most `K_field` participants and a horse at most `K_career` career
   neighbours, so `|N(e)| ≤ K_max := K_field + K_career` for every node and
   every graph size. (Pad to a fixed `K_max` slots with masking; extra nodes
   only add *masked-out* slots.)

2. **Masked softmax has a graph-size-independent spectral bound.** The
   attention weight vector `p = softmax(q·kᵀ/√d)` over `K_max` slots (absent
   peers masked to `−∞` ⇒ weight 0) has Jacobian `diag(p) − ppᵀ` with
   `‖diag(p) − ppᵀ‖₂ ≤ 1/2`, equality at `p = (1/2, 1/2, 0, …, 0)`. The bound
   is independent of `K_max` and of how many slots are masked. Hence the
   attention block's Lipschitz constant is bounded by
   `‖V‖ · (1/√d)‖QKᵀ‖ · (1/2)`, uniform in `K_max`.

3. **Global Jacobian is banded with bounded bandwidth.** Stack
   `X ∈ R^{|V|·d}`. The Jacobian `J_F = ∂F/∂X` has blocks
   `∂(F(X))_e/∂X_{e'}` non-zero only for `e' = e` or `e' ∈ N(e)`, so each
   block row has at most `(1 + K_max)` non-zero blocks, each of norm `≤ ℓ` (the
   fixed local Lipschitz from §3.2). A matrix-norm bound then gives

   ```
   ‖J_F‖₂  ≤  (1 + K_max) · ℓ ,
   ```

   a constant independent of `|V|` and `|E|`.

4. **Uniform contraction.** Choose the self-loop cap `γ` so
   `(1 + K_max)·ℓ < 1` (equivalently §3.2's
   `σ_max(W_o) ≤ γ < 1 − Σ(attention bounds)` with `K_max` fixed). Then `F_θ`
   is a contraction with the SAME constant on every graph size, and Banach's
   fixed-point theorem gives existence and uniqueness of `X*` uniformly in
   `|V|`, `|E|`.

**What grows vs what does not.** Adding a race adds `K_field` nodes and their
peer edges; adding a career start adds one career edge. Neither changes
`K_max`, the masked-softmax bound, or `ℓ`. What grows is (i) the ambient
dimension `|V|·d` and (ii) the solver cost per iteration (`∝ |V|·K_max`). The
contraction *quality* is unchanged; only the solve cost grows, linearly.

**Caveat.** `(1 + K_max)·ℓ` is a coarse worst-case row-sum bound; the effective
spectral radius is usually far below it. The operative question — whether the
residual `‖X_k − X_{k−1}‖` stays under the §3.3 tolerance as `|V|` grows at a
fixed checkpoint — is the Q-1 empirical arm, open until Step 21. If it ever
grows, the lever is a tighter `γ`, not a different solver.

---

## 4. Bayesian pairwise belief propagation (Idea J) — exact spec

TrueSkill (Herbrich+ 2007) is a tabular BP over the match graph with
Gaussian factors and a sigmoid link. The novelty here is that both the
likelihood AND the message functions are learned neural networks, operating
on continuous entry embeddings rather than just a scalar skill.

### 4.1 Latent model

Let every entry `e` have a latent **skill vector** `s_e ∈ R^{d_s}`. Observed
pairwise outcome for `(e_i, e_j)` in the same race:

```
y_{ij}  ∈  {+1, −1, 0}       with +1 = i beat j, −1 = j beat i, 0 = masked.
```

Likelihood:

```
P(y_{ij} = +1 | s_i, s_j, c_{r(i)})  =  sigmoid( g_φ(s_i − s_j, c_{r(i)}) )
```

where `c_r ∈ R^{d_c}` is a learned race-context vector (surface, distance,
weather, pace shape — derivable from §5-§7 of the main plan) and
`g_φ : R^{d_s + d_c} → R` is a small MLP (3 layers, d_h = 32).

TrueSkill is the special case `d_s = 1, g_φ(Δs, c) = Δs / β` with fixed β.

### 4.2 Variational family

Posterior over skills is approximated with a factorised Gaussian:

```
q(s_e | all observations) = N(μ_e, diag(σ_e²))
```

with `(μ_e, σ_e)` emitted by an inference network `h_ψ(x_e, context of e)`.
Factorised across entries; the correlation structure is carried by the
message passing, not the variational family.

### 4.3 Expectation-propagation messages

Standard factor-graph BP with Gaussian beliefs produces closed-form updates
when `g_φ` is approximated locally as linear:

```
message from factor (i, j) to variable i:
  m_{(ij)→i}(s_i)  ∝  ∫ P(y_{ij} | s_i, s_j, c) · q(s_j)  ds_j .
```

For the sigmoid link this integral has no closed form. Two tractable options:

**Option 1 (EP-style moment matching).** Approximate the message as Gaussian
with mean/variance matched to the first two moments of the exact message.
Closed-form approximations exist for sigmoid-Gaussian integrals (MacKay
1992; TrueSkill's implementation uses this).

**Option 2 (Amortised).** Train a neural network `μ_msg, σ_msg = m_ψ(μ_j, σ_j,
y_{ij}, c_r)` to emit the message mean/variance directly. Training signal: at
each BP step, require the amortised message to minimise the KL from the true
posterior (estimated by Monte Carlo on a sub-sample).

**Choice for this project: Option 2.** Option 1 inherits TrueSkill's
Gaussian-link constraint, which collapses `d_s = 1`. Option 2 preserves
multi-dimensional skill while retaining the BP structure.

### 4.4 Forward pass

For a batch containing target race `R`:

1. Gather all participants' entry embeddings and run inference network to
   get `(μ_e, σ_e)` for every entry in the subgraph (participants of `R`
   and their recent career races, limited by a horizon H).
2. For `K` BP rounds (K=3 default):
   a. For each factor `(i, j)` in the subgraph, compute amortised message
      from factor to each incident variable.
   b. Update each variable belief as the product of incoming messages,
      renormalised.
3. Read out the final belief `q(s_e)` for each participant of `R`.
4. Compute race-level softmax over participants using the mean skills
   projected to a scalar: `logit_e := w · μ_e + b`.

### 4.5 Training objective

Variational lower bound on the log-likelihood of observed pairwise outcomes:

```
ELBO  =  Σ_{(i,j) observed}  E_{q(s_i), q(s_j)}[ log P(y_{ij} | s_i, s_j, c_{r(i)}) ]
         −  Σ_e  KL( q(s_e)  ||  p(s_e) )
```

with `p(s_e) = N(0, I)`. Standard mean-field VI. Reparameterisation trick
(Kingma & Welling 2014) for gradient of the expectation term.

### 4.6 Reduction to TrueSkill

In the limit `d_s = 1`, `g_φ(Δs, c) = Δs / β`, no amortisation of messages
(Option 1), no neural inference network (replace `h_ψ` with the prior
`N(0, 1)`), this reduces exactly to TrueSkill. The BP update equations
become TrueSkill's update rules. This is the paper's honesty statement:
TrueSkill is a well-known special case of the architecture.

### 4.7 Novelty angle for NA-1

Three claims, each independent:

1. **Multi-dimensional learned skill.** `d_s > 1` with amortised inference.
2. **Context-conditioned likelihood.** `c_r` is a learned race-context
   vector; traditional Elo / TrueSkill has no context.
3. **End-to-end training through BP** via amortised messages. This replaces
   EP moment matching with a neural network, which handles non-Gaussian
   likelihoods without the EP closed-form constraint.

### 4.8 Prior-art position

- TrueSkill (Herbrich+ 2007): Gaussian, scalar skill, no context, closed-form
  EP.
- Blei-style amortised VI (Kingma & Welling 2014; Rezende+ 2014): amortised
  posteriors but no graph-structured observation model.
- Graph-based skill models (e.g. ChessVector, Chen+ 2019): embed chess
  players in R^d via matrix factorisation; no BP, no context.
- GNNs for player ranking (Huang+ 2020): message passing on match graphs
  but no variational semantics.

The combination (multi-d skill + context + learned messages + factor-graph
structure) is, to my knowledge, not published.

### 4.9 Risks

- **Message collapse.** If `m_ψ` emits messages with tiny variance, the
  posterior collapses to the prior. Regularise via a KL floor on messages.
- **Compute.** `O(|E| · K · d_s)` per batch; for a 100-node subgraph with
  ~500 pairs, K=3, d_s=8 → 12k · 8 = 96k scalar ops per batch. Negligible.
- **Training instability** at the interface between the amortised inference
  network and BP. Warm-start the inference network from a non-BP pass
  (K=0) for the first 5 epochs.

### 4.10 Loopy-BP convergence under amortised messages (Q-2)

*Resolves Q-2. Gives a rate bound under a linearised (Gaussian) likelihood and
a criterion for when K > 3 is needed. The full non-linear case (d_s > 1,
context-conditioned `g_φ`) is flagged, not closed.*

**Linearised model.** In the TrueSkill limit (§4.6), the sigmoid link
`sigmoid((s_i − s_j)/β)` linearises to an effective Gaussian factor with
inverse-variance `κ = 1/σ_obs²` (σ_obs the linearisation noise, `σ_obs² ∝ β²`).
The joint over skills is then Gaussian with information (precision) matrix

```
J = I + κ L_G ,    L_G = graph Laplacian of the match graph G.
```

The match graph is a union of race-cliques (each race contributes a
`K_field`-clique of pairwise edges), so it is **far from a tree** — this is
why K > 1 is needed at all.

**Convergence condition (Weiss & Freeman 2001).** Gaussian loopy BP converges
to the correct means iff the diagonal-shifted random-walk matrix

```
M = (D + σ_obs² I)^{-1} A_G ,    D = diag(deg(i)),
```

has spectral radius `ρ(M) < 1`. For a `d`-regular graph (`A_G` top eigenvalue
`d`),

```
ρ(M) = d / (d + σ_obs²) < 1 ,
```

and in general `ρ(M) ≤ d_max / (d_max + σ_obs²)` with `d_max` the max degree.
This is the walk-summability condition; our §4.4 subgraph (horizon H, ~100
nodes, ~500 pairs) is finite and small, so `ρ` is bounded by the worst local
coupling.

**Rate bound.** In the linearised model the message-mean iteration is affine,
so the mean error decays geometrically:

```
‖μ^{(K)} − μ*‖ = O(ρ^K) ,   hence   K* ≈ log(1/ε) / log(1/ρ)
```

for target error ε.

**When is K = 3 enough?** K = 3 suffices iff `ρ³ ≤ ε`. Because
`ρ = d_max/(d_max + σ_obs²)`:

- `σ_obs²` large (uninformative pairwise signal) → `ρ → 0` → K = 3 (even
  K = 1–2) suffices.
- `σ_obs²` small (confident pairwise signal) **or** `d_max` large (tightly
  coupled subgraph, horses that meet repeatedly) → `ρ → 1` → K = 3 leaves
  error `≈ ρ³` well above ε, and K must grow.

**Recommendation for Step 22.** Do not fix K = 3. Treat K = 3 as a floor and
grow it until the message-residual norm `‖μ^{(k)} − μ^{(k−1)}‖` drops below ε
(mirroring the DEQ residual check in §3.3), or set K from the estimated `ρ`
via `K ≈ log(1/ε)/log(1/ρ)`. In the confident/tightly-coupled regime, K = 5–8
is the expected operating point.

**Where amortisation helps vs hurts.** The amortised message `m_ψ` emits
`(μ_msg, σ_msg)`; its variance feeds the precision update, so a variance error
propagates like a mean error.

- **Helps** when (i) the true message is non-Gaussian (`d_s > 1`, context `c_r`
  enters `g_φ` nonlinearly) — closed-form EP is unavailable and `m_ψ`'s
  non-linearity is necessary; or (ii) `m_ψ` learns to approximate *multiple*
  rounds in one call (a learned Anderson/multi-step), giving effective rate
  `ρ^m` per network call.
- **Hurts** in the near-Gaussian / tightly-coupled regime: Ihler et al. (2005)
  show a fixed message error δ saturates at `δ/(1−ρ)`. As `ρ → 1` a small
  amortisation error is amplified by `1/(1−ρ)` and can dominate — there the
  closed-form Gaussian message (Option 1, §4.3) is exact and cheap, and
  preferable.

**Prior-art position.** Weiss & Freeman (2001) give the walk-summability /
spectral-radius condition for Gaussian loopy BP; Weiss (2000) the tree case.
Ihler, Fisher & Willsky (2005) quantify message-error propagation. Q-2's
contribution is the `ρ = d/(d+σ_obs²)` rate in the *amortised* regime and the
K-adaptation rule tied to the DEQ residual check.

---

## 5. Synthetic DGP for model recovery (§8.1) — rigorous design

The main plan's §8.1 says "assert R² > 0.5 on `pace_hardness` and
`front_tax`". That's cheap talk unless the synthetic DGP is specified
precisely enough to compute the TRUE latent and the required sample size.
Here is the specification.

### 5.1 Generative model

Hyperparameters: `N_races = 100,000`, `N_horses_universe = 20,000`, average
`n_field = 10`.

For each race `r`:

- Sample race latent `θ_r ∈ R^4` as:
  - `pace_hardness_r      ~ N(0, 1)`
  - `class_r              ~ N(0, 1)`
  - `front_tax_r          ~ N(0, 1)`    (positive = pace burned the front)
  - `closer_advantage_r   = 0.5 · pace_hardness_r + ε_r, ε_r ~ N(0, 0.5)`
    (correlated with pace_hardness — so the probe test for `closer_advantage`
    has a slightly lower R² ceiling by construction)
- Sample field size `n_r ~ Uniform{6, …, 16}`.

For each horse `h` in the universe, sample `skill_h ~ N(0, 1)` (persistent).

For each participant `e = (h, r)` of race `r`:

- Sample running style `style_e ~ Categorical({front: 0.3, mid: 0.4,
  back: 0.3})`.
- Compute "performance" scalar:
  ```
  perf_e := skill_h
            + class_r
            − 0.5 · pace_hardness_r
            + front_tax_r · 1[style_e = front] · (−1)
            + closer_advantage_r · 1[style_e = back] · (+1)
            + N(0, σ_noise²)
  ```
  default `σ_noise = 0.6`.
- Finish rank is the within-race ranking of `perf_e`.
- Corner-4 position is a function of `style_e` plus noise.
- "Time" is `perf_e` offset by `pace_hardness_r` with noise.

### 5.2 What the model sees

Each training example is a target race; the model sees participants' entry
features `x_e` (which contain horse_id proxies — a look-up table to a
one-hot over 20k horses — plus style and draw) and the outcomes of their
past races. It never sees `θ_r`, `skill_h`, or the perf noise directly.

### 5.3 Probe design

After training, for the eval split:

1. Extract the trained `z̃_r` for each race.
2. Fit a linear probe `θ_r = W z̃_r + b` on the train split, evaluate R² on
   the eval split.

### 5.4 Pass criterion — calibrated by information-theoretic ceiling

For a single race, the model observes `n_r` participants × average 5 past
races each × ~6 scalars per past race = `~30 · n_r` observations. The true
latent `θ_r` has 4 dimensions. Shannon capacity:

```
I(θ_r ; observations_r)  ≤  H(θ_r)  ·  (1 − 2^{−2 · C})
```

for a Gaussian channel. With `σ_noise = 0.6` per observation and Gaussian
θ, the achievable R² for a 4-d latent from `30 · n_r` noisy observations is
roughly:

```
R²_ceil  ≈  1  −  σ_noise² / (σ_noise² + n_obs · σ_signal²)
           ≈  1  −  0.36 / (0.36 + 300 · 1)   (for n_r = 10, n_obs = 300)
           ≈  0.999 .
```

So the ceiling is near-unity; the model has plenty of signal. A pass
criterion of `R² > 0.5` is thus LOOSE — even a weak model should clear it.
Tighten to:

- `R² > 0.85` for `pace_hardness` and `front_tax`.
- `R² > 0.70` for `class`.
- `R² > 0.50` for `closer_advantage` (the correlated-with-pace latent).

### 5.5 Statistical power

Given `N_eval = 20,000` eval races, standard error of R² at population R² = p
is `≈ sqrt( (1 − p²)² · 4 / N )`. For p = 0.85, SE ≈ 0.004; for p = 0.50,
SE ≈ 0.014. So a difference of R² ≥ 0.05 between two configs is detectable
at p < 0.001. The pass criteria above are thus distinguishable with
effectively zero false-discovery risk.

### 5.6 Ablations buyable from this DGP

The DGP cleanly supports the following structural ablations at zero extra
design cost:

1. Set `front_tax_r ≡ 0` for half of races; confirm model's recovery of
   `front_tax` drops to the floor for those races.
2. Permute `skill_h` across horses for the test split; confirm model's
   downstream pred_prob drops by the expected margin given skill carries
   `≈ 50%` of the variance.
3. Turn off `closer_advantage` correlation with `pace_hardness`; confirm
   the probe's R² on `closer_advantage` rises to the pace_hardness level.

Each ablation gives a signed, pre-registered prediction — the kind of thing
the reviewers will want in an NA-1 submission.

### 5.7 Deliverables

A file `tests/model_recovery_synthetic.py` with:

1. `SyntheticRaceDGP` class with the specification above as config.
2. `generate(n_races, seed) → (races_df, horse_df, θ_df)`.
3. `train_and_probe(config, dgp) → dict of R² per latent dim`.
4. CLI: `python -m tests.model_recovery_synthetic --n_races 100000 --seed 0`.
5. A pass/fail verdict printed at the end tied to the §5.4 thresholds.
6. All ablations in §5.6 runnable via `--ablation {null_tax, perm_skill,
   decorrelate}`.

### 5.8 Front-runner upgrade probe — statistical power (plan §8.4 / Step 14)

*Resolves Q-14. Deliverable: the minimum sample size for the
`tests/front_runner_upgrade_probe.py` fixture, and the "min n" value in the
task-board handoff `A: Q-14 → D: Step 36`.*

**Design recap.** The probe curates a set of races where domain analysis
says "pace burned the front." For each front-runner horse in those races,
compare two next-race logits: `L⁺` (posterior `z̃_r` of the pace-biased race
present in memory) vs `L⁻` (memory lookup ablated for that one race). The
test is paired and one-sided: mean uplift `δ̂ = mean_i(L⁺_i − L⁻_i)`,
`H0: δ = 0` vs `H1: δ > 0`, p < 0.05 under a paired bootstrap.

**Model.** `n` horses, `D_i = L⁺_i − L⁻_i`, assumed `D_i ~ N(δ, σ_d²)` iid.
For `n ≥ 30` the paired bootstrap and the z-approximation agree within the
error we care about, so use the z form for closed-form power.

**Minimum detectable effect.** One-sided, α = 0.05, power 1−β = 0.80:

```
δ_MDE = (z_{1−α} + z_{1−β}) · σ_d / √n = 2.4865 · σ_d / √n
```

with `z_{0.95} = 1.6449`, `z_{0.80} = 0.8416`.

**The controlling input is σ_d, NOT the logit std σ_L.** The Q-14 question
quotes σ_L = 0.5 (the logit std). But a paired difference has variance

```
σ_d² = 2 σ_L² (1 − ρ)
```

where ρ = corr(L⁺, L⁻). The two logits differ only by one ablated memory
token, so ρ is close to 1. With σ_L = 0.5: ρ = 0.90 → σ_d ≈ 0.224;
ρ = 0.80 → σ_d ≈ 0.316; ρ = 0.98 → σ_d ≈ 0.100. If the posterior memory
genuinely dominates one logit, ρ drops and σ_d rises — which *helps* power
(the effect is larger relative to σ_d) but also signals the ablation is
more than a token-level perturbation.

**δ_MDE (power 0.80) vs n and σ_d:**

| σ_d \ n | 50    | 100   | 200   | 400   |
|---------|-------|-------|-------|-------|
| 0.10    | 0.035 | 0.025 | 0.018 | 0.012 |
| 0.22    | 0.077 | 0.055 | 0.039 | 0.027 |
| 0.32    | 0.113 | 0.080 | 0.056 | 0.040 |

**Plausible true effect δ_true.** A single posterior-memory token is one
input to a large encoder, so the next-race logit uplift is small. In
probability terms, near p ≈ 0.1, `dp/dlogit = p(1−p) ≈ 0.09`, so δ = 0.05 is
a ~0.45pp absolute (~4.5% relative) win-probability shift and δ = 0.10 is
~0.9pp. A single memory token moving win probability by a full point would
be a large, publishable effect. Prior: `δ_true ∈ [0.02, 0.10]`, most likely
~0.05.

**Power at δ_true, ρ = 0.90 (σ_d = 0.22):** `power = Φ(δ_true √n / σ_d − z_{0.95})`.

| δ_true \ n | 50    | 100   | 200   | 400   |
|-----------|-------|-------|-------|-------|
| 0.03      | 0.25  | 0.39  | 0.61  | 0.86  |
| 0.05      | 0.49  | 0.74  | 0.94  | 1.00  |
| 0.10      | 0.94  | 1.00  | 1.00  | 1.00  |

**Recommendation.**

1. **n = 50 is underpowered** for a realistic effect: at ρ = 0.90 it detects
   only `δ_true ≥ 0.08` at 80% power and has ~49% power for δ = 0.05. Treat
   n = 50 as a screening pass (large effects only).
2. **n = 200 is the target**: ≥ 0.94 power for δ = 0.05, ≥ 0.61 for δ = 0.03.
   If 200 curated races are infeasible, **n = 100 is a floor** (0.74 power
   for δ = 0.05).
3. **Curation quality is a hidden multiplier.** n = 200 races that only
   weakly satisfy "pace burned the front" have the same power as n = 50
   strongly-curated races. The hand-curated fixture
   (`tests/fixtures/pace_biased_races.json`) should target n ≥ 200, each race
   vetted for the front-burn pattern.
4. **Do not lean on this probe alone for small effects.** Pair it with the
   synthetic model-recovery test (§5.1-§5.6), whose `n_eval = 20,000` gives
   effectively zero false-discovery risk (see §5.5). The front-runner probe
   is the *causal/behavioural* confirmation; model recovery is the *powered*
   confirmation.

**Handoff value.** The `A: Q-14 → D: Step 36` min-n entry is **n = 200**
(100 as a floor), not 50.

---

## 6. Four further novel directions (K, L, M, N)

These are ideas the user asked me to reason about. Each is stated here with
enough detail to be converted into a §10-style implementation step by the
agent IF the user approves. Each has an explicit prior-art position.

### 6.1 Idea K — Variational Event Autoencoder (VEA)

**Statement.** Treat each race `r` as a sample from a latent-variable model
with latent `z_r ∈ R^d`. Observed variables are (a) pre-race entries
`{x_e : e ∈ participants(r)}`, (b) outcomes `{y_e}`, (c) future-career
observations of participants within some window after `r`. The amortised
posterior is

```
q_φ(z_r | observations_r)  =  N( μ_φ(obs_r), diag(σ_φ(obs_r)²) )
```

and the prior is the pre-race context:

```
p_θ(z_r | entries_r)  =  N( ẑ_r, diag(σ_prior(entries_r)²) ) .
```

ELBO:

```
L(r)  =  E_{z_r ~ q_φ}[ log p(outcomes_r | z_r, entries_r) ]
         +  E_{z_r ~ q_φ}[ log p(future_r | z_r, entries_r) ]
         −  KL( q_φ(z_r | obs_r) || p_θ(z_r | entries_r) ) .
```

**Why this is theoretically clean.** The main plan's §7.4 Y-architecture
produces `ẑ_r` and `z̃_r` as POINT estimates with a consistency loss. VEA
makes the prior/posterior relationship EXPLICITLY probabilistic: the prior
IS a distribution `p_θ(z_r | entries_r)` and the posterior IS
`q_φ(z_r | obs_r)`. The KL term is no longer a heuristic — it is the natural
regulariser from the ELBO, with a principled weight of 1.

**Novelty over existing VAE literature.**
- Event-level latent — standard VAEs are per-datapoint. Here one latent per
  race, with multiple participants contributing to the posterior. This
  is a *set-conditional* amortised posterior.
- The prior is NOT fixed (standard VAE) and NOT fully learned (β-VAE) but
  CONDITIONAL on observable pre-race features. Analogous to CVAE
  (Sohn+ 2015) but with a structurally meaningful conditioning set.
- Encoding of CAUSALITY via the future-career observation window: this
  takes neural processes' context/target split and ties it to a temporal
  admissibility constraint.

**Risks.** Posterior collapse is the perennial VAE problem. Standard
mitigations: KL annealing, free bits, or the InfoVAE objective (Zhao+ 2017)
which lower-bounds `I(z_r ; obs_r)`.

**Isolation.** A drop-in replacement for §7's z̃_r / ẑ_r pair. Config flag
`use_vea`. Mutually exclusive with pure consistency-loss anchoring (the
anchor loss becomes the KL term).

**Relation to §7.5 contrastive consistency (Q-4).** If VEA is adopted, does
the §7.5 InfoNCE become redundant? **No — they control orthogonal failure
modes and are non-equivalent (not even modulo constants).**

The ELBO's KL term anchors the posterior to the prior *pointwise*:

```
KL(q_φ ‖ p_θ) = (1/2) [ σ_φ²/σ_p² + ‖μ_φ − ẑ_r‖²/σ_p² − 1 − log(σ_φ²/σ_p²) ],
```

minimised at `μ_φ = ẑ_r, σ_φ = σ_p`. In the sharp-posterior limit `σ_φ → 0`
(forced by a strong reconstruction term) this tends to
`(1/2σ_p²)‖μ_φ − ẑ_r‖² + (1/2)log(σ_p²/σ_φ²)` — i.e. the deterministic §2.4 L2
anchor plus a variance penalty. So in the sharp limit VEA's KL **reduces to the
L2 anchor**, a *pointwise, absolute* constraint.

The §7.5 InfoNCE is a *relative, discriminative* constraint:

```
L_con = −log  e^{s(ẑ_r, z̃_r)/τ} / Σ_{r'} e^{s(ẑ_r, z̃_{r'})/τ},
```

whose batch expectation is a variational lower bound on the mutual information
`I(ẑ_r; z̃_r)` (`I ≥ log N − E[L_con]`). It asks that `z̃_r` match *its own* `ẑ_r`
more than *every other* race's. The pointwise anchor is satisfiable by the
trivial collapse `ẑ_r = z̃_r = const` for all `r`; the InfoNCE is exactly the
term that forbids it, by forcing the prior/posterior pair to carry
discriminative race identity.

**Conclusion — complementary, not redundant:**
- KL / L2 anchor — prevents the posterior from *drifting* from the prior.
- InfoNCE — prevents the shared anchor from *collapsing* to a constant.

If VEA is adopted, §7.5's contrastive loss retains independent value as the
anti-collapse / race-identity term, competing with the standard VAE toolkit
(KL annealing, free bits, InfoVAE — see Q-12). The "specialised test
(conditional InfoNCE)" reading in Q-4 is correct: it is a conditional InfoNCE
whose conditioning set is race identity, not derivable from the KL modulo
constants.

### 6.2 Idea L — Neural Process over Race Fields (NPRF)

**Statement.** A race is a *context set* of (entry, outcome) pairs; new
predictions are the *target set* (the same race's entries without outcomes,
for the pre-race prediction task). Use a Conditional Neural Process
(Garnelo+ 2018a,b):

```
r_e := h_ψ(x_e, y_e)                      per context-entry representation
R   := pooling({r_e : e ∈ context})       set-level representation
pred(target entry e') := g_θ(x_{e'}, R) . target-entry prediction
```

The "context" here is NOT just the current race — it is the full career
graph of all participants aggregated into the pooled `R`. Permutation
invariance over context entries is built in.

**Novelty over NP literature.**
- Hierarchical context: entries → races → careers. Current NPs are flat.
- Attention-based pooling (Attentive NP; Kim+ 2019) extended to the
  multi-hop context (entry → race → career).

**Relationship to CFPT/ENGT.** NPRF is an alternative to BOTH CFPT and
ENGT. It drops the explicit graph and keeps only the pooled representation,
which is cheaper but loses the interpretability of the entry-node graph.
Useful as an ablation against ENGT (§9.4).

**Isolation.** Config flag `use_nprf`. Mutually exclusive with CFPT and
ENGT.

### 6.3 Idea M — Causal Influence Pruning (CIP)

**Statement.** A past race `r` contributes to target `T` ONLY via a learned
gate

```
gate(r, T) := sigmoid( MLP(r_summary, T_summary) ) .
```

The contribution of `r` to any career encoding or memory attention for `T`
is scaled by `gate(r, T)`. Impose L1 sparsity on `gate`:

```
L_sparse  =  λ_s · E_batch[ |gate(r, T)| ] .
```

**Theoretical motivation.** Not every past race is informative for every
target race. A 2-year-old sprint race is uninformative for a 5-year-old
Classic. A race on a different surface may be uninformative even recently.
Learning this explicitly imposes an attention-weighted information
bottleneck (Tishby & Zaslavsky 2015): the model is forced to identify the
causal pathway `r → T` and prune everything else.

**Novelty.** Prior gating mechanisms (highway networks, LSTM gates) operate
at the FEATURE level; CIP gates at the RACE level — an event-granular
information bottleneck. I am not aware of prior work doing this on
multi-participant event data.

**Diagnostic bonus.** For any target race T, `gate(r, T)` is a scalar per
past race — directly interpretable as "how much the model considers r
relevant to predicting T." Useful for debugging and for the paper's
qualitative-analysis section.

**Isolation.** Config flag `use_cip`. Composes with everything (§9's CFPT,
ENGT, dual-path, DEQ, Bayesian BP, VEA, NPRF).

### 6.4 Idea N — Information-Theoretic Posterior Compression (ITPC)

**Statement.** A posterior `z̃_r` should be a *sufficient statistic* of race
`r` for downstream prediction, not a complete reconstruction. Impose

```
I(z̃_r ; outcomes_r)  ≤  I(z̃_r ; target_winner)  +  slack .
```

Equivalently, minimise `I(z̃_r ; outcomes_r | target_winner)`: the
posterior should carry no outcome information beyond what is useful for
future prediction.

**Operationalisation via the IB bound** (Alemi+ 2017):

```
L_ITPC  =  −I(z̃_r ; target_winner)  +  β · I(z̃_r ; outcomes_r)
```

with the first term lower-bounded by a contrastive InfoNCE between `z̃_r`
and target-race outcomes (across the batch), and the second term
upper-bounded by a tractable KL between `q(z̃_r | outcomes_r)` and a
marginal `p(z̃_r)`.

**Novelty.** The Information Bottleneck has been applied to standard
representation learning (Tishby 2015, Alemi 2017). Its application to a
posterior over EVENT latents, where the "sufficient statistic" criterion
is defined relative to DOWNSTREAM EVENTS (future races), is new.

**Benefit over §7.5 contrastive consistency.** §7.5 asks that `ẑ_r` and
`z̃_r` be close. ITPC asks that `z̃_r` not memorise `outcomes_r` beyond what
helps future prediction. These are complementary; both can be active
simultaneously.

**Isolation.** Config flag `use_itpc`. Composes with everything.

---

## 7. Open theoretical questions

Collected for later work or human attention:

1. **DEQ well-posedness on a growing graph.** The contractivity analysis in
   §3 assumes a FIXED graph. During training, as new races arrive, the
   graph grows. Does the fixed-point exist uniformly across graph sizes?
   Probably yes under the spectral-norm cap, but a formal proof would be a
   paper subsection.

2. **BP convergence for non-tree graphs.** The entry-node graph has many
   cycles (e.g., two horses race each other repeatedly). Loopy BP is not
   guaranteed to converge; empirically it usually does but can oscillate.
   Mitigation in §4: K=3 fixed rounds (bounded iteration). Open: can we
   prove a convergence rate under the amortised-message learning regime?

3. **Identifiability of `z_r^*` under the two-paths claim (§2.3).** The
   sufficiency-of-each-path assumption is strong. In practice, a horse with
   no career observations provides zero information to career-path
   reconstruction of `z_e^*`. Is the claim rescued by distributional
   assumptions (sampling many horses per race), or does it genuinely fail
   on the long-tail of debut horses?

4. **VEA vs §7 consistency loss.** If VEA is adopted, what role (if any)
   does the §7.5 contrastive loss retain? Possibly: the contrastive loss
   becomes a specialised test for amortised-posterior quality (equivalent
   to a conditional InfoNCE under certain limits).

5. **Compute budget for Bayesian BP at full training scale.** §4's cost
   analysis assumed per-batch. The amortised-message warm-start phase (§4.9)
   is additional. Need an end-to-end benchmark before claiming Bayesian BP
   is production-ready.

---

## 8. What the implementation agent should do with this document

**Nothing new.** The implementation agent must continue following
`POSTERIOR_RACE_CTX_PLAN.md`'s step list. This document only exists to
settle the math BEHIND specific steps:

- §3 of this document is the authoritative spec for implementing Step 21
  (DEQ).
- §4 is the authoritative spec for Step 22 (Bayesian BP).
- §5 is the authoritative spec for Step 14 (§8.1 synthetic model recovery).
- §6 (Ideas K-N) is NOT yet in the plan's step list and the agent should
  NOT implement it without a corresponding Step 25+ being added by the
  design author.

If a line in this document conflicts with `POSTERIOR_RACE_CTX_PLAN.md`,
this document wins for the math; the plan doc wins for file paths and
config keys.

---

## 9. SDFI — formal spec (Idea in `POSTERIOR_RACE_CTX_PLAN.md` §15)

### 9.1 Setup

Let `H` be the set of horses and `R` the set of races. For a horse `h` and
race `r ∈ R_h` (the set of races h participated in), the full
representation currently used by the model is a single learned function
`φ(h, r) ∈ R^d`. SDFI's claim is that the TRUE representation factorises:

```
φ(h, r)  =  (s_h, d_{h, r})
```

where `s_h ∈ R^{d_s}` is **static** (depends on h but not on r) and
`d_{h,r} ∈ R^{d_d}` is **dynamic** (depends on both). Pairwise interaction
between h and h' in race r:

```
I(h, h' | r)  =  Ψ( s_h, s_{h'}, d_{h,r}, d_{h',r}, c_r )
```

for some function Ψ that is symmetric in (h, h') and depends on a race-
level context `c_r`.

### 9.2 Linear factorisation

Decompose Ψ linearly in its four horse arguments. The result, by symmetry
and the bilinear structure of attention logits, is

```
I(h, h' | r)  =   f_SS( s_h, s_{h'} )
                 + α(c_r) · f_DD( d_{h,r}, d_{h',r}, c_r )
                 + β(c_r) · [ f_SD( s_h, d_{h',r} ) + f_SD( s_{h'}, d_{h,r} ) ] .
```

The two scalars `α(c_r), β(c_r) ∈ R` are race-dependent gates learned from
`c_r`. When `α = 0, β = 0` the model reduces to pure static matching.
When `α → ∞, β = 0` it reduces to a conditional-dynamic-only model.

### 9.3 Identifiability of (s_h, d_{h,r})

The split is not a-priori identifiable: any transformation

```
s_h   →   s_h + u(h, r)
d_{h,r} → d_{h,r} − u(h, r)
```

with `u(h, r)` a function satisfying `u(h, r) = 0` on average over r
leaves `φ(h, r) = (s_h, d_{h,r})` invariant up to the sum. Identifiability
is restored by TWO mechanisms:

1. **Gradient decay on `s_h`** (§9.4) — the prior on `s_h` dynamically
   contracts as the horse accumulates races. This enforces `s_h` to be a
   race-r-INVARIANT summary, i.e. the average of `φ(h, r) − d_{h,r}` over
   r. Under mild assumptions this uniquely identifies `s_h` up to a
   constant.
2. **Dimensionality asymmetry** — set `d_d > d_s`. The dynamic embedding
   gets the extra capacity; the static embedding, being bottlenecked,
   must encode only the most compressible signal. Information-theoretic
   analogue: the bottleneck forces `s_h` to be the sufficient statistic
   of `{φ(h, r)}_{r ∈ R_h}` for predicting race outcomes with the rest
   of the input integrated out.

### 9.4 Gradient-decay dynamics

Let the gradient update for the static embedding be scaled by
`γ(n_h, k) = 1 − exp(−n_h / τ)` where `n_h` is the number of races horse h
has participated in and `τ` is a learned (or fixed) time constant.

Under SGD with learning rate η:

```
Δ s_h  =  −η · γ(n_h) · ∂L/∂s_h .
```

For a debut horse (`n_h` small), updates are nearly zero — `s_h` stays
close to its initialisation, which is a learned prior. For a veteran
(`n_h → ∞`), updates are at full strength — `s_h` is fully adapted.

This is a RESEARCH-DEBT-FREE way to implement a "race-averaged" static
representation. It is mathematically equivalent to EMA (exponential moving
average) of the gradients at `τ-large` limit; see Karras et al. (2018)
Progressive GAN's EMA discussion for the gradient-side version.

**Adam interaction (Q-23).** §9.4 is written for plain SGD. With Adam (the
repo's default optimiser), the question is whether scaling `∂L/∂s_h` by the
constant `γ(n_h)` corrupts the running statistics. It does **not**, provided
`γ(n_h)` is **constant over the run** (true offline: `n_h` is fixed per horse
in a fixed dataset). Two placements are possible:

- **Pre-scaling** — feed `g̃ = γ g` into Adam. The first moment `m` is linear
  and the second moment `v` quadratic in `g`, so `m_t = γ m_t⁰`,
  `v_t = γ² v_t⁰`, giving (bias-corrected)
  ```
  Δ_pre = η · γ m̂_t / ( γ √v̂_t + ε ).
  ```
- **Post-scaling** — scale the Adam step: `Δ_post = γ · η m̂_t / ( √v̂_t + ε )`.

They differ only through the `ε` floor:

```
Δ_pre / Δ_post = ( √v̂_t + ε ) / ( γ √v̂_t + ε )   (coordinate-wise).
```

Consequences:

1. **Bias-correction is preserved under both.** The `1/(1−β^t)` factors act on
   `m_t, v_t` independently of the constant `γ`, so `m̂_t, v̂_t` stay correctly
   de-biased. There is no corruption of the running statistics; they simply
   live at a scaled magnitude.
2. **In the `ε → 0` limit the two coincide**, and Adam reduces to a
   per-parameter learning rate `η_h = η γ(n_h)` on the normalised direction
   `m̂/√v̂` — exactly the §9.4 semantics.
3. **For finite `ε` and `γ < 1`, pre-scaling over-amplifies** by `≤ 1/γ`
   (the `ε` floor becomes proportionally larger once `γ√v̂` shrinks toward it).
   With `ε = 1e-8` and typical `√v̂ ~ 1e-2…1e-1` this is negligible; it only
   matters when gradients have already collapsed (both ≈ 0 there).

**Recommendation for Step 37.** Implement gradient decay as **post-scaling of
the Adam step** (scale the update, not the gradient), which realises
`Δs_h = −η γ(n_h) ∂L/∂s_h` exactly and preserves Adam's normalisation. If
post-scaling is inconvenient, pre-scaling is acceptable with `ε` small
(`1e-8`, not the `1e-3` seen in some code) so `γ√v̂ ≫ ε`. In **AdamW**, scale
only the adaptive term, not the weight-decay term:
`Δs_h = −η( γ m̂/(√v̂+ε) + λ s_h )`, since the decay targets the data gradient,
not the L2 regulariser. Caveat: this assumes `γ(n_h)` frozen per horse; if `τ`
is learned and evolves mid-run, or `n_h` grows online, second-moment
stationarity is perturbed and the two placements diverge measurably — keep `γ`
frozen over a run.

### 9.5 Reduction to known architectures

- `α = β = 0`: pure static embedding model, equivalent to an embedding
  table + dot-product interaction (e.g. LightFM, matrix factorisation for
  recommender systems).
- `γ(n_h) = 1` (no decay) and `s_h = d_{h,r}`: reduces to a standard
  Transformer with single per-token embedding. Current POA is in this
  regime.
- `d_{h,r} = 0, α = β = 0`: pure static GNN with no per-event modulation.

The architecture family interpolates between these three regimes. The
learnable gates `α(c_r), β(c_r)` and the gradient schedule jointly
determine where each race lands.

### 9.6 Cross-domain invariance claim

**Claim.** For any two domains D, D' whose entities admit a well-defined
static/dynamic decomposition in the sense of §9.1, and whose pairwise
interactions follow §9.2, a single SDFI trunk (fixed α, β gating function,
fixed Ψ parameterisation) is approximately universal: a small per-domain
input head and output head are the only components that must be re-trained
on D'.

**Sketch.** The static-static interaction `f_SS` operates on the static
embeddings alone, which ARE the universal horse/station/agent identities
within their domain. `f_DD` operates on the dynamic embeddings alone with
a domain-specific condition `c_r`. The gating scalars encode how much the
domain "weights" static vs dynamic — a high-noise domain weights static
less. Fixing the ARCHITECTURE of `f_SS, f_DD, f_SD` constrains the family
of representable interactions to the universal bilinear class; only the
WEIGHTS of these functions and the input/output heads need re-fitting.

This claim is empirical, not theoretically proven. §15.7 Step 41 (weather
transfer) is the experimental test.

**Transfer theorem — representation-sufficiency sketch (Q-8).** *Resolves
Q-8. Elevates §9.6 from empirical hunch to a consequence of the Information
Bottleneck under the shared-factor-structure assumption. Still a sketch; the
experimental test remains Step 41.*

The argument in four steps:

1. **The task is interaction prediction.** For a pair `(h, h')` in context `r`,
   the generative interaction is (by §9.1–9.2)
   `Ψ = f_SS(s_h, s_h') + α(c_r) f_DD(d_{h,r}, d_{h',r}, c_r) + β(c_r) f_SD(…)`.
   The static block `f_SS` depends *only* on the static embeddings — the
   identity-relevant part of any interaction is a function of `s_h` alone.

2. **`s_h` is a minimal sufficient statistic of entity identity.** In the
   Information-Bottleneck framework (Tishby et al. 1999; Achille & Soatto
   2018), the optimal representation for a task is *sufficient*
   (`I(x_h; y) = I(s_h; y)` for the interaction task `y`) and *minimal*
   (minimising `I(x_h; s_h)`). SDFI enforces both: sufficiency because `f_SS`
   must read all identity information off `s_h`; minimality because the
   bottleneck `d_s < d_d` (§9.3) plus gradient-decay/centring (§9.4, §9.9)
   force `s_h` to discard race-specific variation that carries no
   identity-relevant interaction information.

3. **Nuisance invariance is automatic.** Achille & Soatto's central result is
   that the optimal representation is invariant to nuisances — inputs that
   carry no task information. Here the race index `r` is a nuisance for the
   *identity* component, so `s_h` must be race-invariant; the gradient-decay
   (§9.4) and centring (§9.9 Remark 9.9.1) are exactly the mechanisms that
   enforce this invariance.

4. **Cross-domain universality follows from task abstraction.** The object
   "minimal sufficient statistic of entity identity for pairwise-interaction
   prediction" is defined *only* by the factor structure (static vs dynamic,
   the form of `f_SS/f_DD/f_SD`), which D and D' share by assumption. It
   references no domain-specific feature. A trunk computing raw-features →
   `s_h` under the shared Ψ therefore computes the same abstract object in
   both domains; only the domain-specific input head (raw features ↔ `s_h`)
   and output head differ.

**Theorem (informal).** Let D, D' share the §9.1–9.2 factor structure, and let
`s_h*` be the minimal sufficient statistic of entity identity for the
interaction task in D. Then `s_h*` is, up to the §9.9 symmetry group (a
bijection), the corresponding minimal sufficient statistic in D', and a fixed
SDFI trunk computing `s_h*` transfers across D, D' with only per-domain
input/output heads re-fitted.

**Limits (honest).** (i) Existence/uniqueness of the minimal sufficient
statistic requires the IB Markov structure `y ⊥ x | s_h` and injectivity
conditions — assumed, not proved, on real data. (ii) Identifying "identity"
with the static block is definitional; a mis-specified static/dynamic split
vacuates the theorem. (iii) Universality is conditional on the shared factor
structure, which is exactly what Step 41 tests. This sketch upgrades §9.6 to
"consequence of IB under the factor-structure assumption" — not a substitute
for the Step 41 experiment.

### 9.7 Residual channel — Bayesian interpretation

The residual `r_{h,r} := g_static(s_h, c_r) − d_{h,r}` is the deviation of
the horse's actual dynamic state from its race-condition-projected
prior. Under a linear-Gaussian model:

```
d_{h,r}  ~  N( g_static(s_h, c_r), Σ_r )
```

the residual is distributed as `r_{h,r} ~ N(0, Σ_r)`. Standardised
residuals `Σ_r^{−1/2} r_{h,r}` are a statistically-clean score of
"surprise" per horse per race. The paper can cite Mahalanobis-distance
scoring (McLachlan 1999) and recent ML applications in anomaly detection
(Ruff et al. 2021).

This generalises CROE (§7.2) which produced a scalar residual against the
horse's own career expectation; SDFI's residual is a vector in the
dynamic-embedding space, interpretable componentwise.

### 9.8 Open theoretical questions on SDFI

1. **Formal proof of identifiability under gradient decay.** §9.3 gives
   two heuristic mechanisms. A rigorous result (e.g. "under assumptions
   X-Y-Z, the gradient-decay scheme identifies `s_h` up to orthogonal
   transformation") would be a paper strengthener.
2. **Optimal `τ` schedule.** Fixed `τ` is a crude choice. A per-horse
   `τ_h`, learned, is cheap to add — does it help?
3. **Transfer-theorem proof.** §9.6's claim is empirical. A formal
   sketch via representation-sufficiency arguments (Achille & Soatto
   2018, "Emergence of Invariance and Disentanglement in Deep
   Representations") would be defensible paper-level theory.

### 9.9 Identifiability of `s_h` under gradient decay (Q-6)

*Resolves Q-6. This is the §9.3 appendix: a theorem + proof sketch showing
that the §9.4 gradient-decay scheme identifies the static component `s_h` up
to an orthogonal transformation, and that this is the tightest possible
statement. The honest division of labour between the two mechanisms in §9.3
is also clarified (Remark 9.9.6).*

**Assumption A1 (linear-Gaussian factorisation — analysis device).** For
horse `h` and race `r ∈ R_h`, the representation is generated as

```
x_{h,r} = s_h* + d_{h,r}*,     d_{h,r}* ~ N(0, σ_d² I_d) iid over r,
                               s_h*     ~ N(0, σ_s² I_d) iid over h,
```

with `{s_h*}`, `{d_{h,r}*}` independent and `d_s = d_d = d` for clarity (the
loading-matrix generalisation is Remark 9.9.2). The model fits embeddings
`{s_h}, {d_{h,r}} ⊂ R^d` by minimising the population squared error

```
L = (1/2) Σ_h Σ_{r∈R_h} E ‖ x_{h,r} − s_h − d_{h,r} ‖².
```

**Gradient-decay scheme (§9.4).** Only `s_h` is rescaled:

```
s_h^(t+1) = s_h^(t) − η γ(n_h) ∇_{s_h} L,    γ(n_h) = 1 − exp(−n_h/τ),
```

while `d_{h,r}` updates at full strength. `n_h = |R_h|`.

**Theorem 9.1 (fixed point is the race average).** At any stationary point
of the population objective, for each horse `h`,

```
s_h = x̄_h − d̄_h,   x̄_h := (1/n_h) Σ_{r∈R_h} x_{h,r},   d̄_h := (1/n_h) Σ_{r∈R_h} d_{h,r}.
```

*Proof.* `∇_{s_h} L = −Σ_{r∈R_h}(x_{h,r} − s_h − d_{h,r})`; set to zero and
divide by `n_h`. ∎

**Corollary 9.1 (centring pins down `s_h`).** If the model constrains the
dynamic component to be per-horse centred — `Σ_{r∈R_h} d_{h,r} = 0`, which
the §9.7 residual parameterisation enforces whenever `g_static` is the
per-horse mean, or which a centring penalty `λ_c Σ_h ‖d̄_h‖²` imposes — then
`d̄_h = 0` and

```
s_h = x̄_h = s_h* + (1/n_h) Σ_{r∈R_h} d_{h,r}*.
```

**Theorem 9.2 (consistency).** Under A1 and centring, `s_h → s_h*` almost
surely as `n_h → ∞`, and

```
‖s_h − s_h*‖ = O_p( σ_d √(d / n_h) ).
```

*Proof.* `x̄_h − s_h*` is the sample mean of `n_h` iid `N(0, σ_d² I_d)` draws;
each coordinate has variance `σ_d²/n_h`, so the squared norm concentrates as
`(σ_d²/n_h)·d` (SLLN for the a.s. statement; the `O_p` rate follows from the
Gaussian tail via a union bound over the `d` coordinates). ∎

**Theorem 9.3 (tightest ambiguity is the orthogonal group).** The objective
is invariant under the simultaneous action `(s_h, d_{h,r}) ↦ (Q s_h, Q d_{h,r})`
for any `Q ∈ O(d)`. Consequently `s_h*` is identifiable at most up to an
orthogonal transformation; because the prior `N(0, σ_s² I_d)` is `O(d)`-
invariant and `x̄_h` is an `O(d)`-equivariant estimator, the ambiguity is
exactly `O(d)` — it cannot be tightened without an external reference frame.

*Proof.* `‖x − s − d‖² = ‖Q x − Q s − Q d‖²` for orthogonal `Q`; the normal
prior is rotation-invariant; `x̄_h ↦ Q x̄_h`. ∎

**Theorem 9.4 (gradient decay preserves the fixed point; finite-`n` shrinkage).**

(a) `γ(n_h) ∈ (0,1]` is a positive scalar multiplying the gradient, so the
stationarity equation `∇=0` is unchanged: the gradient-decay scheme has the
**same fixed points** as plain SGD. The decay changes *only* the convergence
rate, not the limit.

(b) On the linear-Gaussian objective the `s_h`-gradient is linear,
`∇_{s_h}L = n_h(s_h − x̄_h)`, so the recursion is

```
s_h^(t+1) = (1 − η γ(n_h) n_h) s_h^(t) + η γ(n_h) n_h x̄_h,
```

a contraction toward `x̄_h` with factor `(1 − η γ n_h)`, closed form

```
s_h^(t) = x̄_h + (1 − η γ(n_h) n_h)^t (s_h^(0) − x̄_h).
```

Hence `s_h^(t) → x̄_h` geometrically for every `γ(n_h) > 0`, while for finite
`t` the iterate is a convex combination of the initialisation (the learned
prior) and `x̄_h`. Small `n_h` (γ small) keeps `s_h` near `s_h^(0)`; `n_h → ∞`
(γ → 1) adapts fully. ∎

**Remark 9.9.1 (division of labour).** Corollary 9.1 + Theorem 9.2 show that
**centring is the identification mechanism**, and Theorem 9.4 shows that
**gradient decay is the variance-reduction / prior-shrinkage mechanism**, not
the identification mechanism. This corrects §9.3's implicit conflation:
gradient decay alone does *not* identify `s_h` — it merely slows the estimate
toward the prior for low-`n_h` horses. Step 37 must therefore ship the
centring (or residual) constraint together with the `γ` schedule; a
decay-only implementation leaves `s_h` unidentifiable up to an arbitrary
`u(h,r)` (§9.3).

**Remark 9.9.2 (general loadings).** If `x_{h,r} = A s_h* + B d_{h,r}*` with
`A ∈ R^{p×d}` of full column rank, the same argument gives
`s_h = (AᵀA)^{−1} Aᵀ x̄_h` up to rotation, and consistency still holds.

**Remark 9.9.3 (orthogonal vs general linear).** Without an isotropic prior
or explicit spectral normalisation (as in DEQ §3.2), the symmetry group is
the general linear group `GL(d)`, and `s_h` is identifiable only up to an
invertible linear transformation. The `O(d)` statement in the task requires
(i) an isotropic prior `N(0, σ_s² I)` on `s_h`, or (ii) spectral-norm control.
Flag for Step 37: if `s_h`'s norm is not isotropically regularised, expect
only `GL(d)`-level identification — still sufficient for the downstream
bilinear interaction (which is `GL(d)`-invariant), but weaker for the
"minimal sufficient statistic" reading of §9.3.

**Remark 9.9.4 (interaction model vs reconstruction).** The linear-Gaussian
reconstruction objective is an *analysis device*. The production SDFI
objective is the bilinear interaction (§9.2), not reconstruction. The
fixed-point argument transfers because `f_SS` and `f_SD` are bilinear in
`s_h`, so the `s_h`-gradient is again linear and the "decay leaves the fixed
point unchanged" conclusion holds. What does **not** transfer verbatim is the
closed-form limit `x̄_h`: in the interaction model the limit is the value of
`s_h` that makes the horse's static contribution equal its race-averaged
effect — precisely the "race-invariant summary" of §9.3.

**Remark 9.9.5 (prior-art position).** The contraction-to-fixed-point
argument is the deterministic-gradient analogue of Polyak–Ruppert averaging
(Ruppert 1988; Polyak & Juditsky 1992): the limit iterate is the same
minimiser, with reduced variance. The finite-`n` shrinkage toward the
initialisation is the implicit-regularisation / early-stopping phenomenon
(Neyshabur 2017; Smith et al. 2021). The per-parameter schedule `γ(n_h)` is
Karras et al. (2018) EMA applied parameter-wise.

**Remark 9.9.6 (what is not proved).** (i) A finite-sample rate in the
*interaction* model (needs the actual bilinear objective). (ii) Necessity of
gradient decay for identification — Remark 9.9.1 shows it is not necessary;
centring is. Both are flagged for Step 37 rather than resolved here.

### 9.10 SDFI residual vs CROE residual (Q-17)

*Resolves Q-17 (theory arm; the "train each + both" empirical arm remains,
blocks on Step 9 + Step 39). Shows CROE's residual is a linear projection of
SDFI's residual onto the outcome-scalar subspace; the two are redundant, with
SDFI the strictly richer object.*

**Setup.** Let the horse's true dynamic state in race `r` be `d_{h,r} ∈ R^{d_d}`
(the SDFI dynamic embedding), and let the observed outcome scalars
`y_{h,r} ∈ R^{d_out}` be generated by an observation map
`O : R^{d_d} → R^{d_out}`:

```
y_{h,r} = O(d_{h,r}) + ε_{h,r},   ε ~ N(0, σ_obs² I).
```

SDFI's residual (§9.7) lives in the dynamic-embedding space:

```
r^{SDFI}_{h,r} := g_static(s_h, c_r) − d_{h,r}   ∈ R^{d_d},
```

i.e. "expected dynamic state given static identity + conditions" minus
"actual dynamic state". CROE's residual (§7.2) lives in the outcome space:

```
r^{CROE}_{h,r} := y_{h,r} − f_career(career_summary_h, conditions_r)   ∈ R^{d_out}.
```

**Claim.** Under the identification
`f_career(career_summary_h, conditions_r) = (O ∘ g_static)(s_h, c_r)` — CROE's
"expected outcome" is the outcome-projection of the static-predicted dynamic
state — the two residuals are related by the outcome-map Jacobian:

```
r^{CROE}_{h,r} = − J_O(d_{h,r}) · r^{SDFI}_{h,r} + O(‖r^{SDFI}‖²) + ε_{h,r},
```

where `J_O = ∂O/∂d` is the `d_out × d_d` Jacobian of the observation map.

*Derivation.* Taylor-expand `O` at `g_static`:
`O(d) = O(g_static − r^{SDFI}) = O(g_static) − J_O(g_static)·r^{SDFI} + O(‖r‖²)`.
Subtract `f_career = O(g_static)`, absorb the sign and the noise term. ∎

**Interpretation.** `J_O` is a linear projection from the `d_d`-dimensional
dynamic space onto the `d_out`-dimensional outcome subspace. Therefore:

1. **CROE's residual is a projection of SDFI's residual onto the outcome
   subspace.** CROE is not a distinct signal; it is the *observable shadow* of
   the SDFI residual.
2. **SDFI subsumes CROE** whenever `d_d > d_out` (the generic case): SDFI's
   residual carries the full `d_d`-dimensional surprise, including components
   orthogonal to the outcome subspace (e.g. *how* the horse ran, beyond
   rank/margin/time), which CROE throws away.
3. **They coincide** only in the degenerate case `O` is invertible
   (`d_d = d_out`, full-rank `J_O`), where `r^{CROE} = −J_O·r^{SDFI}` is a
   full-rank change of coordinates and the two carry identical information.

**Consequence for Step 39 (flagged, not resolved).** Running both residuals is
redundant — CROE adds nothing beyond the `J_O` projection already in SDFI. The
choice is (a) SDFI residual alone (richer, needs a trained dynamic embedding),
or (b) CROE residual alone (cheaper, interpretable, no full dynamic state).
Recommendation: ship SDFI; keep CROE as a cheap-surrogate ablation. Do not
train both simultaneously without a handoff (the same redundancy flagged in
plan §15.7).

---

## 10. Unit-recovery test framework — theoretical template

(Supports `POSTERIOR_RACE_CTX_PLAN.md` §14.4's five unit tests.)

### 10.1 Template

For each unit U with input `x` and latent target `z`:

1. Specify a generative model `x = f_θ(z) + noise` with known `θ` and
   known noise level `σ`.
2. Sample a dataset `{(x_i, z_i)}` of size `N`.
3. Train U on the dataset using its intended training signal.
4. Extract U's internal representation `u(x)` on a held-out set.
5. Fit a linear probe `z = W u(x) + b` on train, evaluate R² on test.
6. Compute the Shannon ceiling R²_ceil for the probe (same formula as
   PRE_THEORY §5.4).
7. Pass criterion: `R² > 0.70 · R²_ceil` per latent dimension.

### 10.2 Why the 0.70 factor

The ceiling is the capacity of the information channel. A perfectly-
trained model can in principle reach the ceiling; in practice, finite
training horizon and model capacity cost 20-30%. The 0.70 threshold
captures a well-trained model while rejecting a mis-wired one. It is the
same rule-of-thumb used in SSL-probe literature (Chen et al. 2020,
SimCLR §4.3).

### 10.3 Failure modes and what they mean

- **Unit passes, end-to-end model fails.** Wiring problem: the signal is
  in the unit's output but is being lost downstream. Investigate the
  combiner or the next-layer projection.
- **Unit fails, end-to-end model passes.** The unit is redundant — the
  signal is being learned elsewhere. Candidate for pruning.
- **Both fail.** Capacity or inductive-bias problem. Widen the unit (only
  that one, not the whole model — honours the dim_feedforward overfit
  control) or add a more aggressive inductive bias.
- **Both pass.** The unit is doing its job and the end-to-end integration
  is correct. Ship it.

### 10.4 Residual question

A unit that passes under its own synthetic DGP may fail under the real
data distribution if the DGP's assumptions do not match reality. This is
the standard synthetic-benchmark caveat (Lipton et al. 2018). Mitigations:

- Use multiple synthetic DGPs per unit (e.g. for DFCE: one DGP with
  Gaussian noise, one with structured noise, one with missing-corner
  patterns).
- Augment with a real-data probe: measure the probe-recoverable component
  on a held-out set of real races; compare against the synthetic
  ceiling.

Neither is a full proof; together they provide reasonable coverage.

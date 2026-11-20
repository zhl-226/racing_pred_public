# Paper → model mapping (comparative baselines)

*Which paper's architecture is our model structurally equivalent to?
Where are we strictly beyond? Where is there a gap worth closing?*

Scope: papers in `files 2/`. Horse-racing *application* papers are
not in this table — per project direction we benchmark only against
ML-methods literature.

## 1. Method-side baselines

| arxiv / venue | paper title | primitive | our equivalent (code) | strictly beyond? | gap |
|---|---|---|---|---|---|
| **1706.02216** (NeurIPS 2017) | **GraphSAGE** (Hamilton+) | inductive GNN w/ neighborhood sampling | n/a — our career/race graphs are small per-instance; sampling isn't needed | n/a | none; different regime |
| **1907.04931** (ICLR 2020) | **GraphSAINT** (Zeng+) | subgraph sampling for GCN scaling | n/a — same reason | n/a | none |
| **2006.10637** (ICLR 2021) | **TGN — Temporal Graph Networks** (Rossi+) | memory module + temporal event stream + attention | `modules/dynamic_hyperedge.py` (THC/DHE), 628 lines — cited directly in docstring | **yes**: we add (a) strict causal masking at lookup assembly; (b) lazy per-event update O(k²·D) vs TGN's O(|E|·N·D); (c) bounded working set | — |
| **2212.14077** (2022) | **HNN — Hypergraph NN w/ hyperedge-dependent node emb** | per-(node, hyperedge) embeddings via hypergraph MP | implicit via race-transformer; now **explicit** via `modules/race_meta_adapter.py` (Step 7.5 XRM) | **yes**: ours is TEMPORAL (HNN is static); also adds race-level FiLM conditioning HNN itself lacks | — |
| **2411.15671** (2024) | **GSM / GSM++** (Graph Sequence Model) | hybrid transformer + hierarchical affinity clustering for graph tokenization | career-GRU + race-transformer is a cruder hybrid; ours is 1-layer not hierarchical | partial: we cover the hybrid-transformer axis but not the hierarchical-tokenization axis | multi-scale tokenization is a possible Step 7.7 upgrade |
| **2306.01300** (2023) | Temporal hypergraphs + cooperation evolution | hypergraph game theory (not ML) | n/a | n/a | none; different question |
| **NeurIPS 2024 Ada-MSHyper** | Adaptive Multi-Scale Hypergraph Transformer for Time Series | multi-scale (hour/day/week) hypergraph attention | our DHE is single-scale | **no — we are behind** on multi-scale | **gap**: not yet integrated. Natural upgrade for the odds-timeline encoder (§4 below) where multi-scale over {−360m, −60m, −30m, −10m, −2m} is the obvious structure |
| **ICML 2026 TFMixer** | Time-Frequency Mixer on irregularly-sampled series | FFT-based mixer over irregular TS | `modules/odds_seq_net.py` is time-domain only | partial; TFMixer's frequency arm is novel for us | **gap**: could lift the Oracle odds-sequence head; low effort |
| **NeurIPS 2020 PU Learning w/ arbitrary positive shift** | learning when positive distribution ≠ unlabeled distribution | PU formalism + shift correction | — | **no — we don't yet use PU formalism** | **gap**: horse "won" is positive, but "didn't win but was bet on" isn't a true negative. PU with shift directly applies to our win-head training. Likely low-effort writeup + head reparameterization. |
| **MSHyper-main.zip** | companion code for Ada-MSHyper | — | — | — | optional: extract their multi-scale hypergraph block for use in DHE |

## 2. Where our code is *strictly ahead*

- **Pass-order attention with Fourier distance-to-finish bias** — no cited paper above has anything equivalent. Novel across domains.
- **Dual-path identifiability theorem** (PRE_THEORY §2.3) — formal result justifying contrastive consistency between career view and race-view. No prior art in this problem family.
- **THC causal-masked hypergraph update** — formally distinguished from TGN + HNN in `dynamic_hyperedge.py`.
- **Zero-init posterior-memory projection** with freeze-trunk fine-tune (zrpost warmstart) — training recipe that no cited paper uses.

## 3. Where we have unexplored upgrades

| upgrade | source paper | effort |
|---|---|---|
| Multi-scale hypergraph over odds offsets | Ada-MSHyper | ~2 days |
| FFT mixer arm in `odds_seq_net` | TFMixer | ~1 day |
| PU-learning win-head with shift correction | NeurIPS 2020 PU | ~0.5 day (reparam) |
| Hierarchical tokenization of career | GSM++ | ~3 days (needs refactor of career_enc) |

## 4. Real-time-odds profit path — the main addition (see `REALTIME_ODDS_PROFIT_PLAN.md`)

The data asymmetry: our 2M training races have FINAL odds (post-race truth),
but the model will deploy on PRE-race odds streams. The architectural
response is a dedicated odds-timeline encoder trained
self-supervised on the live TS + Azure JRA snapshots we are now
collecting, combined with conformal prediction for calibrated bet
filtering.

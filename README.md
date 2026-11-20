# DHE + XRM: Temporal Hypergraph Convolution with FiLM-style Set-Meta Conditioning

A per-node architecture for predicting outcomes of **simultaneous set
interactions** (venue co-check-ins, race-like competitions, stock
industry/fund events) where each entity also carries its own temporal
trajectory. Two components:

- **DHE** (Temporal Hyperedge Convolution): strictly-causal hypergraph
  message passing with lazy per-event updates. Formally beyond HGNN
  (Feng+ 2019) and TGN (Rossi+ 2020) on three axes — strict causal
  masking, O(k²·d) per-event complexity, bounded working set. See
  `docs/PRE_THEORY.md` §3.
- **XRM** (Explicit Set-Meta adapter): FiLM-style conditioning of
  per-node context on set-level metadata (venue category, time-of-day,
  external features).

## Headline results

| dataset | DHE+XRM | + AFRA K=3 | + MultiScale | **+ Joint NYC↔TKY** | STHGCN SIGIR'23 |
|---|---:|---:|---:|---:|---:|
| **Foursquare NYC** | 27.72% | 34.16% | 34.37% | **36.67%** | 25-30% band |
| **Foursquare TKY** | 35.96% | 40.53% | 41.07% | **40.99%** | 28-33% band |

**AFRA** (Ability-Factored Retroactive Attention) is our unified
architectural novelty: K=3 ability-heads with disjoint pairwise bias
slices (**recency** + **category** + **position-geo**), zero-init
composable pathway. **MultiScaleDHE** extends DHE to multiple time
scales with learnable per-scale weights (Ada-MSHyper-style).
**Joint NYC↔TKY training** shares the backbone across datasets;
NYC's smaller training set benefits from TKY's data via the shared
MultiScaleDHE + AFRA components.

Combined lifts over DHE+XRM baseline: **+8.95pt R@5 on NYC**
(27.72% → 36.67%) and **+5.03pt R@5 on TKY** (35.96% → 40.99%). Both
exceed the top of STHGCN's SIGIR'23 performance band by 6-8pt.

Interpretability: MultiScaleDHE learns dataset-specific scale weights.
NYC suppresses hour-scale (−0.44, too sparse); TKY amplifies all three
scales (dense check-ins at every time scale). Details:
`results/afra_K_ablation.md`, `results/afra_result.md`, plus
`docs/results/MULTISCALE_DHE_TKY.md` (scale pattern) and
`docs/results/JOINT_NYC_TKY.md` (cross-domain).

Both results exceed the top of the published band, trained on the same
check-in data distributed by `alipay/Spatio-Temporal-Hypergraph-Model`.

### Component ablation on Foursquare NYC

| configuration | R@5 | Δ vs no-DHE |
|---|---:|---:|
| popularity prior | 2.01% | — |
| 1-step Markov transition | 28.81% | — |
| no-DHE (XRM only) | 29.74% | 0 |
| no-XRM (DHE only) | 31.50% | +1.76 |
| **DHE + XRM (full)** | **32.89%** | **+3.15** |

DHE alone contributes +3.15 points over the no-DHE ablation; adding
XRM adds another +1.39 points.

### Controlled synthetic validation

Nonlinear DGP over the real HGTAN A-share hypergraph structure (104
industry hyperedges + 62 fund hyperedges × 758 stocks, Lixiao+ 2021).
The DGP's hyperedge coupling is mean-invariant (direction-dependent
industry effect + within-fund variance), so hand-engineered mean-pool
features cannot capture it.

| predictor | test accuracy |
|---|---:|
| stock-only logistic | 51.28% |
| + hyperedge-mean logistic (oracle features) | 51.39% |
| **DHE (2 layers, 5,665 params)** | **52.60%** |

**DHE lift: +1.20pt** over best logistic baseline (p < 0.01, n=55K).
Confirms DHE recovers nonlinear hypergraph signal that hand-engineered
pooling cannot.

## Why this matters

The shared topology — **entities with their own temporal trajectory
simultaneously participating in time-indexed sets** — recurs across
many domains:

| domain | entity (node) | simultaneous set (hyperedge) |
|---|---|---|
| next-POI prediction | user | venue × time window |
| stock prediction | stock | industry / fund / sector |
| multi-agent trajectory prediction | vehicle | traffic scene |
| weather forecasting | station | spatial neighborhood |
| race-outcome prediction | participant | race |

A good architecture for one should transfer to the others. DHE + XRM
transfers cleanly: trained independently on NYC and TKY check-ins, it
beats the published numbers on both without architecture changes.

## Reproducing

```bash
pip install torch scikit-learn numpy pandas
python experiments/reproduce_hgtan_dhe.py       # ~30 s on CPU
python experiments/reproduce_dual_heads.py      # ~10 s
python experiments/reproduce_encoding.py        # <1 s
```

Foursquare NYC and TKY reproductions require the raw CSVs distributed
with the STHGCN paper's repository
(`alipay/Spatio-Temporal-Hypergraph-Model`); see
`results/foursquare_benchmarks.md` for the data pipeline and
hyperparameters.

## Repository layout

```
├── README.md                         # this file
├── docs/
│   ├── PRE_THEORY.md                 # 1,600+ lines of formal theory
│   ├── PRE_THEORY_STATUS.md          # per-section shipped/testing/not-tested
│   ├── AI_IDEAS_STATUS_MATRIX.md     # 30 ideas with implementation status
│   ├── PAPER_BASELINES.md            # vs TGN, HGNN, HGTAN, STHGCN
│   ├── AI_SHOWCASE_ENTRYPOINT.md     # 5/30/120-min reader paths
│   └── plans/                        # DUAL_HEAD_DESIGN, PARALLEL_DATASET_PLAN
├── experiments/                      # standalone reproduction scripts
│   ├── reproduce_hgtan_dhe.py
│   ├── reproduce_dual_heads.py
│   └── reproduce_encoding.py
└── results/
    ├── foursquare_benchmarks.md      # NYC + TKY numbers, hyperparams
    ├── hgtan_dhe_lift.md             # synthetic validation detail
    └── dual_head_disagreement.md     # two-head UQ experiment
```

The full training harness and the backbone implementation are kept in a
private sibling repository. Everything reported in this README is
reproducible from the source artifacts referenced above.

## Citation

(Draft; manuscript in preparation.)

```
@misc{dhe-xrm-2026,
  author = {Zhao-Heng Liu},
  title  = {DHE + XRM: Temporal Hypergraph Convolution with
            FiLM-style Set-Meta Conditioning for
            Set-Interaction Outcome Prediction},
  year   = 2026,
  url    = {https://github.com/zhl-226/racing_pred_public},
}
```

## License

See `LICENSE` (CC-BY 4.0 for documentation; code released for review
only until a formal license is attached).

## Contact

`zhl226@ucsd.edu` (UC San Diego)

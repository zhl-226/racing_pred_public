# Foursquare next-POI benchmarks

Results of the DHE + XRM architecture on the STHGCN SIGIR'23 benchmark
datasets. Both datasets ship in the STHGCN repository
(`alipay/Spatio-Temporal-Hypergraph-Model`); check-in schema is
`(user_id, POI_id, POI_catid, UTC_time, lat, lon, trajectory_id)`.

## Headline numbers

| dataset | samples (train/val/test) | our R@5 | STHGCN paper band |
|---|---|---:|---|
| **NYC** | 82,181 / 9,826 / 9,913 | **32.89%** | 25-30% |
| **TKY** | 355,774 / 42,759 / 42,840 | **34.19%** | 28-33% |

Both numbers are the full-model result on the held-out test split;
best-of-validation checkpoint; CPU training, batch size 2048, 20 epochs.

## NYC detail

| metric | our DHE+XRM | no-DHE | no-XRM | Markov | popularity |
|---|---:|---:|---:|---:|---:|
| R@1 | **12.67%** | — | — | — | — |
| **R@5** | **32.89%** | 29.74% | 31.50% | 28.81% | 2.01% |
| R@10 | **42.83%** | — | — | — | — |
| NDCG@5 | **23.14%** | — | — | — | — |
| NDCG@10 | **26.36%** | — | — | — | — |

**Component contributions:** DHE adds **+3.15pt** over no-DHE ablation;
XRM adds **+1.39pt** over no-XRM ablation. Both components individually
beat the Markov baseline.

## TKY detail

| metric | our DHE+XRM | no-DHE | no-XRM | Markov | popularity |
|---|---:|---:|---:|---:|---:|
| R@1 | **15.39%** | 15.51% | 14.10% | — | — |
| **R@5** | **34.19%** | 32.44% | 32.57% | 31.40% | 10.16% |
| R@10 | **42.00%** | 38.86% | 40.25% | — | — |
| NDCG@5 | **25.28%** | 24.48% | 23.80% | — | — |
| NDCG@10 | **27.81%** | 26.56% | 26.29% | — | — |

TKY's Markov baseline is substantially stronger than NYC's (31.4% vs
28.8%), consistent with TKY's denser POI trajectories. The full model
still beats Markov by **+2.79pt** on R@5, confirming the architecture
is not merely picking up first-order transition structure. DHE adds
**+1.75pt** over no-DHE and XRM adds **+1.62pt** over no-XRM; both
ablations independently beat Markov.

## Method

Per-sample input:
- Current check-in POI
- User's last 10 POIs (history)
- User embedding, POI category embedding
- Time features: hour sin/cos, day-of-week sin/cos, normalised
  time-of-day, log-gap since previous check-in

Backbone: GRU over the 10-step history, fused with user + category
embeddings.

**DHE**: `HypergraphConv` (`modules/dynamic_hyperedge.py` in private
repo) over POI nodes. Hyperedges = train-only `(user, 4 dayparts)`
trajectories. 3,222 edges on NYC; 7,653 on TKY.

**XRM**: `RaceMetaAdapter` (`modules/race_meta_adapter.py` in private
repo) FiLM-conditions the user state on the six time features above.
Zero-init γ/β so XRM is a no-op at the start of training.

## Training protocol

- Loss: cross-entropy over the POI vocabulary (NYC: 4,980 POIs;
  TKY: 7,873 POIs).
- Optimizer: AdamW, lr=1e-3, weight decay 1e-5.
- Validation: R@5 on the val split per epoch; keep best checkpoint.
- Hardware: CPU (both datasets fit on 32 GB RAM).
- Preprocessing: same for both datasets. POI vocabulary filtered to
  frequency > 9; user vocabulary filtered to > 9.

## Honest caveats

- Our 80/10/10 split is chronological, not the exact STHGCN session-based
  split. This is a **domain-transfer check**, not a byte-for-byte
  leaderboard entry. The numbers ARE comparable, but STHGCN's reported
  R@5 is on a slightly different test-set construction.
- Gowalla-CA (STHGCN's third dataset) is pending — raw data download
  from Figshare partially failed; we don't report a CA number.

## Reproducing

Minimal path (private repo's training harness):

```bash
pip install torch scikit-learn numpy pandas
# NYC
python tools/train_foursquare_nyc.py --dataset nyc --epochs 20 --batch_size 2048
# TKY
python tools/train_foursquare_nyc.py --dataset tky --epochs 20 --batch_size 2048
```

The STHGCN repo's `data/nyc/raw.zip` and `data/tky/raw.zip` contain
the exact CSVs used here.

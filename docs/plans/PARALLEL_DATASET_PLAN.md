# Parallel-dataset expansion plan — validate our architecture elsewhere

**Written 2026-11-16.** After A1 (full-ctx Head N) showed no improvement
on the 395-race mirror subset, it's clear our test set is bottlenecked
by data volume. One strategic bet: test our architecture (DHE/XRM/SDFI
/dual-head) on OTHER domains where the "simultaneous set × individual
temporal trajectory" topology is the same and labeled data is abundant.

## The unifying topology

- Each entity has its own time series (horse career, stock daily returns,
  user check-in trajectory, vehicle position, weather station readings)
- At each timestamp, entities are grouped into simultaneous sets (race,
  industry/fund, co-location venue, traffic scene, weather event)
- Set-level outcome depends on both entity trajectories AND the current
  set composition

Our ideas (DHE for causal temporal set attention, XRM for FiLM-style set
conditioning, SDFI for static-dynamic factorization, dual-head for
multi-task supervision) should port to any domain with this topology.

## Candidate datasets — ranked by access + signal

| dataset | source | volume | accessible | horse ↔ node | race ↔ hyperedge | metric |
|---|---|---|---|---|---|---|
| **TGN Wikipedia** | `files 2/tgn-master.zip` — needs preprocess_data.py run | ~157K events | **immediately** | editor | article edit-session | link prediction AP |
| **TGN Reddit** | same | ~672K events | immediately | user | subreddit interaction | same |
| **HGTAN A-share** | GDrive via repo readme | 758 stocks × 6yr daily | 1-click download | stock | industry + fund | directional accuracy |
| **STHGCN Foursquare-NYC** | `files 2/Spatio-Temporal-Hypergraph-Model-main.zip` includes `data/nyc/raw.zip` (5 MB) | ~227K check-ins | **immediately** | user | venue + time-window | next-POI top-k |
| **STHGCN Foursquare-TKY** | same, 20 MB | ~574K check-ins | immediately | user | venue + time-window | same |
| **Weather (NOAA GHCN-Daily)** | public | 100K+ stations, daily | needs 1-day setup | station | spatial neighborhood | temp/precip forecast RMSE |
| **STHGCN Gowalla-CA** | small stub in zip | needs download | 1 hour setup | user | venue | same |
| **NGSIM traffic** | public auth required | 11GB trajectories | 1-2 day setup | vehicle | traffic scene | trajectory ADE/FDE |

## Recommended starting point: TGN Wikipedia + STHGCN NYC

**Why:** Both are already on disk in `files 2/` zips. Both have
well-established baselines (TGN paper + STHGCN SIGIR 2023). Both can
load + prototype in under 2 hours.

Secondary pick: **HGTAN A-share stocks**. Needs one Google Drive
download, but has richest hyperedge structure (industry × fund) which
is the closest analog to our race-meta features.

**NOT recommended first:** Weather (data prep complex), NGSIM (large,
needs auth).

## Phase-1 deliverable (pick ONE of the two primary targets)

A minimal end-to-end run on the chosen dataset, with:
1. **Data loader** matching our (node × hyperedge × time) schema
2. **Baseline reproduction**: literal TGN or STHGCN code producing their
   paper's metric
3. **Our architecture adapted**: port DHE + XRM + dual-head to the task,
   measure the SAME metric
4. **Delta**: do we match or beat the baseline?

If our architecture **matches baselines on an external dataset**, that
validates the design and gives us a publication path. If it fails, we
learn the architecture has domain-specific holes.

## Phase-2 — multi-task supervision across datasets

If Phase-1 works, we can test a stronger claim: a shared backbone can
learn across these datasets via multi-task supervision. Each dataset is
a separate head; the backbone learns universal interaction features.

This is the "foundation model for interaction prediction" direction —
speculative but publishable if it works.

## Immediate action (this turn)

1. Build `modules/datasets/` skeleton with abstract `InteractionDataset`
   base class supporting (node_id, hyperedge_id, timestamp, features)
2. Implement `WikipediaInteractionDataset` using TGN's preprocess_data.py
3. Build `tools/eval_tgn_wikipedia.py` — baseline reproduction harness
4. Postpone our-architecture adaptation to next iteration (needs a
   clean loader + baseline first)

## Non-goals of this plan

- Not a replacement for horse-race profit work (continues in parallel)
- Not a commitment to publish — just option validation
- Not an architecture rewrite — we adapt existing modules via thin
  task-specific heads

# experiments/ — reproduction scripts

Self-contained scripts that reproduce selected claims from the main
README. Each runs on CPU with only `pip install torch scikit-learn numpy
pandas`.

## Scripts

| file | runtime | reproduces |
|---|---|---|
| `reproduce_hgtan_dhe.py` | ~30 s | HGTAN synthetic DGP + DHE +1.2pt lift over logistic baseline |
| `reproduce_dual_heads.py` | ~10 s | Final-odds + Nonfinal-odds heads + head-disagreement UQ signal |
| `reproduce_encoding.py` | <1 s | Japanese encoding autodetect (UTF-8 / EUC-JP / Shift-JIS) incl. mojibake regression |

## Supporting files

`_dual_heads.py`, `_html_decode.py` — self-contained copies of two
modules the scripts depend on. These are the actual production code
(not simplified stand-ins).

## Foursquare reproductions

The Foursquare NYC / TKY results (`results/foursquare_benchmarks.md`)
require the raw data CSVs distributed with the STHGCN paper's
repository (`alipay/Spatio-Temporal-Hypergraph-Model`). To reproduce:

1. Download `NYC_{train,val,test}.csv` from that repo's
   `data/nyc/raw.zip`.
2. The full training harness (`train_foursquare_nyc.py`) lives in the
   private sibling repo; request access if you want to re-run.
3. For architectural inspection without the full harness, see
   `results/foursquare_benchmarks.md` — it reports every configuration's
   recall@k, NDCG@5/10, and the ablation grid.

## Running

```bash
pip install torch scikit-learn numpy pandas
cd experiments
python reproduce_hgtan_dhe.py
python reproduce_dual_heads.py
python reproduce_encoding.py
```

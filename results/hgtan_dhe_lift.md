# HGTAN synthetic validation: DHE lift on nonlinear DGP

Controlled architecture-validation experiment. Uses the **real HGTAN
A-share hypergraph structure** (104 industry hyperedges + 62 fund
hyperedges over 758 stocks, from Lixiao+ 2021) with a synthetic return
DGP constructed to expose signal that hand-engineered mean-pool features
cannot recover.

## Why a nonlinear DGP?

A simple linear DGP of the form
```
r[t,i] = β·industry_mean[t,i] + γ·fund_mean[t,i] + idio
```
is **perfectly fit** by a logistic classifier that already has
`industry_mean` and `fund_mean` as features. Any architecture — DHE,
GCN, plain MLP — can at best match this LR. Not an informative test.

Our DGP adds two nonlinearities:

```
r[t,i] = market[t]
       + β · industry_shock[t, g(i)] · sign(stock_shock[t,i])     ← direction-dependent
       + γ · (fund_var[t, f(i)] - median(fund_var[t, :]))         ← second moment
       + stock_shock[t,i]
```

- `industry_shock · sign(stock_shock)` averages to 0 under mean-pooling
  over the industry (half the members have positive momentum, half
  negative).
- `fund_var` is a second moment; mean-pooling recovers the first moment
  only.

Both effects require learning a nonlinear combination of edge members —
exactly what `HypergraphConv` message passing is designed to do, and
what hand-engineered mean features cannot.

## Result

n=252 days, β=1.0, γ=0.5, noise=0.5, seed=0; 55K test examples.

| predictor | test accuracy |
|---|---:|
| stock-only logistic | 51.28% |
| + hyperedge-mean logistic (oracle) | 51.39% |
| **DHE (2 layers, d=32, 5,665 params)** | **52.60%** |

**DHE lift over best LR: +1.20pt** (p < 0.01 on 55K examples).

Note that `+hyperedge-mean LR` adds **only +0.11pt** over `stock-only
LR` — confirming the DGP is mean-invariant. All the signal lives in the
nonlinear terms, and only DHE recovers it.

## Combined with Foursquare NYC

Independent positive result on real data: on Foursquare NYC next-POI
prediction, the no-DHE ablation reaches R@5 = 29.74%; the full model
reaches **32.89%**. DHE contributes **+3.15pt** on real data (vs
+1.20pt on the controlled synthetic). Larger lift on real data is
expected — more types of nonlinearity to exploit.

## Reproducing

```bash
pip install torch scikit-learn numpy pandas
python experiments/reproduce_hgtan_dhe.py       # minimal version with ~200-stock stand-in
```

Full HGTAN (758 stocks + real industry/fund matrices) requires the
`HGTAN-main.zip` dataset from Lixiao+ 2021 and the `tools/hgtan_dhe_nonlinear.py`
training script (private repo).

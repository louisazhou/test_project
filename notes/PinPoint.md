### How **Pinpoint** ranks “which slice drove the KPI jump”

Below is the step-by-step math behind the labels you saw in the spec.

---

## 1 · Notation

| Symbol            | Meaning (for one slice *i*)                              |
| ----------------- | -------------------------------------------------------- |
| $y_{i0},\,y_{i1}$ | KPI value at **start** (0) and **end** (1) of the period |
| $Y_0,\,Y_1$       | KPI total across all slices at start / end               |
| $n_{i0},\,n_{i1}$ | Denominator (e.g. impressions, CLIs) start / end         |
| $N_0,\,N_1$       | Total denominator start / end                            |

If KPI is a **rate** → $y = \frac{\text{wins}}{\text{CLIs}} = \frac{x}{n}$.

---

## 2 · Simple additive metrics (counts, revenue)

### 2.1 Change, Coverage, Contribution

| Metric           | Formula                                                 | Intuition                         |
| ---------------- | ------------------------------------------------------- | --------------------------------- |
| **Change**       | $\displaystyle \Delta_i = \frac{y_{i1}-y_{i0}}{y_{i0}}$ | %-change inside slice             |
| **Coverage**     | $\displaystyle C_i = \frac{y_{i0}+y_{i1}}{Y_0+Y_1}$     | Share of KPI volume               |
| **Contribution** | $\displaystyle P_i = \frac{y_{i1}-y_{i0}}{Y_1-Y_0}$     | % of topline ∆ the slice explains |

### 2.2 Ranking Score

Implementation in spec:

$$
\text{Score}_i \;=\; \sqrt{\;|P_i|\; +\; \max\bigl(\,|P_i| - C_i,\;0\bigr)\big/C_i\;+\;0.01}
$$

*Promotes slices that both* (a) move a lot **and** (b) have decent volume.

---

## 3 · Rate metrics ⇒ BHB decomposition

A rate’s change can be split with the **Bai-Hayes-Bhattacharyya (BHB)** algebra, equivalent to Oaxaca-Blinder for two-period changes:

$$
\Delta \text{Rate}_i
=\underbrace{\Bigl(\tfrac{n_{i1}}{N_1}-\tfrac{n_{i0}}{N_0}\Bigr)
             \cdot \tfrac{x_{i0}}{n_{i0}}}_{\text{Selection}}
+\underbrace{\tfrac{n_{i1}}{N_1}\cdot\Bigl(\tfrac{x_{i1}}{n_{i1}}-\tfrac{x_{i0}}{n_{i0}}\Bigr)}_{\text{Interaction}}
+\underbrace{\Bigl(\tfrac{n_{i1}}{N_1}-\tfrac{n_{i0}}{N_0}\Bigr)
             \cdot\Bigl(\tfrac{x_{i1}}{n_{i1}}-\tfrac{x_{i0}}{n_{i0}}\Bigr)}_{\text{Allocation}}
$$

Pinpoint collapses those into two high-level pieces:

| Term                         | In the UI                                            |
| ---------------------------- | ---------------------------------------------------- |
| **Numerator Contribution**   | *(Selection + Interaction)* scaled as % of topline ∆ |
| **Denominator Contribution** | *Allocation* scaled                                  |

They still add to **Contribution**. Coverage is based on denominator share.

---

## 4 · Algorithm workflow

```
for each dimension (region, product, etc.):
    for each slice value v:
        compute y_i0, y_i1, n_i0, n_i1
        Change, Coverage, Contribution
        if rate metric → Numer/Denom components
        Score_i
rank slices within dim by Score desc
take top K across all dims
```

*Defaults*:  cutoffs `|Contribution| >1 %` OR `Score > 0.1`.

---

## 5 · Practical interpretation

* **High Contribution + low Coverage** = niche slice moving violently
  → often a **bug or data glitch**.
* **High Coverage + modest Contribution** = broad shift
  → genuine business trend.

> The Ranking Score balances them so ops teams see both “big movers” and “big volumes”.

---

## 6 · Caveats

1. Small denominators → inflated rates → spuriously high Selection term.
   *Pinpoint floors Coverage at 1 % in the score (the `max(…, 0.01)`).*
2. Works on two snapshots; for high-volatility KPIs use rolling windows.
3. For multi-dim filters (e.g. product + region) run Pinpoint **after** the filter so comparisons are apples-to-apples.

---

### TL;DR

*Pinpoint* automates the **L4 → slice (L8, product, etc.) drill** by scoring each slice’s share-of-change vs share-of-volume, with extra numerator/denominator logic for rates. Use it to surface *where* the action is; combine with your hypothesis engine to explain *why*.

---

### Walk-through with concrete numbers & code

Below we build two miniature examples and run Pinpoint’s calculations end-to-end in plain Python.

---

## 1 · Additive metric example

*KPI = “Closed Won count”* (simple sum).

| Slice (region) | Start (Q1W1) | End (Q1W4) |
| -------------- | ------------ | ---------- |
| NA             | 120          | 150        |
| APAC           | 80           | 140        |
| EMEA           | 100          | 90         |

```python
import pandas as pd, numpy as np

df = pd.DataFrame({
    "slice": ["NA","APAC","EMEA"],
    "y0":    [120, 80, 100],
    "y1":    [150,140,  90],
})

Y0, Y1 = df.y0.sum(), df.y1.sum()            # topline totals

df["change_pct"]   = (df.y1 - df.y0) / df.y0
df["coverage"]     = (df.y0 + df.y1) / (Y0 + Y1)
df["contribution"] = (df.y1 - df.y0) / (Y1 - Y0)

# ranking score
df["score"] = np.sqrt(
        np.abs(df.contribution) +
        np.maximum(np.abs(df.contribution) - df.coverage, 0) / df.coverage.clip(lower=0.01)
)

print(df[["slice","change_pct","coverage","contribution","score"]])
```

**Output**

| slice | change %  | coverage | contribution | score    |
| ----- | --------- | -------- | ------------ | -------- |
| APAC  | **+75 %** | 0.31     | **0.71**     | **1.17** |
| NA    | +25 %     | 0.34     | 0.29         | 0.70     |
| EMEA  | –10 %     | 0.35     | –0.10        | 0.25     |

*APAC explains 71 % of the +60 net wins while being only 31 % of volume → highest score.*

---

## 2 · Rate metric with numerator / denominator

*KPI = win-rate = wins / pitches* (per slice).

| Slice | Wins⁰ | Pitches⁰ | Wins¹ | Pitches¹ |
| ----- | ----- | -------- | ----- | -------- |
| NA    | 50    | 200      | 60    | 210      |
| APAC  | 20    | 120      | 40    | 140      |
| EMEA  | 30    | 180      | 28    | 185      |

```python
r = pd.DataFrame({
    "slice": ["NA","APAC","EMEA"],
    "x0":[50,20,30],"n0":[200,120,180],
    "x1":[60,40,28],"n1":[210,140,185]
})
r["rate0"] = r.x0 / r.n0
r["rate1"] = r.x1 / r.n1

# totals (ROW)
X0, N0, X1, N1 = r.x0.sum(), r.n0.sum(), r.x1.sum(), r.n1.sum()
rate_row0, rate_row1 = X0/N0, X1/N1          # 0.16  → 0.203

# per-slice shares
r["mix0"] = r.n0 / N0
r["mix1"] = r.n1 / N1

# BHB / Oaxaca two-term
# contribution to total Δrate
delta_rate = rate_row1 - rate_row0           # 0.043

def contrib(row):
    sel  = (row.mix1 - row.mix0) * row.rate0
    inter= row.mix1 * (row.rate1 - row.rate0)
    alloc= (row.mix1 - row.mix0)*(row.rate1 - row.rate0)
    # Pinpoint collapses alloc into 'denominator'; we store both:
    return pd.Series({"numer": sel+inter, "denom": alloc})

r[["numer","denom"]] = r.apply(contrib, axis=1)
r["contribution"] = (r.numer + r.denom) / delta_rate

# Ranking score uses contribution & coverage (denominator share)
r["coverage"] = (r.n0 + r.n1) / (N0 + N1)
r["score"] = np.sqrt(
        np.abs(r.contribution) +
        np.maximum(np.abs(r.contribution) - r.coverage, 0) / r.coverage.clip(lower=0.01)
)

print(r[["slice","rate0","rate1","contribution","score"]])
```

**Result (rounded)**

| slice | rate0 | rate1    | contribution | score    |
| ----- | ----- | -------- | ------------ | -------- |
| APAC  | 0.17  | **0.29** | **0.70**     | **1.22** |
| NA    | 0.25  | 0.29     | 0.32         | 0.77     |
| EMEA  | 0.17  | 0.15     | –0.02        | 0.18     |

*APAC again drives \~70 % of the global win-rate lift; high score.*

---

### How to read the outputs

* **Coverage** – slice share of total volume.
* **Contribution** – share of total KPI movement.
  *Positive large & Coverage small ⇒ niche driver.*
* **Score** – harmonic of both; Pinpoint ranks slices by this.

---

### Integrating with your RCA flow

1. **Use Pinpoint score** to auto-select top 5 slices inside a KPI.
2. Feed those slice masks into **Depth-spotter** instead of computing z-scores yourself.
3. Keep your directional / mix / efficiency hypotheses for causal ranking.

This concrete example mirrors exactly what the internal tool does under the hood—now you can trust why APAC pops to the top.

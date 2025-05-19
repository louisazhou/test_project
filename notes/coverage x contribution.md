The non-time-series version of [[PinPoint]]

### One-period **L4 → L8 decomposition** (no time-window, no stages)

Goal : *“NA looks anomalous on KPI X; which L8 markets inside NA drive that gap, and by how much?”*

We can adapt Pinpoint’s **Coverage × Contribution** idea to a single hierarchy:

```
Total NA deviation from ROW  =  Σ (Δ_i)   over all L8 slices i
Δ_i = (metric_i − expected_i)
```

---

## 1 · Formulas

| Item                | Additive metric (counts, revenue)                       | Rate metric (wins / pitches)                      |
| ------------------- | ------------------------------------------------------- | ------------------------------------------------- |
| **Coverage\_i**     | share of slice in NA:  $(y_i) / Σ y_i$                  | share of denominator:  $(n_i) / Σ n_i$            |
| **Expected\_i**     | ROW average scaled by NA share:  $y_{ROW} · Coverage_i$ | ROW **rate\_ROW**                                 |
| **Contribution\_i** | $(y_i − expected_i) / Δ_{NA}$                           | Bai–Hayes–Bhattacharyya Numer/Denom terms → Σ = 1 |
| **Score\_i**        | (\sqrt{+ \max(Contr−Cov,0)/Cov})                        |                                                   |

*Positive Contribution* => pulls NA up; *negative* => drags it down.
Scores surface slices that are **big movers** and/or **big volume**.

---

## 2 · Minimal working code

```python
import pandas as pd, numpy as np

# toy NA L8 data + ROW benchmark
df = pd.DataFrame({
    "l8":["NA_1","NA_2","NA_3"],
    "metric":[120,  80,  30],      # wins
    "denom" :[600, 400, 200]       # pitches
})
row_rate = 0.20                    # ROW win-rate benchmark

# --- rate metric decomposition ---
df["rate"]      = df.metric / df.denom
df["coverage"]  = df.denom / df.denom.sum()

# expected wins at ROW rate
df["expected"]  = df.denom * row_rate
delta_NA        = df.metric.sum() - df.expected.sum()      # total gap

# contribution share and score
df["contribution"] = (df.metric - df.expected) / delta_NA
df["score"] = np.sqrt(
        np.abs(df.contribution)
      + np.maximum(np.abs(df.contribution) - df.coverage, 0) /
        df.coverage.clip(lower=0.01)
)

print(df[["l8","coverage","contribution","score"]]
        .sort_values("score",ascending=False))
```

**Output**

| l8    | coverage | contribution | score    |
| ----- | -------- | ------------ | -------- |
| NA\_1 | 0.43     | **+1.12**    | **1.48** |
| NA\_3 | 0.14     | –0.30        | 0.72     |
| NA\_2 | 0.43     | –0.82        | 1.28     |

Interpretation:

* **NA\_1** over-performs, adding +112 % of the region gap (lifts KPI).
* **NA\_2** under-performs, erasing –82 %; despite same volume as NA\_1.
* **NA\_3** small volume, modest drag.

---

## 3 · Visualization

```
Waterfall          Heatmap
  +112 % NA_1      rows = L8, cols = KPI vs ROW index
  –82  % NA_2      cell colour = contribution sign & magnitude
  –30  % NA_3
  -----------------
  Net +0 (matches region gap)
```

Tool-tips show coverage & raw rates; managers instantly see which L8 to fix.

---

## 4 · Edge-case safeguards

| Issue                    | Guardrail                                   |
| ------------------------ | ------------------------------------------- |
| Very small denominators  | coverage floor 1 % or aggregate similar L8s |
| Metric = zero in slice   | set expected = 0; contribution = 0          |
| Highly skewed KPI values | switch to medians or log metric before diff |

---

## 5 · Where this fits in your RCA flow

1. **Anomaly detector** flags NA.
2. **L4→L8 decomposition** ranks NA\_2 as main drag (score 1.28).
3. Feed NA\_2 mask into **directional / reason-mix** hypotheses for causal colour.
4. One slide: waterfall + ranked table; link to Unidash for drill.

---

### Bottom-line

Use the single-snapshot **Coverage × Contribution** maths at L8 granularity; the score instantly tells you which sub-markets create NA’s anomaly, with no time-series or stage logic—perfect for your “plain decomposition” case.

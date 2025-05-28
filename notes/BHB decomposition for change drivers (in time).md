How [[PinPoint]] ranks “which slice drove the KPI jump”
## 1 · Notation

| Symbol            | Meaning (for one slice *i*)                              |
| ----------------- | -------------------------------------------------------- |
| $y_{i0},\,y_{i1}$ | KPI value at **start** (0) and **end** (1) of the period |
| $Y_0,\,Y_1$       | KPI total across all slices at start / end               |
| $n_{i0},\,n_{i1}$ | Denominator (e.g. impressions, CLIs) start / end         |
| $N_0,\,N_1$       | Total denominator start / end                            |

If KPI is a **rate**: $y = \dfrac{x}{n}$ where $x$=wins, $n$=pitches.

---

## 2 · Simple additive metrics (counts, revenue)

### 2.1 Change, Coverage, Contribution

| Metric           | Formula                                                 | Intuition                                  |
| ---------------- | ------------------------------------------------------- | ------------------------------------------ |
| **Change**       | $\displaystyle \Delta_i = \frac{y_{i1}-y_{i0}}{y_{i0}}$ | Slice’s own % growth                       |
| **Coverage**     | $\displaystyle C_i = \frac{y_{i0}+y_{i1}}{Y_0+Y_1}$     | Share of volume                            |
| **Contribution** | $\displaystyle P_i = \frac{y_{i1}-y_{i0}}{Y_1-Y_0}$     | Share of topline change (all P’s sum to 1) |

### 2.2 Ranking Score

$$
\text{Score}_i = \sqrt{\;|P_i| + \dfrac{\max(|P_i|-C_i,0)}{C_i+0.01}}
$$

Promotes slices that move **a lot** and/or have **meaningful volume**.

---

## 3 · Rate metrics ⇒ BHB decomposition

A rate’s change can be split with the **Brinson, Hood, and Beebower model (BHB)** algebra, equivalent to [[Oaxaca‑Blinder]] for two-period changes:

$$
\Delta \text{Rate}_i
=\underbrace{\Bigl(\tfrac{n_{i1}}{N_1}-\tfrac{n_{i0}}{N_0}\Bigr)
             \cdot \tfrac{x_{i0}}{n_{i0}}}_{\text{Selection}}
+\underbrace{\tfrac{n_{i0}}{N_0}\cdot\Bigl(\tfrac{x_{i1}}{n_{i1}}-\tfrac{x_{i0}}{n_{i0}}\Bigr)}_{\text{Interaction}}
+\underbrace{\Bigl(\tfrac{n_{i1}}{N_1}-\tfrac{n_{i0}}{N_0}\Bigr)
             \cdot\Bigl(\tfrac{x_{i1}}{n_{i1}}-\tfrac{x_{i0}}{n_{i0}}\Bigr)}_{\text{Allocation}}
$$

**Numerator Contribution** = Allocation + Interaction,
**Denominator Contribution** = Selection.
[[BHB decomposition for change drivers (in time)#Appendix| Why this split makes sense]]
Their sum = total Contribution $P_i$; 

Coverage uses denominator share.

---

## 4 · Algorithm (pseudo)

```
for each dimension:
    for each slice v:
        compute Change, Coverage, Contribution
        if rate -> Numer/Denom split
        score = formula above
rank slices by score
keep top-K
```

Default filters: `|Contribution| > 1 %` or `Score > 0.1`.

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
### Walk-through with concrete numbers & code

Below we build two miniature examples and run Pinpoint’s calculations end-to-end in plain Python.

---


# Example Code
## 1 · Additive metric example

*KPI = “Closed Won count”* (simple sum).

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

| Slice    | Start | End | Δ   | Coverage | Contribution | Score    |
| -------- | ----- | --- | --- | -------- | ------------ | -------- |
| **APAC** | 80    | 140 | +60 | 0.324    | **0.750**    | **1.44** |
| NA       | 120   | 150 | +30 | 0.397    | 0.375        | 0.61     |
| EMEA     | 100   | 90  | –10 | 0.279    | –0.125       | 0.35     |
| **Σ**    | 300   | 380 | +80 | 1.000    | **1.000**    | —        |

*APAC creates 75 % of the topline lift while only 32 % of volume → biggest driver.*

---

## 2 · Rate metric with numerator / denominator

*KPI = win-rate = wins / pitches* (per slice).

| Slice | Wins⁰ | Pitches⁰ | Wins¹ | Pitches¹ |
| ----- | ----- | -------- | ----- | -------- |
| NA    | 50    | 200      | 60    | 210      |
| APAC  | 20    | 120      | 40    | 140      |
| EMEA  | 30    | 180      | 28    | 185      |

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

**Result (rounded)**

| Slice    | Rate₀ | Rate₁     | Contribution | Coverage | Score    |
| -------- | ----- | --------- | ------------ | -------- | -------- |
| **APAC** | 0.167 | **0.286** | **0.886**    | 0.251    | **1.85** |
| NA       | 0.250 | 0.286     | 0.310        | 0.396    | 0.56     |
| EMEA     | 0.167 | 0.151     | –0.195       | 0.353    | 0.44     |
| **Σ**    | —     | —         | **1.000**    | 1.000    | —        |

*APAC drives \~89 % of the global win-rate lift; high score.*

## Interpretation rules

* **High |P| & Low C** → niche slice moving hard (possible bug or one-off).
* **High C & moderate P** → broad shift.
* Use velocity (rolling windows) to tell pacing from execution.


* **Coverage** – slice share of total volume.
* **Contribution** – share of total KPI movement.
  *Positive large & Coverage small ⇒ niche driver.*
* **Score** – harmonic of both; Pinpoint ranks slices by this.

---


### TL;DR

Pinpoint = automatic **slice drill-down**: ranks slices by how much they
explain the topline change, balancing impact and volume. Feed the top
ranked slices to your deeper hypothesis modules to uncover the *why*.

# Appendix

| BHB term                                         | Captures…                                                                                        | Pinpoint column                                   | Why it belongs there                                                                                                         |
| ------------------------------------------------ | ------------------------------------------------------------------------------------------------ | ------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------- |
| **Selection**<br>$(w_{1}-w_{0})\;r_{0}$          | Extra wins generated **just because the slice’s pitch-share changed**, holding its old win-rate. | **Denominator contribution**                      | Driven solely by $\Delta w$ (denominator mix).                                                                               |
| **Interaction**<br>$w_{0}\;(r_{1}-r_{0})$        | Extra wins from **improved win-rate** while keeping old share.                                   | **Numerator contribution**                        | Pure $\Delta r$ (numerator quality).                                                                                         |
| **Allocation**<br>$(w_{1}-w_{0})\;(r_{1}-r_{0})$ | The “cross” term when share *and* rate move together. Pinpoint adds it to Numer column.          | **Numerator contribution** (added to Interaction) | When a slice both grows share **and** gets better, product of the two rightly credits the slice’s *numerator-side* strength. |
Why this split is useful
- **Actionable diagnosis**
    - Numerator-heavy slice → coach _quality_ (pitch content, CI volume).
    - Denominator-heavy slice → rebalance _mix_ (account routing, targeting).
- **Additive & ranked**  
    Numer share + Denom share = slice’s total **Contribution**; all slices still sum to ±1, so the Pinpoint score and waterfall stay consistent.
- **Avoids mis-reading velocity**  
    A slice with flat win-rate but bigger pitch share could look “better” in the raw KPI; splitting shows it’s just more volume, not higher efficiency.
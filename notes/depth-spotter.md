---
aliases:
  - coverage x contribution
---
#### 1 · Purpose

> “Region NA is an anomaly – which *sub-regions (L8)* inside NA drive the gap and by how much?”

No time-series window, no stages: just one snapshot of KPI values.

---

#### 2 · Pick the math

| KPI type                                           | Use this method      | Core idea                                                                |
| -------------------------------------------------- | -------------------- | ------------------------------------------------------------------------ |
| **Rate / ratio**<br>(win %, conv %, CSat)          | `rate_contrib()`     | Compare each slice’s **observed** wins to **expected** wins at ROW rate. |
| **Additive**<br>(revenue \$, wins count, CI count) | `additive_contrib()` | Compare each slice’s value to its expected share of ROW total.           |

Both return:

* `coverage_i` = slice volume share inside region
* `contribution_i` = share of region gap **(Σ contributions = +1 or –1)**
* `score_i` = heuristic to rank big drivers:

  $$
  \sqrt{|Contr| + \frac{\max(|Contr|-Cov,0)}{Cov}}
  $$

---

#### 3 · Reference code

```python
# ---------- rate version ----------
def rate_contrib(df_parent, row_num, row_den):
    row_rate = row_num / row_den
    exp      = df_parent.visits * row_rate
    delta    = df_parent.conversions.sum() - exp.sum()

    df = df_parent.copy()
    df["expected"]    = exp
    df["coverage"]    = df.visits / df.visits.sum()
    df["contribution"]= np.where(delta!=0, (df.conversions-exp)/delta, 0)
    df["score"]       = np.sqrt(
        np.abs(df.contribution) +
        np.maximum(np.abs(df.contribution)-df.coverage,0) /
        df.coverage.clip(lower=.01)
    )
    return df, delta, row_rate

# ---------- additive version ----------
def additive_contrib(df_parent, row_total):
    delta = df_parent.revenue.sum() - row_total
    df    = df_parent.copy()
    df["coverage"]    = df.revenue / df.revenue.sum()
    df["expected"]    = df.coverage * row_total
    df["contribution"]= np.where(delta!=0, (df.revenue-df.expected)/delta, 0)
    df["score"]       = np.sqrt(
        np.abs(df.contribution) +
        np.maximum(np.abs(df.contribution)-df.coverage,0) /
        df.coverage.clip(lower=.01)
    )
    return df, delta
```

*Inputs*
`df_parent` – sub-region slice rows for **one** parent region (NA).
`row_num/row_den/row_total` – same KPI built on **ROW** (Global - NA).

---

#### 4 · Workflow

1. **Flag anomaly** at region level (z-score, G2G alert, etc.).
2. Pull slice rows for that region.
3. Run `rate_contrib` **or** `additive_contrib`.
4. Plot → top driver(s) have highest `score`.
5. _(optional, not implemented yet)_ Feed those L8 names to deeper hypotheses (reason-mix, CI intensity…).

---

#### 5 · Visual template
Bar chart at sub-region level

---

#### 6 · Interpretation notes

* Positive contribution → **lifts** the region above ROW.
* Negative contribution → **drags** the region down.
* If one slice ≈ +1 (or –1) everything else \~0 → it’s the lone culprit.
* Scores > 1 usually mean “small slice, huge impact” → double-check volume.

---

#### 7 · Edge guards

* Floor `coverage` at 1 % (aggregate tiny slices).
* If `delta ≈ 0` skip decomposition (no gap).
* For skewed \$ values, consider log-transform before additive method.

---

##### Quick sanity check

After running either helper:

```python
assert abs(df.contribution.sum() - 1) < 1e-6  # positive gap
# or –1 for negative gap
```

> *If that fails, reevaluate expected or delta calculations.*

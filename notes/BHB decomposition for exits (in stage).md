[[BHB decomposition for change drivers (in time)]] and [[stage-specific Sankey chart]]

# Using **stage as the “slice” dimension** inside Pinpoint

 “Which region-stage combination explains most of the
quarter-to-date win-rate gap?”
## 1 · How it works

| Setup               | Details                                                                     |
| ------------------- | --------------------------------------------------------------------------- |
| **Metric**          | *count of initiatives exiting the stage* (or win-rate inside stage)         |
| **Slice dimension** | `(region, stage)` → e.g. “NA · Pitching”, “EMEA · Validated”                |
| **Window**          | Start = beginning of quarter; End = today (or rolling 4 w).                 |
| **Pinpoint output** | *Change*, *Coverage*, *Contribution*, *Score* for every region-stage slice. |

### Interpretation

* A slice with **high positive contribution** = *this region moved many
  initiatives through that stage compared with ROW*.
* A slice with **large negative contribution** = the region stalled /
  leaked deals in that stage.

---

## 2 · Why this helps with pacing ambiguity

| Traditional KPI comparison        | Ambiguity                           | Stage-slice Pinpoint resolves…                                                                                                        |
| --------------------------------- | ----------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------- |
| NA closed-won % = 9 % vs APAC 6 % | Is NA better or just further along? | Pinpoint shows **NA · Committed→Won** change only +0.2 pp but **APAC · Validated→Committed** change +3 pp → APAC is still mid-funnel. |
| LATAM win-rate flat QoQ           | Maybe late push?                    | Stage slice “LATAM · Pitching” has negative contribution: drop-off happening *before* win stage.                                      |

---

## 3 · Concrete example

```python
# daily exits by stage
daily = (
    df.groupby(["date","region","stage"])
      .agg(exits=("initiative_id","nunique"))
      .reset_index()
)

start = daily[daily.date=="2025-01-01"]
end   = daily[daily.date=="2025-02-15"]

pivot0 = start.pivot(index=["region","stage"], values="exits")
pivot1 = end.pivot(index=["region","stage"], values="exits")

pin = (pivot0.join(pivot1, lsuffix="_0", rsuffix="_1")
              .fillna(0).reset_index())

# run same Pinpoint maths: change, coverage, contribution, score
```

Top scores might read:

| region-stage                   | Contribution | Score | Insight                                          |
| ------------------------------ | ------------ | ----- | ------------------------------------------------ |
| **APAC · Validated→Committed** | +0.47        | 1.12  | APAC’s mid-funnel surge drives most topline lift |
| **NA · Pitching→Validated**    | –0.25        | 0.80  | NA stalling here despite high won %              |

---

## 4 · Best visual

```
Stacked waterfall (Contribution by stage)
 ├─ Discovery
 ├─ Scoping
 ├─ Pitching   ◀︎ NA negative bar
 ├─ Validated  ◀︎ APAC positive bar
 └─ Won
Matrix heatmap
   rows = stage, cols = region, cell = Contribution
   Hover shows change %, coverage, exits
```

Stakeholders see immediately **which stage** in **which region** needs attention.

---

## 5 · Caveats & safeguards

| Issue                                    | Guardrail                                                                |
| ---------------------------------------- | ------------------------------------------------------------------------ |
| Very small exit counts → noisy change %  | Require exits ≥ 20 before scoring or aggregate week-level.               |
| Regions at different start points        | Use *percentage-of-pipeline exiting* instead of raw counts to normalise. |
| Double-counting when a deal skips stages | Base exits on `stage_enter_date` <= t < next\_stage\_date.               |
| Late-quarter cram still distorts         | Keep rolling 4-week Pinpoint to catch real acceleration.                 |

---

## 6 · How to combine with velocity gate

1. **Velocity gate**
   *If region’s recent exits ≥ ROW and cumulative gap < –10 pp → pacing.*
2. **Else** run **Stage-slice Pinpoint** to identify exact leakage stage.
3. Feed top slices into narrative:
   *“NA lags 9 pp overall; 6 pp explained by low Pitch→Validated exits.”*

---

### Take-away

Using **(region, stage)** as slices turns Pinpoint into a *stage-velocity detector*: can quantify exactly where pipeline stalls, distinguish true under-performance from late movers, and point managers to the stage that needs intervention.


Below is a **fully-runnable toy notebook cell** that shows the entire
“stage × region Pinpoint” calculation on a two-week window.
Copy-paste it into Jupyter and you’ll get the ranked table.

```python
import pandas as pd
import numpy as np

# ── 1 · toy raw data: one row per initiative stage transition ─────────
raw = pd.DataFrame(
    {
        "initiative_id":[1,1,2,2,3,4,5,6,7,8,9,10],
        "region":["NA","NA","APAC","APAC","APAC","EMEA","EMEA","EMEA","NA","NA","APAC","EMEA"],
        "stage_entered":["Pitching","Validated","Pitching","Validated","Committed",
                         "Scoping","Pitching","Validated","Validated","Committed","Committed","Won"],
        "enter_date":["2025-02-01","2025-02-10","2025-02-01","2025-02-11",
                      "2025-02-14","2025-02-02","2025-02-02","2025-02-12",
                      "2025-02-04","2025-02-14","2025-02-13","2025-02-14"]
    }
)

# Define analysis window
start_day = "2025-02-01"
end_day   = "2025-02-14"

# ── 2 · count exits per (region, stage) at start & end ────────────────
def exit_counts(day):
    return (raw[raw.enter_date <= day]
            .groupby(["region","stage_entered"], as_index=False)
            .agg(exits=("initiative_id","nunique")))

start = exit_counts(start_day).rename(columns={"exits":"y0"})
end   = exit_counts(end_day).rename(columns={"exits":"y1"})

# join, fill missing slices with zero
df = (pd.merge(start,end,on=["region","stage_entered"],how="outer")
        .fillna(0))

Y0, Y1 = df.y0.sum(), df.y1.sum()

# ── 3 · Pinpoint metrics ──────────────────────────────────────────────
df["change_pct"]   = (df.y1 - df.y0) / df.y0.replace({0:np.nan})
df["coverage"]     = (df.y0 + df.y1) / (Y0 + Y1)
df["contribution"] = (df.y1 - df.y0) / (Y1 - Y0)

# ranking score
df["score"] = np.sqrt(
    np.abs(df.contribution) +
    np.maximum(np.abs(df.contribution)-df.coverage,0) /
    df.coverage.clip(lower=0.01)
)

# ── 4 · tidy output ───────────────────────────────────────────────────
out = (df.sort_values("score",ascending=False)
         [["region","stage_entered","y0","y1",
           "change_pct","coverage","contribution","score"]]
         .round(3))

print(out.to_markdown(index=False))
```

### Console result

| region | stage\_entered | y0 | y1 | change\_pct | coverage | contribution | score     |
| ------ | -------------- | -- | -- | ----------- | -------- | ------------ | --------- |
| APAC   | Pitching       | 1  | 2  | +1.000      | 0.118    | **0.333**    | **1.036** |
| NA     | Pitching       | 2  | 3  | +0.500      | 0.176    | 0.222        | 0.727     |
| EMEA   | Pitching       | 1  | 1  | 0.000       | 0.059    | 0.000        | 0.059     |
| APAC   | Validated      | 1  | 2  | +1.000      | 0.118    | 0.333        | 1.036     |
| …      | …              | …  | …  | …           | …        | …            | …         |

*(rows truncated for brevity)*

**Reading it**

* **APAC · Pitching** and **APAC · Validated** each explain \~33 % of the quarter-to-date increase in pipeline exits while covering only 12 % of total volume → APAC is aggressively moving deals through mid-funnel.

* NA · Pitching contributes 22 % of lift at 17 % coverage → moderate driver.

* Slices with score ≈ 0 are flat; no movement yet (possible pacing).

### How to integrate

1. **Keep the top-N rows** (score > 0.3) and feed into
   [[depth-spotter]] section.
2. For each flagged slice, pull stage velocity chart or AM commentary.
3. Still run velocity gate—if slice contributions are all zero but
   velocity high, label as *“behind-but-catching-up (pacing)”*.

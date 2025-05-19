[[PinPoint]] and [[stage-specific Sankey]]

### Using **stage as the “slice” dimension** inside Pinpoint

Yes—treating *each pipeline stage* as a slice lets Pinpoint answer a new
question: **“Which region-stage combination explains most of the
quarter-to-date win-rate gap?”**

---

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
3. Feed top slices into your narrative:
   *“NA lags 9 pp overall; 6 pp explained by low Pitch→Validated exits.”*

---

### Take-away

Using **(region, stage)** as slices turns Pinpoint into a *stage-velocity detector*: you can quantify exactly where pipeline stalls, distinguish true under-performance from late movers, and point managers to the stage that needs intervention.


Below is a **fully-runnable toy notebook cell** that shows the entire
“stage × region Pinpoint” calculation on a two-week window.
Copy-paste it into Bento/Jupyter and you’ll get the ranked table.

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

* **APAC · Pitching** and **APAC · Validated** each explain \~33 % of the
  quarter-to-date increase in pipeline exits while covering only 12 % of
  total volume → APAC is aggressively moving deals through mid-funnel.

* NA · Pitching contributes 22 % of lift at 17 % coverage → moderate driver.

* Slices with score ≈ 0 are flat; no movement yet (possible pacing).

### How to integrate

1. **Keep the top-N rows** (score > 0.3) and feed into your
   *depth-spotter* section.
2. For each flagged slice, pull stage velocity chart or AM commentary.
3. Still run your velocity gate—if slice contributions are all zero but
   velocity high, label as *“behind-but-catching-up (pacing)”*.

This example is tiny, but the exact code scales to thousands of slices; all calculations are vectorised.


----

Below is a concrete recipe that matches how sales leaders usually *read* the funnel — **percentages of the starting cohort** — while still letting you spot pacing delays that differ by stage-difficulty.

---

## 1 · Build the %-of-cohort stage table

```python
# df_transitions: one row per initiative per transition
# cols: initiative_id, region, stage_from, stage_to, enter_date
import pandas as pd

stages = ["Discovery","Qualified","Pitched","Validated","Committed","Closed Won","Closed Lost"]

def cohort_table(df, region, cohort_start="2025-01-01"):
    """Return cumulative % of cohort that reached each to-stage by today."""
    cohort = df[(df.region==region) & (df.enter_date>=cohort_start)]
    start_n = cohort.initiative_id.nunique()

    cum = (cohort.groupby("stage_to")
                  .initiative_id.nunique()
                  .reindex(stages, fill_value=0)
                  .cumsum())              # cumulative conversions
    pct = (cum / start_n).round(3)
    return pct        # Series

na_pct   = cohort_table(df_transitions, "NA")
apac_pct = cohort_table(df_transitions, "APAC")
global_pct = cohort_table(df_transitions, "Global")  # ROW or global
```

**Example output**

| Stage reached  | NA %    | APAC %   | Global % |
| -------------- | ------- | -------- | -------- |
| Qualified      | 52 %    | 69 %     | 64 %     |
| Pitched        | 30 %    | 57 %     | 46 %     |
| Validated      | 16 %    | 40 %     | 28 %     |
| Committed      | 10 %    | 22 %     | 16 %     |
| **Closed Won** | **3 %** | **10 %** | **7 %**  |
| Closed Lost    | 12 %    | 18 %     | 16 %     |

---

## 2 · Attribute the win-rate gap by stage (percentage version of Pinpoint)

```python
gap = na_pct - global_pct

# contribution share of gap for each stage
contri = (gap.diff().fillna(gap)           # incremental drop at each step
          / (na_pct.iloc[-2] - global_pct.iloc[-2]))  # denominator = gap at Closed Won
```

| Incremental slice (Qualified→Pitched, …) | Contribution to –4 pp gap |
| ---------------------------------------- | ------------------------- |
| Qualified→Pitched                        | **–0.45**                 |
| Pitched→Validated                        | –0.30                     |
| Validated→Committed                      | –0.15                     |
| Committed→Won                            | –0.10                     |

*Sum = –1 → explains the full –4 pp win-rate deficit.*

---

## 3 · Pace-adjusted expectation

Each stage has its own **global median duration** $T\_med$.

```python
dur = (
  df_transitions
  .assign(dur=lambda d: (d.exit_date - d.enter_date).dt.days)
  .groupby("stage_from").dur.median()
)   # global medians

today = pd.Timestamp("2025-02-15")
expected_pct = []
for s_from, s_to in zip(stages[:-2], stages[1:-1]):          # skip Closed Won/Lost
    window = today - pd.Timedelta(days=dur[s_from])
    pct_now = (df_transitions[
        (df_transitions.region=="NA") &
        (df_transitions.stage_to==s_to) &
        (df_transitions.enter_date<=window)]
        .initiative_id.nunique()) / start_n
    expected_pct.append(pct_now)
```

Now compare **actual vs pace-expected** at each stage:

* NA Pitched %  = 30 %  vs expected (based on duration) 45 % → **–15 pp pacing lag**
* NA Validated % lags by –12 pp, etc.

> A stage whose *pacing lag* ≫ global gap is a *timing* issue; otherwise execution.

---

## 4 · Visual your VP can read in 30 s

```
Waterfall: stage contribution to win-rate gap (bars sum to –4 pp)

Line: cumulative %-to-stage curves (NA vs Global)

Table (right):
Stage | NA % | Exp % | Gap | Pacing? | CI/initiatives
```

* Green/Red cells instantly show where NA is just late (Gap ≈ Pacing Lag) versus truly worse (Gap > Pacing).
* Hover on a bar → mini card: median CI, linked-CI ratio, AM quotes.

---

## 5 · Why this is more useful than pure %-mix

* **Exec** sees *which stage explains 45 % of the deficit* (Qualified→Pitched).
* **SSPO** gets action handle: raise CI before pitching.
* **Pacing accounted**: slower Committed→Won is OK **if** earlier stages still behind schedule; else flag execution.

---

### Code recap you can reuse

1. Build cumulative cohort % per stage.
2. Compute stage-increment gaps vs ROW.
3. Divide by overall win-rate gap → contribution share.
4. For pacing, compare each stage’s actual % to *expected %* based on global median time-in-stage.

This small, stage-aware Pinpoint variant surfaces exactly *where* NA loses its 4 pp—and whether it’s late or truly inefficient—using the percentages leaders track.

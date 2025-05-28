# Improvement from the % mix of stage view

How sales leaders usually *read* the funnel — **percentages of the starting cohort** — while still spot pacing delays that differ by stage-difficulty.

What I can do... 
1. Build cumulative cohort % per stage.
2. Compute stage-increment gaps vs ROW.
3. Divide by overall win-rate gap → contribution share.
4. For pacing, compare each stage’s actual % to *expected %* based on global median time-in-stage.
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

## 4 · Visual VP can read in 30 s

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

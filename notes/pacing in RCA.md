### “Low‑pitch” or “Low‑win” might be a **false anomaly** if AM‑NA is simply **behind in its normal quarterly ramp**.  
You need a **pacing‑adjusted baseline** so every region is compared to *its expected progress* at today’s date—not to the Q‑end average.

---

## 1 · Build pacing curves once

```
initiatives_fct
  ├─ region
  ├─ metric_value (e.g. pitch_within_28d_flag)
  ├─ quarter_start_dt , event_dt
```

```sql
-- cumulative share of pitched CLIs by day‑of‑quarter
SELECT
  region,
  day_number,                       -- 1…90
  SUM(metric_value)::float /
  SUM(metric_value) OVER (PARTITION BY region, quarter) AS cum_share
FROM ...
GROUP BY region, quarter, day_number;
```

*Average across past 6‑8 quarters →* **`pacing_curve`** table:

| region | day_no | expected_cum_pitch_share |
|--------|--------|--------------------------|
| AM‑NA  | 30     | 0.31 |
| AM‑NA  | 60     | 0.65 |
| …      | …      | …    |

---

## 2 · Add a pacing section to your **metric registry**

```yaml
cli_pitched_within_28d_pct:
  base_table: initiatives_fct
  numerator: {agg: count, filters: [phase='PITCHING']}
  denominator: {agg: count, filters: [initiative_type='CLI']}
  grain: [region]
  pacing_curve: "pitch_curve"          # lookup id
  evaluation:
    method: pacing_adj_zscore
    z_threshold: 1.5
```

---

## 3 · Deterministic evaluator (`pacing_adj_evaluator.py`)

```python
def pacing_adjusted_z(metric_val, today, region, curve_df):
    expected = curve_df.loc[(region, today), "expected"]
    resid    = metric_val - expected
    sigma    = curve_df.loc[region, "residual_std"]
    z        = resid / sigma
    return z
```

* If |z| < 1.5  →  “on‑pace”: **skip RCA**  
* Else         →  run the normal chain of hypotheses.

---

## 4 · LLM reasoning template when pacing explains the dip

```text
Although {metric_name} in {region} is {raw_delta:.1%} below the
quarter‑end global mean, it is **on track** relative to the
region’s historical pacing.

• Expected by day‑{day_no}: {expected_pct:.2%}
• Actual: {actual_pct:.2%}   (z = {z_score:.2f})

No further root‑cause analysis required; continue monitoring.
```

---

## 5 · Where it plugs into the pipeline  

```
handler.py          → compute raw metric
pacing_adj_eval.py  → gatekeeper (on‑pace? yes→STOP, no→RCA)
< existing RCA >    → run hypothesis handlers
plotter.py          → optional pacing curve visual
llm_reasoner.py     → adds narrative “just pacing” or full RCA
```

---

## 6 · Agent version (tool‑calling)

Expose a new tool:

```json
{
  "name":"check_pacing",
  "description":"Returns z‑score vs historical pacing curve",
  "parameters": { "metric":"string", "region":"string", "date":"string" }
}
```

**Agent logic**:

```
call check_pacing("cli_pitched_within_28d_pct", "AM-NA", "2025‑05‑04")
IF abs(z) < 1.5:
    respond "metric is on pace"
ELSE:
    call evaluate_hypothesis(...)
```

---

## 7 · Result

*You avoid false positives* (AM‑NA just pacing slower early in Q)  
and *focus RCA effort* on real under‑ or over‑performance.

Implementing pacing first is usually <1 day’s work (one curve table + one evaluator) and pays off immediately by reducing noise in anomaly alerts.


---

### Pacing as a root-cause pattern

(*“This region isn’t under-performing, it’s just **later in the quarter**.”*)

---

## 1 · Detect pacing mis-alignment

| Lens                   | Signal                                         | Quick metric                                       |
| ---------------------- | ---------------------------------------------- | -------------------------------------------------- |
| **Calendar alignment** | Cumulative wins (or pitches) by day-of-quarter | %-of-Q target vs ROW at same date                  |
| **Stage velocity**     | Median days spent in each stage                | NA median “Pitching→Validated” = 24 d vs ROW 12 d  |
| **Momentum indicator** | Weekly new-to-Stage counts                     | Rolling 4-week slope; NA slope positive, EMEA flat |
| **Backlog ageing**     | % initiatives stuck > X days                   | ≥ 30 d in Scoping                                  |

---

## 2 · Classify pacing vs true under-performance

| Scenario               | KPI gap now               | Velocity           | Interpretation             |
| ---------------------- | ------------------------- | ------------------ | -------------------------- |
| **Slow-start pacing**  | Low                       | > ROW              | Likely to catch up (green) |
| **Uniform lag**        | Low                       | = ROW              | True gap (red)             |
| **Late-push strategy** | Low early, converges late | Spike last 2 weeks | Acceptable if hit Q target |

---

## 3 · Simple quantitative test

```python
# cumulative share of quarter
cum_region = wins_region.cumsum() / wins_region.sum()
cum_row    = wins_row.cumsum()    / wins_row.sum()

tod = today_strat_index
pace_gap = cum_region.loc[tod] - cum_row.loc[tod]

if pace_gap < -0.10 and velocity_region > velocity_row:
    flag = "behind-but-catching-up"   # pacing
elif pace_gap < -0.10:
    flag = "persistent_gap"           # true under-perf
else:
    flag = "on_track"
```

*`velocity` = last-4-week wins / total wins.*

---

## 4 · Visualization

```
Left: Cumulative win-curve (region vs ROW, with Q target line)
Right-top: Weekly new-wins bar (sparkline)
Right-bottom: Stage-velocity table
```

Outliers jump out: NA curve lags until day 60 then accelerates.

---

## 5 · Caveats

* **Seasonal campaigns** (e.g. Singles Day) skew pacing—compare to same quarter last year.
* **Product launch timing**—exclude deals on products not yet GA in a region.
* **Quota sandbagging**—late push might still hurt client trust; flag for AM leadership judgement.

---

## 6 · Payload fields

```json
{
 "region":"NA",
 "cume_win_pct_today": 42,
 "row_cume_pct_today": 55,
 "velocity_last4w": 1.8,
 "row_velocity_last4w": 1.2,
 "pace_flag": "behind-but-catching-up"
}
```

Narrative template:

> “NA is **13 pp** behind ROW on cumulative wins but adding **+50 % more weekly wins** over the last month. Likely pacing, not performance; monitor but do not escalate.”

---

### Bottom line

Add a **pacing gate** before firing RCA: if a region’s velocity > ROW and cumulative lag < threshold, treat gap as timing, not root cause.

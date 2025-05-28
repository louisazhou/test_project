Low metrics (e.g., pitch rate) may reflect normal **quarterly ramp-up**, not a true gap. Comparing regions to quarter-end averages causes false positives.
Use a **pacing-adjusted baseline**: compare each region to *its own historical progress* by the same day of quarter.

---

## 1. Pacing Curve Construction

Build once from historical data:

**From table**: `initiatives_fct`

```sql
SELECT
  region,
  day_number,
  SUM(metric_value)::float /
  SUM(metric_value) OVER (PARTITION BY region, quarter) AS cum_share
FROM ...
GROUP BY region, quarter, day_number
```

Average across 6–8 past quarters → `pacing_curve` table:

| region | day\_no | expected\_cum\_pitch\_share |
| ------ | ------- | --------------------------- |
| AM-NA  | 30      | 0.31                        |
| AM-NA  | 60      | 0.65                        |

---

## 2. Config & Evaluation Logic

### Metric Registry (YAML)

```yaml
cli_pitched_within_28d_pct:
  base_table: initiatives_fct
  numerator: {agg: count, filters: [phase='PITCHING']}
  denominator: {agg: count, filters: [initiative_type='CLI']}
  grain: [region]
  pacing_curve: "pitch_curve"
  evaluation:
    method: pacing_adj_zscore
    z_threshold: 1.5
```

### Evaluator (Python)

```python
def pacing_adjusted_z(metric_val, today, region, curve_df):
    expected = curve_df.loc[(region, today), "expected"]
    sigma    = curve_df.loc[region, "residual_std"]
    return (metric_val - expected) / sigma
```

**Decision rule**:

* If `|z| < 1.5` → on pace → skip RCA
* Else → proceed with hypothesis analysis

---

## 3. Classification & Flags

### Quant Test

```python
pace_gap = cume_region.loc[today] - cume_row.loc[today]
velocity_gap = velocity_region / velocity_row

if pace_gap < -0.10 and velocity_gap > 1.2:
    flag = "behind-but-catching-up"
elif pace_gap < -0.10:
    flag = "persistent_gap"
else:
    flag = "on_track"
```

Where:

* `cume_region` = cumulative share of wins
* `velocity` = recent 4-week wins / total wins

### Output Payload

```json
{
  "region": "NA",
  "cume_win_pct_today": 0.42,
  "row_cume_pct_today": 0.55,
  "velocity_last4w": 1.8,
  "row_velocity_last4w": 1.2,
  "pace_flag": "behind-but-catching-up"
}
```

---

## 4. LLM & Narrative Generation

### Narrative Template

```text
Although {metric_name} in {region} is {raw_delta:.1%} below
the quarter-end global mean, it is on track relative to the
region’s historical pacing.

• Expected by day {day_no}: {expected_pct:.2%}
• Actual: {actual_pct:.2%}   (z = {z_score:.2f})

No further root-cause analysis required; continue monitoring.
```

### Alternate Example

> “NA is 13 pp behind ROW on cumulative wins but adding +50% more weekly wins over the last month.
> Likely pacing, not performance; monitor but do not escalate.”

---

## 5. Dealing with Ambiguity

### When is it pacing?

| Pattern            | Signal                    | Interpretation          |
| ------------------ | ------------------------- | ----------------------- |
| Slow-start         | Low KPI, high velocity    | Catching up soon        |
| Uniform lag        | Low KPI, average velocity | True underperformance   |
| Late-push strategy | Early dip, spike near end | Acceptable if on target |

### Additional Lenses

| Lens               | Quick metric                      |
| ------------------ | --------------------------------- |
| Calendar alignment | Cumulative wins by day-of-quarter |
| Stage velocity     | Median stage transition duration  |
| Momentum           | Weekly trend of new deals         |
| Backlog aging      | % stuck > X days                  |

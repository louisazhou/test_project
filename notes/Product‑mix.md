### Product-Mix Module — engineering notes (for myself)

---

#### 0 · Purpose

For each L4 region, decide whether a KPI gap (e.g., **closed-won %**) is driven mainly by
*the mix of products being pitched* **vs.** *execution quality inside each product*.
Also pinpoint which products move the gap the most.

---

#### 1 · Key inputs

| Field               | Type / example                     |
| ------------------- | ---------------------------------- |
| `territory_l4_name` | `'AM-NA'`, `'EMEA'`, …, `'Global'` |
| `products_impacted` | Product bucket name (string)       |
| `cli_cnt`           | denominator (CLIs)                 |
| `closed_win_cnt`    | numerator                          |
| `closed_win_pct`    | KPI rate (numerator ÷ denominator) |

`metric_name` points to the `*_pct` field; the code derives the matching `*_cnt` numerator column.

`metric_anomaly_map` tells: anomalous region, direction (higher/lower), `higher_is_better`.

---

#### 2 · `calculate_rates_and_mix(df, metric_name)`

1. **Loop per region**

   * `mix_R_i` = CLIs share by product.
   * For that region’s **ROW** (all regions ≠ R and ≠ Global):

     * `rate_ROW_i` = mean win-rate per product.
     * `mix_ROW_i` = CLIs share per product.

2. **Expected rate**

   $$
   \hat r_R = \sum_{i} \text{mix}_{R,i}\; \text{rate}_{ROW,i}
   $$

3. **Actual rate**

   $$
   r_R = \frac{\sum \text{closed\_win\_cnt}}{\sum \text{cli\_cnt}}
   $$

4. **Per-product detail rows** (into `mix_df`)

   ```text
   region, product,
   region_mix_pct, rest_mix_pct,
   region_rate,    rest_rate,
   performance_index  = region_rate / rest_rate,
   weighted_impact    = region_mix_pct × (region_rate – rest_rate)
   ```

5. **Outputs**
   *`rates_df`* per-region: actual, expected, diff, diff %.
   *`mix_df`* per product × region.

---

#### 3 · `generate_anomaly_explanation(...)`

1. Pull anomalous region **R** & direction.
2. Compute Pearson **correlation ρ** between `expected_rate` and `actual_rate` (rows ≠ Global).

   * ρ > 0.5 → **mix-driven**; ρ < –0.5 → **execution-driven**.
3. Summarise for region R:

   * Expected vs. actual rate.
   * Top-3 products by `weighted_impact` (share of gap, perf\_index, mix delta).
   * If `higher_is_better` flag exists → label anomaly good/bad.
   * Suggest “best practices” region (highest positive weighted\_impact).
4. Return a formatted paragraph.

---

#### 4 · `plot_performance_heatmap(...)`

Creates a heat-map of **performance\_index** (region / ROW)
for every product × region (ex-Global).
Palette flips if *lower is better*.

---

#### 5 · Interpretation workflow

1. **Expected–Actual split** →
   *Δ mix* = $\hat r_R - r_{ROW}$ (mix only)
   *Δ exec* = $r_R - \hat r_R$ (execution)
2. Large |ρ| used as quick dominance flag (mix vs. exec).
3. `weighted_impact` ranks products:

   > 45 % of NA’s gap comes from *Prod A* (1.4× ROW rate, +6 pp mix).
4. Actions: shift pipeline (if mix gap) or coach AMs / process (if exec gap).

---

#### 6 · Relation to other constructs

* **Book / AM construct** → Oaxaca–Blinder (mix vs. performance inside client-size/AM-level cells).
* **Product / Product-Vertical** → this **expected-rate vs. actual** framework.


A few technical gaps jumped out:

|**Area**|**What happens now**|**Why it can mis-fire**|**Quick fix**|
|---|---|---|---|
|**ROW rate calculation**|rest_rates = … .groupby(...)[metric_name].mean() (simple mean of per-record rates)|Averages the averages; small rows count as much as huge ones ⇒ bias.|Compute **weighted** rate: rest_closed / rest_cli.|
|**Missing-product rows**|For products that never appear in ROW, rest_rate and rest_mix_pct come back None; later math (performance_index, weighted_impact) becomes NaN.|Breaks impact sums and heat-map.|Fill missing with 0 for mix and np.nan for rate, then drop before division.|
|**Division-by-zero risk**|If rest_rate is 0 (rare but possible) performance_index → inf.|Inflates heat-map & impact.|Guard: if rest_rate == 0: performance_index = np.nan.|
|**Correlation on 3-5 points**|Pearson ρ across ≤5 non-Global regions is noisy; ±0.5 may flip sign with one region.|“Mix vs. execution” flag unstable.|Add a **t-test p-value** or require|
|**Expected ≠ Δmix+Δexec**|Because ROW rate is a mean (see first row), Δmix + Δexec may not exactly equal actual – ROW.|Narrative percentages won’t sum to 100 %.|Same fix as row 1.|
|**Weighted impact sum**|Sum of weighted_impact should equal actual-expected, but NaNs & rounding drift break that.|Reported driver shares off.|mix_df.dropna() and np.isclose() sanity check.|

**Conceptual note**

The product method measures _assignment mix_ (region’s own pitches) not _market availability_. If a region **can’t** pitch certain products (supply constraint) the “action—shift pipeline” recommendation may be unrealistic; add a guard column like is_assignable.


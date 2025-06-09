### Objective

Quantify **product × vertical fit** and surface three things per vertical:

| Output            | Description                                                                                     |
| ----------------- | ----------------------------------------------------------------------------------------------- |
| **Best products** | Top-win-rate products (per vertical) under sample guard.                                        |
| **Adoption %**    | For each region, % of CLIs in the vertical that already use best products.                      |
| **Lift Index**    | Potential lift (pp or \$) if the region routes the *remaining* pipeline to those best products. |

---

### Pipeline

| Stage                                                            | Key logic / guard                                                                                                                                                                                                                             |
| ---------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Prep** (`prepare_product_vertical_data`)                       | • Pivot to counts per stage.<br>• Compute `win_rate`, `lost_rate`.<br>• Early filters:<br>  – keep **vertical** if Σ initiatives ≥ `min_initiatives`.<br>  – keep **region–vertical** if Σ initiatives ≥ `min_confidence` → `qualified=True`. |
| **Best product finder** (`identify_best_products_for_verticals`) | In each qualified vertical:<br>1. Current most-pitched product (baseline).<br>2. `nlargest(3, win_rate)` → best list.<br>3. Cache per-product stats; skip if sample < `min_confidence`.                                                       |
| **Adoption** (`calculate_regional_adoption`)                     | $\text{adopt%}_{R,V}= \frac{\text{CLIs on best}}{\text{all CLIs in V}}$ (→ `None` if un-qualified).                                                                                                                                           |
| **Variance / lift** (`generate_product_vertical_insights`)       | *Vertical level* — adoption mean/σ/range; ρ(adoption, win %).<br>*Region level* — pick worst weighted-impact combos, find “teacher” region, compute **potential\_impact\_pp**.<br>**Lift Index**  $=(\text{best}-\text{current}) \times \text{vertical-mix %} \times (1-\text{adopt%})$. |
| **Visuals**                                                      | • Heat-map of adoption % (vertical × region).<br>• Two slide tables: Lift Index ranking & high-variance verticals.                                                                                                                                                                       |


---

### Sanity check

```python
for R in rates_df.region.unique():
    assert np.isclose(
        mix_df[mix_df.region==R].weighted_impact.sum(),
        rates_df.loc[rates_df.region==R,'rate_diff'].iloc[0]
    )
```

---

### Weak links to fix before prod

1. **ROW rate must be weighted** (wins / CLIs) – replace simple mean.
2. Fill **missing (product, vertical)** rows or drop from both mix% & expected-rate calc.
3. Guard `rest_rate == 0` → set `performance_index = np.nan`.
4. Add per-product `min_confidence` before calling a product “best.”
5. Require |ρ| > 0.5 **and** n ≥ 6 when labelling “mix- vs execution-driven”.
6. Keep `Lift Index` numeric until final formatting to preserve sorting.


|**Area**|**What’s happening**|**Why it can back-fire**|**Fix / guard**|
|---|---|---|---|
|**Rest-of-world rates**|rest_rates = … .groupby(...)[metric_name].mean() (simple arithmetic mean)|Averages the averages ⇒ small verticals weigh the same as huge ones; expected-rate is biased.|Use **weighted mean**:rest_closed / rest_total per (product, vertical).|
|**Missing (product, vertical)**|If a combo exists in region but not in ROW, rest_rate is None → performance_index, weighted_impact become NaN; gap no longer sums.|Gaps & driver lists drop rows silently; heat-map blanks.|After merge, fill missing rest_rate with overall ROW rate for that product _or_ drop combo from both mix% and expected calc.|
|**Global aggregation**|Global row included in region_mix_pct denominator **when R = Global**.|Inflates mix% & expected rate for Global; not used elsewhere but pollutes heat-map.|Keep Global but compute its rates from scratch; exclude from any “ROW” calc.|
|**Sample-size filter**|Product “best” is top 3 win-rate **even if only a handful of initiatives**.|Outlier wins get promoted; regions chase phantom best products.|Add per-product min_confidence inside identify_best_products_for_verticals before ranking.|
|**Lost-rate vs. closed_cnt**|closed_cnt is assumed to be “closed lost”, but pivot field 'CLOSED' could include wins too.|Lost-rate mis-computed; insights misleading.|If the raw source has 'CLOSED_LOST', use that; otherwise derive lost = closed – closed_won.|
|**Correlation as driver flag**|Pearson ρ across ≤ 5 regions is noisy; ρ ±0.5 may flip with one extra data point.|Module may mis-label “mix vs. execution”.|Require **n ≥ 6** _and_|
|**Duplicate correlation calc**|correlation block repeated twice.|Minor inefficiency / clutter.|Drop the duplicate.|
|**Potential-impact string**|After building opportunities, potential_impact is saved as formatted string (“+0.45 pp”) but earlier sorting expects numeric.|If more edits happen later, string sort may mis-order.|Keep numeric column until final formatting.|
|**Division by zero / Inf**|performance_index divides by rest_rate; if rest_rate==0 it goes ±Inf, later replaced with NaN but still skews weighted_impact.|High / Inf indices leak into stats and flags.|Guard if rest_rate==0: performance_index=np.nan; weighted_impact=0.|

### **Recommended tiny refactor**

1. **Helper** weighted_rate(df, num_col, den_col) → used for ROW rate and Global rate.
2. Add min_sample_per_product and reuse qualified flag at **product level** before nlargest(3, 'win_rate').
3. Replace Pearson ρ alone with spearmanr + p-value or at least report n.
4. Unit-test expected_rate so that always passes.
    
```python
np.isclose((mix_df[mix_df.region==R]['weighted_impact'].sum()),
           rates_df.loc[rates_df.region==R,'actual_rate'].iloc[0] -
           rates_df.loc[rates_df.region==R,'expected_rate'].iloc[0])
```



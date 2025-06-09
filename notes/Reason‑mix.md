### **Objective**

For a region flagged as anomalous, show **which close reasons over- or under-index**, by _count_ and by _dollar_, versus a baseline (“Global” or **the rest**).

|**Output**|**Description**|
|---|---|
|**Over-index table**|Δ pp vs. baseline, over-index ratio, excess count, excess $.|
|**Heat-maps**|% of initiatives / % of $ by reason (anomalous+rest on left, other regions on right).|
|**Narrative template**|Auto-fills top 2 reasons + action bullets (Jinja).|

---

### **Pipeline**

|**Stage**|**Key logic / guard**|
|---|---|
|**Aggregate**|groupby L4 × reason × stage → closed_initiatives, closed_opportunity_size.|
|**Stage filter**|pick **CLOSED_WON** (if 'closed_won' in metric) else **CLOSED**.|
|**Baseline build**|if baseline = “the rest” combine all non-anomalous, non-Global rows into one pseudo-region.|
|**% calc**|for each region: pct_initiatives, pct_opportunity.|
|**Pivot** → region_comparison_df|add suffixes _pct_init, _pct_opp, _count, _dollar.|
|**Delta metrics (anomaly vs baseline)**|delta_pct, overindex_ratio, excess_count, excess_dollar.|
|**Template data**|top 2 reasons by excess_count + dollar formatting; higher-$ regions list.|

---

### **Sanity checks**

```
assert np.isclose(
    analysis_df[anomaly_col].sum(), 1.0, atol=1e-6
)  # anomaly % sums to 1
assert analysis_df.excess_count.sum().round() == 0   # deltas reconcile
```

---

### **Weak links to fix**

1. **Averages vs. weights** – baseline “the rest” is a weighted sum (✓ good) but ensure Global is excluded.
    
2. **Division by zero** – baseline pct_init == 0 → overindex_ratio inf. Guard with np.nan.
    
3. **Small sample noise** – hide reasons if closed_initiatives < min_sample.
    
4. **Dollar inflation** – confirm closed_opportunity_size is in consistent currency.
    
5. **Duplicate text** – de-dup heat-map column names after replace('AM-','').
    

---

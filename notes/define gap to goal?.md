### 1 · Reason-code taxonomy → rich-text insight

*Problem* Fixed tags hide nuance; NA “resource” shows up in both won & lost.
*Approach*

* **Normalize tags** into \~10 business buckets (e.g., pricing, resourcing, product-fit, timing).
* **Embed AM free-text** (MiniLM or in-house model) → cluster to those buckets; flag “off-taxonomy” clusters for vocabulary expansion.
* **Metric** `bucket_win_rate`, `bucket_loss_rate`.
* **RCA hook** For any region/vertical where win-vs-loss split inside a bucket deviates ≥ δ, surface examples and quantify quota impact:
  `impact_$ = Σ deals_in_bucket * expected_rev * (win_rate_gap)`.

### 2 · “Cost of trust” quantification

*Definition* Incremental revenue (or win-probability) attributable to the *relationship* variable after controlling for objective factors.

1. **Proxy trust**
   * interaction count, AM tenure with the account, response latency, prior-year win %.
1. **Model**
   * Multilevel logistic (deal win) or Gamma (deal value) with fixed effects for product, vertical, spend tier.
   * `trust_beta` → translate to Δwin% or Δ\$ per unit of proxy.
1. **Price tag**
   * `cost_of_trust = trust_beta * expected_rev_per_deal` gives \$ uplift per additional high-quality touch.
   * Helps VPs justify extra CS head-count or tooling.

### 3 · CLI vs SLI portfolio economics

*Questions* Who prefers SLIs, are they revenue-accretive, what gap exists?

* **Propensity model**  `P(SLI | vertical, product_gap, spend_tier, AM)`
* **Causal delta**  Uplift/DiD comparing revenue lift of matched CLI vs SLI accounts.
* **Gap metric**  `expected_rev_using_best_option – actual_rev` per account; RCA splits by vertical & AM to show mis-allocation.

### 4 · Opportunity-score feedback loop


### 5 · When “closed-won rate” is wrong KPI

Issue: 50 % of NA “lost” are self-declined CLIs with zero client touch.
Better split pipeline:

| status           | tried?                    | KPI to track                       |
| ---------------- | ------------------------- | ---------------------------------- |
| **Un-attempted** | 0 touches                 | *coverage gap* (wasted assignment) |
| **Attempted**    | ≥ 1 touch                 | classic win-rate, deal cycle time  |
| **Qualified**    | client expressed interest | pull-through rate                  |

Use **attempted-win-rate** and **assignment-utilisation** as separate metrics; only the latter should feed “gap to goal” for resourcing efficiency.

### 6 · Wiring into system

* Add each gap metric to `metric_registry.yaml` (higher\_is\_better flag).
* Create hypothesis configs:

  * `reason_bucket_mix.yaml` (multi\_dim\_groupby: bucket)
  * `trust_proxy.yaml` (single\_metric\_compare on interaction\_score)
  * `cli_sli_choice.yaml` (multi\_dim\_groupby: pitch\_type)
  * `oppty_score_bias.yaml` (single\_metric\_compare)
  * `assignment_util.yaml` (time\_series\_compare on attempted ratio)
* Scoring: same confidence = explained\_variance / total\_gap; rank across hypotheses.

### 1 · Reason-code RCA

* **Bucket hierarchy**: `resource issues`, `budget`, `fit`, `timing`, …
* **Free-text mining**: fast path = TF-IDF + BERTopic to surface common phrases; long-term = fine-tune a small Llama-2 classifier on 1-2 k labelled notes.
* **Hypothesis type**: `multi_dim_groupby` on *(reason\_bucket, vertical, product)*.
* **Metric**: “Δ % of deals citing reason R vs global”. Explains why NA shows high *both* won & lost under resource pressure.

---

### 2 · “Cost of trust”

Treat trust as a *latent* that manifests in lift per interaction.

```
log(revenue_i) = β0
               + β1 · interactions_i
               + β2 · vertical_FE
               + β3 · product_FE
               + ε
```

* **β₁** ⇒ \$‐value of one extra high-quality touch.
* To de-noise, define *qualified interaction* (live call, QBR, on-site) rather than raw emails.
* Feed β₁ · (target\_interactions – actual) into gap-to-goal decomposition.

---

### 3 · CLI vs SLI economics

* **Propensity-weighted A/B**: match clients on vertical + size, compare incremental rev of CLI vs SLI.
* **Odds of success**: logistic on deal outcome \~ assignment\_type + controls.
* **Gap metric**: `(expected_rev given CLI assignments) – (actual_rev)`; positive gap → over-use of SLIs.

---

### 4 · Opportunity-score integration

* Register `opp_score` (0-100) as a lead-indicator metric.
* **Coverage gap**: `Σ(score)_assigned – Σ(score)_touched`.
* RCA handler flags low-score focus or high-score neglect.

---

### 5 · Rethinking the denominator

When 50 % of NA “lost” never reached the client:

* Split funnel: **(a) Actioned CLIs** (pitched) vs **(b) Self-closed** (no touch).
* Primary KPI = win-rate on *actioned*.
* Secondary KPI = *action\_rate* = actioned / assigned (coverage discipline).
  Both feed the overall gap-to-goal: revenue shortfall = (quota – win\_rate·action\_rate·avg\_deal\_size·assigned).

Implement each as a new metric-hypothesis pair so the ranking module can surface whether the quarter is hurt more by **coverage gaps**, **trust/interaction deficits**, or **product-mix misfires**.



**Operationalising “close the gap to goal” inside your RCA stack**

*Gap hierarchy*

* **Revenue-attainment gap**
  `rev_gap = quota – closed_rev`

* **Pipeline-coverage gap**
  `coverage_gap = target_pipeline – Σ(weighted_deal_value)`

* **Action-rate gap** (assignment utilisation)
  `action_rate_gap = target_action_rate – (actioned_CLIs / assigned_CLIs)`

* **Trust/interaction gap**
  `trust_gap = target_touch_count – qualified_touches`
  \$-impact per missing touch = `β₁` from your trust model × avg deal value.

* **Product-fit gap**
  `fit_gap = expected_rev_from_best(CLI/SLI) – actual_rev`

* **Opportunity-score execution gap**
  `opp_score_gap = Σ(score)_assigned – Σ(score)_touched`

* **Reason-code health gap**
  For each bucket: `reason_gap = global_win% – region_win%` (won and lost separately).

---

*Metric registry updates*

```yaml
revenue_gap:
  higher_is_better: false
  unit: USD
  target_source: quota_table

coverage_gap:
  higher_is_better: false
  unit: USD
  target_source: planning_table

action_rate_gap:
  higher_is_better: false
  unit: pct
  target: 0.80       # 80 % of CLIs must be actioned

trust_gap:
  higher_is_better: false
  unit: touches
  target: 3

fit_gap:
  higher_is_better: false
  unit: USD

opp_score_gap:
  higher_is_better: false
  unit: score_points

reason_gap_resource:
  higher_is_better: false
  unit: pct
```

---

*Hypothesis configs (examples)*

* `pipeline_mix.yaml` → `multi_dim_groupby` on stage, vertical, deal\_size.
* `action_util.yaml` → `single_metric_compare` on `action_rate`.
* `trust_model.yaml` → `single_metric_compare` on `qualified_touches`.
* `cli_sli_choice.yaml` → `multi_dim_groupby` on `assignment_type`.
* `opp_score_bias.yaml` → `time_series_compare` tracking score calibration.
* `reason_bucket.yaml` → `multi_dim_groupby` on `(reason_bucket, outcome)`.

---

*Handler sketch for action-rate gap*

```python
def evaluate_action_gap(df_region, target=0.80):
    rate = df_region.actioned.sum() / df_region.assigned.sum()
    gap  = target - rate
    conf = abs(gap) / target
    return {
        "metric": "action_rate_gap",
        "value": gap,
        "confidence": conf,
        "direction": "negative" if gap > 0 else "positive"
    }
```

---

*Ranking logic*

1. Compute all gap metrics for the focal region.
2. Pass each through its hypothesis handler to get `explained_ratio`.
3. Rank by `confidence × impact_$`.
4. Surface top drivers; narrative template:
   “Revenue miss = \$12 M. Key drivers:
   • 45 % from low action-rate on high-score CLIs (utilisation 52 % vs 80 % target)
   • 30 % from pipeline coverage shortfall (3.1× vs 4× quota)
   • 15 % from resource-issue losses (win-rate −8 pp in ‘resource’ bucket).”

---

*Design notes*

* Use **pacing curves** to time-adjust all gap metrics; flag risk early.
* Keep targets in a central table so VPs change goals without code edits.
* Store *self-closed* CLIs as a separate funnel stage; exclude them from win-rate but include them in action-rate gap.
* Retrain trust and opportunity-score models quarterly; feed new β and scores into the same pipeline—config ≠ code.

This structure lets the RCA engine pinpoint whether the current shortfall is coverage, execution, workload, or product-mix—and quantify exactly how much each fixes the VP’s goal gap.

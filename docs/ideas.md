# Advanced RCA Implementation Ideas

## Dependency Chains in Root Cause Analysis

### Metric Dependency Graphs
Many root-cause metrics (e.g., Closed Won %) are functions of upstream inputs:

```
Closed Won % = f(Pipeline Quality, Workload per AM, Vert-Fit Score, ...)
```

When an upstream metric like Workload per AM is itself anomalous, the analysis must drill deeper:

```
Workload per AM = f(Active_AM_Count, Pipeline_Load)
```

This necessitates a system that supports dependency graphs rather than flat hypothesis lists.

### DAG-Based Implementation
Hypotheses can be expressed as a Directed Acyclic Graph (DAG) in YAML configuration:

```yaml
metrics:
  closed_won_pct:
    compute: direct              # leaf metric from table
    hypotheses:
      - workload_per_am
      - vert_fit_score

  workload_per_am:
    compute: derived             # derived metric
    formula: "cli_count / active_am"
    source_metrics: [cli_count, active_am]
    hypotheses:
      - active_am_change
      - cli_mix

  active_am_change:
    compute: direct
    hypotheses: []               # leaf

  vert_fit_score:
    compute: handler             # custom handler
    handler: fit_by_vertical
    hypotheses: []
```

The `compute` key determines metric calculation method:
- `direct`: Read directly from data source
- `derived`: Calculate using formula with other metrics
- `handler`: Use custom Python handler

The engine walks the graph depth-first, evaluating each node in sequence.

### Recursive Evaluation Logic
```python
def evaluate(node):
    # Calculate metric value based on type
    if node.compute == 'derived':
        node.value = eval(node.formula)          
    elif node.compute == 'handler':
        node.value = HANDLERS[node.handler](...)
    
    # Determine anomaly status
    node.is_anomalous = is_outlier(node.value)

    # Recurse through children if anomalous
    for h in node.hypotheses:
        child = evaluate(h)
        node.children.append(child)

    # Rank children by support score
    node.children.sort(key=lambda c: c.support_score, reverse=True)
    node.support_score = combine(node.children)
    return node
```

## LLM-Powered Analysis

### Tool-Based Approach
An LLM agent could navigate the analysis graph using specialized tools:

| Tool | Purpose |
|------|---------|
| `eval_metric(name)` | Returns value & baseline z-score |
| `eval_hypothesis(name)` | Executes handler/formula, returns support score & visualization |

### Agent Reasoning Flow
1. Query metric status: "Is closed_won_pct anomalous?"
2. Based on result, query relevant hypotheses
3. Dynamically choose which branches to explore based on support scores
4. Generate narrative explanations for findings

### Safeguards
- Depth/call limits (max 3 levels, 15 tool calls)
- Schema enforcement for tool returns
- Support score thresholds for exploration

## Narrative Generation
Each node can populate a templated explanation:

```
{metric_name} is {anomaly_dir} by {delta:.1%} vs global.
Primary driver: {top_child.name} (support score {top_child.support:.2f})
{top_child.insight_sentence}

Secondary factors: {secondary_list}
```

If no strong explanations are found (all support_scores < 0.3):

```
No quantified child hypothesis explains the anomaly.
Escalate for qualitative investigation (AM feedback).
```

## Pacing-Adjusted Analysis

### Problem Statement
Metrics like "Low-pitch" or "Low-win" might represent false anomalies if a region is simply following its normal quarterly progression pattern, requiring pacing-adjusted baselines.

### Implementation Approach
1. Build pacing curves from historical data:
   ```sql
   -- Calculate cumulative share by day-of-quarter
   SELECT
     region,
     day_number,
     SUM(metric_value)::float /
     SUM(metric_value) OVER (PARTITION BY region, quarter) AS cum_share
   FROM initiatives_fct
   GROUP BY region, quarter, day_number;
   ```

2. Update metric registry with pacing information:
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

3. Create pacing-adjusted evaluator:
   ```python
   def pacing_adjusted_z(metric_val, today, region, curve_df):
       expected = curve_df.loc[(region, today), "expected"]
       resid    = metric_val - expected
       sigma    = curve_df.loc[region, "residual_std"]
       z        = resid / sigma
       return z
   ```

4. Skip RCA for on-pace metrics (|z| < threshold), focusing analysis on true anomalies

## Semantic Layer for Metrics

### Challenge
Pre-materializing tables for all combinations of filters (phase, time windows, verticals) is impractical.

### Solution: Dynamic Metric Generation
Create a thin semantic layer with:
- Base fact tables (`initiatives_fct`)
- Dimension tables (calendar, AM, product, vertical)
- Metric registry in YAML

Example registry entry:
```yaml
closed_won_pct:
  base_table: initiatives_fct
  numerator:
    agg: count
    filters:
      - phase = 'CLOSED_WON'
  denominator:
    agg: count
    filters:
      - initiative_type = 'CLI'
      - phase = 'CLOSED'
  grain: [region]
  window: quarter_to_date
```

### Dynamic Query Generation
A SQL generator converts registry entries to queries at runtime:

```python
from metrics_sdk import get_metric

cli_closed_won = get_metric("closed_won_pct", region="AM-NA")
workload_per_am = get_metric("workload_per_am", region="AM-NA")
```

### LLM Agent Integration
Expose semantic layer through tools:
```json
{
  "name":"query_metric",
  "description":"Run a registered metric with optional filters",
  "parameters":{
    "metric_name":{"type":"string"},
    "region":{"type":"string"},
    "phase":{"type":"string","optional":true},
    "vertical":{"type":"string","optional":true}
  }
}
```

### Optimization Techniques
- SQL template caching
- Incremental materializations
- Dimension pruning
- Registry-based governance

This approach enables flexible analysis without the complexity of pre-building thousands of data slices, while ensuring reproducibility and governance. 
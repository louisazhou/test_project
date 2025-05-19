### 1 · Why chains appear  
Many “root‑cause” metrics ( e.g. **Closed Won %** ) are *functions* of upstream inputs:

```
Closed Won %  =
   f(  Pipeline Quality,
       Workload per AM,
       Vert‑Fit Score,
       … )
```

If **Workload per AM** is itself anomalous, you need to drill one layer deeper:

```
Workload per AM =
   f( Active_AM_Count ,
      Pipeline_Load )
```

So an RCA engine must support **dependency graphs**, not a flat list of hypotheses.

---

## 2 · Deterministic implementation

### 2‑A.  Express hypotheses as a DAG in YAML

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
    compute: handler             # custom handler (fit_by_vertical)
    handler: fit_by_vertical
    hypotheses: []
```

* `compute` key tells the engine **how to obtain** the metric:
  * `direct` → read from table  
  * `derived` → run formula on other metrics  
  * `handler` → call a custom Python handler (you already generated two)

* `hypotheses` is a list of child nodes; the engine walks the graph depth‑first.

### 2‑B.  Engine logic (pseudo‑code)

```python
def evaluate(node):
    if node.compute == 'derived':
        node.value = eval(node.formula)          # uses source_metrics values
    elif node.compute == 'handler':
        node.value = HANDLERS[node.handler](...)
    # determine anomaly flag
    node.is_anomalous = is_outlier(node.value)

    # recurse if anomalous and there are children
    for h in node.hypotheses:
        child = evaluate(h)
        node.children.append(child)

    # rank children by support_score
    node.children.sort(key=lambda c: c.support_score, reverse=True)
    node.support_score = combine(node.children)
    return node
```

* `combine()` could be max, weighted average, or your concentration score.  
* The graph (a `networkx.DiGraph`) guarantees no circular references.

---

## 3 · LLM‑agent version

### 3‑A. Expose **two tool types**

| Tool | Purpose |
|------|---------|
| `eval_metric(name)` | returns value & baseline z‑score |
| `eval_hypothesis(name)` | executes handler / formula, returns support_score & figure |

### 3‑B.  Agent reasoning pattern

1. **Ask**: “Is *closed_won_pct* anomalous?”  
   → calls `eval_metric("closed_won_pct")`
2. sees it is low → queries top hypotheses list from config.
3. **Chooses** the “workload_per_am” hypothesis  
   → calls `eval_hypothesis("workload_per_am")`
4. The tool returns `support_score=0.85` **and** cites that workload is high because `active_am_change` is −18 %.  
   The agent may now recursively test that child:
   → `eval_hypothesis("active_am_change")` …

The LLM acts like a DFS walker, deciding dynamically **which branch** to expand further, stopping when support scores fall below threshold or depth limit.

### Safeguards

* **Depth / call‑limit**—max 3 levels or 15 tool calls.  
* **Schema enforcement**—each tool returns JSON with mandatory keys; the orchestrator validates.

---

## 4 · Narrative template

At each node the engine (or LLM) can fill a generic template:

```
{metric_name} is {anomaly_dir} by {delta:.1%} vs global.
Primary driver: {top_child.name} (support score {top_child.support:.2f})
{top_child.insight_sentence}

Secondary factors: {secondary_list}
```

If **all children** have `support_score < 0.3`, the engine outputs:

```
No quantified child hypothesis explains the anomaly.
Escalate for qualitative investigation (AM feedback).
```

---

## 5 · Putting it together in your repo

1. **Add** the new DAG YAML under `config/metric_graph.yaml`.  
2. **Register** the two new handlers (`reason_overindex_handler`, `fit_by_vertical_handler`) in `src/handlers/__init__.py`.  
3. **Refactor** existing `automation_pipeline.py`:

```python
from graph_runner import evaluate
root = build_graph("closed_won_pct", yaml_config)
result_tree = evaluate(root)
render_report(result_tree)
```

4. **Optional agent mode**: wrap the same graph runner functions as callable tools and deploy an `openai` function‑calling chat.

---

### TL;DR

* **Deterministic path** → a YAML‑defined DAG + recursive evaluator.  
* **LLM‑agent path** → expose evaluator tools; agent walks the same DAG interactively.  

Both reuse your current config style and new handlers with minimal churn.
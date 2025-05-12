**Data Engineering Alignment Meeting: Structured Agenda**

**Objective:**
Ensure alignment between DS and DE on the architecture, data format, and handoff-readiness of the diagnostics framework. Confirm that the current output and data structures are suitable for future engineering ownership and downstream integration.

---

### 1. Project Overview

* Goal: Scalable DS-driven root cause diagnostics framework
* Design principles: Config-driven, structured outputs, downstream-ready
* Weekly reports are generated as Google Slides for now, but will migrate to Unidash eventually

---

### 2. Data Flow DAG

```mermaid
flowchart TD
  A[Raw Inputs - SQL Queries] --> B[DataCatalog - Column Filtering and Renaming]
  B --> B2[Load Configs - metrics.yaml & hypotheses.yaml]
  B2 --> C[Metric Frame - Aggregated by Region]
  C --> D[Anomaly Detection - Global Z-Score]
  D --> E{Is Region Anomalous?}
  E -- Yes --> F[Run Hypotheses Loop - Defined per Metric]
  F --> G[Evaluate by Type - e.g. single_dim, reason_mix]
  G --> H[Generate Narrative, Score (if any), Figures]
  H --> I[Append to Hypothesis Results]
  E -- No --> J[Record 'No Anomaly']
  I --> K[Assemble Metric-Region Output JSON]
  J --> K
  K --> L[Write Structured Output - JSON / SQLite / DataFrame]
  L --> M1[Google Slide Generator]
  L --> M2[LLM Agent]
  L --> M3[Unidash Dashboard]
```

**To confirm with DE:**

* Should the SQL logic per metric/hypothesis live in the same YAML or external repository?
* Do we cache/store those results in Parquet or just consume them immediately?
* Is this graph complete for how you'd manage pipeline triggers and orchestration?

---

### 3. Output Structure Design

*Visual aid: Sample JSON structure*

* Each output represents a (metric, region) pair
* Fields include:

  * `metric` (str): name of the KPI being evaluated
  * `metric_df` (list of dicts): full metric-level dataframe stored as `df.to_dict("records")`, containing one row per region
  * `metric_anomaly_dict`: a nested dictionary with region-level anomaly information, containing the following keys:

    * `region` (str): L4 region name (e.g., "AM-NA") being evaluated
    * `is_anomalous` (bool): whether this region shows abnormal metric behavior
    * `metric_value` (float): actual value of the metric in this region
    * `z_score` (float): z-score of the metric vs global baseline
  * `figures` (list of dicts): each with keys `path` (str), `purpose` (str), `order` (int)
  * `hypotheses` (list of dicts): each hypothesis result includes:

    * `name` (str): hypothesis name
    * `type` (str): hypothesis type (e.g., "single\_dim", "reason\_mix", etc.)
    * `score` (float or null): confidence score (if applicable)
    * `selected` (bool): whether it was chosen as most plausible explanation
    * `narrative_template` (str): Jinja-style string to be rendered
    * `parameters` (dict): named parameters to fill into template
    * `payload` (dict): hypothesis-type-specific results (e.g., score breakdown, top N reasons, mini tables)
    * `figures` (list): same structure as above, may include hypothesis-specific visualizations
* Payload may contain:

  * score components (if applicable)
  * table-like results: stored as `df.to_dict("records")`

Example of a payload DataFrame as markdown:

| reason         | pct  | loss   |
| -------------- | ---- | ------ |
| Not responsive | 32.1 | 120000 |
| Trust issue    | 18.4 | 85000  |
| Other          | 12.5 | 50000  |

**To confirm with DE:**

* Is this structured payload format acceptable?
* Any constraints on saving list-of-dict data inside JSON fields?
* Do you recommend flattening or nesting?

*Visual aid: Sample JSON structure*

```json
{
  "metric": "cli_closed_pct",
  "metric_df": [
    {"region": "AM-NA", "metric_value": 0.087, "z_score": 2.1},
    {"region": "AM-APAC", "metric_value": 0.043, "z_score": -0.9},
    {"region": "AM-EMEA", "metric_value": 0.034, "z_score": -1.2}
  ],
  "metric_anomaly_dict": {
    "region": "AM-NA",
    "is_anomalous": true,
    "metric_value": 0.087,
    "z_score": 2.1
  },
  "figures": [
    {"path": "figs/cli_closed_pct/AM-NA_summary.png", "purpose": "overview", "order": 1}
  ],
  "hypotheses": [
    {
      "name": "SLI_per_AM",
      "type": "single_dim",
      "score": 0.82,
      "selected": true,
      "narrative_template": "Region {{ region }} has {{ delta_pct }}% higher SLI/AM workload than global mean.",
      "parameters": {
        "region": "AM-NA",
        "delta_pct": 34.6
      },
      "figures": [
        {"path": "figs/cli_closed_pct/SLI_per_AM_bar.png", "purpose": "support", "order": 1}
      ],
      "payload": {
        "score_components": {
          "z": 1.8,
          "direction_match": 1,
          "consistency": 0.6
        }
      }
    },
    {
      "name": "Top_closed_lost_reasons",
      "type": "reason_mix",
      "score": null,
      "selected": false,
      "narrative_template": "Top reasons for loss: {{ reasons[0].reason }} ({{ reasons[0].pct }}%), {{ reasons[1].reason }}...",
      "parameters": {
        "reasons": [
          {"reason": "Not responsive", "pct": 32.1},
          {"reason": "Trust issue", "pct": 18.4}
        ]
      },
      "figures": [
        {"path": "figs/cli_closed_pct/reasons_pie.png", "purpose": "descriptive", "order": 1}
      ],
      "payload": {
        "reason_breakdown": [
          {"reason": "Not responsive", "pct": 32.1, "loss": 120000},
          {"reason": "Trust issue", "pct": 18.4, "loss": 85000},
          {"reason": "Other", "pct": 12.5, "loss": 50000}
        ]
      }
    }
  ]
}
```

**To confirm with DE:**

* Is this structured payload format acceptable?
* Any constraints on saving list-of-dict data inside JSON fields?
* Do you recommend flattening or nesting?

---

### 4. Figure Management

* Each metric-region may produce:

  * metric-level figures (e.g., z-score bar charts)
  * hypothesis-specific figures (bar charts, pie charts, etc.)
* Figures saved as .png with purpose & order metadata
* Stored as relative paths inside JSON

**To confirm with DE:**

* Do you recommend a naming convention or directory structure?
* Will these be mounted somewhere for dashboard use?

---

### 5. Handling Hypothesis Diversity

* Currently supporting 9+ hypothesis types, each with different logic
* Output is unified by wrapping hypothesis-specific results into `payload`
* Only some types produce a `score`; others are descriptive

**To confirm with DE:**

* Is this strategy (type-dispatched payload) maintainable?
* Any schema validation suggestions?

---

### 6. Downstream Integration

* Immediate consumers:

  * Google Slides report generator
  * Google Sheet intermediate table(s)
  * LLM agent (structured insight generation)
  * Future Unidash dashboards

**To confirm with DE:**

* Will Unidash support:

  * Hover interactions (e.g., reason breakdowns)?
  * Drill-downs from KPI to hypothesis?
  * Image preview from path?

---

### 7. Ownership Boundary

* DS handles: config logic, hypothesis templates, scoring rules
* DE expected to: build pipelines, manage I/O, support format standardization, connect to dashboards

**To confirm with DE:**

* What early decisions should we lock in to minimize future refactors?
* How would you prefer we deliver config, outputs, and supporting assets?
* Where should SQL queries per hypothesis/metric be stored, versioned, and triggered?

---

### Wrap-Up Questions

> Are there any infra constraints or storage standards we should align to now?
> Does our current schema and output strategy feel scalable and maintainable from your side?
> What do you need from us next to build a clean handoff path?

---

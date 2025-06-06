# 1 · Status
> Only "Directional" hypothesis are root-cause and can be directly comparable. The rest are more "descriptive". They add visuals but how exactly do they compare to the rest requires further ranking. 

| Metric layer       | Hypothesis type                                                   | Output today                   |
| ------------------ | ----------------------------------------------------------------- | ------------------------------ |
| L4 Operational KPI | [[directional\|Directional]] (ranked score)                       | ✔ auto‑refresh into WBR slides |
|                    | [[depth-spotter\|Depth‑spotter]] (lagging / outperforming L8)     | ✔ auto‑refresh into WBR slides |
|                    | [[Product‑mix]] (index vs ROW)                                    | in Bento Notebook              |
|                    | [[Reason‑mix]] (these reasons are abnormally high in this region) | in Bento Notebook              |
|                    | [[Product‑vertical fit]]                                          | in Bento Notebook              |
|                    | [[Oaxaca‑Blinder\|Mix‑vs‑Efficiency decomposition]]               | in Bento Notebook              |

---

## 2 · Root-Cause Hypothesis Types

### 2.1 [[directional|Directional]] ("D‑score")

Most common, fully standardized – works when KPI and hypothesis can be **ranked & compared across regions**.

**Scoring formula** 

```python
final_score = 0.6*direction_alignment \
            + 0.4*explained_ratio
explains = final_score > 0.5
```

| Component       | Meaning                            | Implementation                                              |
| --------------- | ---------------------------------- | ----------------------------------------------------------- |
| Sign Agreement  | Metric & hypo move in expected dir | `score_sign = (regions where sign(Δmetric)==sign(Δhypo))/n` |
| Explained ratio |                                    | min(Δ\_h/Δ\_m,1)                                            |
Output:  **Ranked Bar chart (based on hypothesis score) with Callout Texts for the highest-score hypothesis (call that root-cause)**

---

### 2.2 [[depth-spotter|Depth‑spotter]]

*What*: slice KPI → L8; flag pockets ±1 σ from region mean.
*Why*: identifies **which sub‑markets actually drag L4**.

*Output*: bar chart of L8s, colored lag vs out‑perform.


---

### 2.3 Mix‑vs‑Efficiency ([[Oaxaca‑Blinder]])

*Use‑case*: How much % of gap is due to better/worse composition, and how much is driven by performance?
*Breaks total gap into*
* **Mix / Composition** (employee job level mix, book construct revenue mix)
* **Efficiency** (within‑cell conversion).

Formula:
`Δ_total = Σ(mix_R−mix_ROW)·rate_ROW + Σ mix_R·(rate_R−rate_ROW)`

Visual: 2‑bar waterfall.

---

### 2.4 Product‑mix index

*What*: region’s win‑rate per product vs ROW baseline.
Outputs green/red index (>1 good, <1 needs review).
Helps answer: *“Switch 5 % pitches from P3 to P1 gives +1 pp uplift.”*

---

### 2.5 Reason‑mix (closed‑lost or closed-won reasons)

% difference & dollar loss by AM‑tagged reason → categorical uplift table.

---

### 2.6 Product‑vertical fit

1. Build global **product × vertical** win‑rate map (cells with wins ≥ 30).
2. Keep top‑k cells per vertical (best‑fit set).
3. Compare each region’s mix & within‑cell rates.
4. Simulate uplift if region re‑allocates 5–10 % pitches to best‑fit.

Visual: heatmap 
_+ waterfall (mix vs exec vs potential uplift)_.

---
## 3 ·Ideas for Next Phase 

[[Slide = "story", Unidash = "evidence & self‑service"]]
[[pacing-adjusted baseline via comparing to region's own historical progress]]
[[pivot from the % stage view to?]]
[[stage-specific Sankey chart]]
[[BHB decomposition for change drivers (in time)]]
[[BHB decomposition for exits (in stage)]]
[[define gap to goal?]]

# 2 · Slides

## Problem Statement

- Current “RCA” work is **reactive & ad-hoc** – leadership spots an odd metric, analysts scramble to pull data and guess why.
    
- WBR & DCMP dashboards overflow with numbers; they surface **what** happened but never the **insight** or **action**.
    
- We need an **automation engine** that turns a dense data library into a one-pager TL;DR:
    
    _“Metric A/B is off; here’s the top reason and next step.”_
    
- Guiding idea: “He who has a _why_ can bear almost any _how_.” Automating the _why_ shrinks the gap from data to action (gap to goal).

## All Hypotheses Types

1. **Directional root-cause** (any metric)
    - Check which intuitive hypotheses are directionally wrong.
    - Rank the “correct” ones by likelihood.
        
2. **Depth spotter** (any metric)
    - Identify whether a single sub-region drives the gap or all are equally weak.
        
3. **Construct mix** (win/loss rate)
    - **Book construct** – higher share of high-revenue clients.
    - **AM construct** – higher share of senior reps.
    - **Product mix** – greater exposure to easy-win products.
    - **Product-vertical fit** – products inherently match local verticals better.
    
4. **Tagged reasons** (win/loss rate) – AM-supplied close reasons.
5. _**Pacing / pipeline** – gap explained by timeline or flow._

## Directional Root-Cause

- **Point-of-Departure (PoD)**: hand-wavy, manual, no formal ranking.
- **Point-of-Arrival (PoA)**: automated, proactive, quantitative.

- Benchmarks:
	- L2 roll-up or “Rest of L4”.
	- _pacing-adjusted baseline (region vs. its own historic curve)._

- **D-score** components    
    1. User-provided directional intuition (e.g., “if A ↓ then B ↑”).
    2. Statistical extremeness of hypothesis metric.
    3. _Actionability weight to emphasize on more controllable shortfalls._

## Depth Spotter
- **PoD**: Analysts eyeball large slices or gaps.
- **PoA**: Unified scoring surfaces culprits by:
	- coverage: volume share within region
	- contribution: share of region-level gap **(Σ contrib = ±1)**
	- _Can be chained after Directional RCA for granular dives._
## Construct Mix 
- **PoD**
    - Book / AM mix shown visually; no quantified impact.
    - Product mix largely ignored?    
- **PoA**
    - Decomposes gap into **mix-induced** vs. **efficiency-induced**.    
    - Performance index vs. Rest of World, e.g.
        - Product X closes 2× faster.
        - 27% of CLIs are easily-winnable product-vertical combo
    - Action levers:
        - Shift pitches to high-win products (gain a pp win-rate, recover $Y).    
        - Standardize CLIs to top-performing combo across regions.
## Tagged Reason

- **PoD**: Raw reason percentages per product; no prioritization.
- **PoA**:
    - Flags abnormally high “Reason X” in Region A → +N lost/won deals → $R revenue delta.
    - Surface 
	1. _relationship dollar_, _measurement dollar_
	2. _sub-reason, LLM surface richer insight_

## Pacing / Pipeline 

**Pacing / Pipeline:** 
*stage-velocity detector*: quantify where pipeline stalls, distinguish true under-performance from late movers, and point managers to the stage that needs intervention.

Directional + Pacing: velocity gate, pacing curve


## Summary

**Insight Modules Delivered**
- **Directional + sub-region deep dives**: ranks root causes and inserts pre-written “what to do” guidance from config.
- **Book-construct / AM-level mix**: isolates win-rate gaps tied to client size and rep seniority.
- **Product vs. AM skill split**: quantifies how much performance comes from inventory quality vs. rep execution.
- **Product-vertical fit**: detects under-performance driven by mismatched product assignments.
- **Closed-reason mix**: flags regions with anomalous win/loss reasons; pairs each with corrective levers.

**Coming Next**

## Infrastructure Supporting The Automation : "RCA_package"

| **Automation**                                          | **Flexible Granularity**                                        | **Deterministic + Extensible**                                     |
| ------------------------------------------------------- | --------------------------------------------------------------- | ------------------------------------------------------------------ |
| YAML / config-driven                                    | Fully parameterized queries                                     | Versioned, deterministic outputs                                   |
| No-code edit needed for new metric-hypothesis pairs     | Today: L4 -> L8 drill-downs, for MM; <br>Ready: L8→L12, for SBG | Same stored results can feed Slides, Unidash, or API without rerun |
| New hypotheses auto-run, rank, and surface side-by-side | Easy switch for grain / market change                           | LLM-ready summarizer layer adds custom prose & reasoning           |

## Infrastructure Supporting The Automation : "slide-generator"
> One-call engine turns any notes + plots into formatted Slide decks—minimal manual PowerPoint work, fully extensible.
### **Core engine (live)**
- Publication-ready decks in minutes—analysts spend time on insight, not formatting.
- Auto-layout titles, paragraphs, tables, images; paginates when out of space.
- Smart defaults: alt-row shaded tables, width-capped images.
### **Next lift**
- Direct Markdown rendering (bold, headers, lists, code).
- Unified layout DSL: ::: columns + YAML front-matter + upgraded parser for side-by-side blocks, full-bleed charts, two-column text—all in one pass.
- Comment feedback loop: pull Google Slides comments/@mentions via Drive API and feed them back into RCA for the next build.

## 1 · What is already automated?

| Metric layer       | Hypothesis type                                                   | Output today                   |
| ------------------ | ----------------------------------------------------------------- | ------------------------------ |
| L4 Operational KPI | **Directional** (ranked score)                                    | ✔ auto‑refresh into WBR slides |
|                    | **Depth‑spotter** (lagging / outperforming L8)                    | in Bento Notebook              |
|                    | **Product‑mix** (index vs ROW)                                    | in Bento Notebook              |
|                    | **Reason‑mix** (these reasons are abnormally high in this region) | in Bento Notebook              |
|                    | **Product‑vertical fit**                                          | in Bento Notebook              |
|                    | **Mix‑vs‑Efficiency decomposition**                               | in Bento Notebook              |
Only "Directional" hypothesis are root-cause and can be directly comparable. The rest are more "descriptive". They add visuals but how exactly do they compare to the rest requires further ranking. 

*May Goal: automate the rest, store all table in Hive → Unidash; 
Slides keep only exec summary texts, ranked numbers and the more important images.*

---

## 2 · Root-Cause Hypothesis Types

### 2.1 Directional ("D‑score")

Most common, fully standardised – works when KPI and hypothesis can be **ranked & compared across regions**.

**Scoring formula** *(weights will be discussed within DS team the Week of May 19th)*

```python
final_score = 0.5*direction_alignment \
            + 0.2*consistency          \
            + 0.2*hypo_z_norm          \
            + 0.1*explained_ratio
explains = final_score > 0.5
```

*Components – 50 / 20 / 20 / 10 %*  (see table)

| Component           | Meaning                                              | Implementation                 |
| ------------------- | ---------------------------------------------------- | ------------------------------ |
| Direction alignment | Metric & hypo move in expected dir                   | sign check (same / opposite)   |
| Consistency         | Strength of correlation across all regions           | `abs(ρ)`                       |
| Hypothesis Z        | How far hypo in anomalous region is from global mean | bucket → 1.0 / 0.7 / 0.6 / 0.3 |
| Explained ratio     |                                                      | min(Δ\_h/Δ\_m,1)               |

Output:  **Ranked Bar chart (based on hypothesis score) with Callout Texts**
Slide sample already in deck.

---

### 2.2 Depth‑spotter

*What*: slice KPI → L8; flag pockets ±1 σ from region mean.
*Why*: identifies **which sub‑markets actually drag L4**.

*Output*: bar chart of L8s sorted by z‑score, coloured lag vs out‑perform.


---

### 2.3 Mix‑vs‑Efficiency (Blinder–Oaxaca)

*Use‑case*: How much % of gap is due to better/worse composition, and how much is driven by performance?
*Breaks total gap into*

* **Mix / Composition** (employee job level mix, book construct revenue mix)
* **Efficiency** (within‑cell conversion).

Formula:
`Δ_total = Σ(mix_R−mix_ROW)·rate_ROW   +   Σ mix_R·(rate_R−rate_ROW)`

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

Visual: heatmap with region dots + waterfall (mix vs exec vs potential uplift).

---
## 3 · Open items for discussion

1. **Where does SSPO need drill/filters?** 
	1. What data needs to be populated in the Manager View spec?
2. **Slides vs Unidash split**. Slides remain the storybook; Unidash is the evidence warehouse.
	1. Which figures or component should remain in deck?
		1. 1. Exec one-pager: KPI heatmap/table & CTA bullets
		2. 1 PNG per flagged KPI 
		3. Narrative auto-filled from Hive Table
	2. What moves exclusively to Unidash? 
		* Hypothesis ranking table with sort / search
		* CI topic uplift matrix
		- Interactive heatmaps (product × vertical)
		* (future) DFS drill from metric → hypothesis → leaf evidence 
		* (future) Stage-flow alluvial with hover counts
		* (future) Pacing curves & velocity tables
	3. How can we make the Unidash better to serve all layers of users?
3. **Dream Next Phase of Interaction** 

| Rank  | Impact | Rationale                                                                                                                                             | Who benefits                  |
| ----- | ------ | ----------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------- |
| **1** | 🔴🔴🔴 | **Self-serve diagnostic filters (region, team, metric, date)** replace 1:1 deck requests; DS time goes to model improvements instead of slide re-cuts | SSPO frontline, Sales leaders |
| **2** | 🔴🔴   | **LLM (Metamate) layer on top of live data**: *“Compare week 3 vs 9”* / *“Show verticals common to all lagging L8s”* / surfacing AM comment excerpts  | All leadership tiers          |
| **3** | 🔴🔴   | **Drill-to-evidence hover** (preview card with mini-waterfall + AM quotes) removes 5–10 slide hops per question                                       | VP reviews, QBR live demos    |
| **4** | 🔴🔴   | **One canonical data source** (Hive / Manifold behind Unidash) → DE can own orchestration; no copy-paste drift                                        | DE, Compliance                |
| **5** | 🔴     | **Manager-view dashboards** reading from the same table power future SSPO roadmap without extra DS coding                                             | SSPO analytics                |
| **6** | 🔴     | **Screenshot → Slide** workflow still possible for weekly WBR deck; DS keeps only 3 static summary pages                                              | WBR owner                     |

---
## 4 · Next steps 

| When        | Deliverable                                                 |
| ----------- | ----------------------------------------------------------- |

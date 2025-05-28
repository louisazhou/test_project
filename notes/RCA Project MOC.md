# 1 · Status
> Only "Directional" hypothesis are root-cause and can be directly comparable. The rest are more "descriptive". They add visuals but how exactly do they compare to the rest requires further ranking. 

| Metric layer       | Hypothesis type                                                   | Output today                   |
| ------------------ | ----------------------------------------------------------------- | ------------------------------ |
| L4 Operational KPI | [[directional\|Directional]] (ranked score)                       | ✔ auto‑refresh into WBR slides |
|                    | [[depth-spotter\|Depth‑spotter]] (lagging / outperforming L8)     | ✔ auto‑refresh into WBR slides |
|                    | **Product‑mix** (index vs ROW)                                    | in Bento Notebook              |
|                    | **Reason‑mix** (these reasons are abnormally high in this region) | in Bento Notebook              |
|                    | **Product‑vertical fit**                                          | in Bento Notebook              |
|                    | [[Oaxaca‑Blinder\|Mix‑vs‑Efficiency decomposition]]               | in Bento Notebook              |

---

## 2 · Root-Cause Hypothesis Types

### 2.1 [[directional|Directional]] ("D‑score")

Most common, fully standardized – works when KPI and hypothesis can be **ranked & compared across regions**.

**Scoring formula** *(weights will be discussed within DS team the Week of May 19th)*

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
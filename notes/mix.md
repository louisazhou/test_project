## 1 · Why “4‑point correlation” is weak & what you can do today

| Fact                                                                      | Implication                                                                   |
| ------------------------------------------------------------------------- | ----------------------------------------------------------------------------- |
| Pearson/Spearman correlation needs many degrees‑of‑freedom to be reliable | With **n = 4** cells, df = 2 ⇒ even ρ = 0.90 is *not* “significant” (p ≈ 0.1) |
| A single outlier flips the sign                                           | The direction (+/–) becomes arbitrary                                         |
| Correlation only measures linear co‑movement                              | It ignores level differences that matter for root‑cause                       |

### 🔧 A drop‑in alternative you can ship now

**Step 1 – Directional screen (still required by TL/manager)**

```text
IF sign(metric_delta) == sign(hypo_metric_delta) ➞ passes direction check
ELSE ➞ fails
```

**Step 2 – Size screen (replace correlation)**
Compute **composition\_gap** and **performance\_gap**:

```
composition_gap = Σ mix_region_i · rate_row_i  –  rate_row_overall
performance_gap = observed_rate_region        –  Σ mix_region_i · rate_row_i
```

> • If |composition\_gap| > X% of global mean → conclude “mix likely driver”
> • Else if |performance\_gap| > Y% of global mean → conclude “efficiency likely driver”
> • Else “weak evidence”

Set X, Y as simple thresholds (e.g., 5 p.p. of win‑rate).
No bootstrap needed; you’ve turned the decision into deterministic rule‑based flags—easy to implement and explain.

*You still record the absolute numbers in payload so later, when you have more historical data, you can run proper statistics.*

---

## 2 · Kitagawa / Oaxaca‑Blinder decomposition—plain‑English walk‑through

Imagine each initiative is bucketed by AM level (`i = L4, L5, L6, …`).
For a given region **R** vs **Rest‑of‑World (ROW)**:

| Symbol       | Meaning                                       |
| ------------ | --------------------------------------------- |
| `mix_R_i`    | Share of initiatives in cell *i* for region R |
| `rate_R_i`   | Win‑rate in cell *i* for region R             |
| `mix_ROW_i`  | Share for ROW                                 |
| `rate_ROW_i` | Win‑rate for ROW                              |

### Total rate difference

```
Δ = ObservedRate_R  –  ObservedRate_ROW
  = Σ mix_R_i * rate_R_i  –  Σ mix_ROW_i * rate_ROW_i
```

### Decompose into two parts

1. **Composition effect (mix difference, rates fixed at ROW):**

   ```
   Δ_mix = Σ (mix_R_i – mix_ROW_i) * rate_ROW_i
   ```

   > *“If region R had ROW’s win‑rates but kept its own mix, this is the gap.”*

2. **Performance (efficiency) effect (rate difference, mix fixed at R):**

   ```
   Δ_perf = Σ mix_R_i * (rate_R_i – rate_ROW_i)
   ```

   > *“If region R kept its mix but used its own rates, this is the gap.”*

By construction: `Δ = Δ_mix + Δ_perf`.

### Interpretation

* **Δ\_mix large** → book or AM‑level composition explains most of the anomaly.
* **Δ\_perf large** → same mix, but R converts better/worse ⇒ execution or efficiency issue.

You can compute both gaps with only the four buckets (L4–L7) **without correlation**.
Store them and compare magnitudes; whichever dominates becomes your “root‑cause flag.”

---

### 🔑  Minimal payload you store

```json
"payload": {
  "composition_gap": 0.005,      // +0.5 pp vs ROW
  "performance_gap": 0.020,      // +2.0 pp
  "dominant_driver": "performance",  // simple if/else rule
  "cell_table": [
    {"level":"L4","mix_R":0.18,"mix_ROW":0.12,"rate_R":0.025,"rate_ROW":0.022},
    ...
  ]
}
```

Downstream systems (Slides, Unidash, LLM) can read:

* Direction check: `dominant_driver`
* Magnitude: `composition_gap`, `performance_gap`
* Detail table for drill‑down

No correlation needed, and the logic is transparent and reproducible.



### How to handle (and communicate) “mixed” or conflicting results across regions

| Situation                                                                  | What you surface                                                                                                              | Why it avoids follow‑up churn                                                                                                                       |
| -------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Region under review shows Δ\_mix > Δ\_perf**                             | “Primary driver: *composition* (book/AM mix). Composition explains **80 %** of this region’s gap.”                            | Gives a single headline + shows share, so stakeholders know it’s dominant, not exclusive.                                                           |
| **Another region shows the opposite pattern**                              | For that region you still report its own dominant driver; your system does **not** compare regions’ drivers unless requested. | Root‑cause analysis is **per region**; conflicting signs aren’t a contradiction—they simply mean different regions are driven by different factors. |
| **Both effects similar in magnitude** (e.g., 0.02 vs 0.025)                | Flag “no single dominant driver” and list both gaps.                                                                          | Pre‑empts questions by acknowledging ambiguity instead of hiding it.                                                                                |
| **Stakeholder asks “why NA is mix‑driven but EMEA is efficiency‑driven?”** | Show the cell detail table; they’ll see NA’s mix skewed to senior AMs, EMEA’s rates lower in every cell.                      | Transparent evidence defuses follow‑up.                                                                                                             |

---

### Concrete rules you can implement

```python
share_mix  = abs(Δ_mix)  / abs(Δ_total)
share_perf = abs(Δ_perf) / abs(Δ_total)

if share_mix >= 0.7:
    dominant = "composition"
elif share_perf >= 0.7:
    dominant = "performance"
else:
    dominant = "mixed"
```

Store in payload:

```json
"composition_gap": 0.005,
"performance_gap": 0.020,
"dominant_driver": "performance",   // or "mixed"
"share_of_total": {"composition": 0.2, "performance": 0.8}
```

Downstream narrative template:

```
*Primary driver for AM‑NA: performance (80 % of gap).*
Book mix accounts for 20 %; both factors shown below.
```

---

### Why this stops endless “what‑about” questions

1. **Region‑specific headline** – Each reader sees the main takeaway for their scope.
2. **Percentage breakdown** – Quantifies how sharp the difference is (or isn’t).
3. **Cell‑level evidence** – Drill‑down table is there if deeper investigation is needed.
4. **Consistent rule** – Same 70 % threshold used everywhere; no ad‑hoc judgement.

You’ll rarely be asked “but EMEA’s numbers look different” because the payload already tells them *why* and *by how much*.


### When **Δ\_mix** and **Δ\_perf** have opposite signs

(*e.g., composition gap = +0.02 pp, performance gap = –0.05 pp*)

---

#### 1 · Interpretation

| Case                               | Meaning                                                                                                                 |
| ---------------------------------- | ----------------------------------------------------------------------------------------------------------------------- |
| **Δ\_mix > 0** (pushes KPI *up*)   | The region’s book / AM‑level mix is **favourable** relative to ROW.                                                     |
| **Δ\_perf < 0** (pulls KPI *down*) | Given that mix, the region converts **worse** than ROW in every cell.                                                   |
| **Total Δ = Δ\_mix + Δ\_perf**     | If signs differ, the two effects **partially cancel**. The observed anomaly is whichever effect dominates in magnitude. |

---

#### 2 · Actionable rule

```python
if sign(Δ_mix) != sign(Δ_perf):
    dominant = "offsetting"
    explanation = (
        "Composition skews positive (+{:.2p}) but is more than offset by "
        "lower within‑cell performance (–{:.2p})."
    ).format(abs(Δ_mix), abs(Δ_perf))
else:
    # previous 70 % rule
```

---

#### 3 · Payload example

```json
"composition_gap": 0.020,
"performance_gap": -0.050,
"dominant_driver": "offsetting",
"share_of_total": {
  "composition": 0.29,
  "performance": 0.71   // in absolute terms
},
"explanation": "Favourable AM mix adds 2 pp to win‑rate, but lower execution subtracts 5 pp; net effect –3 pp."
```

---

#### 4 · Narrative template

> **Offsetting effects:**
> Region {{region}}’s senior‑AM mix would raise win‑rate by **+2 pp**, but within‑cell efficiency is **–5 pp** vs benchmark, producing the net decline.

---

#### 5 · Why this matters

* **Stakeholders see both levers.** They may keep the mix but fix execution.
* **No confusion about sign.** You explicitly tell them the forces cancel.

---

#### 6 · Optional visual

A stacked bar:

```
+2 pp  (composition)   ▮▮
–5 pp  (performance)   ▮▮▮▮▮
-----------------------------
Net –3 pp
```

---

**Bottom‑line:**
Opposite‑sign gaps are perfectly valid; treat them as *offsetting* and highlight which one overrides the other.

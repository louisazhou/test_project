## 1. Rule-Based Threshold Check (Quick Filter)

**Step 1: Compute two gaps**

```python
composition_gap = Σ mix_region_i · rate_row_i  –  rate_row_overall
performance_gap = observed_rate_region         –  Σ mix_region_i · rate_row_i
```

**Step 2: Apply simple thresholds**

```text
If |composition_gap| > X% of global mean → "Mix likely driver"
Elif |performance_gap| > Y% of global mean → "Efficiency likely driver"
Else → "Weak evidence"
```

Use `X = Y = 5pp` (example). This is a fast pre-check before decomposition.

---

## 2. Oaxaca–Blinder Decomposition Logic

**Goal**: Break down total KPI difference between region R and benchmark (e.g., ROW).

### Notation

| Symbol       | Meaning                           |
| ------------ | --------------------------------- |
| `mix_R_i`    | Share in cell *i* for Region R    |
| `rate_R_i`   | Win-rate in cell *i* for Region R |
| `mix_ROW_i`  | Share for ROW (or benchmark)      |
| `rate_ROW_i` | Win-rate for ROW                  |

### Total KPI gap

```python
Δ_total = Σ mix_R_i * rate_R_i – Σ mix_ROW_i * rate_ROW_i
```

### Decomposition

```python
Δ_mix  = Σ (mix_R_i – mix_ROW_i) * rate_ROW_i         # Composition effect
Δ_perf = Σ mix_R_i * (rate_R_i – rate_ROW_i)          # Performance effect
Δ_total = Δ_mix + Δ_perf
```

---

## 3. Driver Detection Rules

```python
share_mix  = abs(Δ_mix)  / abs(Δ_total)
share_perf = abs(Δ_perf) / abs(Δ_total)
```

```python
if share_mix >= 0.7:
    dominant = "composition"
elif share_perf >= 0.7:
    dominant = "performance"
else:
    dominant = "mixed"
```

### Handle Opposite Sign Gaps

```python
if sign(Δ_mix) != sign(Δ_perf):
    dominant = "offsetting"
    explanation = (
        "Composition skews positive (+{:.2p}) but is more than offset by "
        "lower within‑cell performance (–{:.2p})."
    ).format(abs(Δ_mix), abs(Δ_perf))
```

---

## 4. Output Payload Example

```json
{
  "composition_gap": 0.020,
  "performance_gap": -0.050,
  "dominant_driver": "offsetting",
  "share_of_total": {
    "composition": 0.29,
    "performance": 0.71
  },
  "explanation": "Favourable AM mix adds 2 pp to win‑rate, but lower execution subtracts 5 pp; net effect –3 pp.",
  "cell_table": [
    {"level":"L4","mix_R":0.18,"mix_ROW":0.12,"rate_R":0.025,"rate_ROW":0.022},
    ...
  ]
}
```

---

## 5. Narrative Templates

### a. Clear dominant driver

> *Primary driver for {{region}}: performance (80% of gap).*
> Composition accounts for 20%; both effects shown below.

### b. Mixed case

> *No single dominant driver.*
> Composition explains +2 pp; performance explains –2.5 pp.

### c. Offsetting signs

> **Offsetting effects:**
> Region {{region}}’s mix would raise win‑rate by **+2 pp**,
> but efficiency lags by **–5 pp**, leading to net **–3 pp**.

---

## 6. Optional Visual Aid

```
+2 pp  (composition)   ▮▮
–5 pp  (performance)   ▮▮▮▮▮
-----------------------------
Net –3 pp
```

### Mock data (APAC vs. ROW)

| Segment | Mix (APAC) | Win-rate (APAC) | Mix (ROW) | Win-rate (ROW) |
| ------- | ---------- | --------------- | --------- | -------------- |
| Junior  | 0.40       | 11 %            | 0.60      | 10 %           |
| Senior  | 0.60       | 22 %            | 0.40      | 20 %           |

* ROW overall win-rate = 0.6 × 10 % + 0.4 × 20 % = **14 %**
* APAC overall win-rate = 0.4 × 11 % + 0.6 × 22 % = **17.6 %**
* **Total gap** (APAC − ROW) = **+3.6 pp**

---

### Oaxaca–Blinder decomposition

| Segment | **Composition term**<br>(Δ mix × rate ROW) | **Performance term**<br>(mix APAC × Δ rate) |
| ------- | ------------------------------------------ | ------------------------------------------- |
| Junior  | (0.40 − 0.60) × 10 % = **–2.0 pp**         | 0.40 × (11 % − 10 %) = **+0.4 pp**          |
| Senior  | (0.60 − 0.40) × 20 % = **+4.0 pp**         | 0.60 × (22 % − 20 %) = **+1.2 pp**          |
| **Sum** | **+2.0 pp (Δ\_mix)**                       | **+1.6 pp (Δ\_perf)**                       |

* **Composition gap (Δ\_mix)** = +2.0 pp
  → more Seniors in APAC lifts the rate.
* **Performance gap (Δ\_perf)** = +1.6 pp
  → APAC managers win more often even within the same tier.
* **Δ\_total = +3.6 pp = Δ\_mix + Δ\_perf**

---

### Quick rule-based check (X = Y = 5 pp)

* |Δ\_mix| = 2 pp  < 5 pp
* |Δ\_perf| = 1.6 pp < 5 pp

Both below the 5 pp threshold ⇒ flag as **“mixed / weak evidence”** in the fast filter; full decomposition above gives the exact split.

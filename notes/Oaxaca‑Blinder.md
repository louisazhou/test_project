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


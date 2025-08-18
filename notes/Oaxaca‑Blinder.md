## Data

| **product**            | **region_mix_pct** | **rest_mix_pct** | **region_rate** | **rest_rate** |
| ---------------------- | ------------------ | ---------------- | --------------- | ------------- |
| A+SC/AC                | 0.02783            | 0.02418          | 0.3209          | 0.5211        |
| Automation             | 0.0917             | 0.08926          | 0.37373         | 0.60451       |
| CAPI                   | 0.07841            | 0.09782          | 0.3843          | 0.32318       |
| CTX                    | 0.01163            | 0.02978          | 0.05357         | 0.32846       |
| Creative               | 0.14134            | 0.12702          | 0.52461         | 0.19605       |
| Gen_AI                 | 0.11317            | 0.05678          | 0.52061         | 0.76876       |
| Lead Ads               | 0.04486            | 0.05495          | 0.6283          | 0.60317       |
| Other                  | 0.04014            | 0.04014          | 0.60123         | 0.16913       |
| Paid Messaging         | 0.00561            | 0.0379           | 0.44083         | 0.15994       |
| Reels                  | 0.17333            | 0.18592          | 0.27778         | 0.48555       |
| Scaling Good Campaigns | 0.01122            | 0.0379           | 0.22222         | 0.46352       |
| Simplification         | 0.19285            | 0.21834          | 0.44642         | 0.58148       |

# Oaxaca Decomposition — Product-Level Walkthrough (NA vs ROW)

This note shows exactly how to attribute the NA–ROW **rate gap** to each product using Blinder-Oaxaca, with terms that **sum to the top line**. Data are from your corrected table.

## Setup & identity

Per product $i$:

* $m_{A,i}, m_{B,i}$: mix shares (NA vs ROW).
* $r_{A,i}, r_{B,i}$: within-product rates.
* Top lines computed **from the same rows**:
  $\bar r_A=\sum_i m_{A,i}r_{A,i}$, $\bar r_B=\sum_i m_{B,i}r_{B,i}$, total gap $G=\bar r_A-\bar r_B$.

Use ROW as baseline (“two-part” Oaxaca). For each product:

$$
\underbrace{\text{mix}_i}_{\text{composition}}=(m_{A,i}-m_{B,i})\,r_{B,i},\qquad
\underbrace{\text{rate}_i}_{\text{within}}=m_{A,i}\,(r_{A,i}-r_{B,i}),\qquad
\text{gap}_i=\text{mix}_i+\text{rate}_i.
$$

Identity (per row and in total):

$$
\text{gap}_i= m_{A,i}r_{A,i}-m_{B,i}r_{B,i},\qquad \sum_i \text{gap}_i = G.
$$

## Sanity checks (from your inputs)

* $\sum m_A=0.99998$, $\sum m_B=0.99999$ (≈1; rounding OK).
* $\bar r_{\text{NA}}=(m_A\cdot r_A)=\mathbf{0.352988}$ (≈ 0.352996 you stated).
* $\bar r_{\text{ROW}}=(m_B\cdot r_B)=\mathbf{0.494539}$.
* Total gap $G=\mathbf{-0.141551}$.
* $\sum_i \text{gap}_i=\mathbf{-0.141551}$ (exact match).

## Worked row examples (showing the arithmetic)

**Reels** (mA=0.17333, mB=0.18592, rA=0.44038, rB=0.55034)

* mix = $(0.17333-0.18592)\cdot0.55034 = -0.01259\cdot0.55034 = \mathbf{-0.006929}$
* rate = $0.17333\cdot(0.44038-0.55034)=0.17333\cdot(-0.10996)=\mathbf{-0.019059}$
* gap = $\mathbf{-0.025988}$

**Simplification** (mA=0.19285, mB=0.21834, rA=0.44642, rB=0.58148)

* mix = $(0.19285-0.21834)\cdot0.58148 = -0.02549\cdot0.58148 = \mathbf{-0.014822}$
* rate = $0.19285\cdot(0.44642-0.58148)=0.19285\cdot(-0.13506)=\mathbf{-0.026046}$
* gap = $\mathbf{-0.040868}$

**Other** (mA=0.10603, mB=0.04014, rA=0.26836, rB=0.16913)

* mix = $(0.10603-0.04014)\cdot0.16913 = 0.06589\cdot0.16913 = \mathbf{+0.011144}$
* rate = $0.10603\cdot(0.26836-0.16913)=0.10603\cdot0.09923=\mathbf{+0.010521}$
* gap = $\mathbf{+0.021665}$

## Full per-product table (sorted by |impact|)

| product                |           mA |           mB |       rA |       rB |        mix\_i |       rate\_i |        gap\_i | share of gap |
| ---------------------- | -----------: | -----------: | -------: | -------: | ------------: | ------------: | ------------: | -----------: |
| Simplification         |     0.192850 |     0.218340 | 0.446420 | 0.581480 |     -0.014822 |     -0.026046 | **-0.040868** |        28.9% |
| Reels                  |     0.173330 |     0.185920 | 0.440380 | 0.550340 |     -0.006929 |     -0.019059 | **-0.025988** |        18.4% |
| Other                  |     0.106030 |     0.040140 | 0.268360 | 0.169130 |     +0.011144 |     +0.010521 | **+0.021665** |       -15.3% |
| Automation             |     0.091700 |     0.089260 | 0.373730 | 0.604510 |     +0.001475 |     -0.021163 | **-0.019688** |        13.9% |
| CAPI                   |     0.078410 |     0.097820 | 0.323180 | 0.433340 |     -0.008411 |     -0.008638 | **-0.017049** |        12.0% |
| Scaling Good Campaigns |     0.011220 |     0.037900 | 0.277780 | 0.465550 |     -0.012421 |     -0.002107 | **-0.014528** |        10.3% |
| Lead Ads               |     0.044860 |     0.054950 | 0.164350 | 0.346870 |     -0.003500 |     -0.008188 | **-0.011688** |         8.3% |
| Creative               |     0.141340 |     0.127020 | 0.524610 | 0.673080 |     +0.009639 |     -0.020985 | **-0.011346** |         8.0% |
| CTX                    |     0.011630 |     0.029780 | 0.053570 | 0.328460 |     -0.005962 |     -0.003197 | **-0.009159** |         6.5% |
| Paid Messaging         |     0.005610 |     0.037900 | 0.037040 | 0.169190 |     -0.005463 |     -0.000741 | **-0.006205** |         4.4% |
| A+SC/AC                |     0.027830 |     0.024180 | 0.320900 | 0.521010 |     +0.001902 |     -0.005569 | **-0.003667** |         2.6% |
| Gen\_AI                |     0.115170 |     0.056780 | 0.070330 | 0.196050 |     +0.011447 |     -0.014479 | **-0.003032** |         2.1% |
| **Totals / checks**    | **0.999980** | **0.999990** |        — |        — | **-0.021901** | **-0.119650** | **-0.141551** |     **100%** |

Interpretation: negative **gap\_i** rows make NA worse than ROW; positive **gap\_i** offset the deficit. Rank by $|\text{gap}_i|$ to prioritize.

## Variants & reporting

* **Symmetric (mid-point) split** (path-independent):
  $\text{mix}_i=(m_A-m_B)\tfrac{(r_A+r_B)}{2},\;\text{rate}_i=\tfrac{(m_A+m_B)}{2}(r_A-r_B)$.
  $\sum \text{gap}_i$ is unchanged; only the mix vs rate split moves.

* **Bridging to external top lines**: if leadership quotes different aggregates (e.g., different filters), add a single “Scope/denominator residual” line:
  $\text{residual} = (\bar r_A^{\text{ext}}-\bar r_B^{\text{ext}}) - \sum \text{gap}_i$.
  Show it separately so the product attributions remain mathematically clean.

## What to remember

* Compute top lines **from the same rows** you decompose; then $\sum \text{gap}_i = \bar r_A-\bar r_B$ exactly.
* Use **ROW baseline** for a benchmarked story; use the **symmetric** split for baseline-agnostic attribution.
* Rank by $|\text{gap}_i|$; use signs and the mix/rate split to drive action (composition vs execution).

----

# Oaxaca Decomposition — One‑Pager (with Mini‑Tables)

## Core identities

For group A vs B by product $i$:
- Top lines: $\bar r_A=\sum_i m_{A,i}r_{A,i},\;\bar r_B=\sum_i m_{B,i}r_{B,i}$.
- Per‑product identity (path‑independent): $\textbf{gap}_i=m_{A,i}r_{A,i}-m_{B,i}r_{B,i}$.
- Totals: $\sum_i \textbf{gap}_i=\bar r_A-\bar r_B$.



**Splits (how you apportion gap into mix vs rate):**

- **B‑baseline (two‑part):** $\text{mix}_i^B=(m_A-m_B)r_B,\;\text{rate}_i^B=m_A(r_A-r_B)$.
- **A‑baseline (reverse):** $\text{mix}_i^A=(m_A-m_B)r_A,\;\text{rate}_i^A=m_B(r_A-r_B)$.
- **Three‑part (expose interaction):** pure‑mix $(m_A-m_B)r_B$ + pure‑rate $m_B(r_A-r_B)$ + interaction $(m_A-m_B)(r_A-r_B)$.
- **Symmetric (path‑independent):** $\text{mix}_i^{sym}=(m_A-m_B)\tfrac{r_A+r_B}{2},\;\text{rate}_i^{sym}=\tfrac{m_A+m_B}{2}(r_A-r_B)$.



## A) Hidden interaction & path dependence

Toy data (two products). All splits sum to the same total gap; **mix vs rate totals move** with the path.

| product   |       mA |       mB |       rA |       rB |     mix_B |   rate_B |     gap_B |     mix_A |   rate_A |     gap_A |   mix_sym |   rate_sym |   gap_sym |
|:----------|---------:|---------:|---------:|---------:|----------:|---------:|----------:|----------:|---------:|----------:|----------:|-----------:|----------:|
| P1        | 0.600000 | 0.400000 | 0.700000 | 0.600000 |  0.120000 | 0.060000 |  0.180000 |  0.140000 | 0.040000 |  0.180000 |  0.130000 |   0.050000 |  0.180000 |
| P2        | 0.400000 | 0.600000 | 0.500000 | 0.400000 | -0.080000 | 0.040000 | -0.040000 | -0.100000 | 0.060000 | -0.040000 | -0.090000 |   0.050000 | -0.040000 |



**Totals by split**

| Split                 |    Σ mix |   Σ rate |    Σ gap |
|:----------------------|---------:|---------:|---------:|
| B-baseline            | 0.040000 | 0.100000 | 0.140000 |
| A-baseline            | 0.040000 | 0.100000 | 0.140000 |
| Symmetric (mid-point) | 0.040000 | 0.100000 | 0.140000 |



## B) Aggregation bias (Simpson’s paradox)

Within each segment, A beats B, but heavier A‑mix on a harder segment drags the product average.

**Segment table**

| segment   |   nA |   xA |       rA |       mA |   nB |   xB |       rB |       mB |
|:----------|-----:|-----:|---------:|---------:|-----:|-----:|---------:|---------:|
| Large     | 1000 |  550 | 0.550000 | 0.200000 |  500 |  260 | 0.520000 | 0.100000 |
| Small     | 4000 | 1200 | 0.300000 | 0.800000 | 4500 | 1200 | 0.266700 | 0.900000 |

**Collapsed to one product row**

| product   |   rate_A |   rate_B |      gap |
|:----------|---------:|---------:|---------:|
| Reels     | 0.350000 | 0.292000 | 0.058000 |



## C) Scope/denominator mismatch & bridging

Per‑product table excludes an external ‘Unknown’ bucket present in B’s headline. Reconcile with an explicit bridge line.

**Table scope (used for decomposition)**

| product   |       mA |       rA |       mB |       rB |
|:----------|---------:|---------:|---------:|---------:|
| P1        | 0.500000 | 0.300000 | 0.400000 | 0.400000 |
| P2        | 0.300000 | 0.600000 | 0.400000 | 0.550000 |
| P3        | 0.200000 | 0.450000 | 0.200000 | 0.500000 |

**Bridge to external headline**

| Line                               |     Value |
|:-----------------------------------|----------:|
| Sum per-product gaps (table scope) | -0.060000 |
| Bridge: scope/denominator residual |  0.043000 |
| External headline gap              | -0.017000 |



**How to report:** Sum your per‑product gaps to the table‑scope gap, then add the single ‘bridge’ line so totals match the external headline without smearing differences across products.
# What we’re ranking—and why (Depth Spotter)

We attribute the gap to **sub-regions (slices)** inside the top-line regional metric and rank slices by how much they **drag** or **lift** the metric, adjusted for how big they are (**coverage**), obtaining a **contribution-aware score**.

# Inputs & notation
- Focal region: e.g. LATAM; comparison: ROW (all other regions).
- For **rate** metrics (e.g., a conversion-rate style metric): numerator = events, denominator = opportunities.
- For **additive** metrics (e.g., revenue): a single metric column.
- Slice column slice, region column region.

# How it works for rate metric

* Benchmark (ROW rate): $r_{ROW}$.
* For slice $s$: opportunities $n_s$, actual events $y_s$.
* **Expected**: $\text{exp}_s = n_s \cdot r_{ROW}$
* **Diff**: $\text{diff}_s = y_s - \text{exp}_s$
* **Gap**: $\Delta = \sum_s \text{diff}_s$ (negative ⇒ region underperforms).
* **Coverage**: $\text{cov}_s = \dfrac{n_s}{\sum_j n_j}$

# Mode selection (the rule)

Let

$$
\text{delta\_ratio}=\frac{|\Delta|}{\sum_s \text{exp}_s}
,\quad
\text{max\_potential}=\frac{\max_s |\text{diff}_s|}{|\Delta|}\ (\text{if } \Delta\neq 0).
$$

* **Two-sided** if $\text{delta\_ratio}<0.05$ **or** $\text{max\_potential}>1.0$.
  Contributions are normalized separately on each side:

  $$
  \text{contrib}_s=
  \begin{cases}
  \text{diff}_s / \sum(\text{diff}_j>0), & \text{if } \text{diff}_s>0\\[2pt]
  \text{diff}_s / \sum|\text{diff}_j<0|, & \text{if } \text{diff}_s<0\\
  0, & \text{if } \text{diff}_s=0
  \end{cases}
  $$

  (positives sum to **+1**, negatives to **−1**; signs read naturally as **lift/drag**).
* **Standard** otherwise:

  $$
  \text{raw\_share}_s=\frac{\text{diff}_s}{\Delta}\,;\quad
  \text{contrib}_s=
  \begin{cases}
  -\,\text{raw\_share}_s,& \text{if metric is higher-is-better and } \Delta<0\\
  \text{raw\_share}_s,& \text{otherwise}
  \end{cases}
  $$

  (so under a shortfall, **drags** are negative, **lifts** positive, and $\sum_s \text{contrib}_s=-1$).

**Score (for prioritization):**

$$
\text{score}_s=\sqrt{\,|\text{contrib}_s|\;+\;\dfrac{\max\big(|\text{contrib}_s|-\text{cov}_s,\;0\big)}{\max(\text{cov}_s,0.01)}}
$$

(rewards large impact and “punching above coverage”).

---

# Math Walkthrough

Let's say we want to identify top drivers to the gap in LATAM (L4) region, where it has three markets: Brazil / North Cone / SSSA. In the hypothetical scenarios of some rate metric, assume the total opportunities for the three markets are $n=\{4000,3000,3000\}\Rightarrow \text{coverage}=\{0.40,0.30,0.30\}$.

The ranking would choose one of the two rules to see the primary driver:
  • **Standard** sums contributions to **−1** under a shortfall.
  • **Two-sided** sums positives to **+1** and negatives to **−1**, preserving direction and preventing overflow.

## A) Example of a Standard Drilldown (gap is material; no slice exceeds 100% of Δ)

ROW rate $r_{ROW}=96.0\%$.
Rates: Brazil **88.0%**, North Cone **90.0%**, SSSA **93.0%**.

| Slice      | Opportunities | Actual | Rate  | Expected @ ROW | $\Delta$ | Contribution | Coverage | Score |
| ---------- | ------------: | -----: | :---- | -------------: | -------: | -----------: | -------: | ----: |
| Brazil     |         4,000 |  3,520 | 88.0% |        3,840.0 |   −320.0 |       −0.542 |     0.40 | 0.948 |
| North Cone |         3,000 |  2,700 | 90.0% |        2,880.0 |   −180.0 |       −0.305 |     0.30 | 0.567 |
| SSSA       |         3,000 |  2,790 | 93.0% |        2,880.0 |    −90.0 |       −0.153 |     0.30 | 0.391 |
**Expected** (calculated at ROW rate): $3840,\ 2880,\ 2880$
**Actual**: $3520,\ 2700,\ 2790$
**Diff**: $-320,\ -180,\ -90$
$\Delta = -320-180-90=-590$

Check mode:
$\text{delta\_ratio}=590/9600=0.0615>0.05$,
$\text{max\_potential}=320/590=0.54\le 1.0$ ⇒ **Standard**

**Contributions (presented, higher-is-better & $\Delta<0$ ⇒ negate raw shares):**

$$
\begin{aligned}
\text{Brazil:}&\ -\frac{-320}{-590}=-0.542\\
\text{North Cone:}&\ -\frac{-180}{-590}=-0.305\\
\text{SSSA:}&\ -\frac{-90}{-590}=-0.153
\end{aligned}
\quad(\text{sum}=-1.000)
$$

**One score example (Brazil):**
$|c|=0.542,\ \text{cov}=0.40\Rightarrow \text{score}=\sqrt{0.542+\frac{0.542-0.40}{0.40}}=\sqrt{0.897}=0.948.$

---

## B) Example of a Two-sided Drilldown (tiny Δ, stability case)

ROW $=96.0\%$.
Rates: Brazil **95.0%**, North Cone **97.0%**, SSSA **96.0%**.

| Slice      | Opportunities | Actual | Rate  | Expected @ ROW | $\Delta$ | Two-sided Contrib | Coverage | Score |
| ---------- | ------------: | -----: | :---- | -------------: | -------: | ----------------: | -------: | ----: |
| Brazil     |         4,000 |  3,800 | 95.0% |        3,840.0 |    −40.0 |            −1.000 |     0.40 | 1.581 |
| North Cone |         3,000 |  2,910 | 97.0% |        2,880.0 |    +30.0 |            +1.000 |     0.30 | 1.826 |
| SSSA       |         3,000 |  2,880 | 96.0% |        2,880.0 |     +0.0 |            +0.000 |     0.30 | 0.000 |
**Expected** (calculated at ROW rate): $3840,2880,2880$
**Actual**: $3800,2910,2880$
**Diff**: $-40,\ +30,\ 0$
$\Delta=-10$

Check mode:
$\text{delta\_ratio}=10/9600=0.0010<0.05$ ⇒ **Two-sided**
($\text{max\_potential}=40/10=4.0>1.0$ also triggers)

Positives: $+30\Rightarrow \sum^+=30$.
Negatives: $40\Rightarrow \sum^- = 40$.

**Two-sided contributions:**

$$
\text{Brazil}=-40/40=-1.000,\quad
\text{North Cone}=+30/30=+1.000,\quad
\text{SSSA}=0/(\cdot)=0.000.
$$

(positives sum to +1; negatives to −1)

**One score example (Brazil):**
$|c|=1.0,\ \text{cov}=0.40\Rightarrow \text{score}=\sqrt{1.0+\frac{1.0-0.40}{0.40}}=\sqrt{2.5}=1.581$.

---

## C) Example of a Two-sided Drilldown (single-slice dominance, overflow case)

ROW $=90.0\%$.
**Expected** (calculated at ROW rate): $3600,2700,2700$

| Slice      | Opportunities | Actual | Rate  | Expected @ ROW |   Diff | Two-sided Contrib | Coverage | Score |
| ---------- | ------------: | -----: | :---- | -------------: | -----: | ----------------: | -------: | ----: |
| Brazil     |         4,000 |  2,900 | 72.5% |        3,600.0 | −700.0 |            −0.875 |     0.40 | 1.436 |
| North Cone |         3,000 |  2,850 | 95.0% |        2,700.0 | +150.0 |            +1.000 |     0.30 | 1.826 |
| SSSA       |         3,000 |  2,600 | 86.7% |        2,700.0 | −100.0 |            −0.125 |     0.30 | 0.354 |
Choose feasible actuals to illustrate dominance:
Brazil **72.5% → 2900** (diff **−700**);
North Cone **95.0% → 2850** (diff **+150**);
SSSA **86.67% → 2600** (diff **−100**).

$\Delta=-700+150-100=-650$

Check mode:
$\text{delta\_ratio}=650/9000=0.0722>0.05$ but
$\text{max\_potential}=700/650=1.08>1.0$ ⇒ **Two-sided**

Positives: $+150\Rightarrow \sum^+=150$.
Negatives: $700+100=800\Rightarrow \sum^-=800$.

**Two-sided contributions:**

$$
\text{Brazil}=-700/800=-0.875,\quad
\text{North Cone}=+150/150=+1.000,\quad
\text{SSSA}=-100/800=-0.125.
$$

**One score example (Brazil):**
$|c|=0.875,\ \text{cov}=0.40\Rightarrow \text{score}=\sqrt{0.875+\frac{0.875-0.40}{0.40}}=\sqrt{2.0625}=1.436$.

----
# How we present it (on output slide)

* Left: **“What is the gap?”**
  “LATAM vs ROW gap Δ = Σ(actual) − Σ(expected at ROW rate).”
* Middle: **“Who’s driving it?”**
  Bar chart of **all slices**: bars labeled with `% or value (+ numerator/denominator for rate)`, color shows **drag/lift**, lines for **ROW** and **Region avg**, **top contributors highlighted**.
  
  Highlight policy (for clarity):
  - We always show every slice as a neutral bar (gray) and color only the most impactful ones to preserve contrast.
  - For rate metrics, candidates are screened against the region’s rate to aid interpretation (e.g., drags when the region underperforms).
  - For additive metrics, candidates are selected purely by contribution sign and magnitude (no “region-average” gating).
  - We highlight up to the top-3 by absolute contribution, but avoid coloring every bar when the number of slices is very small (i.e., at most `min(3, number_of_slices − 1)`).
* Right: **Top-3 table**
  `slice | metric | contribution (±%) | coverage | score`
  * a one-liner: “Top-3 explain X% of total contribution mass; action focus.”
* Footer: **Mode note** _not yet added to code but should be there_
  “Two-sided normalization used when Δ is very small or when any single diff would exceed 100% of Δ—prevents overflow and preserves signs.”

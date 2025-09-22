# Business-Safe Category Impact Decomposition

## What We’re Ranking — and Why

We rank categories by **their contribution (impact\_pp)** to the region’s overall performance gap vs. a baseline (e.g. rest of world rate). Each category’s contribution should reflect:

* **Performance** (did this category convert better or worse than baseline?),
* **Composition** (did this region carry more or less of this category?).

The "impact_pp" must:

* Read intuitively (better performers don’t look negative; worse performers don’t look positive),
* Scale with share (among similarly better/worse performers, **bigger share ⇒ bigger impact**),
* Preserve totals exactly (sum of category impacts equals the observed region-vs-baseline gap, with a clear rate–vs–mix split).
* If the reason for a region to under-perform is having an unfairly larger share of categories that inherently doesn't perform well, we surface that the main action items is to shift composition and move away from very weak baselines.

---

## Notation & Background

### Vanilla rate/mix decomposition (intuition)

A classic rate/mix split (Kitagawa; Oaxaca–Blinder) attributes the region–baseline gap to:

* **Rate (execution) effect:** $E_c = w^R_c \cdot (r^R_c - r^B_c)$. “At your current share, how much better/worse did you perform?”
* **Anchored mix effect:** $M_c = (w^R_c - w^B_c) \cdot (r^B_c - \bar r^B)$, where $\bar r^B$ is the baseline’s overall rate. “Did you shift share toward above- or below-average baseline categories?”
* **Net:** $I_c = E_c + M_c$.

This is precise mathematically, but raw outputs can read wrong to business stakeholders. A common failure mode (Simpson-type): a category with **worse** performance still gets **positive** contribution because the region moved away from a very weak baseline slice.

### Inputs & Notation

For each category $c$:

* Region share $w^R_c$, baseline share $w^B_c$,
* Region rate $r^R_c$, baseline rate $r^B_c$,
* Baseline overall rate $\bar r^B = \dfrac{\sum_c w^B_c r^B_c}{\sum_c w^B_c}$,
* Shorthands: $\Delta r_c = r^R_c - r^B_c$, $\Delta w_c = w^R_c - w^B_c$.

**Core split we compute:**

* **Rate (execution) effect:** $E_c = w^R_c \cdot \Delta r_c$
* **Anchored mix effect:** $M_c = \Delta w_c \cdot (r^B_c - \bar r^B)$
* **Net:** $I_c = E_c + M_c$

**Preserved totals (per region):**

$$
\sum_c E_c = \text{total rate effect},\quad
\sum_c M_c = \text{total mix effect},\quad
\sum_c I_c = \text{observed region–baseline gap}.
$$

This means, if there are 10 product categories, then summing over the ten `impact_pp` we observe from each, we'd obtain the regional gap `impact_pp`.

---

## What We Changed and Why (business → math)

We add three **small, zero-sum adjustments** to make the category-level story read correctly while preserving all totals and the rate/mix split.

* **Use pooling to change the distribution of positive and negative impact:** We re-weight positive/negative **mix** inside the $\Delta r$ groups by share-aware weights, so large, better-performing slices carry proportionally more of the favorable composition. This is **zero-sum** inside each group; no totals move.
* **Use projection to change the paradox row where worse performance gives positive impact:** After pooling, we **project** to enforce: better-rate $\Rightarrow I\ge\varepsilon$, worse-rate $\Rightarrow I\le-\varepsilon$ (ties ignored). We adjust **only mix**, zero-sum across donors, so **totals don’t change**.

---

### A) Share-aware pooling **within** rate groups (shape magnitudes, not signs)

**Business rule:** Among similarly better (or worse) performers, **bigger share ⇒ bigger magnitude** of contribution.

**Math:** Let $\Delta r_c=r^R_c-r^B_c$. Define “better” $P_+=\{\Delta r_c>\eta\}$ and “worse” $P_-=\{\Delta r_c<-\eta\}$, with near-tie tolerance $\eta$ (e.g., 0.5pp). Start from the anchored split $E_c=w^R_c\Delta r_c$, $M_c=(w^R_c-w^B_c)(r^B_c-\bar r^B)$.

* In $P_+$: pool **positive** mix mass and redistribute by normalized weights
  $\displaystyle \omega_c \propto (w^R_c)^{\alpha}\,\big|r^B_c-\bar r^B\big|^{\beta}\,\big|\Delta r_c\big|^{\gamma}$
  (with small-share damping, e.g., if $w^R_c<2\%$, halve $\omega_c$).
* In $P_-$: pool **negative** mix mass and redistribute by the same $\omega_c$.

Zero-sum **inside each group**; totals and signs unchanged. Denote the result $M^{\text{pool}}_c$ and $I^{\text{pool}}_c=E_c+M^{\text{pool}}_c$.

> **Example (Vertical, LATAM):** Before pooling, “Other” had most of the positive mix (1.5pp) despite tiny share. After pooling, “Creative” and “Advantage+” (both $\Delta r>0$ and larger $w_R$) retain more of the positive mix (their $M$ become less negative), lifting their nets to **1.525pp** and **0.964pp** respectively.

| Category   | w\_R   | r\_R   | r\_B   | Δr (pp)    | E (pp) | M₀ (pp) | I₀ (pp) | M\_pool (pp) | I\_pool (pp) |
| ---------- | ------ | ------ | ------ | ---------- | ------ | ------- | ------- | ------------ | ------------ |
| Advantage+ | 0.1441 | 0.7437 | 0.6760 | **+6.77**  | +0.975 | −0.297  | +0.678  | −0.012       | **+0.964**   |
| Creative   | 0.1414 | 0.8254 | 0.6865 | **+13.89** | +1.963 | −0.742  | +1.221  | −0.438       | **+1.525**   |
| Other      | 0.0159 | 0.1304 | 0.1169 | **+1.35**  | +0.021 | +1.498  | +1.520  | +0.057       | **+0.078**   |

*(pooling is zero-sum over the entire $\Delta r>0$ group, not necessarily over this 3-row subset, but these three-row entry serves as a good example.)*

**Worked example (“Other”):**

Shared baseline:

$$
\bar r^B = \frac{\sum w^B r^B}{\sum w^B} = 0.554585
$$

Inputs: $w_R=0.015877,; w_B=0.050115,; r_R=0.130408,; r_B=0.116943$

Step-by-step:

$$
\Delta r = 0.130408 - 0.116943 = 0.013465 \;\Rightarrow\; 1.346 \text{ pp}
$$

$$
E = 0.015877 \times 0.013465 = 0.000214 \;\Rightarrow\; 0.021 \text{ pp}
$$

$$
\Delta w = 0.015877 - 0.050115 = -0.034238
$$

$$
r_B - \bar r^B = 0.116943 - 0.554585 = -0.437643
$$

$$
M = (-0.034238)\times(-0.437643)=0.014984 \;\Rightarrow\; 1.498 \text{ pp}
$$

$$
I_0 = 0.021 + 1.498 = 1.520 \text{ pp}
$$

**Pooling step (inside the $\Delta r>0$ group only):** redistribute the group’s total positive mix (≈1.5pp) across categories with larger $w_R$. “Other” shrinks from +1.498pp → +0.057pp; Creative and Advantage+ nets lift

* Build weights $\omega_c \propto (w_R)^\alpha \,|r_B-\bar r^B|^\beta \,|\Delta r|^\gamma$ with small-share damp (here $\alpha=1,\beta=1,\gamma=0$).
* Compute the **total positive** anchored mix mass over *all* $\Delta r>0$ categories:
  $M^+_{\text{total}}=\sum_{\Delta r>0}\max(M,0)$ (a scalar).
* Set a **target** for each $\Delta r>0$ category: $T_c = M^+_{\text{total}}\cdot \frac{\omega_c}{\sum_{\Delta r>0}\omega}$.
* Update mix inside the group:
  $M^{\text{pool}}_c \leftarrow M_c + \big(T_c - \max(M_c,0)\big)$.
  (This adds target to entries that had no positive mass and removes it from those that had too much; the adjustment is zero-sum over the group.)

**Effect we see in the table:**

* “Other” had most of the positive mix (1.498pp) despite tiny share → after pooling it keeps **+0.057pp** of that mass;
* “Creative” and “Advantage+” (both larger $w_R$) retain more of the positive mix (their $M$ become less negative), lifting their nets to **1.525pp** and **0.964pp** respectively.
* The **sum of $M$** over the whole $\Delta r>0$ group is unchanged; pooling only **redistributes** composition to be share-sensible.

---

### B) Sign projection (fix direction after pooling)

**Business rule:**

* If a category’s rate is **better** than baseline (beyond $\eta$), its net impact should be **non-negative** (or $\ge\varepsilon$ if you set a margin). ---> If $\Delta r_c>\eta$: enforce $I^{final}_c \ge \varepsilon$.
* If **worse** (beyond $\eta$), its net should be **non-positive** (or $\le-\varepsilon$). --> If $\Delta r_c<-\eta$: enforce $I^{final}_c \le -\varepsilon$.
* Near-ties $(|\Delta r_c|\le\eta)$: don’t force a sign (treat as \~0). --> If $|\Delta r_c|\le \eta$: do nothing.
> Parameters: $\eta=0.005$ (0.5pp), $\varepsilon=0.005$pp.

**Math:** With margin $\varepsilon\in[0,\;0.05\text{pp}]$:

* **Negative ceiling:** For violators $V^-=\{\Delta r_c<-\eta,\;I^{\text{pool}}_c>-\varepsilon\}$, lower to $-\varepsilon$ by decreasing $M^{\text{pool}}_c$ by $\delta^-_c = I^{\text{pool}}_c+\varepsilon$. Sum need $N^-=\sum_{V^-}\delta^-_c$. **Add** $N^-$ back to donors using $\omega_c$: first within $\{\Delta r<-\eta,\;I^{\text{pool}}<-\varepsilon\}$; if none, widen to region donors $\{I^{\text{pool}}<-\varepsilon\}$; if none, relax (no change).
* **Positive floor:** Recompute $I$. For violators $V^+=\{\Delta r_c>\eta,\;I^{\text{pool}}_c<\varepsilon\}$, raise to $\varepsilon$ by increasing $M^{\text{pool}}_c$ by $\delta^+_c = \varepsilon-I^{\text{pool}}_c$. Sum need $N^+=\sum_{V^+}\delta^+_c$. **Subtract** $N^+$ from donors with $\omega_c$: first within $\{\Delta r>\eta,\;I^{\text{pool}}>\varepsilon\}$; else region donors $\{I^{\text{pool}}>\varepsilon\}$; else relax.

It is zero-sum, so $\sum M$, $\sum E$, $\sum I$ are preserved. Denote the final outputs $M^{\text{final}}_c$ and $I^{\text{final}}_c=E_c+M^{\text{final}}_c$.

> **Example (Vertical, LATAM):**
> **“Value optimization for purchase ROAS”** had $\Delta r<0$ and positive net pre-projection (due to composition). Unified projection brings it to **$\le 0$** (≈0 with tiny $\varepsilon$), fixing direction without changing totals.

| Category                             | w\_R  | w\_B  | r\_R  | r\_B  | Δr (pp)    | E (pp) | M₀ (pp) | I₀ (pp) | M\_pool (pp) | I\_pool (pp) | M\_final (pp) | **I\_final (pp)** |
| ------------------------------------ | ----- | ----- | ----- | ----- | ---------- | ------ | ------- | ------- | ------------ | ------------ | ------------- | ----------------- |
| Value optimization for purchase ROAS | 0.004 | 0.029 | 0.128 | 0.292 | **−16.35** | −0.065 | +0.659  | +0.594  | +0.656       | +0.592       | **−0.436**    | **−0.500**        |

**How the projection is computed:**

* Anchored split (pre-pool):
  $E = w_R\cdot\Delta r = 0.003962\times(-0.163483) = -0.000648 \Rightarrow E_{pp}=-0.065$
  $M_0 = \Delta w\cdot(r_B-\bar r^B) = (-0.025051)\times(0.291624-0.554585) = +0.006587 \Rightarrow M_{0,pp}=+0.659$
  $I_0=E+M_0 = +0.00594 \Rightarrow I_{0,pp}=+0.594$ (paradox: worse rate but positive net)

* After pooling (inside dr<0 the negative mass pooling doesn’t change this row materially):
  $M_{\text{pool}} \approx +0.006563 \Rightarrow I_{\text{pool}}=+0.005915$ (**still** positive, violates negative ceiling).

* **Sign projection:**
  This row is a **violator** of the negative ceiling ($\Delta r<-\eta$ but $I_{\text{pool}}>-\varepsilon$).
  Required reduction:

$$
\delta^- = I_{\text{pool}} + \varepsilon = 0.592\text{pp} + 0.500\text{pp} = 1.092\text{pp}
$$

Adjust mix:

$$
M_{\text{final}} = M_{\text{pool}} - \delta^- = 0.656\text{pp} - 1.092\text{pp} = -0.436\text{pp}
$$

Final net:

$$
I_{\text{final}} = E + M_{\text{final}} = (-0.065) + (-0.436) = -0.501\text{pp} \approx \mathbf{-0.500\text{pp}}
$$

The removed 1.092pp is redistributed to donors (zero-sum), so region totals stay fixed.

---

## Outputs & How We Rank

For each category $c$ we publish:

* **rate\_pp** $=\;100\times E_c$,
* **mix\_pp** $=\;100\times (I^{(2)}_c - E_c)$ (after all adjustments),
* **impact\_pp** $=\;100\times I^{(2)}_c$  ⟵ **use this to rank**.

Group views (e.g., by product family/vertical) sum **impact\_pp** within the group; region-level sum equals the observed gap. For interpretation:

* **Positive impact\_pp** → category lifts the region vs. baseline.
* **Negative impact\_pp** → category drags the region.
* Among similarly better (or worse) performers, the **bigger share** category tends to have **bigger magnitude** (thanks to pooling).

Use **rate\_pp** to diagnose execution vs. **mix\_pp** for composition coverage. 
E.g., negative impact driven mostly by rate\_pp → performance issue; 
negative impact with mild rate\_pp but very negative mix\_pp → composition/coverage issue.

---

## Edge Cases We Cover (business-focused)

* **Tiny slices:** Down-weighted in $\omega_c$ (share floor + damping); small segments can’t dominate via mix quirks.
* **Near-ties in rate:** $|\Delta r_c|\le \eta$ treated as ties; we don’t force a sign.
* **No donors for caps:** We widen donors to the region; if still none, cap relaxes to 0 (no change) to preserve totals.
* **Single-category regions or degenerate baseline shares:** We surface the unadjusted split (skip rebalancing) to avoid artifacts; totals obviously still match.
* **Simpson’s paradox:** Discussed separately in a Section  

---

## Validation & Monitoring (we run per region)

**Conservation (exact):**

* $\sum \text{impact\_pp} =$ observed region gap,
* $\sum \text{rate\_pp} = \sum E_c$,
* $\sum \text{mix\_pp} = \sum M_c$.

**Direction (with tolerances $\eta,\varepsilon$):**

* If $\Delta r_c>\eta$, then $I^{(2)}_c \ge \varepsilon$.
* If $\Delta r_c<-\eta$, then $I^{(2)}_c \le -\varepsilon$.
* If $|\Delta r_c|\le \eta$, we do not force a sign (or clamp to \~0 if configured).

**Share-ordering smoke test:**

* Within the $\Delta r>0$ group, increasing $\alpha$ (share weight) increases correlation between **impact\_pp** and $w^R$.

**Stability guards:**

* Pools < $10^{-12}$ or weight sums near zero → **no-op**, with logging.
* If baseline share sum $\sum w^B$ is \~0 → skip rebalancing (surface raw split).

---

## Implementation Notes (what the code does)

**Per-region workflow**

1. **Compute anchored split and gaps**
   $E_c=w^R_c(r^R_c-r^B_c),\quad M_c=(w^R_c-w^B_c)\,(r^B_c-\bar r^B),\quad I_c=E_c+M_c$
   with $\bar r^B=\frac{\sum w^B r^B}{\sum w^B}$ and deltas $\Delta r_c=r^R_c-r^B_c,\;\Delta w_c=w^R_c-w^B_c.$

2. **Share-aware pooling (shape magnitudes, not signs)**
   Split categories by a near-tie tolerance $\eta=0.005\;(\text{0.5pp}):\;P_+=\{\Delta r_c>\eta\},\;P_-=\{\Delta r_c<-\eta\}.$

   * In $P_+$: pool **positive** mix mass only.
   * In $P_-$: pool **negative** mix mass only.
     Redistribute zero-sum **inside each group** using normalized weights
     $\omega_c \propto (w^R_c)^{\alpha}\,\big|r^B_c-\bar r^B\big|^{\beta}\,\big|\Delta r_c\big|^{\gamma},$
     with small-share damping (if $w^R_c<2\%$, halve $\omega_c$). Outputs: $M_c^{\text{pool}},\,I_c^{\text{pool}}=E_c+M_c^{\text{pool}}$.

3. **Sign projection (directional backstop)**
   Enforce intuitive signs while keeping totals exact, with margin $\varepsilon$ (we use $\varepsilon=0.005$pp in Simpson-sensitive reporting):

   * If $\Delta r_c>\eta$, require $I_c^{\text{final}}\ge\varepsilon$.
   * If $\Delta r_c<-\eta$, require $I_c^{\text{final}}\le-\varepsilon$.
     For violators, adjust **only** mix by

   $$\delta^-_c=I_c^{\text{pool}}+\varepsilon\quad(\text{for } \Delta r<-\eta,;I^{\text{pool}}>-\varepsilon),\qquad
     \delta^+_c=\varepsilon-I_c^{\text{pool}}\quad(\text{for } \Delta r>\eta,\;I^{\text{pool}}<\varepsilon),$$
   and redistribute the **total** adjustment zero-sum to donors (first within the same rate group; if none exist, region-wide). Outputs: $M_c^{\text{final}},\,I_c^{\text{final}}=E_c+M_c^{\text{final}}$.

4.**Publish columns (pp)**

   * **rate\_pp** $=100\,E_c$
   * **impact\_pp** $=100\,I_c^{\text{final}}$
   * **mix\_pp** $=100\,(I_c^{\text{final}}-E_c)$
     Plus diagnostics (pool mass, cap need, donor scope) and invariants (all sums preserved via `np.isclose`).

**Weights & parameters (defaults that work well)**
$\alpha\ge1$ (share), $\beta\approx1$ (baseline leverage), $\gamma\in[0,1]$ (optional performance tilt);
small-share damping at $w^R<2\%$ → ×0.5;
$\eta=0.005$ (0.5pp), $\varepsilon\in[0,0.05]$pp (we use 0.5pp for strict sign clarity in Simpson cases);
numeric guards $10^{-12}$.

**Fail-safe skips**
Single-category regions or degenerate baselines $(\sum w^B\approx 0)$ bypass rebalancing and surface the raw anchored split (keeps interpretability and exact totals).

---

## References (and how this relates)

**What Kitagawa–Oaxaca–Blinder does (vanilla, why it’s used)**

* **Kitagawa (1955)** formalized **rate–composition** decomposition for two aggregates: the gap in overall rates equals a **rate effect** (holding composition fixed) plus a **composition/mix effect** (holding rates fixed). This is the foundation of demographic standardization and “shift–share” logic.
* **Blinder (1973); Oaxaca (1973)** extended this to linear models for wage gaps: total mean gap = **endowments (composition)** + **coefficients (rate)** (plus an optional interaction term). In practice, OB is prized because (i) it is **additive and exact** at the top line and (ii) the two terms map to an intuitive story: “how much is mix vs performance.”

**Known issues with the raw decomposition (why we adjust)**

* **Path/anchor dependence:** different choices of the “overall” (rest vs pooled vs geometric mean) or allocation of the interaction term yield different category-level attributions even though the **total** is invariant.
* **Sign weirdness at granular level:** a category can show **positive mix** even when it’s **worse** on rate (e.g., moving weight **away** from a low baseline segment yields a positive mix term), which can flip the **net** positive despite underperformance—this is exactly the **Simpson paradox** flavor users find counter-intuitive.
* **Scale insensitivity:** tiny-share segments can accrue outsized mix due to leverage against a far-from-overall baseline, overstating their perceived “pull.”

**How others address these issues (and where our approach fits)**

* **Standardization choices (Kitagawa / Das Gupta)**: Use an agreed anchor (e.g., baseline overall) for composition—our **anchored mix** $M=(w^R-w^B)\,(r^B-\bar r^B)$ follows this playbook to stabilize signs relative to a single, interpretable reference.
* **Allocation of interaction & path independence:** Variants allocate the interaction symmetrically or adopt **Shapley-value–based decompositions** to ensure **order-invariance and aggregation consistency** (see applications in productivity and inequality accounting). Shapley OB is principled but heavier to communicate and compute at scale; it also doesn’t directly encode the **“bigger share ⇒ bigger magnitude”** business rule inside a rate group.
* **Reweighting counterfactuals (DiNardo–Fortin–Lemieux / Fortin–Lemieux–Firpo):** Build a counterfactual distribution for rates or composition by propensity reweighting, then read the gap. These methods are strong for distributional analysis, but they **replace** rather than **reshape** the category ledger your users consume.
* **Index-number fixes (Bennett, Fisher, Törnqvist)**: Use symmetric means/indices to reduce base bias; again, principled but less transparent for non-technical readers and still not a guarantee of **category-level sign intuition**.

**What we add (constrained, zero-sum redistribution)**

* **Share-aware pooling inside rate groups** makes magnitudes scale with business impact drivers: **share**, **baseline leverage**, and (optionally) **how much better/worse** the rate is. It’s **zero-sum within the group**, so totals and the OB split are preserved.
* **Sign projection** is a **minimal, zero-sum correction** that enforces:

  * if a category is **better** than baseline (beyond $\eta$), its net is **non-negative (≥ε)**;
  * if **worse**, its net is **non-positive (≤−ε)**;
  * near-ties untouched.
    This directly resolves Simpson-type surprises **without** changing the region totals or the OB rate/mix sums.

**Bottom line**
Our method **keeps the Oaxaca–Blinder backbone** (exact additivity; clean rate vs mix story) while layering a **transparent, zero-sum, share-aware** redistribution that 
(i) aligns category signs with performance, 
(ii) makes magnitudes scale with share, and 
(iii) handles Simpson paradox cases by rule, not by exception;
all **without** sacrificing mathematical conservation.

**Key citations** 
* Kitagawa, E.M. (1955). *Components of a Difference Between Two Rates*. **JASA**.
* Das Gupta, P. (1994). *Standardization and Decomposition of Rates*. **International Statistics Review**.
* Blinder, A.S. (1973). *Wage Discrimination: Reduced Form and Structural Estimates*. **JHR**.
* Oaxaca, R. (1973). *Male–Female Wage Differentials in Urban Labor Markets*. **IJF**.
* DiNardo, J., Fortin, N.M., & Lemieux, T. (1996). *Labor Market Institutions and the Distribution of Wages*. **Econometrica**.
* Fortin, N., Lemieux, T., & Firpo, S. (2011). *Decomposition Methods in Economics*. **Handbook of Labor Economics**.
* Shapley-based decompositions in productivity/inequality accounting (order-invariant attribution background).




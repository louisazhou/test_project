
| **Hypothesis / Metric**                                                    | **Global** | **APAC** | **EMEA** | **LATAM (L4)** | **NA** | **Score** | **Rank** | **Expected Direction** |
| -------------------------------------------------------------------------- | ---------- | -------- | -------- | -------------- | ------ | --------- | -------- | ---------------------- |
| **% Qualified RS Pitched Within 42 days of Creation**                      | 0.966      | 0.963    | 0.986    | 0.915          | 0.963  |           |          |                        |
| **RS to Pitch Workload / AM**                                              | 126.99     | 138.28   | 125.06   | 156.06         | 116.91 | 0.85      | 1.0      | opposite               |
| **Interaction Duration (min)/AM/Working Day**                              | 77.27      | 116.95   | 69.74    | 105.09         | 55.06  | 0.67      | 2.0      | opposite               |
| **% Interactions that are not budget or up-sell related**                  | 0.217      | 0.244    | 0.239    | 0.324          | 0.156  | 0.66      | 3.0      | opposite               |
| **Avg CIs - Early Stage -> Pitched/Committed**                             | 2.2        | 2.25     | 2.13     | 2.21           | 2.24   | 0.64      | 4.0      | same                   |
| **New (SBG->MM or GBG->MM) to Assess %**                                   | 0.151      | 0.081    | 0.182    | 0.222          | 0.157  | 0.58      | 5.0      | opposite               |
| **Outreach %**                                                             | 0.995      | 0.986    | 0.999    | 0.993          | 0.999  | 0.58      | 7.0      | same                   |
| **Avg Duration Per Live CI (mins)**                                        | 36.68      | 47.63    | 33.03    | 40.56          | 31.59  | 0.57      | 6.0      | opposite               |
| **Avg CIs - Pitched/Committed -> Closed Lost**                             | 2.08       | 2.34     | 1.98     | 2.15           | 2.02   | 0.55      | 8.0      | opposite               |
| **% of attempted interactions that were not successful**                   | 0.121      | 0.087    | 0.115    | 0.131          | 0.145  | 0.54      | 10.0     | opposite               |
| **SCS to Pitch Workload / AM**                                             | 3.71       | 5.31     | 3.86     | 5.09           | 2.39   | 0.54      | 9.0      | opposite               |
| **% Client Interactions (that are linked to KI) that moved stage forward** | 0.512      | 0.48     | 0.529    | 0.576          | 0.51   | 0.25      | 11.0     | same                   |
| **% Solutions moved from Pitched/Committed -> Actioned**                   | 0.55       | 0.59     | 0.507    | 0.469          | 0.577  | 0.23      | 12.0     | opposite               |
| **% Clients Deemed Uncontactable (Clients Unresponsive)**                  | 0.033      | 0.034    | 0.027    | 0.016          | 0.046  | 0.23      | 13.0     | opposite               |
| **Goaled CIs/AM/Working Day**                                              | 2.32       | 2.93     | 2.27     | 3.43           | 1.82   | 0.15      | 14.0     | same                   |
| **Avg CIs - Pitched/Committed -> Actioned**                                | 2.50       | 2.67     | 2.39     | 2.46           | 2.47   | 0.11      | 15.0     | opposite               |
| **Avg CIs - Actioned -> Partially Adopted**                                | 2.75       | 2.98     | 2.66     | 2.87           | 2.67   | 0.10      | 16.0     | same                   |
| **% Client Interactions that moved stage forward**                         | 0.198      | 0.24     | 0.196    | 0.206          | 0.172  | 0.07      | 17.0     | same                   |

# What we’re ranking—and why

Goal: explain why **LATAM’s “% Qualified RS Pitched within 42 days”** is **lower** than peers (91.5% vs Global 96.6%).
We rank each hypothesis by how well it 
(a) moves in the expected direction across regions and 
(b) quantitatively accounts for LATAM’s shortfall.

# Scoring logic

1. Compute regional **deltas vs Global** for both the metric and hypothesis.
   Δ = (Region − Global) / Global.
   If `expected_direction == "opposite"`, flip the hypothesis deltas’ sign.

2. **Sign agreement (60% weight):** fraction of regions where `sign(Δ_metric) == sign(Δ_hypothesis_adj)`.
   Intuition: when the metric is lower than global in a region, the hypothesis should also be “worse” than global (in the expected direction), and vice-versa.

3. **Explained ratio (40% weight):** for **LATAM only**, compute z-scores
   `z_m = Δ_metric_LATAM / std(Δ_metric_all_regions)`
   `z_h = Δ_hypothesis_adj_LATAM / std(Δ_hypothesis_adj_all_regions)`
   Then `explained_ratio = min(|z_h| / |z_m|, 1)`.
   Intuition: how much of LATAM’s standardized anomaly magnitude the hypothesis can match.

4. **Guardrails + focal check:**

   * require `sign_agreement ≥ 0.5` and `explained_ratio ≥ 0.2`
   * require **LATAM**’s sign to agree; otherwise apply a penalty.
   * Final score: `0.6*sign_agreement + 0.4*explained_ratio`, then ×0.3 if LATAM disagrees.

This yields a single **final score** used for ranking.

# How to narrate the top hypotheses (examples)

Use this 3-line template for each top item:

* **Direction check:** “Across regions, metric and hypothesis move in the expected direction in X/Y regions; LATAM also matches the direction.”
* **Magnitude check:** “LATAM’s hypothesis deviation explains \~Y% of LATAM’s metric z-score.”
* **Takeaway:** “This supports the hypothesis as a dominant driver.”

Applied to the top few:

* **RS to Pitch Workload / AM (expected: opposite)**
  Direction: 3/4 regions match; **LATAM matches** (metric ↓, workload ↑ vs Global).
  Magnitude: explained-ratio high (≈0.8 range).
  Takeaway: elevated workload plausibly depresses timely pitching in LATAM → **highest score**.

* **Interaction Duration / AM / Working Day (expected: opposite)**
  Direction: broad sign consistency; **LATAM longer** interactions vs Global while metric is lower.
  Magnitude: strong; yields a top score.
  Takeaway: longer average interaction time correlates with slower/less timely pitching.

* **% Interactions not budget/up-sell (expected: opposite)**
  Direction: consistent; **LATAM more** non-budget interactions where metric is lower.
  Magnitude: solid; passes guardrails.
  Takeaway: interaction mix may be less conversion-oriented in LATAM.

* **New (SBG→MM or GBG→MM) to Assess % (expected: opposite)**
  Direction: consistent and focal matches.
  Magnitude: moderate; still clears thresholds.
  Takeaway: higher share of “new-to-MM” assessments correlates with lower timely pitching.

* **Avg Duration Per Live CI (mins) (expected: opposite)**
  Direction: generally consistent; LATAM higher.
  Magnitude: moderate; included in top cohort.
  Takeaway: longer live CI duration aligns with lower timely pitch rates.

Items that fall in rank typically do so because of:

* **Low sign agreement** (<50%) ⇒ “wrong hypothesis direction.”
* **Low explained ratio** (<0.2) ⇒ “coincidental moves.”
* **LATAM sign mismatch** ⇒ “focal region direction mismatch” and 0.3× penalty.

# Failure patterns to highlight

1. Wrong hypothesis direction (**Low sign agreement** (<50%) )

* What it shows: peers don’t co-move the way the hypothesis claims.
* Why we block: with 4–5 regions, one off-pattern peer can falsely look compelling; the 50% threshold protects against spurious direction.
%%* Slide text: “Peers disagree on sign (e.g., APAC/EMEA up while metric down). Guardrail: require ≥50% sign agreement.”%%

1. Coincidental moves (**Low explained ratio** (<0.2))

* What it shows: overall direction looks right, but LATAM’s hypothesis deviation is too small (or noisy) vs LATAM’s metric deviation.
* Why we block: prevents rewarding hypotheses that only weakly vary at the focal region.
%%* Slide text: “LATAM effect size is too small vs metric z-score. Guardrail: explained ratio ≥0.2.”%%

1. Focal region direction mismatch (**LATAM sign mismatch**)

* What it shows: peers line up, but **LATAM** moves opposite to what the hypothesis predicts.
* Why we penalize: we’re explaining LATAM; if LATAM’s sign doesn’t match, the story collapses.
%%* Slide text: “LATAM sign mismatch ⇒ 0.3× penalty and fails guardrail.”%%

4. Combination failures (e.g., direction + magnitude)

* What it shows: mixed peer signals and small LATAM magnitude.
* Why we block: avoids cherry-picking in small-N settings.


For each failed row in the Failure Gallery:
* “**\[Hypothesis]** (expected: same/opposite): sign agreement **X/4**, explained ratio **Y**; **LATAM agrees?** Yes/No.
  **Reason:** {wrong hypothesis direction / coincidental moves / focal region direction mismatch}.
  **Peers disagree:** {list}.
  **LATAM vs Global:** {magnitude}.”
This directly answers “but APAC is even lower/higher than us!”—because we show the **peer sign pattern** and the **LATAM magnitude vs metric**.

| **Hypothesis**                                                             | **Expected** | **LATAM vs Global (pretty)** | **LATAM_minus_Global_pp** | **LATAM_vs_Global_pct** | **Sign agreement** | **Explained ratio** | **Focal agrees** | **Final score** | **Explains?** | **Failure reason**                                                              | **reason_bucket**               |
| -------------------------------------------------------------------------- | ------------ | ---------------------------- | ------------------------- | ----------------------- | ------------------ | ------------------- | ---------------- | --------------- | ------------- | ------------------------------------------------------------------------------- | ------------------------------- |
| **Avg CIs - Pitched/Committed -> Actioned**                                | opposite     | -1.60%                       |                           | -1.6000000000000000     | 0.5                | 0.195               | FALSE            | 0.113           | FALSE         | coincidental moves, focal region direction mismatch                             | coincidental moves              |
| **Outreach %**                                                             | same         | -20.00pp                     | -20.000000000000300       |                         | 0.75               | 0.189               | TRUE             | 0.526           | FALSE         | coincidental moves                                                              | coincidental moves              |
| **% Clients Deemed Uncontactable (Clients Unresponsive)**                  | opposite     | -170.00pp                    | -170.00000000000000       |                         | 0.75               | 0.791               | FALSE            | 0.23            | FALSE         | focal region direction mismatch                                                 | focal region direction mismatch |
| **% Solutions moved from Pitched/Committed -> Actioned**                   | opposite     | -810.00pp                    | -810.0000000000000        |                         | 0.75               | 0.825               | FALSE            | 0.234           | FALSE         | focal region direction mismatch                                                 | focal region direction mismatch |
| **% Client Interactions (that are linked to KI) that moved stage forward** | same         | +640.00pp                    | 640.0000000000000         |                         | 0.75               | 0.931               | FALSE            | 0.247           | FALSE         | focal region direction mismatch                                                 | focal region direction mismatch |
| **Avg CIs - Early Stage -> Pitched/Committed**                             | same         | 0.45%                        |                           | 0.4545454545454450      | 0.0                | 0.108               | FALSE            | 0.013           | FALSE         | wrong hypothesis direction, coincidental moves, focal region direction mismatch | wrong hypothesis direction      |
| **% Client Interactions that moved stage forward**                         | same         | +80.00pp                     | 80.00000000000010         |                         | 0.25               | 0.166               | FALSE            | 0.065           | FALSE         | wrong hypothesis direction, coincidental moves, focal region direction mismatch | wrong hypothesis direction      |
| **Avg CIs - Actioned -> Partially Adopted**                                | same         | 4.36%                        |                           | 4.363636363636370       | 0.25               | 0.448               | FALSE            | 0.099           | FALSE         | wrong hypothesis direction, focal region direction mismatch                     | wrong hypothesis direction      |
| **Goaled CIs/AM/Working Day**                                              | same         | 47.84%                       |                           | 47.84482758620690       | 0.25               | 0.915               | FALSE            | 0.155           | FALSE         | wrong hypothesis direction, focal region direction mismatch                     | wrong hypothesis direction      |


%%
# What to show on the slide: high score hypo
* One **formula box** (the 4 steps above).
* One **row** worked example (e.g., Workload): show APAC/EMEA/LATAM/NA deltas, the sign ticks, z-scores, and the calculated score.
* **Top-N table** with: Expected dir, Sign agree, Explained ratio, Focal agrees, Final score.
* **Footnote:** reference = Global; LATAM focal; deltas are relative to Global; opposite hypotheses are sign-flipped before evaluation.
# What to show on the slide: low score hypo
* Left: the 3 guardrails (bulleted, one line each).
* Middle: a **good** example (top-ranked) with tiny evidence table (peer deltas + sign ticks) and z-score ratio.
* Right: **three failed examples**, one per failure type, using the one-liner above.
* Footer: “Reference is Global; deltas vs Global; ‘opposite’ hypotheses pre-flipped before sign checks.”
%%


# Math Walk-through 

## "RS to Pitch Workload / AM" (ranked #1 as root-cause)

### Per-region deltas (vs Global)

| **Region**     | **Metric (M_r)** | **Δ_metric formula**    | **Δ_metric value** | **Hypothesis (H_r)** | **Δ_hypo_raw formula**     | **Δ_hypo_raw value** | **Δ_hypo_adj formula**        | **Δ_hypo_adj value** | **Sign check**                 |
| -------------- | ---------------- | ----------------------- | ------------------ | -------------------- | -------------------------- | -------------------- | ----------------------------- | -------------------- | ------------------------------ |
| **APAC**       | 96.30            | (96.30 - 96.60) / 96.60 | -0.31%             | 138.28               | (138.28 - 126.99) / 126.99 | 8.89%                | -((138.28 - 126.99) / 126.99) | -8.89%               | sign(ΔM) ?= sign(ΔH_adj) → Yes |
| **EMEA**       | 98.60            | (98.60 - 96.60) / 96.60 | 2.07%              | 125.06               | (125.06 - 126.99) / 126.99 | -1.52%               | -((125.06 - 126.99) / 126.99) | 1.52%                | sign(ΔM) ?= sign(ΔH_adj) → Yes |
| **LATAM (L4)** | 91.50            | (91.50 - 96.60) / 96.60 | -5.28%             | 156.06               | (156.06 - 126.99) / 126.99 | 22.89%               | -((156.06 - 126.99) / 126.99) | -22.89%              | sign(ΔM) ?= sign(ΔH_adj) → Yes |
| **NA**         | 96.30            | (96.30 - 96.60) / 96.60 | -0.31%             | 116.91               | (116.91 - 126.99) / 126.99 | -7.94%               | -((116.91 - 126.99) / 126.99) | 7.94%                | sign(ΔM) ?= sign(ΔH_adj) → No  |

Columns:

* Metric: % Qualified RS Pitched Within 42 Days
* Δ\_metric\_vs\_Global = (region−Global)/Global
* Hypothesis: workload value
* Δ\_hypo\_vs\_Global (raw)
* Δ\_hypo\_vs\_Global (adj): sign-flipped because relationship is “opposite”
* Sign match? = sign(Δ\_metric) == sign(Δ\_hypo\_adj)

### Summary components

| **Component**                | **Formula**                                          | **Value** |
| ---------------------------- | ---------------------------------------------------- | --------- |
| **Expected direction**       | opposite                                             | opposite  |
| **Sign agreement**           | 3/4                                                  | 0.750     |
| **σ(metric Δ)**              | std([-0.003106, 0.020704, -0.052795, -0.003106])     | 0.026779  |
| **σ(hypo Δ, adj)**           | std([-0.088905, 0.015198, -0.228916, 0.079376])      | 0.116596  |
| **z_m (LATAM)**              | ΔM_LATAM / σ(metric Δ) = -0.052795 / 0.026779        | -1.971498 |
| **z_h (LATAM)**              | ΔH_adj_LATAM / σ(hypo Δ, adj) = -0.228916 / 0.116596 | -1.963327 |
| **Explained ratio**          | min(\|-1.963327\| / \|-1.971498\|, 1)                | 0.996     |
| **Final score (no penalty)** | 0.6\*0.750000 + 0.4\*0.995856                        | 0.848     |
| **Penalty rule**             | ×0.3 if LATAM sign mismatches (not applied here)     | —         |
| **Final score (applied)**    | 0.848342                                             | 0.848     |

* Sign agreement = mean(Sign match?)
* σ(metric Δ), σ(hypo Δ, adj) = population std across regions
* z\_m(LATAM) = Δ\_metric\_LATAM / σ(metric Δ)
* z\_h(LATAM) = Δ\_hypo\_adj\_LATAM / σ(hypo Δ, adj)
* Explained ratio = min(|z\_h|/|z\_m|, 1)
* Final score = 0.6·(sign agreement) + 0.4·(explained ratio)
  (No penalty applied because LATAM sign matches.)


## "% Outreach"(deemed "coincidental moves")

### Per-region deltas (vs Global)

| **Region**     | **Metric (M_r)** | **Δ_metric formula**       | **Δ_metric value** | **Hypothesis (H_r)** | **Δ_hypo_raw formula**     | **Δ_hypo_raw value** | **Δ_hypo_adj formula**     | **Δ_hypo_adj value** | **Sign check**                 |
| -------------- | ---------------- | -------------------------- | ------------------ | -------------------- | -------------------------- | -------------------- | -------------------------- | -------------------- | ------------------------------ |
| **APAC**       | 0.9630           | (0.9630 - 0.9660) / 0.9660 | -0.31%             | 0.9860               | (0.9860 - 0.9950) / 0.9950 | -0.90%               | (0.9860 - 0.9950) / 0.9950 | -0.90%               | sign(ΔM) ?= sign(ΔH_adj) → Yes |
| **EMEA**       | 0.9860           | (0.9860 - 0.9660) / 0.9660 | 2.07%              | 0.9990               | (0.9990 - 0.9950) / 0.9950 | 0.40%                | (0.9990 - 0.9950) / 0.9950 | 0.40%                | sign(ΔM) ?= sign(ΔH_adj) → Yes |
| **LATAM (L4)** | 0.9150           | (0.9150 - 0.9660) / 0.9660 | -5.28%             | 0.9930               | (0.9930 - 0.9950) / 0.9950 | -0.20%               | (0.9930 - 0.9950) / 0.9950 | -0.20%               | sign(ΔM) ?= sign(ΔH_adj) → Yes |
| **NA**         | 0.9630           | (0.9630 - 0.9660) / 0.9660 | -0.31%             | 0.9990               | (0.9990 - 0.9950) / 0.9950 | 0.40%                | (0.9990 - 0.9950) / 0.9950 | 0.40%                | sign(ΔM) ?= sign(ΔH_adj) → No  |
### Summary components

| **Component**                | **Formula**                                          | **Value**   |
| ---------------------------- | ---------------------------------------------------- | ----------- |
| **Expected direction**       | same                                                 | same        |
| **Sign agreement**           | 3/4                                                  | 0.750       |
| **σ(metric Δ)**              | std([-0.003106, 0.020704, -0.052795, -0.003106])     | 0.026779    |
| **σ(hypo Δ, adj)**           | std([-0.009045, 0.00402, -0.00201, 0.00402])         | 0.005383    |
| **z_m (LATAM)**              | ΔM_LATAM / σ(metric Δ) = -0.052795 / 0.026779        | -1.971498   |
| **z_h (LATAM)**              | ΔH_adj_LATAM / σ(hypo Δ, adj) = -0.002010 / 0.005383 | -0.373408   |
| **Explained ratio**          | min(\|-0.373408\| / \|-1.971498\|, 1)                | 0.189       |
| **Final score (no penalty)** | 0.6*0.750000 + 0.4*0.189403                          | 0.526       |
| **Penalty rule**             | ×0.3 if LATAM sign mismatches                        | not applied |
| **Final score (applied)**    | 0.525761                                             | 0.526       |

## % Client Interactions that moved stage forward (deemed "wrong hypothesis direction")
### Per-region deltas (vs Global)

| **Region**     | **Metric (M_r)** | **Δ_metric formula**       | **Δ_metric value** | **Hypothesis (H_r)** | **Δ_hypo_raw formula**     | **Δ_hypo_raw value** | **Δ_hypo_adj formula**     | **Δ_hypo_adj value** | **Sign check**                 |
| -------------- | ---------------- | -------------------------- | ------------------ | -------------------- | -------------------------- | -------------------- | -------------------------- | -------------------- | ------------------------------ |
| **APAC**       | 0.9630           | (0.9630 - 0.9660) / 0.9660 | -0.31%             | 0.2400               | (0.2400 - 0.1980) / 0.1980 | 21.21%               | (0.2400 - 0.1980) / 0.1980 | 21.21%               | sign(ΔM) ?= sign(ΔH_adj) → No  |
| **EMEA**       | 0.9860           | (0.9860 - 0.9660) / 0.9660 | 2.07%              | 0.1960               | (0.1960 - 0.1980) / 0.1980 | -1.01%               | (0.1960 - 0.1980) / 0.1980 | -1.01%               | sign(ΔM) ?= sign(ΔH_adj) → No  |
| **LATAM (L4)** | 0.9150           | (0.9150 - 0.9660) / 0.9660 | -5.28%             | 0.2060               | (0.2060 - 0.1980) / 0.1980 | 4.04%                | (0.2060 - 0.1980) / 0.1980 | 4.04%                | sign(ΔM) ?= sign(ΔH_adj) → No  |
| **NA**         | 0.9630           | (0.9630 - 0.9660) / 0.9660 | -0.31%             | 0.1720               | (0.1720 - 0.1980) / 0.1980 | -13.13%              | (0.1720 - 0.1980) / 0.1980 | -13.13%              | sign(ΔM) ?= sign(ΔH_adj) → Yes |
### Summary components

| **Component**                | **Formula**                                         | **Value** |
| ---------------------------- | --------------------------------------------------- | --------- |
| **Expected direction**       | same                                                | same      |
| **Sign agreement**           | 1/4                                                 | 0.250     |
| **σ(metric Δ)**              | std([-0.003106, 0.020704, -0.052795, -0.003106])    | 0.026779  |
| **σ(hypo Δ, adj)**           | std([0.212121, -0.010101, 0.040404, -0.131313])     | 0.123376  |
| **z_m (LATAM)**              | ΔM_LATAM / σ(metric Δ) = -0.052795 / 0.026779       | -1.971498 |
| **z_h (LATAM)**              | ΔH_adj_LATAM / σ(hypo Δ, adj) = 0.040404 / 0.123376 | 0.327487  |
| **Explained ratio**          | min(\|0.327487\| / \|-1.971498\|, 1)                | 0.166     |
| **Final score (no penalty)** | 0.6*0.250000 + 0.4*0.166111                         | 0.216     |
| **Penalty rule**             | ×0.3 if LATAM sign mismatches                       | applied   |
| **Final score (applied)**    | 0.064933                                            | 0.065     |
## % Clients Deemed Uncontactable (deemed as "focal region direction mismatch")

### Per-region deltas (vs Global)

| **Region**     | **Metric (M_r)** | **Δ_metric formula**       | **Δ_metric value** | **Hypothesis (H_r)** | **Δ_hypo_raw formula**     | **Δ_hypo_raw value** | **Δ_hypo_adj formula**        | **Δ_hypo_adj value** | **Sign check**                 |
| -------------- | ---------------- | -------------------------- | ------------------ | -------------------- | -------------------------- | -------------------- | ----------------------------- | -------------------- | ------------------------------ |
| **APAC**       | 0.9630           | (0.9630 - 0.9660) / 0.9660 | -0.31%             | 0.0340               | (0.0340 - 0.0330) / 0.0330 | 3.03%                | -((0.0340 - 0.0330) / 0.0330) | -3.03%               | sign(ΔM) ?= sign(ΔH_adj) → Yes |
| **EMEA**       | 0.9860           | (0.9860 - 0.9660) / 0.9660 | 2.07%              | 0.0270               | (0.0270 - 0.0330) / 0.0330 | -18.18%              | -((0.0270 - 0.0330) / 0.0330) | 18.18%               | sign(ΔM) ?= sign(ΔH_adj) → Yes |
| **LATAM (L4)** | 0.9150           | (0.9150 - 0.9660) / 0.9660 | -5.28%             | 0.0160               | (0.0160 - 0.0330) / 0.0330 | -51.52%              | -((0.0160 - 0.0330) / 0.0330) | 51.52%               | sign(ΔM) ?= sign(ΔH_adj) → No  |
| **NA**         | 0.9630           | (0.9630 - 0.9660) / 0.9660 | -0.31%             | 0.0460               | (0.0460 - 0.0330) / 0.0330 | 39.39%               | -((0.0460 - 0.0330) / 0.0330) | -39.39%              | sign(ΔM) ?= sign(ΔH_adj) → Yes |
### Summary components

| **Component**                | **Formula**                                         | **Value** |
| ---------------------------- | --------------------------------------------------- | --------- |
| **Expected direction**       | opposite                                            | opposite  |
| **Sign agreement**           | 3/4                                                 | 0.750     |
| **σ(metric Δ)**              | std([-0.003106, 0.020704, -0.052795, -0.003106])    | 0.026779  |
| **σ(hypo Δ, adj)**           | std([-0.030303, 0.181818, 0.515152, -0.393939])     | 0.330133  |
| **z_m (LATAM)**              | ΔM_LATAM / σ(metric Δ) = -0.052795 / 0.026779       | -1.971498 |
| **z_h (LATAM)**              | ΔH_adj_LATAM / σ(hypo Δ, adj) = 0.515152 / 0.330133 | 1.560438  |
| **Explained ratio**          | min(\|1.560438\| / \|-1.971498\|, 1)                | 0.791     |
| **Final score (no penalty)** | 0.6*0.750000 + 0.4*0.791499                         | 0.767     |
| **Penalty rule**             | ×0.3 if LATAM sign mismatches                       | applied   |
| **Final score (applied)**    | 0.229980                                            | 0.230     |

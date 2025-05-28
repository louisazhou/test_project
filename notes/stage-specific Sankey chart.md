### How to turn stage-change data into actionable RCA

| Layer                           | What to compute                                                     | Why it’s useful                                                       |
| ------------------------------- | ------------------------------------------------------------------- | --------------------------------------------------------------------- |
| **1. Basic funnel**             | Stage-to-stage conversion rates and drop-off counts                 | Flags where a region / vertical leaks most value                      |
| **2. CI intensity by stage**    | `avg_CI_per_initiative` in each stage bucket                        | Tests *“are we quitting too early or over-servicing?”*                |
| **3. CI quality**               | Proportion of CIs tagged to a CLI *vs* unlinked “relationship” CIs  | Low linked-CI share in Scoping/Pitching often predicts drop-off       |
| **4. Time-to-stage (survival)** | Kaplan-Meier curve: probability of still *not* Closed Won over days | Spots regions that stall unusually long in a stage                    |
| **5. Topic efficacy**           | Win-rate uplift by CI topic (e.g. Measurement demo +3 pp)           | Guides enablement content; feed into next-best-action engine          |
| **6. Stage Markov model**       | Transition matrix *P(stage t → stage t+1)* per region               | Quantifies whether a region’s “probability of going backward” is high |

---

### Concrete analysis workflow

1. **Build a long table**
   `initiative_id | region | stage | enter_date | exit_date | CI_count | CI_linked | CI_topic`

2. **Per region**

   * Funnel bar chart (`#enter – #exit`)
   * Stage-specific `CI_linked / CI_total` ratio
   * Mean *linked CI* in Scoping & Pitching; z-score vs global

3. **Regression to flag causal signals**

   ```text
   logit(Closed_Won) ~ CI_linked_Pitching
                     + Days_in_Pitching
                     + vertical dummies
   ```

   > A 1-unit ↑ in linked CI in Pitching raises win odds by 12 % (p < 0.01).

4. **Sankey / Alluvial diagram**
   Shows full flow: Discovery → … → Closed Won / Lost; thickness ∝ # initiatives.

5. **Topic-efficacy matrix**
   Heatmap: CI\_topic on y-axis, stage on x-axis, cell = Δwin-rate vs baseline.

---

### Interpreting insights

* **Low Pitching linked-CI but high “relationship” CI**
  → Re-train AMs to tie calls to concrete proposals.
* **High drop at Scoping but normal CI volume**
  → Problem is *quality* of proposal, not quantity of calls.
* **Survival curve: NA stalls 2× longer in Validated**
  → Escalate legal / contracting resources.

---

### Caveats & safeguards

| Caveat                                        | Mitigation                                             |
| --------------------------------------------- | ------------------------------------------------------ |
| CIs logged late or mis-tagged                 | Use only CIs within ±X days of stage timeframe         |
| Reverse causality (“good deals draw more CI”) | Stage-lag CI count (CI up to enter\_date(Pitching))    |
| Sparse topics                                 | Group rare topics into “Other”; bootstrap CI for rates |
| Region size disparity                         | Weight comparisons by initiative count                 |

---

### Payload snippet to store

```json
{
  "region": "NA",
  "stage_metrics": [
    {"stage":"Discovery","enter":500,"exit":350,"ci_linked_mean":0.8},
    {"stage":"Pitching","enter":220,"exit":140,"ci_linked_mean":2.1}
  ],
  "conversion_gaps_pp": {
    "Discovery→Scoping": -4.2,
    "Pitching→Validated": -6.6
  },
  "ci_topic_uplift": [
    {"topic":"Measurement demo","delta_win_pp": +3.1},
    {"topic":"Roadmap deep-dive","delta_win_pp": -0.9}
  ],
  "survival_median_days": {
    "Validated": 28,
    "Committed": 12
  }
}
```

**Visuals**

* Funnel bar with red drop-off highlights
* CI intensity strip plot per stage
* Alluvial flow per region
* Heatmap of topic uplift

Feed values into Jinja narrative:

> “NA loses **6.6 pp** more deals in Pitching→Validated than ROW.
> They log only **2.1 linked CIs** vs global **3.4**. Boosting pitch-related CIs by 1 could raise win-rate \~ +4 pp.”

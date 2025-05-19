# Directional Sign-Agreement Test

Checks whether metric and hypothesis deltas move in the same direction more often than chance (p = 0.5) when we have very few regions.

```python
from scipy import stats
import numpy as np

# Δmetric, Δhypo: arrays of deviations from mean for each region
k = np.sum(np.sign(delta_metric) == np.sign(delta_hypo))
n = len(delta_metric)

# two-sided exact binomial test
p_value = stats.binom_test(k, n, p=0.5, alternative='two-sided')

# interpret: low p-value → directional consistency unlikely by chance
```

- **H₀**: sign-agreement rate = 50%
- **H₁**: rate ≠ 50% (or use 'greater'/'less')
- returns p-value via exact tail sums of Binomial(n, 0.5)

# Correlation-Based Scoring (`score_hypothesis`)
Applies when you have ≥5 points or care about magnitude and direction.

* **Inputs**

  * `df[region]`: metric & hypothesis series
  * `metric_anomaly_info`: {anomalous\_region, metric\_val, global\_val}

* **Components & Weights**

  1. **Direction Alignment (30%)**
     1.0 if `sign(corr)` matches `expected_direction`, else 0
  2. **Consistency (30%)**
     \= |Pearson corr(metric, hypo)|
  3. **Hypothesis Z-score (20%)**
     bucketed to {0.3, 0.6, 0.7, 1.0} based on |z|>1/2/3
  4. **Explained Ratio (20%)**
     \= min(|Δ\_hypo|/|Δ\_metric|, 1) for the anomalous region

* **Final Score**

  ```python
  0.3*direction_alignment
  + 0.3*consistency
  + 0.2*hypo_z_score_norm
  + 0.2*explained_ratio
  ```

* **Magnitude**

  * `%` columns → absolute pp diff ×100
  * others → relative % diff ×100

---

# Sign-Based Scoring (`sign_based_score_hypothesis`)
Robust for n=4–5 regions; ignores magnitude except in explained-ratio.

* **Inputs**

  * same `df`, `metric_anomaly_info`, `expected_direction`

* **Steps & Weights**

  1. **Sign Agreement**

     ```python
     score_sign = (#regions where sign(Δmetric)==sign(Δhypo))/n
     ```

     (flip hypo sign if expecting “opposite”)
  2. **Binomial p-value**
     as in (1), for context but not in final score
  3. **Explained Ratio**
     \= min(|Δ\_hypo|/|Δ\_metric|, 1) at the anomalous region

* **Final Score**

  ```python
  0.6*score_sign
  + 0.4*explained_ratio
  ```

* **Output Flags**

  * `explains` = final\_score > 0.5
  * `is_sign_based` = True

# Data Flow

## System Pipeline

### 1. Configuration Loading
- Loads metric definitions from `config/metrics.yaml`
- Processes hypothesis configurations from `config/hypotheses.yaml`
- Applies system settings from `config/settings.yaml`
- Registers datasets via data catalog

### 2. Data Ingestion
- Reads metric data from input files
- Validates data against expected schema
- Performs initial data cleaning and normalization
- Registers data in the data registry

### 3. Metric Processing
- Calculates core statistics for each metric
- Computes deltas from reference values
- Applies transformations as configured
- Prepares metrics for anomaly detection

### 4. Anomaly Detection
The system uses five methods to detect anomalies:

1. **Z-score Analysis**
   - Calculates deviation from mean in standard deviations
   - Flags metrics exceeding threshold (typically 2-3 σ)
   - Accounts for distribution shape

2. **Delta Percentage**
   - Computes percentage difference from reference value
   - Applies configurable threshold (typically 10-20%)
   - Direction-aware (improvement vs. degradation)

3. **Interquartile Range (IQR)**
   - Identifies outliers beyond Q1-1.5×IQR and Q3+1.5×IQR
   - Robust to non-normal distributions
   - Less sensitive to extreme outliers

4. **Confidence Interval**
   - Constructs 95% CI around reference value
   - Flags metrics outside confidence interval
   - Accounts for sample size and variation

5. **Percentile Bounds**
   - Checks if metric falls outside 10th-90th percentile range
   - Provides non-parametric bounds
   - Useful for skewed distributions

### 5. Hypothesis Testing
For each anomalous metric, the system evaluates configured hypotheses:

#### Scoring Components
1. **Direction Alignment (30%)**
   - Checks if metric and hypothesis movements align
   - Based on expected direction ('same' or 'opposite')
   ```
   if expected_direction == 'opposite':
       if consistency_sign < 0:
           direction_alignment = 1.0
   elif expected_direction == 'same':
       if consistency_sign > 0:
           direction_alignment = 1.0
   ```

2. **Consistency (30%)**
   - Measures correlation strength
   - Uses absolute correlation value
   ```
   consistency = abs(raw_consistency)
   ```

3. **Hypothesis Z-score (20%)**
   - Evaluates statistical significance
   - Normalized based on magnitude
   ```
   if abs_hypo_z > 3:
       hypo_z_score_norm = 1.0
   elif abs_hypo_z > 2:
       hypo_z_score_norm = 0.7
   elif abs_hypo_z > 1:
       hypo_z_score_norm = 0.6
   else:
       hypo_z_score_norm = 0.3
   ```

4. **Explained Ratio (20%)**
   - Measures proportion of metric deviation explained
   ```
   explained_ratio = min(abs(hypo_delta) / abs(metric_delta), 1.0)
   ```

#### Final Score Calculation
```
final_score = (
    0.3 * direction_alignment +
    0.3 * consistency +
    0.2 * hypo_z_score_norm +
    0.2 * explained_ratio
)
explains = final_score > 0.5  # Threshold for explanation
```

### 6. Anomaly Classification
The system classifies anomalies as "good" or "bad" based on:
- Metric's preferred direction (higher_is_better flag)
- Observed delta (positive or negative)
```
# Good/Bad Anomaly Classification
is_anomaly = anomaly_df['is_anomaly']
delta_positive = anomaly_df['delta'] > 0
wants_higher = anomaly_df['higher_is_better']

# Determine anomaly type
good_anomaly = is_anomaly & ((wants_higher & delta_positive) | (~wants_higher & ~delta_positive))
bad_anomaly = is_anomaly & ((wants_higher & ~delta_positive) | (~wants_higher & delta_positive))
```

### 7. Visualization Generation
- The Plot Engine generates visualizations based on analysis results
- Visualization type depends on configuration (detailed/succinct)
- Formats include charts for metric performance and hypothesis comparison

### 8. Narrative Creation
- The Narrative Engine creates text explanations for anomalies
- Generates insights based on hypothesis evaluation
- Formats findings for inclusion in reports

### 9. Report Generation
- The Presentation Engine compiles results into structured reports
- Creates PowerPoint or Google Slides document
- Organizes content by metrics and hypotheses
- Applies consistent styling and formatting

## Data Paths

### Input Path
`input/metrics_input/` → `data_registry.py` → `metric_engine.py` → `anomaly_gate.py`

### Hypothesis Path
`config/hypotheses.yaml` → `hypothesis_engine.py` → `scoring system` → `ranked results`

### Output Path
`analysis results` → `plot_engine.py` → `narrative_engine.py` → `presentation.py` → `output/` 
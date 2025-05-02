# Data Flow

## Input Processing

### 1. Configuration Loading
```python
# src/data_processor.py
def _load_metrics_config(self) -> None:
    """Loads metrics configuration and settings"""
```
- Loads metric definitions
- Sets evaluation parameters
- Configures visualization options

### 2. Data Preparation
```python
# src/data_processor.py
def prepare_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Processes input data for analysis"""
```
- Validates input data format
- Performs data cleaning
- Prepares analysis datasets

## Analysis Flow

### 1. Anomaly Detection
```python
# src/automation_pipeline.py
def detect_anomaly(self) -> Tuple[float, float, float, int]:
    """Detects anomalies using multiple statistical methods"""
```
Uses 5 methods to detect anomalies:
1. Z-score: Deviation from mean in standard deviations
2. Delta: Percentage difference from reference value
3. IQR: Outlier detection using interquartile range
4. 95% CI: Outside confidence interval
5. 10/90 Percentile: Outside expected range

### 2. Hypothesis Evaluation
```python
# src/automation_pipeline.py
def calculate_score(self) -> Tuple[float, float, float, float, float, bool]:
    """Evaluates hypothesis validity using multiple components"""
```

## Scoring System
The system uses a weighted scoring approach:

### Components
1. **Direction Alignment (30%)**
   - Checks if metric and hypothesis movements align
   - Based on expected direction ('same' or 'opposite')
   ```python
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
   ```python
   consistency = abs(raw_consistency)
   ```

3. **Hypothesis Z-score (20%)**
   - Evaluates statistical significance
   - Normalized based on magnitude
   ```python
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
   ```python
   explained_ratio = min(abs(hypo_delta) / abs(metric_delta), 1.0)
   ```

### Final Score Calculation
```python
final_score = (
    0.3 * direction_alignment +
    0.3 * consistency +
    0.2 * hypo_z_score_norm +
    0.2 * explained_ratio
)
explains = final_score > 0.5  # Threshold for explanation
```

## Result Processing

### 1. Anomaly Classification
```python
# Good/Bad Anomaly Classification
is_anomaly = anomaly_df['is_anomaly']
delta_positive = anomaly_df['delta'] > 0
wants_higher = anomaly_df['higher_is_better']

# Determine anomaly type
good_anomaly = is_anomaly & ((wants_higher & delta_positive) | (~wants_higher & ~delta_positive))
bad_anomaly = is_anomaly & ((wants_higher & ~delta_positive) | (~wants_higher & delta_positive))
```

### 2. Result Consolidation
- Combines all hypothesis evaluations
- Ranks hypotheses by score
- Generates final insights

## Output Generation
1. Analysis Results
   - Anomaly detection summary
   - Hypothesis evaluation scores
   - Explanation texts

2. Visualizations
   - Metric performance plots
   - Hypothesis comparison charts
   - Score component breakdowns

3. Reports
   - PowerPoint/Google Slides
   - Future: Dashboard integration 
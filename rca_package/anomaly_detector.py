import pandas as pd

def detect_snapshot_anomaly_for_column(df: pd.DataFrame, ref_index: str, 
                              z_thresh=1.5, delta_thresh=0.1, column: str = 'Global') -> dict:
    """
    Detect anomalies in a snapshot of data for a specific column using multiple statistical methods.
    
    This function uses a voting system across multiple statistical methods to identify anomalies:
    1. Z-score: Identifies values beyond z_thresh standard deviations from the mean
    2. Delta: Identifies values with absolute difference > delta_thresh from reference
    3. IQR: Identifies values beyond 1.5 * IQR from Q1/Q3
    4. 95% CI: Identifies values outside the 95% confidence interval
    5. Percentile: Identifies values outside 10th/90th percentiles
    
    Args:
        df (pd.DataFrame): DataFrame containing the data to analyze
        ref_index (str): Reference index to compare against (usually 'Global')
        z_thresh (float, optional): Z-score threshold for anomaly detection. Defaults to 1.5.
        delta_thresh (float, optional): Delta threshold for anomaly detection. Defaults to 0.1.
        column (str, optional): Column name to analyze. Defaults to 'Global'.
        
    Returns:
        dict: Dictionary containing anomaly information with keys:
            - anomalous_region: Index of the detected anomaly
            - metric_val: Value at the anomalous region
            - global_val: Reference value
            - direction: 'higher' or 'lower'
            - magnitude: Float representing the percentage difference
            - higher_is_better: Boolean indicating if higher values are better
            Returns empty dict if no anomaly is detected.
    """
    values = df[column].dropna()
    if values.std() < 0.01 or values.nunique() == 1:
        return {}

    ref_value = df.loc[ref_index, column]
    anomaly_scores = {}

    for idx in df.index:
        if idx == ref_index:
            continue

        current_value = df.loc[idx, column]
        mean = values.mean()
        std = values.std()
        z_score = (current_value - mean) / std if std > 0 else 0
        delta = current_value - ref_value

        votes = 0

        # 1. Z-score
        if abs(z_score) > z_thresh:
            votes += 1

        # 2. Delta
        if abs(delta) > delta_thresh:
            votes += 1

        # 3. IQR
        q1 = values.quantile(0.25)
        q3 = values.quantile(0.75)
        iqr = q3 - q1
        if current_value < q1 - 1.5 * iqr or current_value > q3 + 1.5 * iqr:
            votes += 1

        # 4. 95% CI
        lower_ci = mean - 1.96 * std if std > 0 else mean
        upper_ci = mean + 1.96 * std if std > 0 else mean
        if current_value < lower_ci or current_value > upper_ci:
            votes += 1

        # 5. 10/90 percentile
        p10 = values.quantile(0.1)
        p90 = values.quantile(0.9)
        if current_value < p10 or current_value > p90:
            votes += 1

        anomaly_scores[idx] = (votes, delta, current_value)

    if not anomaly_scores:
        return {}

    max_votes = max(score[0] for score in anomaly_scores.values())
    candidates = [idx for idx, (votes, _, _) in anomaly_scores.items() if votes == max_votes]
    anomaly_idx = max(candidates, key=lambda idx: abs(anomaly_scores[idx][1])) if len(candidates) > 1 else candidates[0]

    delta = anomaly_scores[anomaly_idx][1]
    current_value = anomaly_scores[anomaly_idx][2]
    direction = 'higher' if delta > 0 else 'lower'

    if 'pct' in column or '%' in column:
        magnitude = f"{abs(delta * 100):.2f}pp"
    else:
        magnitude = f"{abs((delta / ref_value) * 100):.2f}%" if ref_value else "N/A"

    higher_is_better = False if column == 'cli_closed_pct' else True

    return {
        'anomalous_region': anomaly_idx,
        'metric_val': current_value,
        'global_val': ref_value,
        'direction': direction,
        'magnitude': magnitude,
        'higher_is_better': higher_is_better,
    }
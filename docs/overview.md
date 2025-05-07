# RCA Automation System Overview

## Purpose
The Root Cause Analysis (RCA) Automation system is designed to automatically detect metric anomalies across regions and identify potential root causes through hypothesis testing. The system aims to reduce manual analysis time and provide consistent, data-driven insights.

## Current Scope
- Operational metrics analysis
- Single-dimensional hypothesis testing
- PowerPoint/Google Slides report generation
- Statistical anomaly detection
- Configurable metrics and hypotheses

## Key Features
- Automated anomaly detection using multiple methods:
  - Z-score: Deviation from mean in standard deviations
  - Delta: Percentage difference from reference value
  - IQR: Outlier detection using interquartile range
  - 95% CI: Outside confidence interval
  - 10/90 Percentile: Outside expected range
- Hypothesis evaluation and scoring with weighted components:
  - Direction Alignment (30%)
  - Consistency (30%)
  - Hypothesis Z-score (20%)
  - Explained Ratio (20%)
- Visualization generation with detailed and succinct options
- Report automation to PowerPoint and Google Slides

## Development Phases

### Phase 1 (Current)
- Core engines implemented:
  - Metric Engine: Processes metrics and calculates statistics
  - Hypothesis Engine: Evaluates hypotheses against metrics
  - Plot Engine: Creates visualizations of results
  - Narrative Engine: Generates explanations
  - Presentation Engine: Formats output reports
- Manual hypothesis configuration
- Basic visualization options (detailed/succinct)
- PowerPoint/Google Slides output
- Single-dimensional analysis

### Phase 2 (Upcoming)
- Multi-dimensional hypothesis support
- Enhanced visualization framework
- Dashboard integration (Unidash)
- Pacing metrics support
- New hypothesis types:
  - Depth Spotter: Granular analysis
  - Reason Mix: Tagged reason analysis
  - Benchmark Comparison: Historical benchmarking
  - History Reference: Temporal analysis

### Phase 3 (Future)
- LLM Agent integration with Llama
- ReAct framework implementation
- Revenue and product metrics
- Time-series and causal analysis
- Advanced hypothesis types
- Automated hypothesis selection and ranking 
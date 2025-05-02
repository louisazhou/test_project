# RCA Automation System Overview

## Purpose
The Root Cause Analysis (RCA) Automation system is designed to automatically detect metric anomalies across regions and identify potential root causes through hypothesis testing. The system aims to reduce manual analysis time and provide consistent, data-driven insights.

## Current Scope
- Operational metrics analysis
- Single-dimensional hypothesis testing
- PowerPoint/Google Slides report generation
- Snapshot-based analysis (non-temporal)

## Key Features
- Automated anomaly detection
- Hypothesis evaluation and scoring
- Visualization generation
- Report automation

## Development Phases

### Phase 1 (Current)
- Manual hypothesis configuration
- Basic visualization options (detailed/succinct)
- PowerPoint output
- Single-dimensional analysis
- Core scoring system implementation
  - Direction Alignment (30%)
  - Consistency (30%)
  - Hypothesis Z-score (20%)
  - Explained Ratio (20%)

### Phase 2 (Upcoming)
- Multi-dimensional hypothesis support
- Enhanced visualization framework
- Dashboard integration (Unidash)
- Pacing metrics support
- New hypothesis types:
  - Depth Spotter
  - Reason Mix
  - Benchmark Comparison
  - History Reference

### Phase 3 (Future)
- LLM Agent integration with Llama
- ReAct framework implementation
- Revenue and product metrics
- Time-series analysis
- Advanced hypothesis types
- Automated hypothesis selection and ranking 
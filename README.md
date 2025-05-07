# RCA Automation System

## Overview
The Root Cause Analysis (RCA) Automation System is a powerful tool designed to automate the detection of metric anomalies and identify their root causes. It streamlines the process of analyzing operational metrics, evaluating hypotheses, and generating insights through automated analysis and visualization.

## Documentation

### Core Documentation
- [Overview](docs/overview.md): Project purpose, scope, and key features
- [Architecture](docs/architecture.md): System design and component details
- [Data Flow](docs/data_flow.md): Data processing and analysis pipeline
- [Components](docs/components.md): Detailed component documentation
- [Setup & Usage](docs/setup.md): Configuration and usage instructions
- [Future Development](docs/future.md): Roadmap and planned features
- [Advanced Ideas](docs/ideas.md): Brainstorming concepts for future enhancement

## System Usage

### Configuration
1. Update metrics in `config/metrics.yaml`
2. Set up hypotheses in `config/hypotheses.yaml`
3. Review system settings in `config/settings.yaml`
4. Place input data in `input/metrics_input/`

### Running the Analysis
Run the main pipeline script to perform the analysis:
```
python src/run_pipeline.py
```

### Output Locations
- Analysis results: `output/analysis_results/`
- Visualizations: `output/visualizations/`
- Reports: `output/reports/`

## Key Features

### Current Capabilities
- Automated anomaly detection using multiple statistical methods
- Single-dimensional hypothesis testing with weighted scoring system
- Visualization generation with detailed and succinct options
- Report automation with PowerPoint/Google Slides integration
- Configurable metrics and hypotheses

### Upcoming Features
- Multi-dimensional analysis for complex relationships
- Enhanced visualization options with interactive elements
- Dashboard integration for real-time monitoring
- LLM-powered insights using advanced NLP techniques
- Advanced analytics with temporal and causal analysis

## Project Structure
```
RCA_automation/
├── config/           # Configuration files
│   ├── metrics.yaml  # Metric definitions
│   ├── hypotheses.yaml # Hypothesis configurations
│   └── settings.yaml # System settings
├── docs/             # Documentation
├── input/            # Input data directory
├── src/              # Source code
│   ├── core/         # Core engine components
│   ├── handlers/     # Data handlers
│   ├── plotting/     # Visualization modules
│   ├── reporting/    # Report generation
│   └── cli/          # Command-line interface
├── output/           # Generated outputs
├── tmp/              # Temporary files
└── hypothesis_input/ # Hypothesis data files
```
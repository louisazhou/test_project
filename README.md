# RCA Automation System

## Overview
The Root Cause Analysis (RCA) Automation System is a powerful tool designed to automate the detection of metric anomalies and identify their root causes. It streamlines the process of analyzing operational metrics, evaluating hypotheses, and generating insights through automated analysis and visualization.

## Documentation

### Core Documentation
- [Overview](docs/overview.md): Project purpose, scope, and key features
- [Architecture](docs/architecture.md): System design and component details
- [Data Flow](docs/data_flow.md): Data processing and analysis pipeline
- [Components](docs/components.md): Detailed component documentation
- [Setup & Usage](docs/setup.md): Installation and usage instructions
- [Future Development](docs/future.md): Roadmap and planned features

## Quick Start

### Prerequisites
- Python 3.8+
- pip package manager
- Virtual environment (recommended)

### Installation
```bash
# Clone the repository
git clone [repository-url]

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Basic Usage
1. Configure metrics in `config/metrics.yaml`
2. Set up hypotheses in `config/hypotheses.yaml`
3. Place input data in `input/metrics_input/`
4. Run the pipeline:
```bash
python src/run_pipeline.py
```

## Key Features

### Current Capabilities
- Automated anomaly detection
- Single-dimensional hypothesis testing
- Visualization generation
- Report automation
- PowerPoint output

### Upcoming Features
- Multi-dimensional analysis
- Enhanced visualization options
- Dashboard integration
- LLM-powered insights
- Advanced analytics suite

## Project Structure
```
RCA_automation/
├── config/           # Configuration files
├── docs/            # Documentation
├── input/           # Input data directory
├── src/             # Source code
├── output/          # Generated outputs
├── tmp/             # Temporary files
└── tests/           # Test suite
```

## Contributing
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

See [Future Development](docs/future.md) for planned enhancements and contribution opportunities.

## Support
For issues and feature requests, please use the issue tracker.

## License
[License Type] - See LICENSE file for details 
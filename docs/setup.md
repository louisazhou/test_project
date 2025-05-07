# Setup & Usage Guide

## Installation

### Prerequisites
- Python 3.8+
- pip package manager
- Virtual environment (recommended)

### Dependencies
```bash
pip install -r requirements.txt
```

Key dependencies:
- pandas: Data processing
- matplotlib: Visualization
- seaborn: Enhanced plotting
- python-pptx: PowerPoint generation
- google-api-python-client: Google Drive integration

## Configuration

### 1. Metrics Configuration
Edit `config/metrics.yaml`:
```yaml
metrics:
  metric_name:
    description: str
    natural_name: str
    dependencies: List[str]
    hypothesis: List[str]
    higher_is_better: bool
```

### 2. Hypothesis Configuration
Edit `config/hypotheses.yaml`:
```yaml
hypotheses:
  - name: str
    description: str
    hypothesis_type: str  # single_dim or multi_dim_groupby
    input_data:
      file: str
      value_column: str
    evaluation:
      direction: str  # same or opposite
```

### 3. Google Drive Integration (Optional)
1. Set up Google Cloud Project
2. Enable Drive API
3. Create credentials
4. Save credentials as `credentials.json`

## Directory Structure
```
RCA_automation/
├── config/
│   ├── metrics.yaml
│   └── hypotheses.yaml
├── input/
│   └── metrics_input/
│       └── *.csv
├── src/
│   ├── automation_pipeline.py
│   ├── data_processor.py
│   ├── visualization.py
│   └── presentation.py
├── output/
│   └── visualizations/
├── tmp/
│   └── debug/
└── docs/
```

## Running the Pipeline

### Basic Usage
```bash
python src/run_pipeline.py
```

### Output Options
Configure in `metrics.yaml`:
```yaml
settings:
  visualization_type: "detailed"  # or "succinct"
  generate_ppt: true
  upload_to_drive: false
  drive_folder_id: null
```

## Development Guidelines

### Adding New Hypothesis Types
1. Define hypothesis configuration in `hypotheses.yaml`
2. Implement analysis logic in `automation_pipeline.py`
3. Add visualization support in `visualization.py`
4. Update documentation

### Planned Enhancements
- See [Future Development](future.md) for details
- Follow issue tracker for updates
- Review roadmap in project documentation 
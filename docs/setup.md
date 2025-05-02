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

### Modifying Visualization
1. Update `RCAVisualizer` class in `visualization.py`
2. Maintain consistent styling
3. Follow existing naming conventions

### Adding New Metrics
1. Add metric configuration to `metrics.yaml`
2. Ensure data file contains required columns
3. Update hypothesis associations if needed

## Troubleshooting

### Common Issues
1. Missing Data Columns
   - Verify input data format
   - Check column names in configuration

2. Visualization Errors
   - Confirm matplotlib backend
   - Check plot dimensions

3. Google Drive Integration
   - Verify credentials
   - Check API permissions

### Debug Information
- Check `tmp/` directory for intermediate results
- Review logging output
- Examine debug visualizations

## Future Development

### Contributing
1. Fork the repository
2. Create feature branch
3. Follow coding standards
4. Submit pull request

### Planned Enhancements
- See [Future Development](future.md) for details
- Follow issue tracker for updates
- Review roadmap in project documentation 
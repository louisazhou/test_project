import pytest
import pandas as pd
import os
from pathlib import Path
from src.core.data_catalog import DataCatalog
from src.core.data_registry import DataRegistry

@pytest.fixture
def sample_data():
    # Create a sample DataFrame matching the real data structure
    return pd.DataFrame({
        'territory_region_name': ['AM-APAC', 'AM-EMEA', 'AM-NA'],
        'CLI_per_active_AM': [153.78, 123.8, 91.58],
        'SLI_per_active_AM': [11.35, 10.08, 4.99],
        'cli_closed_pct': [0.043, 0.034, 0.087]
    })

@pytest.fixture
def mock_config_dir(tmp_path):
    # Create a temporary directory with mock config files
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    
    # Create datasets.yaml with proper indentation
    datasets_yaml = config_dir / "datasets.yaml"
    datasets_yaml.write_text("""
datasets:
  test_dataset:
    path: "test_data.csv"
    rename:
      territory_region_name: region
      CLI_per_active_AM: CLI_per_active_AM
      SLI_per_active_AM: SLI_per_active_AM
      cli_closed_pct: cli_closed_pct
""")
    
    return config_dir

@pytest.fixture
def mock_input_dir(tmp_path, sample_data):
    # Create a temporary directory with mock input files
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    
    # Save sample data
    sample_data.to_csv(input_dir / "test_data.csv", index=False)
    
    return input_dir

def test_load_subset_columns(mock_config_dir, mock_input_dir):
    """Test that load() with subset of columns returns df with exactly those columns"""
    catalog = DataCatalog(str(mock_config_dir), str(mock_input_dir))
    cols = ["region", "CLI_per_active_AM"]
    
    # Load data and get from registry
    registry_key = catalog.load("test_dataset", cols)
    df = DataRegistry.get(registry_key)
    
    assert list(df.columns) == cols
    assert len(df.columns) == 2

def test_cache_hit(mock_config_dir, mock_input_dir):
    """Test that same call twice hits cache"""
    catalog = DataCatalog(str(mock_config_dir), str(mock_input_dir))
    cols = ["region", "CLI_per_active_AM"]
    
    # First load
    initial_cache_len = len(catalog._cache)
    catalog.load("test_dataset", cols)
    after_first_load = len(catalog._cache)
    
    # Second load
    catalog.load("test_dataset", cols)
    after_second_load = len(catalog._cache)
    
    assert after_first_load == initial_cache_len + 1  # Cache size increased after first load
    assert after_second_load == after_first_load      # Cache size unchanged after second load

def test_registry_data_integrity(mock_config_dir, mock_input_dir):
    """Test that registry.get(key) returns identical DataFrame"""
    catalog = DataCatalog(str(mock_config_dir), str(mock_input_dir))
    cols = ["region", "CLI_per_active_AM"]
    
    # Load data
    registry_key = catalog.load("test_dataset", cols)
    
    # Get same data twice from registry
    df1 = DataRegistry.get(registry_key)
    df2 = DataRegistry.get(registry_key)
    
    # Verify they are identical
    pd.testing.assert_frame_equal(df1, df2)

# Integration tests with real data
@pytest.fixture
def real_config_dir():
    """Fixture providing the real config directory"""
    return Path(__file__).parent.parent / "config"

@pytest.fixture
def real_input_dir():
    """Fixture providing the real input directory"""
    return Path(__file__).parent.parent / "input"

def test_load_cli_ci_ki_hypothesis(real_config_dir, real_input_dir):
    """Test loading the CLI_CI_KI_hypothesis dataset with specific columns"""
    catalog = DataCatalog(str(real_config_dir), str(real_input_dir))
    
    # Test loading with columns from a hypothesis
    cols = ["CLI_per_active_AM", "region"]
    registry_key = catalog.load("cli_ci_ki_hypothesis", cols)
    df = DataRegistry.get(registry_key)
    
    # Verify columns and data
    assert list(df.columns) == cols
    assert len(df) > 0
    assert "AM-APAC" in df["region"].values
    assert df["CLI_per_active_AM"].dtype in [float, int]

def test_load_pipeline_metrics(real_config_dir, real_input_dir):
    """Test loading the pipeline metrics dataset"""
    catalog = DataCatalog(str(real_config_dir), str(real_input_dir))
    
    # Test loading with columns from metrics.yaml
    cols = ["region", "cli_closed_pct"]
    registry_key = catalog.load("pipeline_metrics", cols)
    df = DataRegistry.get(registry_key)
    
    # Verify columns and data
    assert list(df.columns) == cols
    assert len(df) > 0
    assert "AM-NA" in df["region"].values
    assert df["cli_closed_pct"].dtype == float
    assert df["cli_closed_pct"].between(0, 1).all()  # Percentages should be between 0 and 1

def test_load_multiple_hypotheses(real_config_dir, real_input_dir):
    """Test loading multiple hypotheses from the same dataset"""
    catalog = DataCatalog(str(real_config_dir), str(real_input_dir))
    
    # Load different column combinations
    cols1 = ["CLI_per_active_AM", "region"]
    cols2 = ["SLI_per_active_AM", "region"]
    
    key1 = catalog.load("cli_ci_ki_hypothesis", cols1)
    key2 = catalog.load("cli_ci_ki_hypothesis", cols2)
    
    df1 = DataRegistry.get(key1)
    df2 = DataRegistry.get(key2)
    
    # Verify different column sets
    assert list(df1.columns) == cols1
    assert list(df2.columns) == cols2
    assert len(df1) == len(df2)  # Same number of rows
    assert not df1.equals(df2)   # Different data

def test_registry_persistence(real_config_dir, real_input_dir):
    """Test that registry maintains data integrity across multiple loads"""
    catalog = DataCatalog(str(real_config_dir), str(real_input_dir))
    
    # Load same data multiple times
    cols = ["CLI_per_active_AM", "region"]
    key1 = catalog.load("cli_ci_ki_hypothesis", cols)
    key2 = catalog.load("cli_ci_ki_hypothesis", cols)
    
    # Should get same registry key
    assert key1 == key2
    
    # Data should be identical
    df1 = DataRegistry.get(key1)
    df2 = DataRegistry.get(key2)
    pd.testing.assert_frame_equal(df1, df2)

@pytest.fixture(autouse=True)
def cleanup():
    """Clean up the registry after each test"""
    yield
    DataRegistry.clear() 
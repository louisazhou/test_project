from typing import Dict, Tuple, Optional, List, Any
import pandas as pd
import yaml
import os
import logging
from utils import convert_to_numeric, map_territory_to_region

class DataProcessor:
    def __init__(self, config_dir: str, input_dir: str, region_column: str = 'L4') -> None:
        """
        Initialize DataProcessor with configuration and data loading.
        
        This class handles the loading and processing of metrics and hypothesis data
        from configuration files and input data sources. It performs initial data
        validation and preprocessing.
        
        Args:
            config_dir: Directory containing hypothesis YAML files and metrics configuration
            input_dir: Directory containing input data files
            region_column: Column name to use for region mapping (default: 'L4')
            
        Raises:
            FileNotFoundError: If required configuration files are not found
            ValueError: If required columns are missing in input data
        """
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Initialize paths and settings
        self.config_dir = config_dir
        self.input_dir = input_dir
        self.region_column = region_column
        
        # Initialize configuration dictionaries
        self.metrics_config: Dict = {}
        self.hypothesis_configs: Dict = {}
        self.settings: Dict = {}
        
        # Initialize data frames
        self.metric_data: Optional[pd.DataFrame] = None
        self.single_dim_df: Optional[pd.DataFrame] = None
        self.multi_dim_df: Optional[pd.DataFrame] = None
        
        # Load all configurations and data
        self._load_metrics_config()
        self._load_hypothesis_configs()
        self._load_metric_data() 
    
    def _load_metrics_config(self) -> None:
        """
        Load metrics configuration and general settings from metrics.yaml.
        Stores the configuration in self.metrics_config.
        
        The metrics configuration defines:
        - Input data file location
        - Region column mapping
        - Metric definitions and thresholds
        - Associated hypotheses
        """
        metrics_file = os.path.join(self.config_dir, 'metrics.yaml')
        with open(metrics_file, 'r') as f:
            config = yaml.safe_load(f)
        
        # Store input data configuration
        self.metrics_config['input_data'] = config['input_data']
        
        # Store metrics configuration
        self.metrics_config['metrics'] = {}
        for metric_name, metric_config in config['metrics'].items():
            self.metrics_config['metrics'][metric_name] = {
                'description': metric_config['description'],
                'natural_name': metric_config['natural_name'],
                'higher_is_better': metric_config['higher_is_better'],
                'dependencies': metric_config.get('dependencies', []),
                'hypothesis': metric_config['hypothesis']
            }
        
        # --- Load general settings --- 
        self.settings = config.get('settings', {})
        # Provide defaults if settings are missing
        self.settings.setdefault('visualization_type', 'detailed')
        self.settings.setdefault('generate_ppt', True)
        self.settings.setdefault('upload_to_drive', False)
        self.settings.setdefault('drive_folder_id', None)
        # --- End Load Settings --- 

        self.logger.info("Loaded metrics configuration:")
        self.logger.info(f"Input data file: {self.metrics_config['input_data']['file']}")
        self.logger.info(f"Region column: {self.metrics_config['input_data']['region_column']}")
        self.logger.info(f"Metrics: {list(self.metrics_config['metrics'].keys())}")
        self.logger.info("Loaded general settings:")
        self.logger.info(f"  Visualization Type: {self.settings['visualization_type']}")
        self.logger.info(f"  Generate PPT: {self.settings['generate_ppt']}")
        self.logger.info(f"  Upload to Drive: {self.settings['upload_to_drive']}")
        self.logger.info(f"  Drive Folder ID: {self.settings['drive_folder_id']}")
    
    def _load_hypothesis_configs(self) -> None:
        """
        Load all hypothesis configurations from the consolidated hypotheses.yaml file.
        Stores configurations in self.hypothesis_configs.
        
        hypotheses.yaml structure:
        hypotheses:
          - name: "hypothesis_name_1"
            description: ...
            hypothesis_type: ...
            input_data:
              ...
            ...
          - name: "hypothesis_name_2"
            description: ...
            ...
        """
        # Load from single consolidated file
        hypotheses_file = os.path.join(self.config_dir, 'hypotheses.yaml')
        if not os.path.exists(hypotheses_file):
            self.logger.error(f"Hypotheses configuration file not found: {hypotheses_file}")
            self.hypothesis_configs = {}
            return
        
        with open(hypotheses_file, 'r') as f:
            config_data = yaml.safe_load(f)
            
        if not config_data or 'hypotheses' not in config_data: 
             self.logger.warning(f"Hypotheses configuration file {hypotheses_file} is empty or malformed.")
             self.hypothesis_configs = {}
             return

        # Process each hypothesis in the list
        self.hypothesis_configs = {}
        for hypothesis in config_data['hypotheses']:
            name = hypothesis['name']
            self.hypothesis_configs[name] = {
                'description': hypothesis['description'],
                'type': hypothesis['hypothesis_type'],
                'input_data': hypothesis['input_data'],
                'evaluation': hypothesis['evaluation']
            }
            
            # Add dimensional analysis if present
            if 'dimensional_analysis' in hypothesis:
                self.hypothesis_configs[name]['dimensional_analysis'] = hypothesis['dimensional_analysis']
            
            # Add insight template if present
            if 'insight_template' in hypothesis:
                self.hypothesis_configs[name]['insight_template'] = hypothesis['insight_template']
        
        self.logger.info("\nLoaded hypothesis configurations from hypotheses.yaml:")
        for name, config in self.hypothesis_configs.items():
            self.logger.info(f"\n{name}:")
            self.logger.info(f"Type: {config['type']}")
            self.logger.info(f"Input file: {config['input_data']['file']}")
            self.logger.info(f"Value column: {config['input_data']['value_column']}")
            
            if 'dimensional_analysis' in config:
                self.logger.info(f"Has dimensional analysis configuration")
    
    def _load_metric_data(self) -> Dict:
        """
        Load and preprocess the metrics data from the configured input file.
        
        This method:
        1. Reads the metrics CSV file
        2. Validates required columns
        3. Cleans up region mapping
        4. Converts metric values to numeric format
        
        Raises:
            FileNotFoundError: If the metrics input file is not found
            ValueError: If required columns are missing
        """
        # Use the input file path from metrics configuration
        metrics_file = os.path.join(self.input_dir, self.metrics_config['input_data']['file'])
        # Read CSV keeping NA as is and not converting data types
        df = pd.read_csv(metrics_file, na_values=None, keep_default_na=False)
        
        # Get region column from config
        region_column = self.metrics_config['input_data']['region_column']
        
        # Only keep required columns
        required_columns = [region_column] + list(self.metrics_config['metrics'].keys())
        if not all(col in df.columns for col in required_columns):
            missing_cols = [col for col in required_columns if col not in df.columns]
            raise ValueError(f"Missing required columns in metrics data: {missing_cols}")
            
        df = df[required_columns]
        
        # Clean up region column and rename to L4
        df[self.region_column] = df[region_column].apply(map_territory_to_region)
        df = df.drop(region_column, axis=1)  # Drop the original region column
        
        # Convert metric columns to numeric
        for metric in self.metrics_config['metrics'].keys():
            df[metric] = df[metric].apply(convert_to_numeric)
        
        self.metric_data = df
        self.logger.info(f"\nLoaded metrics data shape: {self.metric_data.shape}")
        self.logger.info(f"Metrics columns: {self.metric_data.columns.tolist()}")
    
    def get_metrics_config(self) -> Dict:
        """
        Get the loaded metrics configuration.
        
        Returns:
            Dictionary containing metrics configuration including input data specs,
            metric definitions, thresholds, and associated hypotheses.
        """
        return self.metrics_config
    
    def get_hypothesis_configs(self) -> Dict:
        """
        Get the loaded hypothesis configurations.
        
        Returns:
            Dictionary containing hypothesis configurations including types,
            input specifications, analysis parameters, and scoring criteria.
        """
        return self.hypothesis_configs
    
    def get_hypothesis_config(self, hypothesis_name: str) -> Optional[Dict]:
        """
        Get configuration for a specific hypothesis.
        
        Args:
            hypothesis_name: Name of the hypothesis to retrieve
            
        Returns:
            Dictionary containing hypothesis configuration if found, None otherwise
        """
        return self.hypothesis_configs.get(hypothesis_name)
    
    def get_settings(self) -> Dict:
        """Get the loaded general settings."""
        return self.settings
    
    def prepare_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Prepare data by processing all hypothesis files and joining with metrics.
        
        This method:
        1. Processes single-dimensional hypotheses
        2. Processes multi-dimensional hypotheses
        3. Merges hypothesis data with metrics data
        4. Validates and organizes columns
        
        Returns:
            Tuple of (single_dimension_df, multi_dimension_df)
            
        Raises:
            ValueError: If metric_data hasn't been loaded
        """
        if self.metric_data is None:
            raise ValueError("Metric data must be loaded before preparing analysis data")
            
        # Initialize empty DataFrames for each type
        single_dim_dfs: List[pd.DataFrame] = []
        multi_dim_dfs: List[pd.DataFrame] = []
        
        # Process each hypothesis
        for name, config in self.hypothesis_configs.items():
            try:
                config['name'] = name  # Add name to config for reference
                if config['type'] == 'single_dim':
                    df = self._process_single_dim_hypothesis(config)
                    single_dim_dfs.append(df)
                elif config['type'] == 'multi_dim_groupby':
                    df = self._process_multi_dim_hypothesis(config)
                    multi_dim_dfs.append(df)
            except Exception as e:
                self.logger.error(f"Error processing {name}: {str(e)}")
                continue
        
        # Combine single dimension hypotheses
        if single_dim_dfs:
            # Start with metrics data
            self.single_dim_df = self.metric_data.copy()
            
            # Merge each hypothesis data
            for df in single_dim_dfs:
                self.single_dim_df = pd.merge(
                    self.single_dim_df,
                    df,
                    on=self.region_column,
                    how='left'
                )
            
            # Reorder columns for single_dim_df
            metric_columns = list(self.metrics_config['metrics'].keys())
            hypothesis_columns = []
            for df in single_dim_dfs:
                hypothesis_columns.extend([col for col in df.columns if col != self.region_column])
            
            column_order = [self.region_column] + metric_columns + hypothesis_columns
            self.single_dim_df = self.single_dim_df[column_order]
        else:
            pass # keep the default (None) so nothing weird is triggered downstream
        
        # Combine multi dimension hypotheses
        if multi_dim_dfs:
            # Start with metrics data
            self.multi_dim_df = self.metric_data.copy()
            
            # Merge each hypothesis data
            for df in multi_dim_dfs:
                self.multi_dim_df = pd.merge(
                    self.multi_dim_df,
                    df,
                    on=self.region_column,
                    how='left'
                )
            
            # Reorder columns based on config
            metric_columns = list(self.metrics_config['metrics'].keys())
            multi_dim_config = next((config for config in self.hypothesis_configs.values() 
                                   if config['type'] == 'multi_dim_groupby'), None)
            
            if multi_dim_config:
                dimension_column = multi_dim_config['dimensional_analysis']['dimension_column']
                value_column = multi_dim_config['input_data']['value_column']
                
                # Get weight column if it exists
                weight_column = None
                if 'metrics' in multi_dim_config['dimensional_analysis']:
                    for metric in multi_dim_config['dimensional_analysis']['metrics']:
                        if metric.get('is_weight', False):
                            weight_column = metric['column']
                            break
                
                # Build column order
                column_order = [self.region_column, dimension_column]
                if weight_column:
                    column_order.append(weight_column)
                column_order.extend([value_column] + metric_columns)
                
                self.multi_dim_df = self.multi_dim_df[column_order]
        else:
            pass # keep the default (None) so nothing weird is triggered downstream
        
        return self.single_dim_df, self.multi_dim_df

    def _process_single_dim_hypothesis(self, config: Dict[str, Any]) -> pd.DataFrame:
        """
        Process a single dimension hypothesis file.
        
        This method:
        1. Loads the hypothesis data file
        2. Validates and selects required columns
        3. Cleans up region mapping
        4. Converts values to numeric format
        
        Args:
            config: Hypothesis configuration dictionary containing:
                   - input_data: file and column specifications
                   - name: hypothesis name
                   
        Returns:
            DataFrame with processed hypothesis data
            
        Raises:
            FileNotFoundError: If the input file doesn't exist
            ValueError: If required columns are missing
        """
        input_file = os.path.join(self.input_dir, config['input_data']['file'])
        
        # Read CSV keeping NA as is and not converting data types
        df = pd.read_csv(input_file, na_values=None, keep_default_na=False)
        self.logger.debug(f"Raw data from {input_file}:")
        self.logger.debug(df)
        
        # Get required columns
        value_column = config['input_data']['value_column']
        region_column = config['input_data']['region_column']
        
        # Validate required columns exist
        required_columns = [region_column, value_column]
        if not all(col in df.columns for col in required_columns):
            missing_cols = [col for col in required_columns if col not in df.columns]
            raise ValueError(f"Missing required columns in {config['name']}: {missing_cols}")
        
        # Select only necessary columns
        df = df[[region_column, value_column]]
        self.logger.debug(f"After selecting columns {region_column}, {value_column}:")
        self.logger.debug(df)
        
        # Clean up region column and rename to for consistency
        df[self.region_column] = df[region_column].apply(map_territory_to_region)
        df = df.drop(region_column, axis=1)
        
        # Convert value column to numeric
        df[value_column] = df[value_column].apply(convert_to_numeric)
        
        self.logger.info(f"Processed single dim hypothesis: {config['name']}")
        self.logger.debug(f"Columns: {df.columns.tolist()}")
        self.logger.debug("Sample data:")
        self.logger.debug(df)
        
        return df
    
    def _process_multi_dim_hypothesis(self, config: Dict[str, Any]) -> pd.DataFrame:
        """
        Process a multi-dimensional hypothesis.
        
        This method:
        1. Loads the hypothesis data file
        2. Validates and selects required columns
        3. Processes dimensional analysis configuration
        4. Handles weight and value columns
        
        Args:
            config: Hypothesis configuration dictionary containing:
                   - dimensional_analysis: dimension specifications
                   - input_data: file and column specifications
                   - name: hypothesis name
                   
        Returns:
            DataFrame containing processed hypothesis data
            
        Raises:
            FileNotFoundError: If the input file doesn't exist
            ValueError: If required columns are missing
        """
        # Load data
        filepath = os.path.join(self.input_dir, config['input_data']['file'])
        df = pd.read_csv(filepath, na_values=None, keep_default_na=False)
        
        # Get column names from config
        dimension_column = config['dimensional_analysis']['dimension_column']
        value_column = config['input_data']['value_column']
        region_column = config['input_data']['region_column']
        
        # Get weight column from dimensional analysis if it exists
        weight_column = None
        if 'metrics' in config['dimensional_analysis']:
            for metric in config['dimensional_analysis']['metrics']:
                if metric.get('is_weight', False):
                    weight_column = metric['column']
                    break
        
        # Select required columns
        required_columns = [region_column, dimension_column, value_column]
        if weight_column:
            required_columns.append(weight_column)
            
        # Validate required columns exist
        if not all(col in df.columns for col in required_columns):
            missing_cols = [col for col in required_columns if col not in df.columns]
            raise ValueError(f"Missing required columns in {config['name']}: {missing_cols}")
        
        df = df[required_columns]
        
        # Clean up region column and rename for consistency
        df[self.region_column] = df[region_column].apply(map_territory_to_region)
        df = df.drop(region_column, axis=1)  # Drop the original region column
        
        # Convert value columns to numeric
        for col in [value_column, weight_column]:
            if col and col in df.columns:
                df[col] = df[col].apply(convert_to_numeric)
        
        return df 
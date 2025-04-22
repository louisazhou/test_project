import os
import json
from data_processor import DataProcessor

def main():
    # Initialize paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_dir = os.path.join(base_dir, 'config')
    input_dir = os.path.join(base_dir, 'input')
    tmp_dir = os.path.join(base_dir, 'tmp')
    
    # Create tmp directory if it doesn't exist
    if not os.path.exists(tmp_dir):
        os.makedirs(tmp_dir)
    
    # Initialize DataProcessor
    processor = DataProcessor(config_dir, input_dir)
    
    # Get configurations
    metrics_config = processor.get_metrics_config()
    hypothesis_configs = processor.get_hypothesis_configs()
    
    # Load and process data
    single_dim_df, multi_dim_df = processor.prepare_data()
    
    # Save processed data
    if single_dim_df is not None:
        single_dim_df.to_csv(os.path.join(tmp_dir, 'single_dim_processed.csv'), index=False)
    else:
        print("No single dimension data found")
    
    if multi_dim_df is not None:
        multi_dim_df.to_csv(os.path.join(tmp_dir, 'multi_dim_processed.csv'), index=False)
    else:
        print("No multi dimension data found")
    
    # Save metrics configuration for reference
    with open(os.path.join(tmp_dir, 'metrics_config.json'), 'w') as f:
        json.dump(metrics_config, f, indent=2)
    with open(os.path.join(tmp_dir, 'hypothesis_config.json'), 'w') as f:
        json.dump(hypothesis_configs, f, indent=2)

if __name__ == '__main__':
    main() 
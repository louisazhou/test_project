import os
import logging
import yaml
from typing import Dict, Any
from data_processor import DataProcessor
from automation_pipeline import RootCauseAnalysisPipeline
from presentation import generate_ppt, upload_to_drive

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
logger = logging.getLogger(__name__)

def load_pipeline_config(config_path: str) -> Dict[str, Any]:
    """Loads output options from the metrics configuration file."""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        # Default values if section or keys are missing
        output_options = config.get('output_options', {})
        return {
            'save_ppt': output_options.get('save_ppt', False),
            'upload_to_drive': output_options.get('upload_to_drive', False),
            'drive_folder_id': output_options.get('drive_folder_id', None) # Optional folder ID
        }
    except FileNotFoundError:
        logger.warning(f"Configuration file not found at {config_path}. Using default output options (no PPT, no upload).")
        return {'save_ppt': False, 'upload_to_drive': False, 'drive_folder_id': None}
    except Exception as e:
        logger.error(f"Error loading configuration from {config_path}: {e}")
        return {'save_ppt': False, 'upload_to_drive': False, 'drive_folder_id': None}

def main():
    """Main script for running the RCA pipeline.

    This script:
    1. Sets up necessary paths and configurations
    2. Initializes the data processor
    3. Prepares input data
    4. Runs the RCA pipeline
    5. Saves results to output files
    6. Generates visualizations
    7. Optionally generates and uploads a PowerPoint presentation based on config
    """
    logger.info("Starting RCA Pipeline...")
    # Initialize paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    config_dir = os.path.join(base_dir, 'config')
    input_dir = os.path.join(base_dir, 'input')
    tmp_dir = os.path.join(base_dir, 'tmp')
    output_dir = os.path.join(base_dir, 'output')

    # Create tmp and output directories if they don't exist
    os.makedirs(tmp_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    # Load pipeline output configuration
    pipeline_config_path = os.path.join(config_dir, 'metrics.yaml') # Assuming config is in metrics.yaml
    output_config = load_pipeline_config(pipeline_config_path)
    save_ppt_flag = output_config['save_ppt']
    upload_drive_flag = output_config['upload_to_drive']
    drive_folder_id = output_config['drive_folder_id']

    logger.info(f"Output Configuration: Save PPT={save_ppt_flag}, Upload to Drive={upload_drive_flag}")

    try:
        # Initialize DataProcessor
        logger.info("Initializing Data Processor...")
        processor = DataProcessor(config_dir, input_dir, region_column="L4")

        # Load and process data
        logger.info("Preparing data...")
        single_dim_df, multi_dim_df = processor.prepare_data()

        # Check if data preparation was successful
        if single_dim_df is None or multi_dim_df is None:
             logger.error("Data preparation failed. Exiting pipeline.")
             return

        # Initialize and run the pipeline
        logger.info("Initializing RCA Pipeline...")
        pipeline = RootCauseAnalysisPipeline(
            processor.get_metrics_config(),
            processor.get_hypothesis_configs(),
            single_dim_df,
            multi_dim_df,
            region_column=processor.region_column
        )

        logger.info("Running metric analysis...")
        pipeline.run_all_metrics()

        # Save output tables (consider making filenames configurable too)
        debug_output_path = os.path.join(output_dir, 'rca_debug_output.csv')
        if not pipeline.evaluator.get_score_calculation_table().empty:
             # Using score table as proxy for final results now
             final_processed_df = pipeline.evaluator.get_score_calculation_table()
             final_processed_df.to_csv(debug_output_path, index=False)
             logger.info(f"Debug score output saved to: {debug_output_path}")
        else:
             logger.warning("No final score results generated to save.")

        anomaly_output_path = os.path.join(output_dir, 'rca_anomaly_detection.csv')
        anomaly_df = pipeline.evaluator.get_anomaly_detection_table()
        if not anomaly_df.empty:
            anomaly_df.to_csv(anomaly_output_path, index=False)
            logger.info(f"Anomaly detection results saved to: {anomaly_output_path}")
        else:
            logger.warning("No anomaly detection results generated to save.")

        # Generate visualizations
        logger.info("Generating visualizations...")
        pipeline.generate_visualizations(output_dir)
        logger.info(f"Visualizations saved to: {output_dir}")

        # Generate and optionally upload PowerPoint
        ppt_path = None
        if save_ppt_flag:
            logger.info("Generating PowerPoint presentation...")
            if pipeline.metric_analysis_results:
                ppt_filename = "RCA_Summary.pptx"
                ppt_path = generate_ppt(pipeline.metric_analysis_results, output_dir, ppt_filename)
                if ppt_path:
                    logger.info(f"PowerPoint presentation generated: {ppt_path}")
                else:
                    logger.warning("PowerPoint generation failed.")
            else:
                logger.warning("Skipping PowerPoint generation as no analysis results were processed.")

            # Upload PowerPoint to Google Drive if flag is set and PPT was created
            if upload_drive_flag and ppt_path:
                logger.info(f"Uploading PowerPoint to Google Drive (Folder ID: {drive_folder_id or 'Root'})...")
                file_id = upload_to_drive(ppt_path, folder_id=drive_folder_id)
                if file_id:
                    logger.info(f"PowerPoint uploaded successfully. File ID: {file_id}")
                else:
                    logger.error("Failed to upload PowerPoint to Google Drive.")
            elif upload_drive_flag and not ppt_path:
                 logger.warning("Skipping Google Drive upload because PowerPoint generation failed or was skipped.")

    except Exception as e:
        logger.critical(f"An critical error occurred during pipeline execution: {e}", exc_info=True)

    logger.info("RCA Pipeline finished.")

if __name__ == "__main__":
    main() 
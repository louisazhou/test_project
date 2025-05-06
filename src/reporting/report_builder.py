import os
import logging
import pandas as pd
from typing import Dict, Any, List, Optional
import matplotlib.pyplot as plt

from ..core.data_registry import DataRegistry
from ..core.types import MetricReport, PlotSpec
from ..plotting.plot_router import plot_summary_report
from ..core.presentation import generate_ppt, upload_to_drive

logger = logging.getLogger(__name__)

class ReportBuilder:
    """Generates reports and visualizations based on analysis results."""

    def __init__(self, data_registry: DataRegistry, settings: Dict[str, Any], output_dir: str):
        """Initialize the report builder.
        
        Args:
            data_registry: DataRegistry instance containing processed data
            settings: Settings from the configuration
            output_dir: Directory where reports should be saved
        """
        self.data_registry = data_registry
        self.settings = settings
        self.output_dir = output_dir
        self.fig_dir = os.path.join(output_dir, 'figs')
        
        # Ensure output directories exist
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.fig_dir, exist_ok=True)
        
        # Extract settings
        output_format_settings = settings.get('output_formats', {})
        self.summary_report_format = output_format_settings.get('summary_report_format', None)
        if self.summary_report_format == 'null':
            self.summary_report_format = None
            
        self.generate_ppt_flag = output_format_settings.get('ppt', False)
        if isinstance(self.generate_ppt_flag, str):
            self.generate_ppt_flag = self.generate_ppt_flag.lower() == 'true'
            
        # Extract Google Drive settings
        gdrive_settings = settings.get('google_drive', {})
        self.upload_gdrive_flag = gdrive_settings.get('upload_enabled', False)
        if isinstance(self.upload_gdrive_flag, str):
            self.upload_gdrive_flag = self.upload_gdrive_flag.lower() == 'true'
            
        self.gdrive_folder_id = gdrive_settings.get('folder_id')
        if self.gdrive_folder_id == 'null':
            self.gdrive_folder_id = None
            
        logger.info(f"ReportBuilder initialized with format={self.summary_report_format}, ppt={self.generate_ppt_flag}")

    def build_analysis_summary(self, 
                              all_metric_reports: List[MetricReport], 
                              metrics_map: Dict[str, Any],
                              hypotheses_map: Dict[str, Any],
                              z_thresh: float = 1.0) -> Dict[str, Dict[str, Any]]:
        """Build a consolidated analysis summary from metric reports.
        
        Args:
            all_metric_reports: List of processed MetricReport objects
            metrics_map: Dictionary mapping metric names to their configurations
            hypotheses_map: Dictionary mapping hypothesis names to their configurations
            z_thresh: Z-score threshold used for anomaly detection
            
        Returns:
            Dictionary containing consolidated analysis summary for each metric
        """
        logger.info("Building analysis summary from metric reports...")
        
        analysis_summary = {}
        
        for report in all_metric_reports:
            metric = report.metric_name
            metric_natural_name = metrics_map.get(metric, {}).get('natural_name', metric)
            
            # Get enrichment data from the report
            metric_enrichment = report.metric_enrichment_data if report.metric_enrichment_data else {}
            
            # Get the metric dataframe
            metric_df = self.data_registry.get(report.metric_data_key) if report.metric_data_key else None
            
            # Initialize with defaults
            primary_region = "NoAnomaly"
            best_hypothesis_result = None
            explanation_text = "No significant anomalies detected for this metric."
            metric_delta_primary = 0.0
            metric_dir_primary = "neutral"
            primary_anomaly_obj = None
            
            # Process anomalies
            true_anomalies = [a for a in report.anomalies if a.is_anomaly]
            
            # Consolidate unique hypotheses across all anomalies for this metric
            all_hypo_results_dict = {}
            for anom in report.anomalies:
                for hypo_res in anom.hypo_results:
                    if hypo_res.name not in all_hypo_results_dict:
                        all_hypo_results_dict[hypo_res.name] = hypo_res
            all_hypotheses_for_metric = list(all_hypo_results_dict.values())
            
            if true_anomalies:
                # Find explaining hypotheses with score > 0.5
                explaining_hypotheses = sorted(
                    [h for h in all_hypotheses_for_metric if h.score is not None and h.score > 0.5], 
                    key=lambda x: x.score, 
                    reverse=True
                )
                
                if explaining_hypotheses:
                    # Found explaining hypotheses
                    best_hypothesis_result = explaining_hypotheses[0]
                    
                    # Find the anomaly object corresponding to the best hypothesis
                    for anom in true_anomalies:
                        if best_hypothesis_result in anom.hypo_results:
                            primary_anomaly_obj = anom
                            break
                    
                    if primary_anomaly_obj:
                        primary_region = primary_anomaly_obj.region
                        explanation_text = best_hypothesis_result.narrative
                        metric_delta_primary = primary_anomaly_obj.delta_pct
                        metric_dir_primary = primary_anomaly_obj.dir
                    else:
                        # Fallback if hypothesis doesn't have a corresponding anomaly
                        primary_region = best_hypothesis_result.key_numbers.get('region', 'Unknown')
                        explanation_text = f"Anomaly likely in {primary_region} explained by {best_hypothesis_result.name}, but anomaly details missing."
                else:
                    # Anomalies exist but none well explained
                    primary_anomaly_obj = max(true_anomalies, key=lambda a: abs(a.z_score))
                    primary_region = primary_anomaly_obj.region
                    explanation_text = f"Anomaly detected in {primary_region} (Z={primary_anomaly_obj.z_score:.2f}), but no hypothesis strongly explains it."
                    metric_delta_primary = primary_anomaly_obj.delta_pct
                    metric_dir_primary = primary_anomaly_obj.dir
            
            # Get hypothesis details if best hypothesis found
            hypo_natural_name = None
            hypo_delta_primary = None
            hypo_dir_primary = None
            hypo_delta_fmt = None
            
            if best_hypothesis_result:
                hypo_natural_name = best_hypothesis_result.natural_name
                if hasattr(best_hypothesis_result, 'key_numbers'):
                    key_numbers = best_hypothesis_result.key_numbers
                    hypo_delta_primary = key_numbers.get('hypo_delta')
                    hypo_dir_primary = key_numbers.get('hypo_dir')
                    hypo_delta_fmt = key_numbers.get('delta_fmt', hypo_delta_primary)
            
            # Store consolidated summary for this metric
            analysis_summary[metric] = {
                'metric_name': metric,
                'metric_natural_name': metric_natural_name,
                'metric_ref_val': report.global_value,
                'metric_ref_val_fmt': report.formatted_global_value,
                'metric_std': report.metric_std if report.metric_std is not None else 0.0,
                'metric_value_col': metric,
                'higher_is_better': metrics_map.get(metric, {}).get('higher_is_better', True),
                'z_score_threshold': z_thresh,
                'primary_region': primary_region,
                'primary_anomaly_obj': primary_anomaly_obj,
                'metric_delta_primary': metric_delta_primary,
                'metric_dir_primary': metric_dir_primary,
                'metric_value_primary': primary_anomaly_obj.value if primary_anomaly_obj else None,
                'metric_value_primary_fmt': primary_anomaly_obj.formatted_value if primary_anomaly_obj else None,
                'metric_deviation_description': primary_anomaly_obj.deviation_description if primary_anomaly_obj else None,
                'best_hypothesis_result': best_hypothesis_result,
                'best_hypothesis_name': best_hypothesis_result.name if best_hypothesis_result else None,
                'hypo_natural_name': hypo_natural_name,
                'hypo_delta_primary': hypo_delta_primary,
                'hypo_dir_primary': hypo_dir_primary,
                'hypo_delta_fmt': hypo_delta_fmt,
                'explanation_text': explanation_text,
                'all_hypotheses_for_metric': all_hypotheses_for_metric,
                'metric_data_key': report.metric_data_key,
                'metric_enrichment_data': metric_enrichment,
                'hypotheses_configs': hypotheses_map,
                'is_percentage_metric': report.is_percentage
            }
        
        return analysis_summary

    def generate_summary_visualizations(self, analysis_summary: Dict[str, Dict[str, Any]]) -> List[str]:
        """Generate summary visualizations for metrics in the analysis summary.
        
        Args:
            analysis_summary: Dictionary containing consolidated analysis summary
            
        Returns:
            List of paths to generated summary plots
        """
        if not self.summary_report_format or self.summary_report_format.lower() == 'null':
            logger.info("No summary report format specified, skipping summary visualization generation.")
            return []
        
        if self.summary_report_format not in ["detailed", "succinct"]:
            logger.warning(f"Unknown summary_report_format: '{self.summary_report_format}'. Skipping summary visualization.")
            return []
        
        logger.info(f"Generating '{self.summary_report_format}' summary visualizations...")
        summary_plot_paths = []
        
        for metric, summary_item in analysis_summary.items():
            summary_plot_path = plot_summary_report(
                analysis_summary_item=summary_item,
                data_registry=self.data_registry,
                output_dir=self.fig_dir,
                report_format=self.summary_report_format
            )
            
            if summary_plot_path:
                summary_plot_paths.append(summary_plot_path)
        
        return summary_plot_paths

    def generate_powerpoint(self, analysis_summary: Dict[str, Dict[str, Any]]) -> Optional[str]:
        """Generate a PowerPoint presentation from the analysis summary.
        
        Args:
            analysis_summary: Dictionary containing consolidated analysis summary
            
        Returns:
            Path to the generated PowerPoint file, or None if generation failed
        """
        if not self.generate_ppt_flag:
            logger.info("PowerPoint generation disabled in settings.")
            return None
        
        if not self.summary_report_format or self.summary_report_format.lower() == 'null':
            logger.info("No summary report format specified, skipping PowerPoint generation.")
            return None
        
        if not analysis_summary:
            logger.warning("No analysis results to include in PowerPoint presentation.")
            return None
        
        logger.info("Generating PowerPoint presentation...")
        ppt_filename = "RCA_Summary.pptx"
        ppt_path = generate_ppt(
            analysis_summary,
            self.fig_dir,
            ppt_filename,
            report_format=self.summary_report_format
        )
        
        if ppt_path:
            logger.info(f"PowerPoint presentation generated: {ppt_path}")
            
            # Upload to Google Drive if enabled
            if self.upload_gdrive_flag:
                logger.info("Uploading PowerPoint to Google Drive...")
                file_id = upload_to_drive(ppt_path, folder_id=self.gdrive_folder_id)
                
                if file_id:
                    logger.info(f"PowerPoint uploaded to Google Drive. File ID: {file_id}")
                else:
                    logger.error("Failed to upload PowerPoint to Google Drive.")
        else:
            logger.error("Failed to generate PowerPoint presentation.")
        
        return ppt_path

    def build_report(self, 
                    all_metric_reports: List[MetricReport], 
                    metrics_map: Dict[str, Any],
                    hypotheses_map: Dict[str, Any],
                    z_thresh: float = 1.0) -> Dict[str, Any]:
        """Build reports and visualizations from metric reports.
        
        Args:
            all_metric_reports: List of processed MetricReport objects
            metrics_map: Dictionary mapping metric names to their configurations
            hypotheses_map: Dictionary mapping hypothesis names to their configurations
            z_thresh: Z-score threshold used for anomaly detection
            
        Returns:
            Dictionary containing report metadata
        """
        # Build the analysis summary
        analysis_summary = self.build_analysis_summary(
            all_metric_reports, 
            metrics_map, 
            hypotheses_map,
            z_thresh
        )
        
        # Generate summary visualizations
        summary_plot_paths = self.generate_summary_visualizations(analysis_summary)
        
        # Generate PowerPoint if enabled
        ppt_path = self.generate_powerpoint(analysis_summary)
        
        # Return report metadata
        return {
            'analysis_summary': analysis_summary,
            'summary_plot_paths': summary_plot_paths,
            'ppt_path': ppt_path
        } 
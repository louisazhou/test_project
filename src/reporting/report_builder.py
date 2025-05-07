import os
import logging
import pandas as pd
from typing import Dict, Any, List, Optional
import matplotlib.pyplot as plt

from ..core.data_registry import DataRegistry
from ..core.types import MetricReport, PlotSpec
from .report_plots.detailed import plot_summary_report
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
        
        # Extract settings directly from the reporting section
        self.summary_report_format = settings.get('summary_report_format', None)
        if self.summary_report_format == 'null':
            self.summary_report_format = None
            
        self.generate_ppt_flag = settings.get('generate_ppt', False)
        if isinstance(self.generate_ppt_flag, str):
            self.generate_ppt_flag = self.generate_ppt_flag.lower() == 'true'
            
        # Extract Google Drive settings directly
        self.upload_gdrive_flag = settings.get('upload_gdrive', False)
        if isinstance(self.upload_gdrive_flag, str):
            self.upload_gdrive_flag = self.upload_gdrive_flag.lower() == 'true'
            
        self.gdrive_folder_id = settings.get('gdrive_folder_id', '')
        if not self.gdrive_folder_id or self.gdrive_folder_id == 'null':
            self.gdrive_folder_id = None
            
        logger.info(f"ReportBuilder initialized with format={self.summary_report_format}, ppt={self.generate_ppt_flag}")

    def _extract_metric_report_context(self, metric_report, all_hypotheses) -> Dict[str, Any]:
        """Extract report context from a metric report.
        
        This centralizes the complex logic of pulling report data from metric and
        hypothesis objects in a format suitable for rendering plots and slides.
        
        Args:
            metric_report: The MetricReport object to extract context from
            all_hypotheses: List of all HypoResult objects for this metric
            
        Returns:
            Dictionary containing the extracted context
        """
        metric_name = metric_report.metric_name
        metric_natural_name = metric_report.natural_name or metric_name
        
        # Get enriched metric data
        metric_data_key = metric_report.metric_data_key
        
        # Extract anomalies
        anomalies = metric_report.anomalies or []
        true_anomalies = [a for a in anomalies if a.is_anomaly]
        
        # Extract metric values
        metric_value_col = metric_name
        global_value = metric_report.global_value
        metric_std = metric_report.metric_std or 0.0
        
        # Find the primary region and best hypothesis
        primary_region = "NoAnomaly"
        primary_anomaly_obj = None
        best_hypothesis_result = None
        explanation_text = f"No anomalies detected for {metric_natural_name}."
        metric_delta_primary = None
        metric_dir_primary = None
        
        # Get all hypotheses for this metric
        all_hypotheses_for_metric = all_hypotheses
        
        # Extract hypothesis configs from all hypothesis results
        hypotheses_configs = {}
        for h in all_hypotheses_for_metric:
            if h.name:
                hypotheses_configs[h.name] = {
                    'natural_name': h.natural_name or h.name
                }
                
                # If we have input_data from configuration, add that
                if hasattr(h, 'input_data'):
                    hypotheses_configs[h.name]['input_data'] = h.input_data
        
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
                    primary_region = best_hypothesis_result.value.region if hasattr(best_hypothesis_result, 'value') and hasattr(best_hypothesis_result.value, 'region') else "Unknown"
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
        
        if best_hypothesis_result:
            hypo_natural_name = best_hypothesis_result.natural_name
            # No need to calculate delta and direction here anymore
            # These will be calculated directly in the visualization components
        
        # Extract score components for visualization
        score_components = {}
        for h in all_hypotheses_for_metric:
            if h.score is not None:
                # Collect score components
                score_components[h.name] = {
                    'direction_alignment': 0.0,  # Default values - these will be populated elsewhere
                    'consistency': 0.0,
                    'hypo_z_score_norm': 0.0,
                    'explained_ratio': 0.0
                }
        
        # Use enrichment data directly from anomaly gate
        metric_enrichment_data = {}
        for anomaly in anomalies:
            # Only store minimal data needed for visualization
            metric_enrichment_data[anomaly.region] = {
                'is_anomaly': anomaly.is_anomaly,
                'good_anomaly': anomaly.good_anomaly,
                'bad_anomaly': anomaly.bad_anomaly,
                'z_score': anomaly.z_score
            }
            # Add direction if available
            if hasattr(anomaly, 'dir'):
                metric_enrichment_data[anomaly.region]['dir'] = anomaly.dir
        
        # Return the consolidated summary item
        return {
            'metric_name': metric_name,
            'metric_natural_name': metric_natural_name,
            'metric_ref_val': global_value,
            'metric_ref_val_fmt': metric_report.get_formatted_global_value(),
            'metric_std': metric_std,
            'metric_value_col': metric_value_col,
            'higher_is_better': metric_report.is_percentage, # Typically % metrics are better when higher
            'z_score_threshold': 1.0,  # Default z-score threshold
            'primary_region': primary_region,
            'primary_anomaly_obj': primary_anomaly_obj,
            'metric_delta_primary': metric_delta_primary,
            'metric_dir_primary': metric_dir_primary,
            'best_hypothesis_result': best_hypothesis_result,
            'all_hypotheses_for_metric': all_hypotheses_for_metric,
            'hypo_natural_name': hypo_natural_name,
            'explanation_text': explanation_text,
            'metric_data_key': metric_data_key,
            'plot_specs': [],  # MetricReport no longer has metric_level_plots
            'score_components': score_components,
            'metric_enrichment_data': metric_enrichment_data,
            'hypotheses_configs': hypotheses_configs
        }

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
            # Consolidate unique hypotheses across all anomalies for this metric
            all_hypo_results_dict = {}
            for anom in report.anomalies:
                for hypo_res in anom.hypo_results:
                    if hypo_res.name not in all_hypo_results_dict:
                        all_hypo_results_dict[hypo_res.name] = hypo_res
            all_hypotheses_for_metric = list(all_hypo_results_dict.values())
            
            # Use the existing extract method to avoid code duplication
            analysis_summary[report.metric_name] = self._extract_metric_report_context(report, all_hypotheses_for_metric)
        
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
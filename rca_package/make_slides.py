#!/usr/bin/env python3
"""
Clean slide system with decorator approach and figure generation functions.
"""

import os
import tempfile
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from datetime import datetime
import pandas as pd
import numpy as np
from typing import Optional, Dict, Any, Callable, Union, Tuple, List
from jinja2 import Template
import logging
import re
import matplotlib.pyplot as plt

# Setup logging
logger = logging.getLogger(__name__)


class SlideContent:
    """Container for slide content with support for figure generation functions."""
    
    def __init__(self, title: str, text_template: str = None, 
                 df: pd.DataFrame = None, 
                 figure_generator: Callable = None,
                 template_params: Dict = None,
                 additional_dfs: Dict[str, pd.DataFrame] = None,
                 show_figures: bool = True):
        self.title = title
        self.text_template = text_template
        self.df = df
        self.figure_generator = figure_generator
        self.template_params = template_params or {}
        self.additional_dfs = additional_dfs or {}
        self.show_figures = show_figures
        self._generated_figure_path = None
        
    def _extract_markdown_tables_and_clean_text(self) -> Tuple[str, List[pd.DataFrame]]:
        """
        Extract markdown tables from template and return clean text + DataFrames.
        
        Returns:
            Tuple of (clean_text, list_of_dataframes)
        """
        if not self.text_template:
            return "", []
        
        # Render the template first
        template = Template(self.text_template)
        rendered_text = template.render(**self.template_params)
        
        # Split into parts and identify tables
        parts = rendered_text.split('\n\n')
        clean_parts = []
        extracted_tables = []
        
        i = 0
        while i < len(parts):
            part = parts[i].strip()
            
            # Check if this part contains a markdown table
            if '|' in part and part.count('|') >= 3:  # Likely a table
                # Try to parse as markdown table
                try:
                    lines = part.split('\n')
                    # Find header and separator
                    header_line = None
                    sep_line = None
                    data_lines = []
                    
                    for line in lines:
                        if '|' in line:
                            if header_line is None:
                                header_line = line
                            elif sep_line is None and ('---' in line or ':-' in line):
                                sep_line = line
                            else:
                                data_lines.append(line)
                    
                    if header_line and data_lines:
                        # Parse the table
                        headers = [h.strip() for h in header_line.split('|')[1:-1]]
                        
                        table_data = []
                        for line in data_lines:
                            row = [cell.strip() for cell in line.split('|')[1:-1]]
                            if len(row) == len(headers):
                                table_data.append(row)
                        
                        if table_data:
                            # Create DataFrame
                            df_table = pd.DataFrame(table_data, columns=headers)
                            # Try to convert numeric columns
                            for col in df_table.columns:
                                try:
                                    df_table[col] = pd.to_numeric(df_table[col])
                                except:
                                    pass  # Keep as string if not numeric
                            
                            extracted_tables.append(df_table)
                            # Don't add this part to clean_parts (remove the table)
                            i += 1
                            continue
                except:
                    pass  # If parsing fails, treat as regular text
            
            # Not a table or parsing failed, add to clean text
            clean_parts.append(part)
            i += 1
        
        clean_text = '\n\n'.join(clean_parts)
        
        # Clean up excessive empty lines and whitespace
        clean_text = clean_text.strip()
        
        # Remove excessive consecutive newlines (more than 2)
        clean_text = re.sub(r'\n{3,}', '\n\n', clean_text)
        
        return clean_text, extracted_tables
        
    def render_text(self) -> str:
        """Render the text using Jinja template, removing markdown tables."""
        clean_text, _ = self._extract_markdown_tables_and_clean_text()
        return clean_text
    
    def get_extracted_tables(self) -> List[pd.DataFrame]:
        """Get DataFrames extracted from markdown tables in the template."""
        _, tables = self._extract_markdown_tables_and_clean_text()
        return tables
    
    def get_all_dataframes(self) -> List[pd.DataFrame]:
        """Get all DataFrames: main df, extracted tables, and additional dfs."""
        all_dfs = []
        
        # Add main DataFrame
        if self.df is not None:
            all_dfs.append(self.df)
        
        # Add extracted tables from template
        all_dfs.extend(self.get_extracted_tables())
        
        # Add additional DataFrames
        all_dfs.extend(self.additional_dfs.values())
        
        return all_dfs
    
    def get_figure_path(self) -> str:
        """Generate figure if needed and return temp path for PowerPoint, optionally display inline"""
        if self._generated_figure_path is None and self.figure_generator is not None:
            # Generate the figure
            fig = self.figure_generator(df=self.df, **self.template_params)
            
            # Create temp file for PowerPoint embedding
            temp_dir = tempfile.gettempdir()
            temp_filename = f"temp_figure_{int(datetime.now().timestamp() * 1000)}.png"
            self._generated_figure_path = os.path.join(temp_dir, temp_filename)
            
            # Save to temp file for PowerPoint
            fig.savefig(self._generated_figure_path, dpi=300, bbox_inches='tight')
            
            # Display inline if requested
            if self.show_figures:
                plt.show()
            
            # Clean up the figure object
            plt.close(fig)
            
        return self._generated_figure_path
    
    def cleanup(self):
        """Clean up generated figure files."""
        if self._generated_figure_path and os.path.exists(self._generated_figure_path):
            os.unlink(self._generated_figure_path)
            self._generated_figure_path = None
    
    def render_console(self) -> str:
        """Render content for console display"""
        output = []
        output.append(f"# {self.title}")
        output.append("")
        
        if self.text_template:
            output.append(self.render_text())
            output.append("")
        
        if self.df is not None:
            output.append("## Data:")
            output.append(self.df.to_markdown())
            output.append("")
        
        # Show figure info
        if self.figure_generator:
            if self.show_figures:
                pass
            else:
                figure_path = self.get_figure_path()
                if figure_path:
                    output.append(f"\nFigure saved to: {figure_path}")
        
        return '\n'.join(output)


class SlideLayouts:
    """Clean slide layout templates."""
    
    def __init__(self):
        self.prs = Presentation()
        self.prs.slide_width = Inches(13.33)  # 16:9
        self.prs.slide_height = Inches(7.5)
    
    def _add_title(self, slide, title: str):
        """Add standardized title."""
        title_box = slide.shapes.add_textbox(Inches(0.3), Inches(0.2), Inches(9), Inches(0.4))
        title_box.text = title
        title_box.text_frame.paragraphs[0].font.size = Pt(18)
        title_box.text_frame.paragraphs[0].font.bold = True
        return title_box
    
    def _add_text(self, slide, text: str, left: float, top: float, 
                  width: float, height: float, font_size: int = 11):
        """Add text box."""
        text_box = slide.shapes.add_textbox(Inches(left), Inches(top), Inches(width), Inches(height))
        text_box.text = text
        text_box.text_frame.word_wrap = True
        for paragraph in text_box.text_frame.paragraphs:
            paragraph.font.size = Pt(font_size)
        return text_box
    
    def _add_table(self, slide, df: pd.DataFrame, left: float, top: float, 
                   width: float, height: float, highlight_cells: Dict = None):
        """Add DataFrame as table with intelligent column sizing."""
        highlight_cells = highlight_cells or {}
        
        include_index = not isinstance(df.index, pd.RangeIndex)
        rows, cols = df.shape[0] + 1, df.shape[1] + (1 if include_index else 0)
        
        # Calculate minimum required width for readability
        min_col_width = 0.8  # Minimum 0.8 inches per column for readability
        required_width = cols * min_col_width
        
        # Use larger width if needed for readability, up to almost full slide width
        actual_width = max(width, min(required_width, 12.0))  # Max 12 inches (almost full slide)
        
        # If table is too wide, adjust font size
        font_size = 12
        if required_width > 12.0:
            font_size = max(8, int(12 * 12.0 / required_width))  # Scale down font if needed
        
        table_shape = slide.shapes.add_table(rows, cols, Inches(left), Inches(top), 
                                            Inches(actual_width), Inches(height))
        table = table_shape.table
        
        # Calculate intelligent column widths based on content
        col_widths = []
        
        if include_index:
            # Calculate index column width
            max_index_length = max(len(str(idx)) for idx in df.index)
            index_width = max(min_col_width, min(max_index_length * 0.12, actual_width * 0.3))
            col_widths.append(index_width)
            
            # Calculate data column widths
            remaining_width = actual_width - index_width
            data_cols = df.shape[1]
            
            # Calculate width for each data column based on content
            for col in df.columns:
                max_content_length = max(
                    len(str(col)),  # Column header
                    max(len(str(val)) for val in df[col]) if len(df) > 0 else 5  # Data values
                )
                col_width = max(min_col_width, min(max_content_length * 0.12, remaining_width / data_cols * 1.5))
                col_widths.append(col_width)
                
            # Normalize if total exceeds available width
            total_width = sum(col_widths)
            if total_width > actual_width:
                scaling_factor = actual_width / total_width
                col_widths = [w * scaling_factor for w in col_widths]
                
        else:
            # No index - distribute evenly with minimum width
            col_width = max(min_col_width, actual_width / cols)
            col_widths = [col_width] * cols
        
        # Apply column widths
        for i, width_val in enumerate(col_widths):
            table.columns[i].width = Inches(width_val)
        
        # Headers
        for col_idx in range(cols):
            cell = table.cell(0, col_idx)
            if include_index and col_idx == 0:
                cell.text = ""
            else:
                actual_col_idx = col_idx - 1 if include_index else col_idx
                cell.text = str(df.columns[actual_col_idx])
            cell.fill.solid()
            cell.fill.fore_color.rgb = RGBColor(0, 0, 0)
            for paragraph in cell.text_frame.paragraphs:
                paragraph.alignment = PP_ALIGN.CENTER
                for run in paragraph.runs:
                    run.font.bold = True
                    run.font.size = Pt(font_size)
                    run.font.color.rgb = RGBColor(255, 255, 255)
        
        # Data
        for row_idx, (index_val, row) in enumerate(df.iterrows()):
            for col_idx in range(cols):
                cell = table.cell(row_idx + 1, col_idx)
                if include_index and col_idx == 0:
                    val = str(index_val)
                else:
                    actual_col_idx = col_idx - 1 if include_index else col_idx
                    cell_value = df.iloc[row_idx, actual_col_idx]
                    if isinstance(cell_value, (int, float)) and not np.isnan(cell_value):
                        col_name = df.columns[actual_col_idx]
                        row_name = str(index_val)
                        
                        is_percent = ('_pct' in col_name or '%' in col_name or 'pct' in col_name.lower() or
                                     '_pct' in row_name or '%' in row_name or 'pct' in row_name.lower() or
                                     'rate' in row_name.lower() or 'ratio' in row_name.lower())
                        
                        if is_percent:
                            val = f"{cell_value*100:.0f}%"
                        elif abs(cell_value) < 0.01 and cell_value != 0:
                            val = f"{cell_value:.4f}"
                        elif abs(cell_value) < 1:
                            val = f"{cell_value:.3f}"
                        elif abs(cell_value) < 100:
                            val = f"{cell_value:.2f}"
                        else:
                            val = f"{cell_value:.0f}"
                    else:
                        val = str(cell_value) if not pd.isna(cell_value) else "N/A"
                cell.text = val
                cell.fill.solid()
                shade = RGBColor(242, 244, 248) if row_idx % 2 == 0 else RGBColor(255, 255, 255)
                cell.fill.fore_color.rgb = shade
                
                if include_index and col_idx == 0:
                    color_rgb = RGBColor(0, 0, 0)
                else:
                    key = (index_val, df.columns[actual_col_idx])
                    color = highlight_cells.get(key)
                    if color == "red":
                        color_rgb = RGBColor(255, 0, 0)
                    elif color == "green":
                        color_rgb = RGBColor(0, 153, 0)
                    elif isinstance(color, tuple) and len(color) == 3:
                        color_rgb = RGBColor(*color)
                    else:
                        color_rgb = RGBColor(0, 0, 0)
                
                for paragraph in cell.text_frame.paragraphs:
                    paragraph.alignment = PP_ALIGN.CENTER
                    for run in paragraph.runs:
                        run.font.size = Pt(font_size)
                        run.font.color.rgb = color_rgb
        
        return table
    
    def _add_figure(self, slide, figure_path: str, left: float, top: float, height: float):
        """Add figure to slide."""
        if os.path.exists(figure_path):
            slide.shapes.add_picture(figure_path, Inches(left), Inches(top), height=Inches(height))
        else:
            self._add_text(slide, f"Figure not found: {figure_path}", left, top, 4.0, 1.0)
    
    def _estimate_text_height(self, text: str) -> float:
        """Estimate height needed for text."""
        if not text:
            return 0.5
        
        # Clean the text first
        text = text.strip()
        if not text:
            return 0.5
            
        lines = text.count('\n') + 1
        # Add wrapping estimation
        for line in text.split('\n'):
            if len(line) > 90:  # Rough chars per line
                lines += (len(line) - 1) // 90
        
        # More conservative height estimation
        estimated_height = lines * 0.18 + 0.4
        return max(min(estimated_height, 3.0), 0.8)  # Cap at 3 inches, minimum 0.8
    
    def _estimate_table_size(self, df: pd.DataFrame) -> tuple:
        """Estimate table dimensions."""
        if df is None:
            return 0, 0
        
        include_index = not isinstance(df.index, pd.RangeIndex)
        cols = df.shape[1] + (1 if include_index else 0)
        
        # Calculate minimum required width for readability
        min_col_width = 0.8
        required_width = cols * min_col_width
        
        # Allow tables to be wider if needed, up to almost full slide width
        width = max(min(1.5 + (cols * 1.2), 12.0), min(required_width, 12.0))
        height = min((len(df) + 1) * 0.35, 4.0)
        
        return width, height
    
    def add_content_slide(self, content: SlideContent, layout_type: str = 'auto', 
                         highlight_cells: Dict = None):
        """Add slide based on content and layout type."""
        title = content.title
        text = content.render_text()  # This now automatically removes markdown tables
        figure_path = content.get_figure_path()
        
        # Get all DataFrames (main + extracted from markdown + additional)
        all_dfs = content.get_all_dataframes()
        primary_df = all_dfs[0] if all_dfs else None
        
        if layout_type == 'auto':
            # Auto-detect layout
            has_text = bool(text)
            has_table = primary_df is not None
            has_figure = bool(figure_path)
            
            if has_text and has_table and has_figure:
                layout_type = 'text_table_figure'
            elif has_text and has_table:
                layout_type = 'text_table'
            elif has_text and has_figure:
                layout_type = 'text_figure'
            elif has_text:
                layout_type = 'text'
            else:
                layout_type = 'text'
        
        return self._create_slide(title, text, primary_df, figure_path, layout_type, highlight_cells, all_dfs)
    
    def _create_slide(self, title: str, text: str, df: pd.DataFrame, 
                     figure_path: str, layout_type: str, highlight_cells: Dict, all_dfs: List[pd.DataFrame]):
        """Create slide with specified layout."""
        slide = self.prs.slides.add_slide(self.prs.slide_layouts[6])
        self._add_title(slide, title)
        
        # Define available content area (after title)
        content_left = 0.3
        content_top = 0.7  # Start below title
        content_width = 12.7  # Almost full width
        content_height = 6.3  # Available height after title
        
        if layout_type == 'text':
            # Use full content area for text
            self._add_text(slide, text, content_left, content_top, content_width, content_height)
            
        elif layout_type == 'summary':
            # Fixed layout for summary (uses specialized function)
            pass  # Will be handled by create_metrics_summary_slide
            
        elif layout_type == 'text_table':
            # Content-driven positioning
            text_height = self._estimate_text_height(text)
            
            # Handle multiple tables if available
            if len(all_dfs) > 1:
                # Multiple tables - stack them vertically
                table_spacing = 0.2
                available_table_height = content_height - min(text_height, 2.0) - 0.3
                table_height_each = min(available_table_height / len(all_dfs) - table_spacing, 2.0)
                
                # Add text at top
                self._add_text(slide, text, content_left, content_top, content_width, min(text_height, 2.0))
                
                # Add tables stacked vertically
                current_top = content_top + min(text_height, 2.0) + 0.3
                for i, table_df in enumerate(all_dfs):
                    table_width, _ = self._estimate_table_size(table_df)
                    self._add_table(slide, table_df, content_left, current_top, 
                                   min(table_width, content_width), table_height_each, highlight_cells)
                    current_top += table_height_each + table_spacing
            else:
                # Single table - use original logic
                table_width, table_height = self._estimate_table_size(df)
                
                if text_height <= 2.0 and table_height <= 3.5:
                    # Stack vertically - text on top, table below
                    self._add_text(slide, text, content_left, content_top, content_width, min(text_height, 2.0))
                    table_top = content_top + min(text_height, 2.0) + 0.3
                    self._add_table(slide, df, content_left, table_top, 
                                   min(table_width, content_width), min(table_height, content_height - table_top + content_top), 
                                   highlight_cells)
                else:
                    # Side by side
                    text_width = content_width * 0.45  # 45% for text
                    table_width_adj = content_width * 0.5  # 50% for table
                    self._add_text(slide, text, content_left, content_top, text_width, content_height)
                    self._add_table(slide, df, content_left + text_width + 0.2, content_top, 
                                   table_width_adj, content_height, highlight_cells)
                
        elif layout_type == 'text_figure':
            text_height = self._estimate_text_height(text)
            if text_height <= 2.0:
                # Text top, figure bottom
                self._add_text(slide, text, content_left, content_top, content_width, min(text_height, 2.0))
                fig_top = content_top + min(text_height, 2.0) + 0.3
                fig_height = content_height - (fig_top - content_top)
                self._add_figure(slide, figure_path, content_left + 1.0, fig_top, max(fig_height, 3.0))
            else:
                # Side by side
                text_width = content_width * 0.35  # 35% for text
                fig_width = content_width * 0.6   # 60% for figure
                self._add_text(slide, text, content_left, content_top, text_width, content_height)
                self._add_figure(slide, figure_path, content_left + text_width + 0.3, content_top, content_height)
                
        elif layout_type == 'text_table_figure':
            # Three-way layout - more compact and ensure figure stays on slide
            text_height = min(self._estimate_text_height(text), 1.5)  # Cap text at 1.5 inches
            
            # Use primary table for layout calculation
            table_width, table_height = self._estimate_table_size(df) if df is not None else (0, 0)
            table_width = min(table_width, content_width * 0.35)  # Limit table to 35% of width
            table_height = min(table_height, 2.5)  # Cap table height at 2.5 inches
            
            # Text across the top - more compact
            self._add_text(slide, text, content_left, content_top, content_width, text_height)
            
            # Table and figure side by side below text
            table_top = content_top + text_height + 0.2  # Small gap after text
            remaining_height = content_height - (table_top - content_top)  # Available height for table+figure
            
            if df is not None:
                # Add table on the left
                self._add_table(slide, df, content_left, table_top, table_width, 
                               min(table_height, remaining_height), highlight_cells)
                fig_left = content_left + table_width + 0.3
                fig_width = content_width - table_width - 0.3
            else:
                fig_left = content_left
                fig_width = content_width
            
            # Ensure figure fits on slide - conservative sizing
            fig_height = min(remaining_height, 3.5)  # Cap at 3.5 inches
            if fig_left + 6.0 > content_left + content_width:  # Check if figure would extend beyond slide
                fig_left = content_left + content_width - 6.0  # Pull back from right edge
            
            self._add_figure(slide, figure_path, fig_left, table_top, fig_height)
        
        return slide
    
    def save(self, filename: str = None, output_dir: str = '.'):
        """Save presentation."""
        os.makedirs(output_dir, exist_ok=True)
        
        if not filename:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"presentation_{timestamp}.pptx"
        elif not filename.endswith('.pptx'):
            filename = f"{filename}.pptx"
        
        filepath = os.path.join(output_dir, filename)
        self.prs.save(filepath)
        return filepath


# The decorator 
def dual_output(console: bool = True, slide: bool = True, 
               slide_builder: SlideLayouts = None,
               layout_type: str = 'auto',
               show_figures: bool = True):
    """
    Decorator that captures function output and creates both console display and slide.
    
    Args:
        console: Whether to print to console
        slide: Whether to create slide
        slide_builder: SlideLayouts instance to add slide to
        layout_type: Layout type for the slide ('text', 'chart', 'table', 'auto')
        show_figures: Whether to display figures inline (True) or just show paths (False)
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            # Execute the function to get SlideContent
            slide_content = func(*args, **kwargs)
            
            # Set show_figures parameter on the content
            slide_content.show_figures = show_figures
            
            results = {}
            
            if console:
                console_output = slide_content.render_console()
                results['console'] = console_output
                print(console_output)
            
            if slide and slide_builder:
                slide_layout = slide_builder.add_content_slide(slide_content, layout_type)
                results['slide'] = slide_layout
            
            return slide_content, results
        
        return wrapper
    return decorator


# Example figure generators
def create_bar_chart(save_path: str, df: pd.DataFrame, **params):
    """Example figure generator - bar chart."""
    try:
        fig, ax = plt.subplots(figsize=(8, 5))
        df.plot(kind='bar', ax=ax)
        ax.set_title(params.get('chart_title', 'Data Overview'))
        ax.set_ylabel(params.get('ylabel', 'Values'))
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    except ImportError:
        # Fallback if matplotlib not available
        with open(save_path, 'w') as f:
            f.write("Matplotlib not available")


def create_scatter_plot(save_path: str, df: pd.DataFrame, **params):
    """Example figure generator - scatter plot."""
    try:
        fig, ax = plt.subplots(figsize=(8, 5))
        
        if len(df.columns) >= 2:
            x_col, y_col = df.columns[:2]
            ax.scatter(df[x_col], df[y_col])
            ax.set_xlabel(x_col)
            ax.set_ylabel(y_col)
            ax.set_title(params.get('chart_title', f'{y_col} vs {x_col}'))
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    except ImportError:
        with open(save_path, 'w') as f:
            f.write("Matplotlib not available")


# Authentication functions for Google Drive (simplified)
def get_credentials_local(credentials_path: str = None, token_path: str = None):
    """Get credentials using local files (development/personal use)."""
    try:
        from google.auth.transport.requests import Request
        from google.oauth2.credentials import Credentials
        from google_auth_oauthlib.flow import InstalledAppFlow
        
        SCOPES = ['https://www.googleapis.com/auth/presentations',
                  'https://www.googleapis.com/auth/drive.file',
                  'https://www.googleapis.com/auth/drive']
        
        creds = None
        if token_path and os.path.exists(token_path):
            creds = Credentials.from_authorized_user_file(token_path, SCOPES)
        
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                if not credentials_path or not os.path.exists(credentials_path):
                    raise FileNotFoundError("credentials.json file is required for OAuth flow")
                flow = InstalledAppFlow.from_client_secrets_file(credentials_path, SCOPES)
                creds = flow.run_local_server(port=0)
                if token_path:
                    with open(token_path, 'w') as token:
                        token.write(creds.to_json())
        
        return creds
        
    except ImportError:
        raise ImportError("Install: pip install google-auth google-auth-oauthlib")


def get_credentials_enterprise(credentials_dict: dict, proxy_info=None, ca_certs: str = None):
    """Get credentials for enterprise environments with proxy support."""
    try:
        from oauth2client.service_account import ServiceAccountCredentials
        import httplib2
        
        SCOPES = ['https://www.googleapis.com/auth/presentations',
                  'https://www.googleapis.com/auth/drive.file',
                  'https://www.googleapis.com/auth/drive']
        
        creds = ServiceAccountCredentials.from_json_keyfile_dict(credentials_dict, scopes=SCOPES)
        http = httplib2.Http(proxy_info=proxy_info, ca_certs=ca_certs)
        authorized_http = creds.authorize(http)
        
        return creds, authorized_http
        
    except ImportError:
        raise ImportError("Install: pip install oauth2client httplib2")


def upload_to_google_drive(file_path: str, user_email: str = None, **auth_kwargs):
    """Upload a file to Google Drive with smart authentication detection."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    try:
        from googleapiclient.discovery import build
        from googleapiclient.http import MediaFileUpload
        
        # Determine auth method and get service
        if 'credentials_dict' in auth_kwargs:
            # Enterprise auth
            creds, authorized_http = get_credentials_enterprise(
                auth_kwargs['credentials_dict'],
                auth_kwargs.get('proxy_info'),
                auth_kwargs.get('ca_certs')
            )
            drive_service = build('drive', 'v3', http=authorized_http, cache_discovery=False)
        else:
            # Local auth
            creds = get_credentials_local(
                auth_kwargs.get('credentials_path'),
                auth_kwargs.get('token_path')
            )
            drive_service = build('drive', 'v3', credentials=creds)
        
        # Create user folder if user_email provided
        folder_id = None
        if user_email:
            username = user_email.split('@')[0] if '@' in user_email else user_email
            folder_name = f"RCA_Analysis_{username}"
            
            query = f"name='{folder_name}' and mimeType='application/vnd.google-apps.folder'"
            results = drive_service.files().list(q=query).execute()
            
            if results['files']:
                folder_id = results['files'][0]['id']
            else:
                folder_metadata = {
                    'name': folder_name,
                    'mimeType': 'application/vnd.google-apps.folder'
                }
                folder = drive_service.files().create(body=folder_metadata).execute()
                folder_id = folder['id']
                
                if 'credentials_dict' in auth_kwargs:
                    permission = {
                        'type': 'user',
                        'role': 'writer',
                        'emailAddress': user_email
                    }
                    drive_service.permissions().create(
                        fileId=folder_id,
                        body=permission,
                        sendNotificationEmail=False
                    ).execute()
        
        # Upload file
        file_metadata = {
            'name': os.path.basename(file_path),
            'mimeType': 'application/vnd.google-apps.presentation'
        }
        if folder_id:
            file_metadata['parents'] = [folder_id]
        
        media = MediaFileUpload(
            file_path,
            mimetype='application/vnd.openxmlformats-officedocument.presentationml.presentation',
            resumable=True
        )
        
        file = drive_service.files().create(
            body=file_metadata,
            media_body=media,
            fields='id,webViewLink'
        ).execute()
        
        result = {
            'gdrive_id': file['id'],
            'gdrive_url': file['webViewLink']
        }
        
        if folder_id:
            result['folder_url'] = f"https://drive.google.com/drive/folders/{folder_id}"
        
        return result
        
    except ImportError:
        raise ImportError("Install: pip install google-api-python-client")


# Additional summary page functions
def _create_highlight_cells_map(
    df: pd.DataFrame,
    metric_anomaly_map: Dict[str, Dict[str, Any]]
) -> Dict[Tuple[str, str], str]:
    """Create a highlight_cells dictionary for the table based on anomaly information."""
    highlight_cells = {}
    
    for metric_name, metric_info in metric_anomaly_map.items():
        if metric_name in df.index:
            anomalous_region = metric_info.get('anomalous_region')
            direction = metric_info.get('direction')
            higher_is_better = metric_info.get('higher_is_better', True)
            
            if anomalous_region and direction and anomalous_region in df.columns:
                is_good = (direction == 'higher' and higher_is_better) or \
                         (direction == 'lower' and not higher_is_better)
                
                color = 'green' if is_good else 'red'
                highlight_cells[(metric_name, anomalous_region)] = color
    
    return highlight_cells


def create_metrics_summary_slide(
    slide_layouts: SlideLayouts,
    df: pd.DataFrame,
    metrics_text: Dict[str, str],
    metric_anomaly_map: Dict[str, Dict[str, Any]],
    title: str = "Metrics Summary"
) -> None:
    """Create a summary slide with a styled table of metrics data and pre-formatted text explanations."""
    slide_layout = slide_layouts.prs.slide_layouts[6]
    slide = slide_layouts.prs.slides.add_slide(slide_layout)
    
    # Add standardized title
    slide_layouts._add_title(slide, title)
    
    # Filter DataFrame to only include metrics we have text for
    metrics = list(metrics_text.keys())
    metrics_df = df[metrics]
    
    # Transpose the DataFrame so regions are columns and metrics are rows
    metrics_df_transposed = metrics_df.T
    
    # Create highlight cells map for transposed data
    highlight_cells = _create_highlight_cells_map(metrics_df_transposed, metric_anomaly_map)
    
    content_top = 0.8
    
    # Calculate table dimensions
    num_rows = len(metrics_df_transposed) + 1
    row_height = 0.4
    table_height = num_rows * row_height
    
    max_index_length = max(len(str(idx)) for idx in metrics_df_transposed.index)
    max_column_length = max(len(str(col)) for col in metrics_df_transposed.columns)
    
    index_width_estimate = max(max_index_length * 0.08, 1.3)
    data_width_estimate = max_column_length * 0.08 * len(metrics_df_transposed.columns)
    total_table_width = min(index_width_estimate + data_width_estimate + 0.5, 6.0)
    
    # Add styled table on the left side using integrated method
    table = slide_layouts._add_table(
        slide=slide,
        df=metrics_df_transposed,
        left=0.5,
        top=content_top,
        width=total_table_width,
        height=table_height,
        highlight_cells=highlight_cells
    )
    
    # Create text area on the right side
    text_left = Inches(0.5 + total_table_width + 0.3)
    text_width = Inches(13.0 - (0.5 + total_table_width + 0.3) - 0.2)
    text_top = Inches(content_top)
    text_height = Inches(6.0)
    
    text_box = slide.shapes.add_textbox(text_left, text_top, text_width, text_height)
    text_frame = text_box.text_frame
    text_frame.word_wrap = True
    
    # Add explanations for each metric
    for metric, explanation_text in metrics_text.items():
        p = text_frame.add_paragraph()
        p.text = f"{metric}:"
        p.font.bold = True
        p.font.size = Pt(12)
        p.space_after = Pt(6)
        
        p = text_frame.add_paragraph()
        p.text = explanation_text
        p.font.size = Pt(12)
        p.space_after = Pt(12)


if __name__ == "__main__":
    # Example usage
    data = {
        'Conversion Rate': [0.12, 0.08, 0.11, 0.13],
        'AOV': [75.0, 65.0, 80.0, 85.0]
    }
    df = pd.DataFrame(data, index=['Global', 'NA', 'EU', 'Asia'])
    
    # Create slide builder
    slides = SlideLayouts()
    
    # Example 1: Text + Table with decorator
    @dual_output(console=True, slide=True, slide_builder=slides, layout_type='summary')
    def create_summary_analysis():
        return SlideContent(
            title="Summary Analysis",
            text_template="Key insight: {{ region }} shows {{ trend }} performance with {{ metric }} being {{ pct }}% {{ direction }}.",
            df=df,
            template_params={
                'region': 'North America',
                'trend': 'concerning',
                'metric': 'conversion rate',
                'pct': 33,
                'direction': 'lower'
            }
        )
    
    # Example 2: Figure generation with decorator
    @dual_output(console=True, slide=True, slide_builder=slides, layout_type='text_figure')
    def create_chart_analysis():
        return SlideContent(
            title="Performance by Region",
            text_template="The chart shows {{ insight }}. {{ region }} is an outlier.",
            df=df,  # Pass the dataframe
            figure_generator=create_bar_chart,  # Function, not path!
            template_params={
                'insight': 'significant regional variation',
                'region': 'North America',
                'chart_title': 'Regional Performance Comparison'
            }
        )
    
    # Run examples
    print("🎯 DECORATOR APPROACH WITH FIGURE GENERATION")
    print("=" * 60)
    
    print("\n📊 SUMMARY ANALYSIS:")
    content1, results1 = create_summary_analysis()
    print(results1['console'])
    
    print("\n📈 CHART ANALYSIS:")
    content2, results2 = create_chart_analysis()
    print(results2['console'])
    
    # Save slides
    filepath = slides.save("decorator_demo", "./output")
    print(f"\n✅ Slides saved: {filepath}")
    print("✅ Figure files automatically cleaned up!") 
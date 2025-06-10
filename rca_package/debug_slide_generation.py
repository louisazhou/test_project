#!/usr/bin/env python3

import pandas as pd
import matplotlib.pyplot as plt
from rca_package.make_slides import SlideLayouts, SlideContent, dual_output
import numpy as np
import logging
import sys
import os
import tempfile
from datetime import datetime

# Set up detailed logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(levelname)s: %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('slide_debug.log', mode='w')
    ]
)

class DebugSlideLayouts(SlideLayouts):
    """Enhanced SlideLayouts with comprehensive debugging."""
    
    def __init__(self):
        super().__init__()
        self.slide_count = 0
        logging.info("🏗️  DebugSlideLayouts initialized")
    
    def add_content_slide(self, content: SlideContent, layout_type: str = 'auto', 
                         highlight_cells = None):
        """Override with comprehensive debugging."""
        self.slide_count += 1
        logging.info(f"\n{'='*60}")
        logging.info(f"🎯 SLIDE #{self.slide_count}: add_content_slide() called")
        logging.info(f"   Title: '{content.title}'")
        logging.info(f"   Layout type: {layout_type}")
        logging.info(f"   Show figures: {content.show_figures}")
        
        # Debug content details
        logging.info(f"   Content details:")
        logging.info(f"     • Text template: {len(content.text_template) if content.text_template else 0} chars")
        logging.info(f"     • Template params: {list(content.template_params.keys())}")
        logging.info(f"     • DataFrames: {list(content.dfs.keys())}")
        logging.info(f"     • Figure generators: {len(content.figure_generators)}")
        
        # Test text rendering
        try:
            rendered_text = content.render_text()
            logging.info(f"     • Rendered text: {len(rendered_text)} chars")
            logging.info(f"     • Text preview: {rendered_text[:200]}...")
        except Exception as e:
            logging.error(f"     ❌ Text rendering failed: {e}")
            rendered_text = ""
        
        # Test text and table parsing
        try:
            text_chunks, table_positions = content.get_text_and_table_positions()
            logging.info(f"     • Text chunks: {len(text_chunks)}")
            for i, chunk in enumerate(text_chunks):
                logging.info(f"       Chunk {i}: {len(chunk)} chars")
            logging.info(f"     • Table positions: {len(table_positions)}")
            for i, (key, df) in enumerate(table_positions):
                logging.info(f"       Table {i} ({key}): {df.shape}")
        except Exception as e:
            logging.error(f"     ❌ Text/table parsing failed: {e}")
            text_chunks, table_positions = [], []
        
        # Call parent method and track result
        try:
            logging.info(f"🚀 Calling parent add_content_slide...")
            result = super().add_content_slide(content, layout_type, highlight_cells)
            logging.info(f"✅ Parent add_content_slide completed successfully")
            return result
        except Exception as e:
            logging.error(f"❌ Parent add_content_slide failed: {e}")
            import traceback
            logging.error(f"   Full traceback: {traceback.format_exc()}")
            raise
    
    def _create_slide_with_structured_content(self, title: str, text_chunks, 
                                            table_positions, figure_path, layout_type, 
                                            highlight_cells):
        """Override with detailed step-by-step debugging."""
        logging.info(f"\n📋 _create_slide_with_structured_content called:")
        logging.info(f"   Title: '{title}'")
        logging.info(f"   Layout type: {layout_type}")
        logging.info(f"   Text chunks: {len(text_chunks)}")
        logging.info(f"   Table positions: {len(table_positions)}")
        logging.info(f"   Figure path provided: {figure_path is not None}")
        if figure_path:
            logging.info(f"   Figure path: {figure_path}")
            logging.info(f"   Figure file exists: {os.path.exists(figure_path)}")
            if os.path.exists(figure_path):
                try:
                    size = os.path.getsize(figure_path) / 1024
                    logging.info(f"   Figure file size: {size:.1f} KB")
                except:
                    logging.warning(f"   Could not get figure file size")
        
        # Log text content details
        if text_chunks:
            for i, chunk in enumerate(text_chunks):
                logging.info(f"     Text chunk {i}: {len(chunk)} chars")
                logging.info(f"       Preview: {chunk[:100]}...")
        
        # Log table details
        if table_positions:
            for i, (key, df) in enumerate(table_positions):
                logging.info(f"     Table {i} ({key}): shape {df.shape}")
                logging.info(f"       Columns: {list(df.columns)}")
        
        # Create slide and track
        logging.info(f"🏗️  Creating basic slide structure...")
        slide = self._create_basic_slide_structure(title)
        logging.info(f"✅ Basic slide structure created")
        
        # Check for two-column layout
        if layout_type == 'text_tables' and self._should_use_two_column_layout(text_chunks, table_positions):
            logging.info(f"🔄 Using two-column layout")
            self._create_two_column_text_tables_layout(slide, text_chunks, table_positions, highlight_cells)
            return slide
        
        # Continue with standard processing
        logging.info(f"🔄 Using standard single-column layout")
        
        # Process text
        full_text = self._combine_text_chunks(text_chunks)
        text_height = self._estimate_text_height(full_text) if full_text else 0
        logging.info(f"   Combined text: {len(full_text)} chars, estimated height: {text_height:.3f}")
        
        # Process tables
        table_width = table_height = 0
        if table_positions:
            table_width, table_height = self._calculate_table_dimensions(table_positions[0][1])
            logging.info(f"   First table dimensions: {table_width:.3f} x {table_height:.3f}")
        
        # Calculate layouts
        logging.info(f"🧮 Calculating content layout...")
        layouts = self._calculate_content_layout(
            content_type=layout_type,
            text_height=text_height,
            table_width=table_width,
            table_height=table_height
        )
        
        # Log calculated layouts
        logging.info(f"   Calculated layouts:")
        for element, layout in layouts.items():
            logging.info(f"     {element}: {layout}")
            
            # Validate layout dimensions
            if layout.get('width', 0) <= 0 or layout.get('height', 0) <= 0:
                logging.error(f"❌ INVALID LAYOUT for {element}: {layout}")
        
        # Add content blocks based on layout
        logging.info(f"📦 Adding content blocks for layout: {layout_type}")
        
        if layout_type == 'text' and full_text:
            logging.info(f"   Adding text block...")
            self._add_content_block(
                slide=slide,
                content_type='text',
                content_data={'text': full_text},
                layout_info=layouts['text']
            )
            
        elif layout_type == 'text_figure':
            if full_text:
                logging.info(f"   Adding text block...")
                self._add_content_block(
                    slide=slide,
                    content_type='text',
                    content_data={'text': full_text},
                    layout_info=layouts['text']
                )
            if figure_path:
                logging.info(f"   Adding figure block...")
                self._add_content_block(
                    slide=slide,
                    content_type='figure',
                    content_data={'figure_path': figure_path},
                    layout_info=layouts['figure']
                )
                
        elif layout_type == 'text_tables':
            if full_text:
                logging.info(f"   Adding text block...")
                self._add_content_block(
                    slide=slide,
                    content_type='text',
                    content_data={'text': full_text},
                    layout_info=layouts['text']
                )
            
            # Add tables sequentially
            if table_positions:
                logging.info(f"   Adding {len(table_positions)} table(s)...")
                current_top = layouts['tables']['top']
                for i, (table_key, df) in enumerate(table_positions):
                    table_width, table_height = self._calculate_table_dimensions(df)
                    table_layout = {
                        'left': layouts['tables']['left'],
                        'top': current_top,
                        'width': table_width,
                        'height': table_height
                    }
                    logging.info(f"     Table {i} layout: {table_layout}")
                    self._add_content_block(
                        slide=slide,
                        content_type='table',
                        content_data={'df': df, 'highlight_cells': highlight_cells},
                        layout_info=table_layout
                    )
                    current_top += table_height + 0.1  # ELEMENT_GAP
                    
        elif layout_type == 'text_tables_figure':
            logging.info(f"   Adding text_tables_figure layout...")
            if full_text:
                logging.info(f"     Adding text...")
                self._add_content_block(
                    slide=slide,
                    content_type='text',
                    content_data={'text': full_text},
                    layout_info=layouts['text']
                )
            
            if table_positions:
                logging.info(f"     Adding table...")
                table_key, df = table_positions[0]
                self._add_content_block(
                    slide=slide,
                    content_type='table',
                    content_data={'df': df, 'highlight_cells': highlight_cells},
                    layout_info=layouts['tables']
                )
            
            if figure_path:
                logging.info(f"     Adding figure...")
                self._add_content_block(
                    slide=slide,
                    content_type='figure',
                    content_data={'figure_path': figure_path},
                    layout_info=layouts['figure']
                )
        
        logging.info(f"✅ Slide creation completed for: {title}")
        return slide
    
    def _add_content_block(self, slide, content_type: str, content_data, layout_info):
        """Override with detailed debugging."""
        logging.info(f"📦 Adding content block: {content_type}")
        logging.info(f"   Layout info: {layout_info}")
        
        # Validate layout info
        if layout_info.get('width', 0) <= 0 or layout_info.get('height', 0) <= 0:
            logging.error(f"❌ INVALID DIMENSIONS for {content_type}: {layout_info}")
            return None
        
        try:
            result = super()._add_content_block(slide, content_type, content_data, layout_info)
            logging.info(f"✅ Content block '{content_type}' added successfully")
            return result
        except Exception as e:
            logging.error(f"❌ Content block '{content_type}' failed: {e}")
            import traceback
            logging.error(f"   Full traceback: {traceback.format_exc()}")
            return None
    
    def _add_text(self, slide, text: str, left: float, top: float, 
                  width: float, height: float, font_size: int = 12):
        """Override with detailed debugging."""
        logging.info(f"🔤 _add_text called:")
        logging.info(f"   Position: ({left:.3f}, {top:.3f})")
        logging.info(f"   Size: {width:.3f} x {height:.3f}")
        logging.info(f"   Text length: {len(text)} chars")
        logging.info(f"   Font size: {font_size}")
        
        # Check for problematic values
        if width <= 0 or height <= 0:
            logging.error(f"❌ INVALID TEXT DIMENSIONS: width={width}, height={height}")
            return None
        
        if left < 0 or top < 0:
            logging.warning(f"⚠️  Negative position: left={left}, top={top}")
        
        if not text or not text.strip():
            logging.warning(f"⚠️  Empty or whitespace-only text")
            return None
        
        try:
            result = super()._add_text(slide, text, left, top, width, height, font_size)
            logging.info(f"✅ Text added successfully")
            return result
        except Exception as e:
            logging.error(f"❌ TEXT ADDITION FAILED: {e}")
            import traceback
            logging.error(f"   Full traceback: {traceback.format_exc()}")
            return None
    
    def _add_table(self, slide, df: pd.DataFrame, left: float, top: float, 
                   width: float, height: float, highlight_cells=None):
        """Override with detailed debugging."""
        logging.info(f"📊 _add_table called:")
        logging.info(f"   Position: ({left:.3f}, {top:.3f})")
        logging.info(f"   Size: {width:.3f} x {height:.3f}")
        logging.info(f"   DataFrame shape: {df.shape}")
        logging.info(f"   DataFrame columns: {list(df.columns)}")
        
        # Check for problematic values
        if width <= 0 or height <= 0:
            logging.error(f"❌ INVALID TABLE DIMENSIONS: width={width}, height={height}")
            return None
        
        if df.empty:
            logging.warning(f"⚠️  Empty DataFrame")
            return None
        
        try:
            result = super()._add_table(slide, df, left, top, width, height, highlight_cells)
            logging.info(f"✅ Table added successfully")
            return result
        except Exception as e:
            logging.error(f"❌ TABLE ADDITION FAILED: {e}")
            import traceback
            logging.error(f"   Full traceback: {traceback.format_exc()}")
            return None
    
    def _add_figure(self, slide, figure_path: str, left: float, top: float, 
                    width=None, height=None):
        """Override with detailed debugging."""
        logging.info(f"🖼️  _add_figure called:")
        logging.info(f"   Figure path: {figure_path}")
        logging.info(f"   Position: ({left:.3f}, {top:.3f})")
        logging.info(f"   Constraints: {width} x {height}")
        
        # Check if file exists
        if not figure_path:
            logging.error(f"❌ FIGURE PATH IS NONE/EMPTY")
            return None
            
        if not os.path.exists(figure_path):
            logging.error(f"❌ FIGURE FILE NOT FOUND: {figure_path}")
            return None
        
        # Check file details
        try:
            file_size = os.path.getsize(figure_path) / 1024  # KB
            logging.info(f"   File size: {file_size:.1f} KB")
            
            from PIL import Image
            with Image.open(figure_path) as img:
                img_width, img_height = img.size
                logging.info(f"   Original image: {img_width}x{img_height} pixels")
        except Exception as e:
            logging.warning(f"⚠️  Could not read image details: {e}")
        
        # Check if position makes sense
        if left < 0 or top < 0:
            logging.warning(f"⚠️  NEGATIVE POSITION: left={left}, top={top}")
        if left > 13 or top > 7:
            logging.warning(f"⚠️  POSITION OUTSIDE SLIDE: left={left}, top={top}")
        if width and width <= 0:
            logging.error(f"❌ INVALID WIDTH: {width}")
        if height and height <= 0:
            logging.error(f"❌ INVALID HEIGHT: {height}")
        
        try:
            result = super()._add_figure(slide, figure_path, left, top, width, height)
            logging.info(f"✅ Figure added successfully")
            
            # Additional verification
            logging.info(f"   PowerPoint shapes count after adding figure: {len(slide.shapes)}")
            
            return result
        except Exception as e:
            logging.error(f"❌ FIGURE ADDITION FAILED: {e}")
            import traceback
            logging.error(f"   Full traceback: {traceback.format_exc()}")
            return None

def debug_dual_output(console: bool = True, slide: bool = True, 
                     slide_builder=None, layout_type: str = 'auto',
                     show_figures: bool = True):
    """Debug version of dual_output decorator with comprehensive logging."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            logging.info(f"\n{'='*60}")
            logging.info(f"🎯 DUAL_OUTPUT DECORATOR: {func.__name__}")
            logging.info(f"   Console: {console}")
            logging.info(f"   Slide: {slide}")
            logging.info(f"   Layout type: {layout_type}")
            logging.info(f"   Show figures: {show_figures}")
            
            # Execute the function
            logging.info(f"🚀 Executing function: {func.__name__}")
            try:
                slide_content = func(*args, **kwargs)
                logging.info(f"✅ Function executed successfully")
            except Exception as e:
                logging.error(f"❌ Function execution failed: {e}")
                import traceback
                logging.error(f"   Full traceback: {traceback.format_exc()}")
                raise
            
            # Validate slide_content
            if not isinstance(slide_content, SlideContent):
                logging.error(f"❌ Function must return SlideContent, got: {type(slide_content)}")
                raise TypeError(f"Expected SlideContent, got {type(slide_content)}")
            
            # Set show_figures
            slide_content.show_figures = show_figures
            logging.info(f"✅ SlideContent configured")
            
            results = {}
            
            # Console output
            if console:
                logging.info(f"🖥️  Processing console output...")
                try:
                    console_output = slide_content.render_console()
                    results['console'] = console_output
                    print(console_output)
                    logging.info(f"✅ Console output completed ({len(console_output)} chars)")
                except Exception as e:
                    logging.error(f"❌ Console output failed: {e}")
            
            # Slide output
            if slide and slide_builder:
                logging.info(f"🎯 Processing slide output...")
                try:
                    slide_layout = slide_builder.add_content_slide(slide_content, layout_type)
                    results['slide'] = slide_layout
                    logging.info(f"✅ Slide output completed")
                except Exception as e:
                    logging.error(f"❌ Slide output failed: {e}")
                    import traceback
                    logging.error(f"   Full traceback: {traceback.format_exc()}")
            
            # Cleanup
            logging.info(f"🧹 Cleaning up temporary files...")
            slide_content.cleanup()
            
            logging.info(f"✅ DUAL_OUTPUT DECORATOR COMPLETED: {func.__name__}")
            return slide_content, results
        
        return wrapper
    return decorator

def create_test_figure(df: pd.DataFrame, title: str) -> plt.Figure:
    """Create a simple test figure."""
    logging.info(f"🖼️  Creating test figure: {title}")
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Simple bar chart
    if 'Conversion Rate' in df.columns:
        try:
            # Convert percentage strings to floats
            values = df['Conversion Rate'].str.rstrip('%').astype('float') / 100
            ax.bar(df.index, values)
            ax.set_ylabel('Conversion Rate')
            ax.set_title(title)
            ax.set_ylim(0, max(values) * 1.2)
            plt.xticks(rotation=45)
            plt.tight_layout()
            logging.info(f"✅ Test figure created successfully")
        except Exception as e:
            logging.error(f"❌ Figure creation failed: {e}")
            ax.text(0.5, 0.5, f'Figure Error: {e}', transform=ax.transAxes, ha='center')
    else:
        ax.text(0.5, 0.5, 'No Conversion Rate data', transform=ax.transAxes, ha='center')
    
    return fig

def main():
    """Main debugging function that mimics example_analysis.py usage pattern."""
    
    print("🔍 COMPREHENSIVE SLIDE DEBUGGING - Mimicking example_analysis.py")
    logging.info("=== COMPREHENSIVE DEBUGGING SESSION STARTED ===")
    
    # Create test data (mimicking example_analysis.py)
    test_df = pd.DataFrame({
        'Region': ['North America', 'Europe', 'Asia'],
        'Conversion Rate': ['8%', '11%', '13%'],
        'Page Load Time': ['3.8s', '2.2s', '1.9s'],
        'Bounce Rate': ['45%', '32%', '28%']
    })
    test_df = test_df.set_index('Region')
    
    logging.info(f"📊 Test data created: {test_df.shape}")
    
    # Create debug slide builder (like example_analysis.py)
    slides = DebugSlideLayouts()
    
    # Test Case 1: Simple text slide (like example_analysis.py title slide)
    print(f"\n🔍 TEST 1: Simple text slide")
    
    @debug_dual_output(console=True, slide=True, slide_builder=slides, 
                      layout_type='text', show_figures=False)
    def create_simple_text_slide():
        return SlideContent(
            title="Debug Test - Simple Text",
            text_template="""
This is a simple text slide for debugging.

Summary:
• Testing basic text rendering
• No tables or figures
• Generated on: {{ timestamp }}
            """,
            dfs={},
            template_params={
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
        )
    
    try:
        content1, results1 = create_simple_text_slide()
        print("✅ Test 1 completed")
    except Exception as e:
        print(f"❌ Test 1 failed: {e}")
    
    # Test Case 2: Text + Table (like example_analysis.py scorer slides)
    print(f"\n🔍 TEST 2: Text + Table slide")
    
    @debug_dual_output(console=True, slide=True, slide_builder=slides, 
                      layout_type='text_tables', show_figures=False)
    def create_text_table_slide():
        return SlideContent(
            title="Debug Test - Text + Table",
            text_template="""
Analysis results for regional performance:

Key insights:
• North America shows concerning performance
• Europe is performing moderately  
• Asia shows strong performance

Table below shows the detailed metrics:

{{ results_table }}
            """,
            dfs={'results_table': test_df},
            template_params={}
        )
    
    try:
        content2, results2 = create_text_table_slide()
        print("✅ Test 2 completed")
    except Exception as e:
        print(f"❌ Test 2 failed: {e}")
    
    # Test Case 3: Text + Figure (like example_analysis.py with figures)
    print(f"\n🔍 TEST 3: Text + Figure slide")
    
    @debug_dual_output(console=True, slide=True, slide_builder=slides, 
                      layout_type='text_figure', show_figures=False)
    def create_text_figure_slide():
        return SlideContent(
            title="Debug Test - Text + Figure",
            text_template="""
Regional performance visualization.

The chart shows significant variation across regions:
• North America: Underperforming 
• Europe: Baseline performance
• Asia: Outperforming expectations

This pattern suggests regional-specific factors are at play.
            """,
            dfs={},
            template_params={},
            figure_generators=[{
                'title_suffix': 'Performance Chart',
                'params': {'df': test_df, 'title': 'Regional Performance'},
                'function': create_test_figure
            }]
        )
    
    try:
        content3, results3 = create_text_figure_slide()
        print("✅ Test 3 completed")
    except Exception as e:
        print(f"❌ Test 3 failed: {e}")
    
    # Test Case 4: Complex slide (text + table + figure)
    print(f"\n🔍 TEST 4: Complex slide (text + table + figure)")
    
    @debug_dual_output(console=True, slide=True, slide_builder=slides, 
                      layout_type='text_tables_figure', show_figures=False)
    def create_complex_slide():
        return SlideContent(
            title="Debug Test - Complex Slide",
            text_template="""
Comprehensive regional analysis:

{{ analysis_table }}

The data reveals critical performance gaps that require immediate attention.
            """,
            dfs={'analysis_table': test_df},
            template_params={},
            figure_generators=[{
                'title_suffix': 'Analysis Chart',
                'params': {'df': test_df, 'title': 'Regional Analysis'},
                'function': create_test_figure
            }]
        )
    
    try:
        content4, results4 = create_complex_slide()
        print("✅ Test 4 completed")
    except Exception as e:
        print(f"❌ Test 4 failed: {e}")
    
    # Test Case 5: User's exact scenario
    print(f"\n🔍 TEST 5: User's exact scenario reproduction")
    
    # User's exact data structure
    user_breakdown_data = {
        'job_level': ['6', '7'],
        'Region Mix': ['Higher', 'Lower'], 
        'ROW Mix': ['71.6%', '28.4%'],
        'Region Lost-Rate': ['53.3%', '46.7%'],
        'ROW Lost-Rate': ['37.8%', '29.6%']
    }
    user_breakdown_df = pd.DataFrame(user_breakdown_data)
    
    def user_figure_generator(**params):
        """Replicate user's figure generation."""
        logging.info(f"🎨 USER FIGURE GENERATOR called with: {params}")
        
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.bar(['Job Level 6', 'Job Level 7'], [53.3, 46.7])
        ax.set_title('CLI Closed Lost - Job Level Analysis')
        ax.set_ylabel('Lost Rate (%)')
        
        # Make figure very obvious for testing
        ax.text(0.5, 40, 'USER FIGURE TEST', ha='center', fontsize=14, color='red', weight='bold')
        
        logging.info(f"✅ User figure created successfully")
        return fig
    
    @debug_dual_output(console=True, slide=True, slide_builder=slides, 
                      layout_type='text_tables_figure', show_figures=False)
    def create_user_scenario():
        return SlideContent(
            title="CLI Closed Lost - Construct (job_level)",
            text_template="""CLI Closed Lost in NA is 7.16pp higher than Global mean.
The main factor for this gap is efficiency (lost-rate within each level).

Job level breakdown:
{{ breakdown_df }}""",
            dfs={'breakdown_df': user_breakdown_df},
            template_params={},
            figure_generators=[{
                'function': user_figure_generator,
                'params': {}
            }]
        )
    
    try:
        content5, results5 = create_user_scenario()
        print("✅ Test 5 (User scenario) completed")
        logging.info(f"🎯 USER SCENARIO RESULTS:")
        logging.info(f"   Console output length: {len(results5.get('console', ''))}")
        logging.info(f"   Slide created: {results5.get('slide') is not None}")
    except Exception as e:
        print(f"❌ Test 5 (User scenario) failed: {e}")
        logging.error(f"❌ USER SCENARIO FAILED: {e}")
        import traceback
        logging.error(f"   Full traceback: {traceback.format_exc()}")
    
    # Save and report
    print(f"\n💾 Saving debug presentation...")
    try:
        filename = slides.save('debug_comprehensive_analysis.pptx')
        print(f"✅ Presentation saved: {filename}")
        
        # Check file size
        if os.path.exists(filename):
            file_size = os.path.getsize(filename) / 1024  # KB
            print(f"   File size: {file_size:.1f} KB")
            print(f"   Total slides: {slides.slide_count}")
        
    except Exception as e:
        print(f"❌ Save failed: {e}")
    
    print(f"\n📋 Debug session completed!")
    print(f"📄 Detailed log saved: slide_debug.log")
    print(f"\n🔍 TO ANALYZE YOUR ISSUE:")
    print(f"   1. Run this script: python notebooks/debug_slide_generation.py")
    print(f"   2. Check the console output for any ERROR messages")
    print(f"   3. Check slide_debug.log for detailed step-by-step analysis")
    print(f"   4. Open the generated .pptx file to see if slides are actually empty")
    print(f"   5. Share the ERROR messages with me if any are found!")

if __name__ == '__main__':
    main() 
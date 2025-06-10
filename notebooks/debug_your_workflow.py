#!/usr/bin/env python3
"""
Debug script that exactly mimics your example_analysis.py workflow with detailed logging.
Use this to debug your specific use case.
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import logging
from datetime import datetime

# Set up detailed logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(levelname)s: %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('your_workflow_debug.log', mode='w')
    ]
)

from rca_package.debug_slide_generation import DebugSlideLayouts, debug_dual_output

def main():
    """Mimic your exact example_analysis.py workflow with debugging."""
    
    print("🔍 DEBUGGING YOUR EXACT WORKFLOW")
    logging.info("=== YOUR WORKFLOW DEBUG SESSION STARTED ===")
    
    # Create test data that matches your structure
    # REPLACE THIS with your actual data structure
    test_data = {
        'conversion_rate_pct': np.array([0.12, 0.08, 0.11, 0.13, 0.10]),
        'avg_order_value': np.array([75.0, 65.0, 80.0, 85.0, 72.0]),
        'bounce_rate_pct': np.array([0.35, 0.45, 0.32, 0.28, 0.34]),
        'page_load_time': np.array([2.4, 3.8, 2.2, 1.9, 2.5]),
    }
    
    regions = ["Global", "North America", "Europe", "Asia", "Latin America"]
    df_technical = pd.DataFrame(test_data, index=regions)
    
    print("✅ Test data created matching your structure")
    logging.info(f"DataFrame shape: {df_technical.shape}")
    logging.info(f"DataFrame columns: {list(df_technical.columns)}")
    logging.info(f"DataFrame index: {list(df_technical.index)}")
    
    # Create debug slides (like your example_analysis.py)
    slides = DebugSlideLayouts()
    
    # Test case that matches your example_analysis.py pattern
    print(f"\n🔍 Testing slide creation that matches your pattern...")
    
    @debug_dual_output(console=True, slide=True, slide_builder=slides, 
                      layout_type='text_tables', show_figures=False)
    def create_analysis_slide_like_yours():
        """This should match your exact slide creation pattern."""
        
        logging.info("Creating slide content like your example_analysis.py")
        
        # Create content structure that matches your pattern
        slide_info = {
            'title': 'Test Analysis - Your Pattern',
            'template_text': """
Analysis results for regional performance:

Key findings:
• {{ region }} shows {{ trend }} performance 
• Conversion rate is {{ direction }} by {{ pct }}%
• Action required: {{ action }}

Data table:

{{ results_table }}

This analysis was generated automatically from your workflow pattern.
            """,
            'dfs': {'results_table': df_technical},
            'template_params': {
                'region': 'North America',
                'trend': 'concerning',
                'direction': 'lower',
                'pct': '33',
                'action': 'immediate investigation'
            },
            'layout_type': 'text_tables'
        }
        
        # Return SlideContent object like your system does
        from rca_package.make_slides import SlideContent
        
        return SlideContent(
            title=slide_info['title'],
            text_template=slide_info['template_text'],
            dfs=slide_info['dfs'],
            template_params=slide_info['template_params']
        )
    
    try:
        logging.info("🚀 Executing slide creation...")
        content, results = create_analysis_slide_like_yours()
        print("✅ Slide creation completed successfully")
        
        # Log what was actually created
        logging.info("📋 Slide creation results:")
        logging.info(f"   Content title: {content.title}")
        logging.info(f"   Console result: {'Yes' if 'console' in results else 'No'}")
        logging.info(f"   Slide result: {'Yes' if 'slide' in results else 'No'}")
        
        if 'console' in results:
            console_output = results['console']
            logging.info(f"   Console output length: {len(console_output)} chars")
            logging.info(f"   Console preview: {console_output[:200]}...")
            
    except Exception as e:
        print(f"❌ Slide creation failed: {e}")
        logging.error(f"Slide creation failed: {e}")
        import traceback
        logging.error(f"Full traceback: {traceback.format_exc()}")
        return
    
    # Save and analyze results
    print(f"\n💾 Saving presentation...")
    try:
        filename = slides.save('your_workflow_debug.pptx')
        print(f"✅ Presentation saved: {filename}")
        
        if os.path.exists(filename):
            file_size = os.path.getsize(filename) / 1024  # KB
            print(f"   File size: {file_size:.1f} KB")
            print(f"   Total slides: {slides.slide_count}")
            
            # Compare with our successful test
            print(f"\n📊 COMPARISON:")
            print(f"   Your workflow: {file_size:.1f} KB, {slides.slide_count} slides")
            print(f"   Our test: 159.2 KB, 4 slides")
            
            if file_size < 50:
                print(f"   ⚠️  Your file is suspiciously small - may indicate empty slides")
            else:
                print(f"   ✅ File size looks normal")
        
    except Exception as e:
        print(f"❌ Save failed: {e}")
        logging.error(f"Save failed: {e}")
    
    print(f"\n🔍 DEBUGGING GUIDANCE:")
    print(f"   1. Open 'your_workflow_debug.pptx' and check if slides are empty")
    print(f"   2. Check 'your_workflow_debug.log' for detailed analysis")
    print(f"   3. If slides are empty, the issue is in the SlideContent parameters")
    print(f"   4. If slides have content, the issue is elsewhere in your workflow")
    print(f"   5. Compare this with your actual code - what's different?")

if __name__ == '__main__':
    main() 
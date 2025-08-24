# %%
"""
Interactive Oaxaca-Blinder Visualization Notebook

Run this file in VS Code or any editor that supports "# %%" cell markers
to interactively explore the visualization capabilities.

Each cell can be run independently to see the results and charts.
"""

# %%
# Setup and imports
import sys
import os
import pandas as pd

# Configure matplotlib for interactive use
import matplotlib
try:
    # Try to use interactive backend
    matplotlib.use('TkAgg')
except:
    try:
        # Fallback to Qt
        matplotlib.use('Qt5Agg')
    except:
        # Last resort - inline for notebooks
        matplotlib.use('inline')

import matplotlib.pyplot as plt

# Add project to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from rca_package.oaxaca_blinder import run_oaxaca_analysis, auto_find_best_slice
from rca_package.viz_oaxaca import (
    show_storyboard, 
    show_comprehensive_analysis,
    plot_waterfall,
    plot_contributions_by_category,
    plot_pooled_vs_common_mix,
    plot_despite_due_to_rates,
    plot_exec_summary_panel
)

print("✅ Setup complete! Ready for interactive visualization.")

# %%
# Cell 1: Performance-Driven Example
print("🎯 PERFORMANCE-DRIVEN SCENARIO")
print("=" * 40)

# Create data where one region just executes worse across all products
perf_df = pd.DataFrame([
    {"region": "UNDERPERFORM", "product": "Premium", "wins": 300, "total": 1000},  # 30%
    {"region": "UNDERPERFORM", "product": "Standard", "wins": 150, "total": 1000}, # 15%
    {"region": "UNDERPERFORM", "product": "Basic", "wins": 100, "total": 1000},    # 10%
    
    {"region": "BASELINE", "product": "Premium", "wins": 450, "total": 1000},      # 45%
    {"region": "BASELINE", "product": "Standard", "wins": 300, "total": 1000},     # 30%
    {"region": "BASELINE", "product": "Basic", "wins": 200, "total": 1000},        # 20%
])

perf_result = run_oaxaca_analysis(
    df=perf_df,
    region_column="region",
    numerator_column="wins", 
    denominator_column="total",
    category_columns=["product"]
)

# Show executive narrative
exec_report = perf_result.generate_executive_report("UNDERPERFORM")
print(f"Executive narrative: {exec_report['narrative_text']}")

# Show automatic storyboard (picks best 2 charts + captions)
show_storyboard(perf_result, "UNDERPERFORM", max_charts=2)

# %%
# Cell 2: Composition-Driven Example  
print("📊 COMPOSITION-DRIVEN SCENARIO")
print("=" * 40)

# Same execution rates, but terrible product mix
comp_df = pd.DataFrame([
    # BAD_MIX: Great at premium (60%), bad at basic (20%), but sells mostly basic
    {"region": "BAD_MIX", "product": "Premium", "wins": 600, "total": 1000},   # 60% rate, 10% mix
    {"region": "BAD_MIX", "product": "Basic", "wins": 180, "total": 900},      # 20% rate, 90% mix
    
    # GOOD_MIX: Same rates (60%/20%) but sells mostly premium
    {"region": "GOOD_MIX", "product": "Premium", "wins": 540, "total": 900},   # 60% rate, 90% mix  
    {"region": "GOOD_MIX", "product": "Basic", "wins": 20, "total": 100},      # 20% rate, 10% mix
])

comp_result = run_oaxaca_analysis(
    df=comp_df,
    region_column="region",
    numerator_column="wins",
    denominator_column="total",
    category_columns=["product"]
)

# Show narrative and storyboard
exec_report = comp_result.generate_executive_report("BAD_MIX") 
print(f"Executive narrative: {exec_report['narrative_text']}")

show_storyboard(comp_result, "BAD_MIX", max_charts=2)

# %%
# Cell 3: Simpson's Paradox Example
print("🔀 SIMPSON'S PARADOX SCENARIO") 
print("=" * 40)

try:
    # Classic paradox: worse overall but better in every segment
    simpson_df = pd.DataFrame([
        # SIMPSON: Better in both High (45% vs 40%) and Low (25% vs 20%) but worse overall
        {"region": "SIMPSON", "product": "A", "vertical": "High", "wins": 45, "total": 100},   # 45%, 10% mix
        {"region": "SIMPSON", "product": "A", "vertical": "Low", "wins": 225, "total": 900},   # 25%, 90% mix
        
        # NORMAL: Worse in both segments but better overall due to better mix
        {"region": "NORMAL", "product": "A", "vertical": "High", "wins": 360, "total": 900},   # 40%, 90% mix  
        {"region": "NORMAL", "product": "A", "vertical": "Low", "wins": 20, "total": 100},     # 20%, 10% mix
    ])

    print("📊 Running Simpson's paradox analysis...")
    simpson_result = run_oaxaca_analysis(
        df=simpson_df,
        region_column="region",
        numerator_column="wins",
        denominator_column="total", 
        category_columns=["product"],
        subcategory_columns=["vertical"],
        detect_edge_cases=True
    )

    print(f"✅ Analysis complete!")
    print(f"Paradox detected: {simpson_result.paradox_report.paradox_detected}")
    if simpson_result.paradox_report.paradox_detected:
        print(f"Affected regions: {simpson_result.paradox_report.affected_regions}")

    # Show narrative and storyboard
    exec_report = simpson_result.generate_executive_report("SIMPSON")
    print(f"Executive narrative: {exec_report['narrative_text']}")

    if simpson_result.paradox_report.paradox_detected and "SIMPSON" in simpson_result.narrative_decisions:
        print("\n📈 Generating storyboard...")
        show_storyboard(simpson_result, "SIMPSON", max_charts=2)
        print("✅ Storyboard complete!")
    else:
        print("⚠️ Storyboard skipped - paradox not detected or region not found")
        
except Exception as e:
    print(f"❌ Error in Simpson's paradox analysis: {e}")
    import traceback
    traceback.print_exc()
    print("\nℹ️ This might be due to matplotlib backend issues in interactive mode.")
    print("Try running the cells individually or check your matplotlib installation.")

# %%
# Cell 4: Individual Chart Testing
print("📈 INDIVIDUAL CHART GALLERY")
print("=" * 40)

try:
    # Use the performance result for individual chart examples
    result = perf_result
    region = "UNDERPERFORM"

    # Waterfall Chart
    print("1. Waterfall decomposition:")
    fig1, ax1 = plot_waterfall(result, region)
    plt.show()
    print("✅ Waterfall chart displayed")

except Exception as e:
    print(f"❌ Error with waterfall chart: {e}")

# %%
try:
    # Contributions by category 
    print("2. Category contributions (stacked):")
    fig2, ax2 = plot_contributions_by_category(result, region, top_n=5)
    plt.show()
    print("✅ Contributions chart displayed")
    
except Exception as e:
    print(f"❌ Error with contributions chart: {e}")

# %%
try:
    # Executive summary panel
    print("3. Executive summary panel:")
    fig3, (ax3a, ax3b) = plot_exec_summary_panel(result, region)
    plt.show()
    print("✅ Executive summary displayed")
    
except Exception as e:
    print(f"❌ Error with executive summary: {e}")

# %%
# Cell 5: Auto Slice Selection + Visualization
print("🔍 AUTO SLICE SELECTION + VISUALIZATION")
print("=" * 40)

# Complex multi-dimensional data
multi_dim_df = pd.DataFrame([
    # WEST region: Strong in Premium+Direct, weak elsewhere
    {"region": "WEST", "product": "Premium", "channel": "Direct", "wins": 500, "total": 1000},
    {"region": "WEST", "product": "Premium", "channel": "Partner", "wins": 200, "total": 1000},
    {"region": "WEST", "product": "Standard", "channel": "Direct", "wins": 150, "total": 1000},
    {"region": "WEST", "product": "Standard", "channel": "Partner", "wins": 100, "total": 1000},
    
    # EAST region: Baseline performance
    {"region": "EAST", "product": "Premium", "channel": "Direct", "wins": 350, "total": 1000},
    {"region": "EAST", "product": "Premium", "channel": "Partner", "wins": 300, "total": 1000},
    {"region": "EAST", "product": "Standard", "channel": "Direct", "wins": 250, "total": 1000},
    {"region": "EAST", "product": "Standard", "channel": "Partner", "wins": 200, "total": 1000},
])

# Test different cuts
candidate_cuts = [
    ["product"],
    ["channel"], 
    ["product", "channel"]
]

# Auto-find best slice for WEST
best = auto_find_best_slice(
    df=multi_dim_df,
    candidate_cuts=candidate_cuts,
    focus_region="WEST",
    numerator_column="wins",
    denominator_column="total"
)

print(f"🎯 Best cut for WEST: {best.best_categories}")
print(f"📊 Score: {best.score:.3f}")
print(f"🔢 Score breakdown: {best.score_breakdown}")

# Show storyboard for the winning cut
print(f"\n📋 Executive storyboard for optimal cut:")
show_storyboard(best.analysis_result, "WEST", max_charts=2)

# %%
# Cell 6: Portfolio Mode (Auto-pick worst region)
print("🏢 PORTFOLIO MODE - AUTO-PICK WORST REGION")
print("=" * 40)

portfolio_best = auto_find_best_slice(
    df=multi_dim_df,
    candidate_cuts=candidate_cuts,
    portfolio_mode=True,  # Auto-select most problematic region
    numerator_column="wins",
    denominator_column="total"
)

print(f"🎯 Auto-selected focus region: {portfolio_best.focus_region}")
print(f"📊 Best cut: {portfolio_best.best_categories}")
print(f"🔢 Score: {portfolio_best.score:.3f}")

show_storyboard(portfolio_best.analysis_result, portfolio_best.focus_region, max_charts=1)

# %%
# Cell 7: Comprehensive Analysis (All Charts)
print("🌟 COMPREHENSIVE ANALYSIS - ALL VISUALIZATIONS") 
print("=" * 40)

# Use composition example for comprehensive view
show_comprehensive_analysis(comp_result, "BAD_MIX")

# %%
# Cell 8: Custom Scenarios
print("⚙️  CUSTOM SCENARIO BUILDER")
print("=" * 40)

# Create your own scenario here and test
custom_df = pd.DataFrame([
    # TODO: Add your own data
    # {"region": "YOUR_REGION", "product": "A", "wins": X, "total": Y},
])

# Uncomment to run:
# custom_result = run_oaxaca_analysis(...)
# show_storyboard(custom_result, "YOUR_REGION")

print("💡 Tip: Modify the custom_df above to test your own scenarios!")
print("📊 Available functions:")
print("   - show_storyboard(result, region)")
print("   - show_comprehensive_analysis(result, region)")  
print("   - auto_find_best_slice(df, candidate_cuts=...)")
print("   - Individual plot functions: plot_waterfall, plot_contributions_by_category, etc.")

# %%
print("✅ Interactive visualization notebook complete!")
print("🎯 Use the cells above to explore different scenarios and visualizations.")
print("📊 Modify the data in any cell to test your own cases!")

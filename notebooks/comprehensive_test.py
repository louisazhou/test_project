#!/usr/bin/env python3

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from rca_package.make_slides import SlideLayouts, SlideContent, dual_output
import numpy as np

def create_bar_chart(df: pd.DataFrame, title: str) -> plt.Figure:
    """Create a bar chart for metrics."""
    # Convert percentage strings to float values
    data = df.copy()
    for col in ['Conversion Rate', 'Page Load Time', 'Bounce Rate']:
        if col in data.columns:
            data[col] = data[col].str.rstrip('%').astype('float') / 100
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Set bar positions
    x = np.arange(len(data['Region']))
    width = 0.25
    
    # Create bars
    if 'Conversion Rate' in data.columns:
        ax.bar(x - width, data['Conversion Rate'], width, label='Conversion Rate', color='#2ecc71')
    if 'Page Load Time' in data.columns:
        ax.bar(x, data['Page Load Time'], width, label='Page Load Time', color='#e74c3c')
    if 'Bounce Rate' in data.columns:
        ax.bar(x + width, data['Bounce Rate'], width, label='Bounce Rate', color='#3498db')
    
    # Customize chart
    ax.set_ylabel('Change from Baseline')
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(data['Region'], rotation=45)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Format y-axis as percentages
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
    
    # Adjust layout
    plt.tight_layout()
    
    return fig

def create_trend_chart(df: pd.DataFrame, title: str) -> plt.Figure:
    """Create a line chart for trends."""
    # Convert percentage strings to float values
    data = df.copy()
    for col in ['NA Growth', 'EU Growth', 'APAC Growth']:
        if col in data.columns:
            data[col] = data[col].str.rstrip('%').astype('float') / 100
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot lines
    if 'NA Growth' in data.columns:
        ax.plot(data['Week'], data['NA Growth'], marker='o', label='North America', color='#2ecc71')
    if 'EU Growth' in data.columns:
        ax.plot(data['Week'], data['EU Growth'], marker='s', label='Europe', color='#e74c3c')
    if 'APAC Growth' in data.columns:
        ax.plot(data['Week'], data['APAC Growth'], marker='^', label='Asia Pacific', color='#3498db')
    
    # Customize chart
    ax.set_ylabel('Growth Rate')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Format y-axis as percentages
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
    
    # Rotate x-axis labels
    plt.xticks(rotation=45)
    
    # Adjust layout
    plt.tight_layout()
    
    return fig

def create_giant_complex_chart(df: pd.DataFrame, title: str) -> plt.Figure:
    """Create a giant, complex chart that should take up most/all of a slide."""
    # Create a very large figure - much bigger than normal
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))  # Giant size: 20x16 inches
    
    # Convert percentage strings to float values for all dataframes
    data = df.copy()
    for col in ['Conversion Rate', 'Page Load Time', 'Bounce Rate']:
        if col in data.columns:
            data[col] = data[col].str.rstrip('%').astype('float') / 100
    
    # Subplot 1: Bar chart
    x = np.arange(len(data['Region']))
    width = 0.25
    if 'Conversion Rate' in data.columns:
        ax1.bar(x - width, data['Conversion Rate'], width, label='Conversion Rate', color='#2ecc71')
    if 'Page Load Time' in data.columns:
        ax1.bar(x, data['Page Load Time'], width, label='Page Load Time', color='#e74c3c')
    if 'Bounce Rate' in data.columns:
        ax1.bar(x + width, data['Bounce Rate'], width, label='Bounce Rate', color='#3498db')
    
    ax1.set_ylabel('Change from Baseline')
    ax1.set_title('Regional Performance Metrics')
    ax1.set_xticks(x)
    ax1.set_xticklabels(data['Region'], rotation=45)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
    
    # Subplot 2: Scatter plot with trend lines
    if len(data.columns) >= 3:
        ax2.scatter(data['Conversion Rate'], data['Page Load Time'], s=200, alpha=0.7, c='red')
        ax2.set_xlabel('Conversion Rate Change')
        ax2.set_ylabel('Page Load Time Change')
        ax2.set_title('Performance Correlation Analysis')
        ax2.grid(True, alpha=0.3)
        # Add trend line
        z = np.polyfit(data['Conversion Rate'].astype(float), data['Page Load Time'].astype(float), 1)
        p = np.poly1d(z)
        ax2.plot(data['Conversion Rate'], p(data['Conversion Rate']), "r--", alpha=0.8)
    
    # Subplot 3: Heatmap-style visualization
    metrics_matrix = data[['Conversion Rate', 'Page Load Time', 'Bounce Rate']].values
    im = ax3.imshow(metrics_matrix, cmap='RdYlGn_r', aspect='auto')
    ax3.set_xticks(range(len(['Conversion Rate', 'Page Load Time', 'Bounce Rate'])))
    ax3.set_xticklabels(['Conversion Rate', 'Page Load Time', 'Bounce Rate'], rotation=45)
    ax3.set_yticks(range(len(data['Region'])))
    ax3.set_yticklabels(data['Region'])
    ax3.set_title('Performance Heatmap')
    plt.colorbar(im, ax=ax3)
    
    # Subplot 4: Pie chart of revenue impact
    if 'Revenue Impact' in data.columns:
        revenue_values = []
        labels = []
        for i, region in enumerate(data['Region']):
            rev_str = data.iloc[i]['Revenue Impact']
            # Extract numeric value from strings like '$-2.5M', '$+4.2M'
            rev_val = float(rev_str.replace('$', '').replace('M', '').replace('+', ''))
            if rev_val != 0:  # Only include non-zero values in pie chart
                revenue_values.append(abs(rev_val))  # Use absolute values for pie chart
                labels.append(f"{region} ({rev_str})")
        
        if revenue_values:
            colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99', '#ff99cc', '#c2c2f0']
            ax4.pie(revenue_values, labels=labels, autopct='%1.1f%%', colors=colors[:len(revenue_values)])
            ax4.set_title('Revenue Impact Distribution')
    
    # Add overall title
    fig.suptitle(title, fontsize=24, fontweight='bold', y=0.98)
    
    # Adjust layout with extra padding
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    return fig

def main():
    """Comprehensive test of slide layout with long text, multiple tables, and figures."""
    
    # Create more comprehensive test data
    metrics_df = pd.DataFrame({
        'Region': ['North America', 'Europe', 'Asia', 'Latin America', 'Middle East', 'Africa'],
        'Conversion Rate': ['-33.33%', '0%', '+15%', '-5%', '+8%', '-12%'],
        'Page Load Time': ['+58.3%', '+5%', '-20%', '+25%', '+15%', '+40%'],
        'Bounce Rate': ['+45%', '+8%', '-10%', '+15%', '+5%', '+25%'],
        'Revenue Impact': ['$-2.5M', '$0.1M', '$+4.2M', '$-0.8M', '$+1.1M', '$-1.2M']
    })
    
    trends_df = pd.DataFrame({
        'Week': ['Week 1', 'Week 2', 'Week 3', 'Week 4', 'Week 5', 'Week 6'],
        'NA Growth': ['-5%', '-8%', '-12%', '-15%', '-18%', '-20%'],
        'EU Growth': ['-2%', '-3%', '-4%', '-5%', '-4%', '-3%'],
        'APAC Growth': ['+8%', '+10%', '+5%', '+3%', '+6%', '+9%']
    })
    
    detailed_metrics_df = pd.DataFrame({
        'Metric': ['Session Duration', 'Pages per Session', 'New User Rate', 'Return Rate', 'Mobile Traffic %'],
        'North America': ['2.3 min', '3.2 pages', '65%', '35%', '72%'],
        'Europe': ['2.8 min', '4.1 pages', '58%', '42%', '68%'],
        'Asia': ['3.1 min', '4.8 pages', '71%', '29%', '85%'],
        'Global Average': ['2.7 min', '4.0 pages', '64%', '36%', '75%']
    })
    
    # Very long text content for testing
    very_long_intro = """Regional Performance Analysis - Q4 2023 Deep Dive Report
    
Executive Summary:
Our comprehensive analysis of Q4 2023 performance data reveals significant variations in key performance indicators across all major geographical regions. This detailed examination encompasses conversion rates, technical performance metrics, user engagement patterns, and revenue impact assessments that require immediate strategic attention and coordinated response efforts.

Critical Findings Overview:
The data presents a complex picture of regional performance disparities that extend beyond simple geographical differences. Our analysis indicates systemic issues in certain markets that correlate with infrastructure limitations, user behavior patterns, and competitive pressures that vary significantly by region.

1. North America Performance Crisis:
- Conversion rates have experienced a catastrophic decline, showing a 33.33% decrease compared to the global baseline performance metrics
- Page load times have deteriorated significantly, registering 58.3% higher than our established global baseline standards
- This performance degradation strongly correlates with decreased conversion rates, indicating a direct causal relationship
- Bounce rates have increased by 45%, representing the highest negative performance indicator across all measured regions
- User session duration has decreased by 23%, indicating reduced engagement and potential user experience issues
- Mobile traffic performance shows particular degradation, with 18% slower load times on mobile devices
- Customer acquisition costs have increased by 28% due to reduced conversion efficiency
- Revenue impact is estimated at $2.5M negative for the quarter, representing the largest regional revenue loss

Current Regional Metrics Overview:
{{ metrics_table }}

2. European Market Stability with Underlying Concerns:
- While conversion rates appear stable at surface level, detailed analysis reveals concerning micro-trends
- New user acquisition has decreased by 10%, indicating potential market saturation or competitive pressure
- Week-over-week analysis demonstrates a consistent decline pattern in engagement metrics over the past six weeks
- Technical performance metrics remain within acceptable operational parameters, suggesting the issues are not infrastructure-related
- Session quality metrics show gradual degradation, with users completing fewer desired actions per session
- Customer lifetime value calculations show a 12% decrease compared to previous quarter performance
- Market penetration rates have plateaued, indicating the need for strategic repositioning efforts

3. Asian Pacific Market Exceptional Performance:
- Demonstrates the strongest performance in new user acquisition metrics, achieving 30% above global baseline averages
- Page load times consistently achieve the fastest performance across all measured regions and device types
- User engagement metrics exceed global averages by 25%, indicating strong product-market fit
- Revenue per user shows consistent growth trends, with 18% increase quarter-over-quarter
- Mobile-first user behavior patterns align well with our technical infrastructure optimization efforts
- However, recent trend analysis indicates potential stability concerns that require proactive monitoring and intervention
- Infrastructure scaling requirements are projected to increase by 40% to maintain current performance levels
- Market expansion opportunities exist but require additional investment in localization efforts"""

    europe_asia_detailed = """Detailed Regional Analysis - European and Asian Markets:

European Market Dynamics Deep Dive:
The European market presents a paradoxical situation where surface-level stability masks underlying structural challenges that require strategic intervention. Our comprehensive analysis reveals that while immediate performance indicators remain within acceptable parameters, longer-term trend analysis suggests the beginning of a concerning decline pattern.

Key Performance Indicators Analysis:
- Conversion funnel analysis shows micro-interruptions at the consideration stage, with 8% more users dropping off before final purchase decisions
- A/B testing results indicate that European users respond differently to our current messaging and visual design elements
- Seasonal adjustment models predict continued decline through Q1 2024 unless corrective measures are implemented
- Competitive analysis reveals increased market pressure from 3 major regional competitors who have enhanced their offerings
- Regulatory compliance costs have increased by 15%, impacting overall profitability metrics

Weekly Performance Trends Analysis:
{{ trends_table }}

Asian Pacific Market Success Factors:
The exceptional performance in the APAC region provides valuable insights into successful market penetration strategies and technical optimization approaches that could potentially be replicated in other geographical markets.

Success Factor Analysis:
- Mobile-optimized user experience design aligns perfectly with regional user behavior patterns and device preferences
- Localized content strategies show 45% higher engagement rates compared to standard global content approaches
- Payment method optimization for regional preferences results in 23% fewer cart abandonment instances
- Customer support response times average 40% faster than global standards, contributing to higher satisfaction scores
- Social media integration strategies leverage region-specific platforms, resulting in 35% higher organic traffic growth

Infrastructure and Technical Considerations:
- Content delivery network optimization specifically for Asian markets reduces load times by average of 1.2 seconds
- Database query optimization for common Asian market use cases improves overall system performance by 18%
- Redundancy systems and failover mechanisms ensure 99.97% uptime, exceeding regional competitor performance
- Security implementations meet all regional regulatory requirements while maintaining optimal user experience flow

Risk Assessment and Monitoring Requirements:
Despite current success, several risk factors require continuous monitoring and proactive management strategies to maintain competitive advantage and market position."""

    recommendations_comprehensive = """Strategic Recommendations and Implementation Roadmap:

4. Latin American Market Development Opportunities:
The Latin American market presents mixed performance indicators that suggest significant potential for improvement through targeted optimization efforts and strategic regional adaptations.

Current Performance Assessment:
- Mixed performance indicators reflect the diverse nature of the Latin American market landscape
- Page load time optimization efforts show moderate success but require additional infrastructure investment
- New user acquisition rates remain steady but fall below target performance levels established in strategic planning
- Mobile traffic dominance (78%) requires specific optimization approaches tailored to regional device capabilities
- Payment method localization shows promise, with 19% improvement in conversion rates after recent updates

Market-Specific Strategic Recommendations:

Infrastructure and Technical Optimization:
1. Immediate Technical Optimization for North American Markets:
   - Deploy emergency CDN optimization specifically targeting major metropolitan areas experiencing the worst performance
   - Implement advanced caching strategies to reduce server response times by target of 40%
   - Establish dedicated server infrastructure for high-traffic periods to prevent performance degradation
   - Conduct comprehensive third-party service audit to identify and eliminate performance bottlenecks
   - Timeline: Implementation within 4-6 weeks, with measurable improvements expected within 8 weeks

2. Enhanced Monitoring and Analytics for European Markets:
   - Deploy advanced user behavior tracking to identify specific points of engagement decline
   - Implement predictive analytics models to anticipate performance issues before they impact user experience
   - Establish region-specific A/B testing protocols to optimize content and design for European preferences
   - Create automated alert systems for early detection of declining trend patterns
   - Timeline: Full implementation within 6-8 weeks, with initial insights available within 3 weeks

3. Infrastructure Scaling Preparation for Asian Market Continued Growth:
   - Conduct capacity planning analysis to ensure infrastructure can support projected 40% traffic growth
   - Implement auto-scaling solutions to handle traffic spikes during peak regional usage periods
   - Establish redundant systems across multiple geographical locations to ensure continued reliability
   - Develop disaster recovery protocols specific to high-performance requirements of Asian markets
   - Timeline: Infrastructure scaling complete within 10-12 weeks, with phased implementation approach

4. Targeted Performance Improvements for Latin American Services:
   - Optimize content delivery for diverse internet infrastructure capabilities across the region
   - Implement progressive loading strategies to improve perceived performance on slower connections
   - Establish regional content caching to reduce latency for frequently accessed materials
   - Develop offline-capable features to improve user experience during connectivity interruptions
   - Timeline: Initial optimizations within 6 weeks, full regional optimization within 12 weeks

Long-term Strategic Initiatives:
- Establish quarterly performance review protocols with specific regional benchmarks and improvement targets
- Develop cross-regional knowledge sharing programs to replicate successful strategies across all markets
- Create unified dashboard systems for real-time monitoring of all key performance indicators across regions
- Implement customer feedback integration systems to ensure user voice drives optimization priorities

Financial Impact Projections:
- North American optimization efforts projected to recover $1.8M of the $2.5M quarterly revenue loss
- European market stabilization expected to prevent estimated $1.2M in potential revenue decline
- Asian market scaling investments projected to generate additional $3.2M in quarterly revenue
- Latin American optimizations expected to unlock $900K in additional quarterly revenue potential

Risk Mitigation Strategies:
- Establish contingency protocols for each regional market to respond quickly to performance degradation
- Create cross-regional backup systems to maintain service availability during localized infrastructure issues
- Develop competitive monitoring systems to quickly respond to market changes and competitor actions
- Implement customer communication protocols to maintain transparency during optimization periods

This comprehensive analysis combines both current metrics and trend data to provide a complete understanding of our global operations, enabling data-driven decision making for sustainable long-term growth across all regional markets."""

    # Create slide deck
    slides = SlideLayouts()
    
    print("Creating comprehensive test presentation using uniform @dual_output decorator pattern...")
    print("Testing various scenarios:")
    print("1. Very long text with single table")
    print("2. Long text with table") 
    print("3. Very long text with multiple tables")
    print("4. Text with tables and multiple figures")
    print("5. Multiple tables with minimal text")
    print("6. Pure text overflow test")
    print("7. Giant figure with text and tables (tests empty slide bug)")
    
    # Test 1: Very long text with table 
    @dual_output(console=False, slide=True, slide_builder=slides, layout_type='text_tables', show_figures=False)
    def create_comprehensive_analysis_part1():
        return SlideContent(
            title="Comprehensive Regional Analysis - Executive Summary",
            text_template=very_long_intro,
            dfs={'metrics_table': metrics_df}
        )
    
    # Test 2: Long text with table 
    @dual_output(console=False, slide=True, slide_builder=slides, layout_type='text_tables', show_figures=False)
    def create_comprehensive_analysis_part2():
        return SlideContent(
            title="Detailed Market Analysis - Europe and Asia",
            text_template=europe_asia_detailed,
            dfs={'trends_table': trends_df}
        )
    
    # Test 3: Very long text with multiple tables 
    @dual_output(console=False, slide=True, slide_builder=slides, layout_type='text_tables', show_figures=False)
    def create_comprehensive_analysis_part3():
        return SlideContent(
            title="Strategic Recommendations and Implementation",
            text_template=recommendations_comprehensive + """

Primary Metrics Analysis:
{{ metrics_table }}

Detailed Performance Breakdown:
{{ detailed_metrics }}

These comprehensive data points drive our implementation roadmap.""",
            dfs={
                'metrics_table': metrics_df,
                'detailed_metrics': detailed_metrics_df
            }
        )
    
    # Test 4: Text with tables and figures 
    @dual_output(console=False, slide=True, slide_builder=slides, layout_type='text_tables_figure', show_figures=False)
    def create_comprehensive_analysis_part4():
        return SlideContent(
            title="Performance Visualization and Analytics",
            text_template="""Comprehensive visual analysis of regional performance data with key insights and actionable recommendations based on quantitative analysis.

Summary metrics for visualization:
{{ summary_metrics }}

The charts provide additional perspective on these performance patterns.""",
            dfs={'summary_metrics': metrics_df},
            figure_generators=[
                {
                    'title_suffix': 'Regional Metrics Overview',
                    'params': {
                        'df': metrics_df,
                        'title': 'Key Performance Metrics by Region'
                    },
                    'function': create_bar_chart
                },
                {
                    'title_suffix': 'Trend Analysis',
                    'params': {
                        'df': trends_df,
                        'title': 'Week-over-Week Performance Trends'
                    },
                    'function': create_trend_chart
                }
            ]
        )
    
    # Test 5: Tables only 
    @dual_output(console=False, slide=True, slide_builder=slides, layout_type='text_tables', show_figures=False)
    def create_comprehensive_analysis_part5():
        return SlideContent(
            title="Detailed Metrics Tables",
            text_template="""Comprehensive data tables showing all performance metrics:

Primary Regional Performance Metrics:
{{ primary_metrics }}

Detailed Performance Indicators by Region:
{{ detailed_metrics }}

Weekly Trend Analysis Data:
{{ trend_data }}

These comprehensive tables provide granular insights into regional performance patterns and trends that inform strategic decision making.""",
            dfs={
                'primary_metrics': metrics_df,
                'detailed_metrics': detailed_metrics_df,
                'trend_data': trends_df
            }
        )
    
    # Test 6: Pure text overflow test 
    @dual_output(console=False, slide=True, slide_builder=slides, layout_type='text', show_figures=True)
    def create_pure_text_overflow_test():
        return SlideContent(
            title="Pure Text Overflow Test",
            text_template=very_long_intro + "\n\n" + europe_asia_detailed + "\n\n" + recommendations_comprehensive,
            dfs={}  # No tables - pure text
        )
    
    # Test 7: Giant figure with text and tables - this should expose the empty slide issue
    @dual_output(console=False, slide=True, slide_builder=slides, layout_type='text_tables_figure', show_figures=False)
    def create_giant_figure_test():
        return SlideContent(
            title="Giant Figure with Text and Tables Test",
            text_template="""This slide contains a giant figure that should take up most of the slide space, along with some explanatory text and data tables.

Key findings from the comprehensive dashboard analysis:
- Multi-dimensional performance view reveals complex regional patterns
- Correlation analysis shows strong negative relationship between conversion rates and page load times
- Heatmap visualization clearly identifies problem areas requiring immediate intervention
- Revenue impact distribution demonstrates the financial magnitude of regional performance variations

Summary metrics for context:
{{ summary_metrics }}

Detailed breakdown for analysis:
{{ detailed_metrics }}

This scenario tests whether the slide generation can handle extremely large figures while still including text and table content.""",
            dfs={
                'summary_metrics': metrics_df,
                'detailed_metrics': detailed_metrics_df
            },
            figure_generators=[
                {
                    'title_suffix': 'Giant Complex Dashboard',
                    'params': {
                        'df': metrics_df,
                        'title': 'Comprehensive Performance Dashboard - All Metrics'
                    },
                    'function': create_giant_complex_chart
                }
            ]
        )
    
    # Execute all tests using the uniform pattern
    print("\n=== Executing Test 1: Very long text with single table ===")
    content1, results1 = create_comprehensive_analysis_part1()
    
    print("\n=== Executing Test 2: Long text with table ===")
    content2, results2 = create_comprehensive_analysis_part2()
    
    print("\n=== Executing Test 3: Very long text with multiple tables ===")
    content3, results3 = create_comprehensive_analysis_part3()
    
    print("\n=== Executing Test 4: Text with tables and multiple figures ===")
    content4, results4 = create_comprehensive_analysis_part4()
    
    print("\n=== Executing Test 5: Multiple tables with minimal text ===")
    content5, results5 = create_comprehensive_analysis_part5()
    
    print("\n=== Executing Test 6: Pure text overflow test ===")
    content6, results6 = create_pure_text_overflow_test()
    
    print("\n=== Executing Test 7: Giant figure with text and tables ===")
    content7, results7 = create_giant_figure_test()
    
    # Save presentation
    slides.save('comprehensive_test.pptx')
    print("\nComprehensive test presentation saved as 'comprehensive_test.pptx'")
    print("This test covers:")
    print("- Very long text content (3000+ words)")
    print("- Multiple tables per slide")
    print("- Mixed content with tables and figures")
    print("- Pure text overflow scenarios")
    print("- Giant figure (20x16 inches) with text and tables")
    print("- Various combinations to test overflow handling")
    print("- Tests for empty slide generation bug")
    print("- Uses uniform @dual_output decorator pattern")

    # File to upload
    file_path = 'comprehensive_test.pptx'
    
    from rca_package.google_drive_utils import upload_to_google_drive

    try:
        # Upload using existing token
        result = upload_to_google_drive(
            file_path=file_path,
            credentials_path='credentials.json',
            token_path="token.json"
        )
        
        print("\nComprehensive test upload successful!")
        print(f"View presentation at: {result['gdrive_url']}")
        if 'folder_url' in result:
            print(f"Folder URL: {result['folder_url']}")
        
        print("\nThis comprehensive test includes:")
        print("- Very long text content (3000+ words)")
        print("- Multiple tables per slide") 
        print("- Mixed content with tables and figures")
        print("- Giant figure (20x16 inches) with text and tables")  
        print("- Various combinations to test overflow handling")
        print("- Tests for empty slide generation bug")
        print("- Tests text_tables layout extensively")
            
    except Exception as e:
        print(f"Error uploading to Google Drive: {e}")

if __name__ == '__main__':
    main() 
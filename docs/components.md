# Component Details

## Hypothesis Types

### Current Implementation
1. **Single Dimensional**
   - Direct metric-to-hypothesis comparison
   - One-to-one relationship between metric and hypothesis values
   - [Implementation](../src/automation_pipeline.py#L385-L432)
   - Example: CLI per AM vs. Closed Won Rate

### Planned Types
1. **Depth Spotter**
   - Status: In Development
   - Purpose: Analyze metric patterns at different organizational levels
   - Features:
     - Multi-level analysis
     - Hierarchical pattern detection
     - Drill-down capabilities

2. **Reason Mix**
   - Status: Planned
   - Purpose: Surface the tagged reasons submitted and show their mix
   - Features:


3. **Benchmark Comparison**
   - Status: Planned
   - Purpose: Compare against industry/historical benchmarks
   - Features:



## Visualization Framework

### Current Implementation
1. **Detailed View**
   - Complete analysis visualization
   - [Implementation](../src/visualization.py)
   - Features:
     - Metric performance plots
     - Hypothesis comparison charts
     - Score component breakdowns
     - Detailed annotations

2. **Succinct View**
   - Condensed summary visualization
   - Features:
     - Key metrics only
     - Simplified visuals
     - Essential annotations

### Future Framework
1. **Modular Components**
   - Reusable visualization elements
   - Customizable layouts
   - Interactive features

2. **Dashboard Integration**
   - Unidash compatibility
   - Real-time updates
   - Interactive filtering

3. **Output Formats**
   - PowerPoint/Google Slides
   - Web dashboards
   - PDF reports
   - API endpoints

## Metric Types

### Current
1. **Operational Metrics**
   - Performance indicators
   - Process metrics
   - Efficiency measures

### Upcoming
1. **Pacing Metrics**
   - Geographic normalization
   - Temporal alignment
   - Progress tracking

### Future
1. **Revenue Metrics**
   - Financial performance
   - Revenue growth
   - Market share

2. **Product Metrics**
   - Usage statistics
   - Feature adoption
   - User engagement

## LLM Integration (Future)

### Agent Framework
1. **ReAct Implementation**
   - Reasoning and acting loops
   - Dynamic hypothesis selection
   - Natural language insights

2. **Function Calling**
   - Automated analysis triggers
   - Dynamic parameter adjustment
   - Result interpretation

3. **Hypothesis Selection**
   - Context-aware selection
   - Confidence scoring
   - Explanation generation

### Integration Points
1. **Data Processing**
   - Input validation
   - Anomaly detection
   - Pattern recognition

2. **Analysis**
   - Hypothesis selection
   - Score interpretation
   - Insight generation

3. **Reporting**
   - Natural language summaries
   - Recommendation generation
   - Action item creation 
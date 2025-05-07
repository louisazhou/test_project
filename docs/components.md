# Component Details

## Hypothesis Types

### Current Implementation
1. **Single Dimensional**
   - Direct metric-to-hypothesis comparison
   - One-to-one relationship between metric and hypothesis values
   - [Implementation](../src/...)
   - Example: CLI per AM vs. Closed Won Rate, when workload (CLI/AM) is too heavy the won rate would be low

### Planned Types
1. **Depth Spotter**
   - Status: In Development
   - Purpose: Analyze metric patterns at a more granular level
   - Features:
     - Drill-down capabilities (e.g. the most lagged)

2. **Reason Mix**
   - Status: Planned
   - Purpose: Surface the tagged reasons (submitted by sales managers)
   - Features:
     - Weighted impact assessment (e.g. compared to the rest of the world, the over-indexed reason in this region cause X more dollar of lost)
     - Recommended action based on the tagged reason type

3. **Benchmark Comparison**
   - Status: Planned
   - Purpose: Compare against historical benchmarks
   - Features:
     - External benchmark integration
     - Peer group analysis
     - Performance gap assessment

4. **History Reference**
   - Status: Planned
   - Purpose: Temporal pattern analysis (e.g. compared to start of quarter, now...)
   - Features:
     - Time series comparison
     - Seasonal adjustment
     - Trend analysis
  
5. 

## Visualization Framework

### Current Implementation
1. **Detailed View**
   - Complete analysis visualization
   - [Implementation](../src/...)
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
  
3. **Reporting Deck**
   - PowerPoint/Google Slides

### Future Framework
1. **Modular Components**
   - Reusable visualization elements
   - Customizable layouts
   - Interactive features

2. **Dashboard Integration**
   - Unidash compatibility
   - Real-time updates
   - Interactive filtering

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
   - Anomaly detection
   - Pattern recognition

2. **Analysis**
   - Hypothesis selection
   - Score interpretation
   - Insight generation

3. **Reporting**
   - Natural language summaries
   - Recommendation generation
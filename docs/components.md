# Component Details

## Core Engines

### Metric Engine
- Processes metric data across dimensions
- Calculates statistical properties
- Detects anomalies using multiple methods
- Manages metric definitions and relationships

### Hypothesis Engine
- Evaluates hypothesis validity against metrics
- Implements scoring algorithm with weighted components
- Ranks hypotheses by explanatory power
- Supports hypothesis configuration via YAML

### Plot Engine
- Generates visualizations for metric performance
- Creates comparison charts for hypotheses
- Implements visualization styles (detailed/succinct)
- Supports static output formats

### Narrative Engine
- Creates natural language explanations
- Generates context-aware insights
- Formats analysis results as readable text
- Supports multiple narrative styles

### Presentation Engine
- Compiles results into structured reports
- Supports PowerPoint and Google Slides output
- Implements templating system for slides
- Manages visual hierarchy and layout

## Hypothesis Types

### Current Implementation
1. **Single Dimensional**
   - Direct metric-to-hypothesis comparison
   - One-to-one relationship between metric and hypothesis values
   - Weighted scoring approach:
     - Direction Alignment (30%)
     - Consistency (30%)
     - Hypothesis Z-score (20%)
     - Explained Ratio (20%)

### Planned Types
1. **Depth Spotter**
   - Status: In Development
   - Purpose: Analyze metric patterns at a more granular level
   - Features:
     - Drill-down capabilities (e.g. the most lagged)

2. **Reason Mix**
   - Status: Planned
   - Purpose: Surface tagged reasons from sales managers
   - Features:
     - Weighted impact assessment
     - Overindexed reason analysis
     - Action recommendations

3. **Benchmark Comparison**
   - Status: Planned
   - Purpose: Compare against historical benchmarks
   - Features:
     - External benchmark integration
     - Peer group analysis
     - Performance gap assessment

4. **History Reference**
   - Status: Planned
   - Purpose: Temporal pattern analysis
   - Features:
     - Time series comparison
     - Seasonal adjustment
     - Trend analysis

## Visualization Framework

### Current Implementation
1. **Detailed View**
   - Complete analysis visualization
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
   - PowerPoint/Google Slides integration
   - Slide templates for different analysis types
   - Visual hierarchy for insights

### Future Framework
1. **Modular Components**
   - Reusable visualization elements
   - Customizable layouts
   - Interactive features

2. **Dashboard Integration**
   - Unidash compatibility


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
   - Anomaly detection enhancement
   - Pattern recognition
   - Anomaly classification

2. **Analysis**
   - Hypothesis selection
   - Score interpretation
   - Insight generation

3. **Reporting**
   - Natural language summaries
   - Recommendation generation
   - Contextual explanations
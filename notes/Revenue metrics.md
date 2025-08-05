1. Run rate – territory vs. finer grains    
    - Leaders track actual vs. goal (run rate) at territory (L4) accurately; at L6/L8/L12 the goal model degrades
    - Gap investigation:
        • Identify which vertical is decelerating
        • Examine vertical mix for environmental factors
        • Use operational proxies (pitch success rate, AM attrition/PTO, performance of large “whale” accounts)     
2. Year-over-year growth rate
    - Leaders want to know why growth is decelerating and if it’s normal (e.g. mature base naturally slows)        
    - Must adjust for last year’s anomalies (an event that depressed volume makes this year’s growth look inflated)
3. Current automation
    - Spreadsheet with red/green flags, z-score anomaly detection
    - Pull in proxy tables (operational metrics, whale accounts) and frontline AM commentary
    - LLM summaries of AM comments
    - Manual validation across spreadsheets and slide assembly at the end

---

# Macro model of budget flow

The macro “budget‐flow” equation is a top‐down way to map advertiser spends all the way to recognized revenue.  By breaking revenue into its upstream levers, you can see exactly where a shift (e.g. a CPM jump) ripples through the system.

---

### The equation

```
(# advertisers)  
× (campaigns per advertiser)  
× (budget per campaign)  
× (delivery rate)  
× (impressions)  
× (CPM)  
± (coupons & other deductions)  
= recognized revenue  
```

…and similarly for performance‐priced ads, substitute the impression x CPM with

```
# conversions × CPC  
# leads       × CPL
```

---

### Component-by-component reasoning

- **# advertisers**
    Number of active buyers on the platform.  If advertisers pause or churn, all downstream revenue collapses.
    
- **Campaigns per advertiser**
    How many simultaneous ad campaigns each runs.  More campaigns → more budget “slots” and more opportunities to spend.
    
- **Budget per campaign**
    The dollar allocation an advertiser sets.  If they cut campaign budgets (e.g. during economic caution), total spend dips.
    
- **Delivery rate**
    Of that budget, the fraction the platform is able to deliver (vs. throttled by pacing controls or audience saturation).  A dropped delivery rate (e.g. audience exhausted) shrinks realized impressions.
    
- **Impressions**
    Total ad views served.  Delivery rate × planned impressions informs volume; if users aren’t online (seasonality) or audience is small, impressions fall.
    
- **CPM (cost per mille)**
    The price advertisers pay per thousand impressions.  Higher CPMs—driven by auction competition or limited inventory—mean the same impression volume generates more revenue, but may also cause advertisers to reduce budgets.
    
- **Coupons & deductions**
    Platform credits, performance rebates, or billing adjustments reduce net revenue.  If you offer big coupons, gross CPM revenue overstates net.
    
- **# conversions × CPC**
    For performance‐priced buying, you’re not paid per impression but per click (CPC) or per action (CPL).  Shifts in click‐through rate, conversion rate, or bids change this line directly.
    
---

### Why this matters for root‐cause analysis

1. **Pinpoint upstream driver**
    If revenue fell, is it because advertisers cut budgets (# advertisers or budget per campaign)?
    
2. **Separate volume vs. price effects**
    Decompose “impressions × CPM” to see if fewer impressions or a lower/higher CPM drove the change.
    
3. **Spot campaign‐level shifts**
    A big promo campaign may boost impressions but at heavy coupon cost—so gross vs. net tells different stories.
    
4. **Bridge to operational metrics**
    If delivery rate dropped, you troubleshoot creative fatigue or auction competitiveness; if CPC soared, you probe bid strategy.
    
# Macro model, simplified

However, the macro model above has too many floating pieces. Simplified revenue-flow could be 

```
Revenue ≈ Advertiser Engagement × Delivery Efficiency × Monetization Yield
```

1. **Advertiser Engagement**
    captures how many budgeted “slots” are live
    - proxy metric: active campaigns × average budget per campaign
    - why it matters: fewer or under-funded campaigns mean less top-of-funnel spend
        
2. **Delivery Efficiency**
    captures how well that budget turns into served impressions
    - proxy metric: impressions / (ad_budget / CPM)
    - why it matters: throttles like pacing controls or audience exhaustion cut volume
        
3. **Monetization Yield**
    captures how much revenue you extract per delivered impression or action
    - proxy metric: effective CPM (net rev / 1000 impr) or blended CPC/CPL
    - why it matters: auction dynamics, bid strategies, coupons all live here

---

### What this means for me:

- **Compute each lever’s gap to its baseline** (historical rolling average or “Global” aggregate)
    
- **Score directional delta**
    - Advertiser Engagement down → attribution to budget cuts or ad-buyer churn
    - Delivery Efficiency down → attribution to pacing, targeting, seasonality
    - Yield down → attribution to CPM/CPC shifts or coupon expansions
    
- **Rank by magnitude** of gap share
    
- **Generate insight**:
    
    > “Revenue lag is 10% vs. pacing. 60% of the shortfall comes from depressed Effective CPM (yield), 30% from decreased delivered impressions (efficiency), and 10% from reduced active campaign budgets (engagement).”
    
---

# A sales version of the macro model

```
Revenue ≈ # Opportunities × Win Rate × Average Deal Size
```

1. **Pipeline Volume** (# Opportunities)
    – how many deals enter the funnel (new pitches)    
    – proxies: total opportunities created, pipeline coverage vs. quota
    
2. **Conversion Efficiency** (Win Rate)
    – % of opportunities that close
    – proxies: win-rate by rep, by vertical, by stage drop-off rates
    
3. **Deal Value** (Average Deal Size)
    – average $ closed per deal
    – proxies: Annual Contract Value or SoW, product mix shifts
    
**RCA process**
- For the current gap, compute each lever’s % shortfall vs. baseline (rolling average or goal curve)
- Score and rank: which lever explains most of the revenue gap
- Drill in: e.g. if Win Rate is down, look at rep activity (calls, meetings), attrition/PTO; if Deal Size is down, examine product mix or discounting
    

----

# Root Cause Approach for Revenue as a % of Goal (QTD Attainment or Run Rate)

## TL;DR

This document outlines a structured plan to apply root cause analysis (RCA) to **Revenue as a % of Goal** using QTD Attainment or run rate as the central metric. This metric allows us to evaluate in-quarter revenue performance relative to target while minimizing ambiguity from forecast assumptions. It is  used for comparing markets and segments with different baselines.

The analysis builds a directional hypothesis space spanning five major dimensions—Mix & Composition, Sales Motion, Advertiser Structure, Business Dynamics, and Macro Factors—each with example metrics and failure modes. These hypotheses help identify where and why revenue shortfalls occur, and offer a grounded framework for RCA automation and stakeholder insight delivery.

---

## Why “Revenue as a % of Goal (QTD Attainment or Run Rate)” as Metric

* Provides a normalized view of revenue performance relative to target

  * e.g. AM-NA 33% attainment QTD vs. AM-APAC 40% attainment QTD is directly comparable
* Avoids ambiguity introduced by end-of-quarter forecast attainment comparisons

  * e.g. AM-NA 95% forecast attainment at EoQ vs. AM-APAC 98% forecast attainment at EoQ—without knowing how reliable the seasonal/SSPO forecast is, this can’t be meaningfully compared  

## What Else Was Considered but Not Recommended for RCA on Revenue

1. **Revenue \$**
   Absolute revenue is not recommended because it cannot be meaningfully compared across regions or markets with different baseline sizes and advertiser structures. Its absolute nature makes it inherently prone to showing anomalies even when performance is normal relative to local context.

2. **Forecast Attainment %, OnTrack/Not OnTrack (with pacing assumption)**
   This approach depends heavily on forecast quality and pacing assumptions, which vary by market and product. Errors in pacing logic can produce misleading anomaly flags, especially early in the quarter.

3. **YoY / QoQ Growth Rate of Revenue**
   While useful for historical trending, YoY or QoQ growth rate requires us to explain not only this quarter’s performance but also validate the prior year’s baseline. If last year was unusually low or high due to one-off events, this can distort the perceived growth. In addition, growth deceleration is expected as businesses scale, so interpreting what is “normal” vs. “anomalous” becomes subjective. For these reasons, growth rates are better used as directional inputs to help explain anomalies in revenue attainment %, not as the core RCA metric.

---

## Root Cause Hypothesis Framework

We break down potential drivers of variance into thematic levers. Example numbers are provided to illustrate how these patterns may appear in practice:

### 1. Deep Dive Into (Mix or Composition Effects)

* **Market:** Particular L8s are lagging
  *e.g. L8-NA-Strategic-Commerce1 at 43% attainment vs. L4-NA average of 58%*
* **Product Mix**: Large gap concentrated in one or few products
  *e.g. 60% of gap driven by underperformance in Product A*
* **Vertical Mix**: Particular product-verticals are lagging
  *e.g. Tech vertical in Product B only reached 22% of goal*
* **Grower Share**: Share of grower Advertiser is lower this quarter
  *e.g. Only 48% of book are Growers this Q, down from 62%*
* **Growth Rate**: Growth rate of Growers is slowing or decline rate of decliners is accelerating
  *e.g. Grower segment growth slowed from +19% last Q to +6% this Q; Decliner segment dropped –18% this Q vs. –9% last Q*
* **Top Advertiser Share**: Top advertisers show lower contribution (of total revenue) or slower momentum (lower growth rate)
  *e.g. Top 10 advertisers contributed 33% of revenue this Q vs. 45% last Q*

### 2. Sales Behavior & Motion

* **Funnel Conversion Pacing Issue**: % Solutions moved from Discovery/Scoping → Pitching is lower than expected based on historical pacing
  *e.g. 28% moved by D5 this Q vs. 39% by D5 last Q*
* **Pitch Effectiveness**: Pitch success rate is low (%Pitched → Actioned low)
  *e.g. Pitch success rate = 41% this Q vs. 54% last Q*
* **Pre-pitch Lost Rate**: High share of opportunities lost before any pitch
  *e.g. 3% marked Not Relevant or Not Actionable, up from 1%*
* **Stage Stalling**: Actioned or Partially Adopted share is high (revenue inflow % is low)
  *e.g. 37% of Actioned RS, \$PRC headroom is \$xM*
* **Effort w/o Conversion**: Advertiser Actioned but decided to drop it mid-way; eventually marked Lost (Committed or Actioned eventually marked as Lost %)
  *e.g. 2% of Actioned or Committed RS later reclassified as Lost*
* **Moved to Next Quarter**: Revenue start date moved to next quarter
  *e.g. 12% of RS now tagged active\_future*
* **Lost Reasons**: Sales-tagged reasons like budget cuts, competition pressure
  *e.g. 39% of Lost RS cited Budget Cut as primary reason*
* **Inventory Issue**: pitched RS is low or \$PRC shrunk
  *e.g. Avg # RS pitched per AM = X this Q v.s. 1.1X last Q, \$PRC beyond Actioned = \$X this Q vs.  \$1.05X last Q*

### 3. Advertiser Structure & Load

* **Advertiser Count**: Total number of advertisers is lower than expected
  *e.g. 118 this Q vs. 136 last Q*
* **Campaign Load**: Fewer campaigns per advertiser
  *e.g. 2.2 per advertiser vs. 2.9 prior Q*
* **Wallet Share**: Lower spend concentration / diluted share of wallet
  *e.g. Top 20 accounts = 37% of spend vs. 40% last Q*
* **Pricing Signals**: Higher CPM/CPC/CPL, indicating price pressure or pullback
  *e.g. Avg CPM rose from \$11.8 → \$13.7*

### 4. Business-Level Shocks

* **Goal Accuracy**: This quarter’s goal may be set too aggressively
  *e.g. Goal = \$18.2M vs. \$13.4M last Q (+36% QoQ)*
* **Book of Business Transfer Effects**:

  * Gained decliners from GBG or lost SBG or lost advertisers from GBG or SBG
    *e.g. 14 new accounts from GBG; contributing x% to the negative growth. 10 growers lost to SBG; could have added \$xM growth (-0.8x% instead of -x% YoY)* 

### 5. External Macro Factors

* % Share of Revenue Contribution at:

  * **Vertical**: e.g. CPG vertical typically contributes 18% ±3%, but dropped to 13% this Q — may signal macro pullback or policy change
  * **Product**: e.g. Product C share dropped from 12% to 6% — could reflect competitive pressure or phase-out
  * **Market**: e.g. APAC market share fell from 14% to 8% — possibly due to regional regulation or advertiser exits

Sudden changes in these shares—especially outside historical fluctuation ranges—may indicate external shocks or broader macroeconomic shifts.
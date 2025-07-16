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
    


- Ties directly to sales activity owners (BDRs for pipeline, AEs for conversion, Finance for pricing)
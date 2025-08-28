import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# ==== COLORS (single source of truth) ====
COLOR_STAGE_NEUTRAL = ['#3498DB', '#4ECDC4', '#FFC300', '#FF9F43', '#AF7AC5']
COLOR_ADVANCED = '#2ECC71'
COLOR_LOST     = '#E74C3C'
COLOR_BLOCKED  = '#FFC300'
COLOR_IN_STAGE = '#5DADE2'
COLOR_LOST_SOFT    = '#E74C3C80'
COLOR_BLOCKED_SOFT = '#FFC30080'

CALLOUT_PCT_INNER = 0.02  # 2% of the donut’s base -> hide inner label if below this

# =============================================================================
# UTILS - Simple helper functions
# =============================================================================

def _fmt_money_short(x):
    """Format money amounts with K/M/B suffix."""
    x = float(x or 0)
    s = '-' if x < 0 else ''
    x = abs(x)
    for unit, div in (('B', 1e9), ('M', 1e6), ('K', 1e3)):
        if x >= div:
            return f"{s}${x/div:.1f}{unit}"
    return f"{s}${x:.1f}"

def _bracket(primary_use, num_n, num_amt, base_n, base_amt):
    """Generate bracket text following the bracket rule."""
    return f" ({0 if primary_use!='n' else (0 if base_amt==0 else num_amt/base_amt):.1%})" \
           if primary_use=='n' else f" ({_fmt_money_short(num_amt)})"

def lighten(hex_color, factor=0.85):
    """Return a lighter hex color (toward white)."""
    try:
        if not (isinstance(hex_color, str) and hex_color.startswith('#') and len(hex_color) == 7):
            return '#DDDDDD'
        r = int(hex_color[1:3], 16); g = int(hex_color[3:5], 16); b = int(hex_color[5:7], 16)
        lr = int(r + (255 - r) * factor); lg = int(g + (255 - g) * factor); lb = int(b + (255 - b) * factor)
        return f"#{lr:02x}{lg:02x}{lb:02x}"
    except Exception:
        return '#DDDDDD'

def _ensure_stage_val_column(tl_dict, use='amt'):
    """Ensure tl_dict['stage_breakdown_df'] has a numeric 'val' column used for totals."""
    st = tl_dict.get('stage_breakdown_df')
    if st is None or st.empty or 'val' in st.columns:
        return
    if use == 'amt' and 'lost_amount' in st.columns:
        tl_dict['stage_breakdown_df']['val'] = st['lost_amount'].astype(float).fillna(0.0)
    elif use == 'n' and 'n' in st.columns:
        tl_dict['stage_breakdown_df']['val'] = st['n'].astype(float).fillna(0.0)
    elif 'amt' in st.columns:
        tl_dict['stage_breakdown_df']['val'] = st['amt'].astype(float).fillna(0.0)
    else:
        tl_dict['stage_breakdown_df']['val'] = 0.0
        
# =============================================================================
# DATA LAYER - Single source of truth for normalization
# =============================================================================

def normalize_funnel(df_lost, df_block, config):
    """Return a single normalized long table for Lost/Blocked with stage origins and rates.
    
    Args:
        df_lost, df_block: raw input DataFrames
        config: dict with 'columns' key mapping logical names to actual column names
        
    Returns:
        df_norm: DataFrame with columns ['region','kind','stage','reason','n','amt',
                'stage_origin_n','stage_origin_amt','rate_n','rate_amt']
    """
    c = config['columns']
    
    # Normalize lost data - handle missing columns gracefully
    lost_cols = [c['lost_territory'], c['lost_stage'], c['lost_reason'],
                 c['lost_count'], c['lost_amount'], c['lost_current_count'], c['lost_current_amount']]
    
    # Add total columns if they exist, otherwise we'll calculate them later
    if c['lost_total_count'] in df_lost.columns:
        lost_cols.extend([c['lost_total_count'], c['lost_total_amount']])
        lost = df_lost[lost_cols].copy()
        lost.columns = ['region','stage','reason','n','amt','curr_n','curr_amt','stage_total_n','stage_total_amt']
    else:
        lost = df_lost[lost_cols].copy()
        lost.columns = ['region','stage','reason','n','amt','curr_n','curr_amt']
        # Add placeholder columns that will be calculated later
        lost['stage_total_n'] = lost['curr_n'] + lost['n']
        lost['stage_total_amt'] = lost['curr_amt'] + lost['amt']
    
    lost['kind'] = 'Lost'

    # Normalize blocked data - only use columns that exist
    block_cols = [c['blocked_territory'], c['blocked_stage'], c['blocked_reason'],
                  c['blocked_count'], c['blocked_amount'], c['blocked_current_count'], c['blocked_current_amount']]
    
    block = df_block[block_cols].copy()
    block.columns = ['region','stage','reason','n','amt','curr_n','curr_amt']
    block['kind'] = 'Blocked'

    # Calculate stage origins for lost data
    # Use 'first' for invariant columns (same per region+stage), 'sum' for reason-level data
    st_l = (lost.groupby(['region','stage'], as_index=False)
                .agg(stage_total_n=('stage_total_n','first'),   # invariant per region+stage
                     curr_n=('curr_n','first'),                # invariant per region+stage  
                     lost_n=('n','sum'),                       # sum across reasons
                     stage_total_amt=('stage_total_amt','first'), # invariant per region+stage
                     curr_amt=('curr_amt','first'),            # invariant per region+stage
                     lost_amt=('amt','sum')))                  # sum across reasons
    st_l['stage_origin_n'] = st_l['stage_total_n'].fillna(st_l['curr_n'] + st_l['lost_n'])
    st_l['stage_origin_amt'] = st_l['stage_total_amt'].fillna(st_l['curr_amt'] + st_l['lost_amt'])
    lost = lost.merge(st_l[['region','stage','stage_origin_n','stage_origin_amt']], on=['region','stage'], how='left')

    # Calculate stage origins for blocked data  
    # Use 'first' for invariant columns (same per region+stage), 'sum' for reason-level data
    st_b = (block.groupby(['region','stage'], as_index=False)
                .agg(curr_n=('curr_n','first'),      # invariant per region+stage
                     block_n=('n','sum'),            # sum across reasons
                     curr_amt=('curr_amt','first'),   # invariant per region+stage
                     block_amt=('amt','sum')))        # sum across reasons
    st_b['stage_origin_n'] = st_b['curr_n'] + st_b['block_n']
    st_b['stage_origin_amt'] = st_b['curr_amt'] + st_b['block_amt']
    block = block.merge(st_b[['region','stage','stage_origin_n','stage_origin_amt']], on=['region','stage'], how='left')

    # Combine and calculate rates
    df = pd.concat([
        lost[['region','kind','stage','reason','n','amt','stage_origin_n','stage_origin_amt']],
        block[['region','kind','stage','reason','n','amt','stage_origin_n','stage_origin_amt']]
    ], ignore_index=True)
    
    df['rate_n'] = (df['n'] / df['stage_origin_n']).fillna(0.0)
    df['rate_amt'] = (df['amt'] / df['stage_origin_amt']).fillna(0.0)
    
    return df

# =============================================================================
# ANALYSIS HELPERS - Pure data processing, no plotting
# =============================================================================

def scope_lost(df_lost, *, config, focus_region, scope='rest_of_world', comparison_region=None):
    """Return a scoped df_lost and label for total-lost analysis.
    
    Args:
        df_lost: input DataFrame
        config: configuration dict
        focus_region: the region we're analyzing
        scope: 'region' (focus only) | 'rest_of_world' | 'global' | 'region_name'
        comparison_region: specific region name when scope is a region
        
    Returns:
        tuple: (scoped_df_lost, label)
    """
    c = config['columns']
    
    if scope == 'region':
        return df_lost[df_lost[c['lost_territory']] == focus_region].copy(), focus_region
    
    if scope == 'rest_of_world':
        df = df_lost[df_lost[c['lost_territory']] != focus_region]
        lab = 'Rest of World'
    elif scope == 'global':
        df = df_lost.copy()
        lab = 'Global'
    else:
        df = df_lost[df_lost[c['lost_territory']] == (comparison_region or scope)]
        lab = comparison_region or scope

    # Aggregate by stage+reason to avoid double-counting across regions
    if df.empty:
        out = df.copy()
        out[c['lost_territory']] = lab
        return out, lab
        
    out = (df.groupby([c['lost_stage'], c['lost_reason']], as_index=False)
             .agg({c['lost_count']: 'sum', c['lost_amount']: 'sum'}))
    out[c['lost_territory']] = lab
    
    return out, lab

def scope_block(df_block, *, config, focus_region, scope='rest_of_world', comparison_region=None):
    """Return a scoped df_block and label for blocked data analysis.
    
    Args:
        df_block: input DataFrame
        config: configuration dict
        focus_region: the region we're analyzing
        scope: 'region' (focus only) | 'rest_of_world' | 'global' | region_name
        comparison_region: specific region name when scope is a region
        
    Returns:
        tuple: (scoped_df_block, label)
    """
    c = config['columns']
    
    if scope == 'region':
        return df_block[df_block[c['blocked_territory']] == focus_region].copy(), focus_region
    elif scope == 'rest_of_world':
        # Use unified aggregation helper
        _, agg_block = _aggregate_funnel_data(
            df_lost=pd.DataFrame(), df_block=df_block, config=config,
            new_territory_name='Rest of World', exclude_regions=[focus_region]
        )
        return agg_block, 'Rest of World'
    elif scope == 'global':
        # Use unified aggregation helper
        _, agg_block = _aggregate_funnel_data(
            df_lost=pd.DataFrame(), df_block=df_block, config=config,
            new_territory_name='Global'
        )
        return agg_block, 'Global'
    else:
        # Specific region comparison
        return df_block[df_block[c['blocked_territory']] == (comparison_region or scope)].copy(), (comparison_region or scope)

def total_lost_data(df_lost_scoped, *, config, use='amt', top_k=5, coverage=0.85, min_share=0.02):
    """Compute total-lost inner/outer data for a single scope.
    
    Args:
        df_lost_scoped: scoped lost data from scope_lost()
        config: configuration dict
        use: 'amt' or 'n' for primary metric
        top_k, coverage, min_share: Pareto filtering parameters
        
    Returns:
        dict: Contains stage_breakdown_df, outer ring data, base totals
    """
    c = config['columns']
    L = df_lost_scoped[[
        c['lost_territory'], c['lost_stage'], c['lost_reason'],
        c['lost_count'], c['lost_amount']
    ]].rename(columns={
        c['lost_territory']: 'region', c['lost_stage']: 'stage',
        c['lost_reason']: 'reason', c['lost_count']: 'n', c['lost_amount']: 'amt'
    })

    if L.empty:
        return {
            'stage_breakdown_df': pd.DataFrame(columns=['stage', 'lost_amount', 'lost_pct']),
            'outer': pd.DataFrame(columns=['stage', 'reason', 'val_n', 'val_amt']),
            'base_n': 0, 'base_amt': 0
        }

    base_n, base_amt = int(L['n'].sum()), float(L['amt'].sum())
    
    # Stage breakdown for inner ring
    stage_breakdown_df = (L.groupby('stage', as_index=False)[['n', 'amt']].sum()
                         .sort_values('amt', ascending=False)
                         .rename(columns={'amt': 'lost_amount'}))
    stage_breakdown_df['lost_pct'] = (0 if base_amt == 0 else 
                                     (stage_breakdown_df['lost_amount'] / base_amt * 100).round(1))

    # Outer ring data with Pareto filtering
    def _collapse(g):
        g2 = (g.groupby('reason', as_index=False)[['n', 'amt']].sum()
                .rename(columns={'n': 'val_n', 'amt': 'val_amt'}))
        key = 'val_amt' if use == 'amt' else 'val_n'
        g2 = g2.sort_values(key, ascending=False)
        
        # Pareto filtering
        keep = set(g2.head(top_k)['reason'])
        tot = g2[key].sum()
        if tot > 0:
            keep |= set(g2.loc[g2[key].cumsum() / tot <= coverage, 'reason'])
        denom = base_amt if use == 'amt' else base_n
        if denom > 0:
            keep |= set(g2.loc[g2[key] / denom >= min_share, 'reason'])
            
        kept, other = g2[g2['reason'].isin(keep)].copy(), g2[~g2['reason'].isin(keep)]
        if not other.empty:
            kept.loc[len(kept)] = {
                'reason': f"Other ({g.name})",
                'val_n': other['val_n'].sum(),
                'val_amt': other['val_amt'].sum()
            }
        kept.insert(0, 'stage', g.name)
        return kept

    outer = (L.groupby('stage', group_keys=False).apply(_collapse).reset_index(drop=True) 
             if not L.empty else pd.DataFrame(columns=['stage', 'reason', 'val_n', 'val_amt']))

    return {
        'stage_breakdown_df': stage_breakdown_df,
        'outer': outer,
        'base_n': base_n,
        'base_amt': base_amt
    }

def total_lost_pair_data(df_lost, *, config, region, comparison_type='rest_of_world', comparison_region=None, use='amt'):
    """Build left/right datasets for plotting comparison charts.
    
    Args:
        df_lost: input DataFrame
        config: configuration dict
        region: focus region
        comparison_type: 'rest_of_world' | 'global' | region name
        comparison_region: specific region for comparison
        use: 'amt' or 'n'
        
    Returns:
        tuple: (left_data, right_data, (left_label, right_label))
    """
    left_df, left_label = scope_lost(df_lost, config=config, focus_region=region, scope='region')
    
    scope = ('rest_of_world' if comparison_type == 'rest_of_world' else
             'global' if comparison_type == 'global' else
             (comparison_region or comparison_type))
    right_df, right_label = scope_lost(df_lost, config=config, focus_region=region,
                                      scope=scope, comparison_region=comparison_region)
    
    left = total_lost_data(left_df, config=config, use=use)
    right = total_lost_data(right_df, config=config, use=use)
    
    return left, right, (left_label, right_label)

def _aggregate_funnel_data(df_lost, df_block, *, config, new_territory_name, include_regions=None, exclude_regions=None):
    """Unified aggregation helper for all funnel data aggregation needs.
    
    Args:
        df_lost, df_block: input DataFrames
        config: configuration dict
        new_territory_name: what to rename aggregated territory to
        include_regions: list of regions to include (None = all)
        exclude_regions: list of regions to exclude (None = none)
        
    Returns:
        tuple: (aggregated_lost_df, aggregated_block_df)
    """
    c = config['columns']
    
    # Filter data based on include/exclude criteria
    if include_regions is not None:
        agg_lost = df_lost[df_lost[c['lost_territory']].isin(include_regions)].copy()
        agg_block = df_block[df_block[c['blocked_territory']].isin(include_regions)].copy()
    elif exclude_regions is not None:
        agg_lost = df_lost[~df_lost[c['lost_territory']].isin(exclude_regions)].copy()
        agg_block = df_block[~df_block[c['blocked_territory']].isin(exclude_regions)].copy()
    else:
        agg_lost = df_lost.copy()
        agg_block = df_block.copy()
    
    # Rename territory for aggregation
    agg_lost[c['lost_territory']] = new_territory_name
    agg_block[c['blocked_territory']] = new_territory_name
    
    # Standard lost data aggregation
    agg_lost_result = agg_lost.groupby([
        c['lost_territory'], c['lost_stage'], c['lost_reason']
    ]).agg({
        c['lost_count']: 'sum',
        c['lost_amount']: 'sum',
        c['lost_current_count']: 'sum',
        c['lost_current_amount']: 'sum',
        c['lost_total_count']: 'sum',
        c['lost_total_amount']: 'sum'
    }).reset_index()
    
    # Standard blocked data aggregation
    agg_block_result = agg_block.groupby([
        c['blocked_territory'], c['blocked_stage'], c['blocked_reason']
    ]).agg({
        c['blocked_count']: 'sum',
        c['blocked_amount']: 'sum',
        c['blocked_current_count']: 'sum',
        c['blocked_current_amount']: 'sum'
    }).reset_index()
    
    return agg_lost_result, agg_block_result

def _create_aggregated_comparison_data(df_lost, df_block, *, focus_region, config, comparison_label):
    """Helper to create aggregated comparison data while preserving structure."""
    
    if focus_region is None:
        # Global aggregation - include all regions
        return _aggregate_funnel_data(
            df_lost, df_block, config=config, 
            new_territory_name=comparison_label
        )
    else:
        # Rest of world - exclude focus region
        return _aggregate_funnel_data(
            df_lost, df_block, config=config,
            new_territory_name=comparison_label,
            exclude_regions=[focus_region]
        )

def stage_flow(df_norm, df_lost, df_block, *, region, stage, config, use='amt',
               top_k=3, coverage=0.8, min_rate=0.03, min_n=5):
    """Pure data computation for a single region-stage funnel.
    
    Args:
        df_norm: normalized data from normalize_funnel()
        df_lost, df_block: raw input DataFrames
        region, stage: focus region and stage
        config: configuration dict
        use: 'amt' or 'n'
        Pareto parameters: top_k, coverage, min_rate, min_n
        
    Returns:
        dict: Flow data for plotting including reasons with percentages
    """
    c = config['columns']
    
    # Get stage-specific data - handle missing columns gracefully
    Ls = df_lost[(df_lost[c['lost_territory']] == region) & (df_lost[c['lost_stage']] == stage)]
    Bs = df_block[(df_block[c['blocked_territory']] == region) & (df_block[c['blocked_stage']] == stage)]

    # Basic totals - safe column access
    lost_n = Ls[c['lost_count']].sum() if not Ls.empty else 0
    lost_amt = Ls[c['lost_amount']].sum() if not Ls.empty else 0
    
    # Current amounts - try lost first, then blocked, handle missing columns
    curr_n = curr_amt = 0
    if not Ls.empty and c['lost_current_count'] in Ls.columns:
        curr_n = Ls[c['lost_current_count']].sum()
        curr_amt = Ls[c['lost_current_amount']].sum()
    elif not Bs.empty and c['blocked_current_count'] in Bs.columns:
        curr_n = Bs[c['blocked_current_count']].sum()
        curr_amt = Bs[c['blocked_current_amount']].sum()
    
    block_n = Bs[c['blocked_count']].sum() if not Bs.empty else 0
    block_amt = Bs[c['blocked_amount']].sum() if not Bs.empty else 0

    # Advanced to later stages (unique per later stage to avoid double-counting)
    stage_list = list(config['stage_transitions'].keys())
    adv_n = adv_amt = 0
    if stage in stage_list and c['lost_total_count'] in df_lost.columns:
        later_stages = stage_list[stage_list.index(stage)+1:]
        final_stage = config['stage_transitions'].get(stage_list[-1])
        if final_stage: 
            later_stages.append(final_stage)

        later = df_lost[(df_lost[c['lost_territory']] == region) &
                        (df_lost[c['lost_stage']].isin(later_stages))]

        if not later.empty:
            # Use 'first' to get one total per stage (avoids multiplying by # of reasons)
            uniq = (later.groupby(c['lost_stage'], as_index=False)
                        .agg(uniq_total_n=(c['lost_total_count'], 'first'),
                             uniq_total_amt=(c['lost_total_amount'], 'first')))
            adv_n = int(uniq['uniq_total_n'].sum())
            adv_amt = float(uniq['uniq_total_amt'].sum())

    # Calculate flow components
    in_unblocked_n = max(curr_n - block_n, 0)
    in_unblocked_amt = max(curr_amt - block_amt, 0)
    base_n = lost_n + block_n + in_unblocked_n + adv_n
    base_amt = lost_amt + block_amt + in_unblocked_amt + adv_amt

    # Get reasons using normalized data with Pareto filtering
    reg = df_norm[(df_norm['region'] == region) & (df_norm['stage'] == stage)]
    use_col = 'amt' if use == 'amt' else 'n'
    rate_col = 'rate_amt' if use == 'amt' else 'rate_n'
    
    def _keep(g):
        g = g.sort_values(use_col, ascending=False).reset_index(drop=True)
        core = set(range(min(top_k, len(g))))
        total = g[use_col].sum()
        if total > 0:
            core |= set(g.index[(g[use_col] / total).cumsum() <= coverage])
        mask = (g.index.isin(core) | (g[rate_col] >= min_rate) | (g['n'] >= min_n))
        return g[mask]

    collapsed = (reg.groupby(['region', 'kind', 'stage'], group_keys=False).apply(_keep)
                if not reg.empty else reg)

    def _reasons(kind, max_reasons=3):
        items = []
        sub = collapsed[collapsed['kind'] == kind]
        
        if sub.empty:
            return items
            
        # Sort by amount/count and take top N for summary text to avoid flooding
        if kind == 'Blocked':
            sub = sub.sort_values('n', ascending=False).head(max_reasons)
        else:
            sub = sub.sort_values('amt', ascending=False).head(max_reasons)
        
        for _, row in sub.iterrows():
            if kind == 'Blocked':
                # For blocked reasons, show count instead of % since blocked amounts are typically small
                items.append(f"{row['reason']} ({int(row['n'])} solutions)")
            else:
                # For lost reasons, continue showing percentages
                pct = 0 if base_amt == 0 else row['amt'] / base_amt * 100
                items.append(f"{row['reason']} ({pct:.1f}%)")
        return items

    return {
        'lost_n': lost_n, 'lost_amt': lost_amt, 'block_n': block_n, 'block_amt': block_amt,
        'in_n': in_unblocked_n, 'in_amt': in_unblocked_amt, 'adv_n': adv_n, 'adv_amt': adv_amt,
        'base_n': base_n, 'base_amt': base_amt, 'stage': stage, 'region': region,
        'lost_reasons_with_pct': _reasons('Lost'), 'blocked_reasons_with_pct': _reasons('Blocked')
    }

# =============================================================================
# PLOTTING LAYER - Consume precomputed data only
# =============================================================================

def plot_total_lost_single(tl_dict, *, label, use='amt', stage_order=None, stage_color_map=None, ax=None):
    """Plot a single total-lost donut chart.
    
    Args:
        tl_dict: output from total_lost_data()
        label: region label for title
        use: 'amt' or 'n'
        stage_order: optional stage ordering
        ax: optional matplotlib axes
        
    Returns:
        matplotlib figure
    """
    st = tl_dict['stage_breakdown_df'].copy()
    outer = tl_dict['outer'].copy()
    bn, ba = tl_dict['base_n'], tl_dict['base_amt']
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 7))
    else:
        fig = ax.figure

    if (use == 'n' and bn == 0) or (use == 'amt' and ba == 0):
        ax.text(0.5, 0.5, "No Lost data", ha='center')
        return fig

    # Stage colors (inner ring)
    stages = stage_order or st['stage'].tolist()
    st = st.set_index('stage').reindex(stages).dropna().reset_index()
    if stage_color_map:
        st['color'] = [stage_color_map[s] for s in st['stage']]
    else:
        st['color'] = [COLOR_STAGE_NEUTRAL[i % len(COLOR_STAGE_NEUTRAL)] for i in range(len(st))]

    base_primary = ba if use == 'amt' else bn

    # Inner ring labels and plot
    inner_labels = []
    for _, row in st.iterrows():
        pct = (row['val'] / base_primary) if base_primary > 0 else 0
        bracket = _bracket(use, getattr(row, 'n', 0), row['lost_amount'], bn, ba)
        inner_labels.append(f"{row['stage']}\n{pct:.1%}{bracket}")
        
    ax.pie(st['val'].to_numpy(), radius=0.74, labels=inner_labels, colors=st['color'],
           wedgeprops=dict(width=0.34), startangle=90, labeldistance=0.45)

    # Outer ring (sub-reasons)
    if not outer.empty:
        outer['val_primary'] = outer['val_amt'] if use == 'amt' else outer['val_n']
        outer = outer.sort_values(['stage', 'val_primary'], ascending=[True, False])
        outer_vals = outer['val_primary'].to_numpy()
        
        # Smart labeling - only show significant slices
        outer_labels = []
        total_val = sum(outer_vals)
        for i, (_, row) in enumerate(outer.iterrows()):
            val = outer_vals[i]
            pct = val / total_val if total_val > 0 else 0
            
            # Only label slices > 3% or top 6 items
            if pct > 0.03 or i < 6:
                reason_short = row['reason'][:15] + '...' if len(row['reason']) > 15 else row['reason']
                label_val = f"{pct:.1%}" if use == 'n' else _fmt_money_short(row['val_amt'])
                outer_labels.append(f"{reason_short}\n{label_val}")
            else:
                outer_labels.append('')
            
        stage2color = {s: c for s, c in zip(st['stage'], st['color'])}
        outer_colors = [lighten(stage2color.get(s, '#DDDDDD')) for s in outer['stage']]
        
        # Plot outer ring with same startangle and ensure segments align with inner ring
        ax.pie(outer_vals, radius=1.0, labels=outer_labels, colors=outer_colors,
               wedgeprops=dict(width=0.26), startangle=90, labeldistance=0.88, 
               textprops={'fontsize': 8})

    # Title and center text
    base_txt = f"{int(bn):,}" if use == 'n' else _fmt_money_short(ba)
    title_metric = "# of Solutions Lost" if use == 'n' else "\\$PRC Lost"
    ax.set(aspect='equal', title=f"{label} — Total Lost ({title_metric}, total={base_txt})")
    ax.title.set_fontsize(10)
    
    center_text = _fmt_money_short(ba) if use == 'amt' else f'{int(bn):,}'
    ax.text(0, 0, f"Total Lost\n{center_text}", ha='center', va='center', 
            fontsize=11, fontweight='bold')
    
    return fig

def plot_total_lost_pair(left, right, *, labels, use='amt', stage_order=None):
    """Side-by-side total-lost donut charts.
    
    Args:
        left, right: dicts from total_lost_data()
        labels: tuple of (left_label, right_label)
        use: 'amt' or 'n'
        stage_order: optional stage ordering
        
    Returns:
        matplotlib figure
    """
    # Guarantee 'val' exists for stage presence / ordering
    _ensure_stage_val_column(left,  use)
    _ensure_stage_val_column(right, use)

    def present_stages(tl):
        st = tl['stage_breakdown_df']
        if st.empty:
            return []
        vals = st['val'] if 'val' in st.columns else (st['lost_amount'] if use=='amt' else st.get('n', st['lost_amount']))
        return st.loc[vals.fillna(0) > 0, 'stage'].tolist()

    left_present  = present_stages(left)
    right_present = present_stages(right)

    if stage_order:
        stages_eff = [s for s in stage_order if s in (set(left_present) | set(right_present))]
        if not stages_eff:
            stages_eff = stage_order[:]
    else:
        stages_eff = list(dict.fromkeys(left_present + right_present))

    # One shared map for both plots + legends
    stage_color_map = {s: COLOR_STAGE_NEUTRAL[i % len(COLOR_STAGE_NEUTRAL)] for i, s in enumerate(stages_eff)}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8), gridspec_kw={'wspace': 0.04})
    plot_total_lost_single(left,  label=labels[0], use=use, stage_order=stages_eff,
                           stage_color_map=stage_color_map, ax=ax1)
    plot_total_lost_single(right, label=labels[1], use=use, stage_order=stages_eff,
                           stage_color_map=stage_color_map, ax=ax2)
    fig.suptitle(f"Total Lost Analysis: {labels[0]} vs {labels[1]}", y=0.98)
    # Legends
    stage_handles  = [Patch(facecolor=stage_color_map[s],           label=s)                for s in stages_eff]
    reason_handles = [Patch(facecolor=lighten(stage_color_map[s]),  label=f"{s} reasons")   for s in stages_eff]

    lg_reasons = fig.legend(handles=reason_handles, loc='lower center', bbox_to_anchor=(0.5, 0.12),
                            ncol=min(len(reason_handles), 4), frameon=False, fontsize=9,
                            title="Reasons (Outer Ring)")
    lg_reasons.get_title().set_fontsize(10); lg_reasons.get_title().set_fontweight('bold')

    lg_stages  = fig.legend(handles=stage_handles,  loc='lower center', bbox_to_anchor=(0.5, 0.05),
                            ncol=min(len(stage_handles), 4), frameon=False, fontsize=9,
                            title="Stages (Inner Ring)")
    lg_stages.get_title().set_fontsize(10); lg_stages.get_title().set_fontweight('bold')

    plt.tight_layout()
    plt.subplots_adjust(top=0.90, bottom=0.20, left=0.04, right=0.96)
    return fig

def plot_stage_donut(flow, *, use='amt', stage_transitions=None, ax=None):
    """Plot single stage funnel donut chart.
       Inner labels are hidden if their slice < CALLOUT_PCT_INNER of base."""

    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 7))

    bn, ba = flow['base_n'], flow['base_amt']
    stage = flow['stage']; region = flow['region']

    # Next stage
    next_stage = stage_transitions.get(stage, 'Final') if stage_transitions else 'Final'
    next_label = f"→ {next_stage}" if next_stage and next_stage != 'Final' else "→ Closed Won"

    # Colors
    OUTER_COLORS = {'in_stage': COLOR_IN_STAGE, 'advanced': COLOR_ADVANCED,
                    'blocked': COLOR_BLOCKED, 'lost': COLOR_LOST}
    INNER_COLORS = {'lost': COLOR_LOST_SOFT, 'blocked': COLOR_BLOCKED_SOFT}

    # ------- Outer ring (main categories) -------
    outer_parts = [
        (next_label,         flow['adv_n'],   flow['adv_amt'],  OUTER_COLORS['advanced']),
        ('Lost',             flow['lost_n'],  flow['lost_amt'], OUTER_COLORS['lost']),
        ('Blocked',          flow['block_n'], flow['block_amt'],OUTER_COLORS['blocked']),
        (f"In {stage}",      flow['in_n'],    flow['in_amt'],   OUTER_COLORS['in_stage']),
    ]
    base_primary = (bn if use == 'n' else ba)
    outer_vals = [max(0, float(p[1] if use == 'n' else p[2])) for p in outer_parts]
    if sum(outer_vals) == 0:
        outer_vals = [1, 0, 0, 0]

    outer_labels = []
    for i, (name, n, amt, _) in enumerate(outer_parts):
        p = 0 if base_primary == 0 else outer_vals[i] / base_primary
        bracket = _bracket(use, n, amt, bn, ba)
        outer_labels.append(f"{name}\n{p:.1%}{bracket}")

    ax.pie(outer_vals, radius=1.0, labels=outer_labels,
           colors=[p[3] for p in outer_parts],
           wedgeprops=dict(width=0.26), startangle=90, labeldistance=0.88)

    # ------- Inner ring (sub-reasons) -------
    inner_labels, inner_vals, inner_colors = [], [], []

    # Lost reasons (strings like "Reason (5.4%)")
    for s in (flow.get('lost_reasons_with_pct') or []):
        val = 0.0
        try:
            pct = float(s.split('(')[1].split('%')[0]) / 100.0
            val = pct * (ba if use == 'amt' else bn)
        except Exception:
            val = 0.0
        inner_vals.append(val)
        inner_colors.append(INNER_COLORS['lost'])
        inner_labels.append(s)  # temporarily store; may blank out below

    # Blocked reasons (strings like "Executive Approval (12)" or "(12 solutions)")
    blocked_items = (flow.get('blocked_reasons_with_pct') or [])
    total_block_n = max(int(bn), 1)  # avoid div-by-zero
    for s in blocked_items:
        # parse integer inside parentheses
        count = 0
        if '(' in s and ')' in s:
            inside = s.split('(', 1)[1].split(')', 1)[0].strip()
            try:
                count = int(inside.split()[0])
            except Exception:
                count = 0
        # convert to value on the base scale so % threshold works
        share = (count / total_block_n) if total_block_n > 0 else 0.0
        val = share * base_primary
        inner_vals.append(val)
        inner_colors.append(INNER_COLORS['blocked'])
        inner_labels.append(s)

    # Decide which inner labels to show (hide if < threshold)
    safe_labels = []
    total_inner_base = float(base_primary) if base_primary else 0.0
    for lbl, v in zip(inner_labels, inner_vals):
        show = (total_inner_base > 0 and (v / total_inner_base) >= CALLOUT_PCT_INNER)
        safe_labels.append(lbl if show else '')

    if inner_vals and sum(inner_vals) > 0:
        ax.pie(inner_vals, radius=0.74, labels=safe_labels, colors=inner_colors,
               wedgeprops=dict(width=0.34), startangle=90, labeldistance=0.45,
               textprops={'fontsize': 9})

    # Title & center
    use_label = "# of Solutions" if use == 'n' else "\\$PRC"
    base_desc = f"funnel start: {_fmt_money_short(ba) if use == 'amt' else f'{int(bn):,}'}"
    ax.set(aspect='equal', title=f"{region}\n{stage} ({use_label}, {base_desc})")
    ax.title.set_fontsize(10)
    ax.title.set_position([0.5, 0.85])

    at_risk_primary = ((flow['lost_n'] + flow['block_n']) / bn if use == 'n' and bn > 0
                       else (flow['lost_amt'] + flow['block_amt']) / ba if ba > 0 else 0)
    ax.text(0, 0, f"{at_risk_primary:.1%}\nBlocked/Lost",
            ha='center', va='center', fontsize=12, fontweight='bold')
    return ax

def plot_funnel_donut(flow_data, *, comparison_flow=None, comparison_label=None, use='amt', stage_transitions=None):
    """Plot funnel donut chart(s) from flow data.
    
    Args:
        flow_data: output from stage_flow()
        comparison_flow: optional comparison flow data
        comparison_label: label for comparison
        use: 'amt' or 'n'
        stage_transitions: dict of stage transitions
        
    Returns:
        matplotlib figure
    """
    if comparison_flow:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7), gridspec_kw={'wspace': 0.05})
        plot_stage_donut(flow_data,      use=use, stage_transitions=stage_transitions, ax=ax1)
        plot_stage_donut(comparison_flow, use=use, stage_transitions=stage_transitions, ax=ax2)
        fig.suptitle(f'Funnel Analysis: {flow_data["region"]} vs {comparison_label} @ {flow_data["stage"]}',
                     fontsize=14, y=0.98)
    else:
        fig, ax = plt.subplots(figsize=(7, 7))
        plot_stage_donut(flow_data, use=use, stage_transitions=stage_transitions, ax=ax)

    # ---------- Legends ----------
    # OUTER ring (categories)
    outer_handles = [
        Patch(facecolor=COLOR_ADVANCED,  label='Advanced'),
        Patch(facecolor=COLOR_LOST,      label='Lost'),
        Patch(facecolor=COLOR_BLOCKED,   label='Blocked'),
        Patch(facecolor=COLOR_IN_STAGE,  label=f'In {flow_data["stage"]}')
    ]
    fig.legend(handles=outer_handles,
               loc='lower center', bbox_to_anchor=(0.5, 0.11),
               ncol=len(outer_handles), frameon=False, fontsize=10)

    # INNER ring legend: keep only Lost sub-reasons (you asked to drop Blocked)
    inner_handles = [Patch(facecolor=COLOR_LOST_SOFT, label='Lost sub-reasons (inner)')]
    fig.legend(handles=inner_handles,
               loc='lower center', bbox_to_anchor=(0.5, 0.04),
               ncol=1, frameon=False, fontsize=10)

    # Layout to make room for the two legend rows
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.18, top=0.88 if comparison_flow else 0.95)
    return fig

# =============================================================================
# HIGH-LEVEL INTERFACES - Main entry points
# =============================================================================

def analyze_funnel_reasons(df_lost, df_block, config, metric_anomaly_map, *, 
                          lost_columns, blocked_columns):
    """
    Main entrypoint for funnel analysis:
    - If metric['drop_off_from'] is a string → single-stage donut (region vs benchmark).
    - If it's a list → total-lost pair (region vs rest).
    
    Args:
        df_lost, df_block: input DataFrames
        config: configuration dict (without column mappings)
        metric_anomaly_map: anomaly detection results
        lost_columns: dict mapping required lost data column names to actual DataFrame column names
        blocked_columns: dict mapping required blocked data column names to actual DataFrame column names
        
    Returns:
        unified_results dict for slide builders.
    """
    # Merge column mappings into config for internal use
    config_with_columns = config.copy()
    config_with_columns['columns'] = {**lost_columns, **blocked_columns}
    metrics_config = config.get('metrics', {})
    unified_results = {}

    for metric_name, metric_config in metrics_config.items():
        if not metric_anomaly_map or metric_name not in metric_anomaly_map:
            continue
        region = metric_anomaly_map[metric_name].get('anomalous_region')
        if not region:
            continue

        try:
            stage_spec = metric_config['drop_off_from']

            # Multi-stage → Total Lost lens
            if isinstance(stage_spec, list):
                left, right, labels = total_lost_pair_data(
                    df_lost, config=config_with_columns, region=region, comparison_type='rest_of_world', use='amt'
                )
                summary_text = build_total_lost_summary(labels[0], left, use='amt')

                unified_results[metric_name] = {
                    'slides': {
                        'Funnel': {
                            'summary': {'summary_text': summary_text},
                            'slide_info': {
                                'title': f"{metric_name} - Funnel Analysis",
                                'template_text': summary_text,
                                'template_params': {},
                                'figure_generators': [
                                    {
                                        "function": plot_total_lost_pair,
                                        "params": {"left": left, "right": right, "labels": labels, "use": "amt",
                                                   "stage_order": list(config['stage_transitions'].keys())},
                                        "caption": f"Total lost breakdown by stage. Left: {labels[0]} vs Right: {labels[1]}.\n Inner ring shows stages distribution; outer ring shows top sub-reasons per stage."
                                    }
                                ],
                                'layout_type': "text_figure",
                                'total_hypotheses': 1
                            }
                        }
                    },
                    'payload': {"left": left, "right": right, "labels": labels, "region": region}
                }
                continue

            # Single-stage → Stage funnel lens
            stage = stage_spec
            # Build benchmark on the fly using existing interface (now fixed for block scoping)
            analysis = analyze_funnel_reasons_data(
                df_lost, df_block, region=region, stage=stage, config=config_with_columns,
                comparison_type='rest_of_world'
            )
            focus_flow = analysis['region_flow']
            bench_flow = analysis['comparison_flow']
            bench_label = analysis['comparison_label']

            summary_text = build_single_stage_summary(focus_flow, use='amt')

            unified_results[metric_name] = {
                'slides': {
                    'Funnel': {
                        'summary': {'summary_text': summary_text},
                        'slide_info': {
                            'title': f"{metric_name} - Funnel Analysis",
                            'template_text': summary_text,
                            'template_params': {},
                            'figure_generators': [
                                {
                                    "function": plot_funnel_donut,
                                    "params": {"flow_data": focus_flow, "comparison_flow": bench_flow,
                                               "comparison_label": bench_label, "use": "amt",
                                               "stage_transitions": config['stage_transitions']},
                                    "caption": f"Funnel breakdown coming from {stage}. Left:{region} vs Right: {bench_label}.\nOuter ring shows main funnel categories (solid colors), inner ring shows top sub-reasons for blocked/lost."
                                }
                            ],
                            'layout_type': "text_figure",
                            'total_hypotheses': 1
                        }
                    }
                },
                'payload': {"analysis_data": analysis, "stage": stage, "anomalous_region": region}
            }

        except Exception as e:
            print(f"[analyze_funnel_depth] {metric_name}: {e}")
            continue

    return unified_results

def analyze_funnel_reasons_all_regions(df_lost, df_block, config, *, 
                                      lost_columns, blocked_columns):
    """
    Analyze funnel reasons for ALL regions in the data, not just anomalous ones.
    Creates dummy anomaly map for each region and generates comprehensive results.
    
    Args:
        df_lost, df_block: input DataFrames
        config: configuration dict (without column mappings)
        lost_columns: dict mapping required lost data column names to actual DataFrame column names
        blocked_columns: dict mapping required blocked data column names to actual DataFrame column names
        
    Returns:
        dict: {metric_name: {region_name: unified_results_dict}}
    """
    # Get all unique regions from the data
    config_with_columns = {**config, 'columns': {**lost_columns, **blocked_columns}}
    all_regions = df_lost[lost_columns['lost_territory']].unique()
    
    all_results = {}
    metrics_config = config.get('metrics', {})
    
    for metric_name in metrics_config.keys():
        all_results[metric_name] = {}
        
        for region in all_regions:
            # Create dummy anomaly map for this region
            dummy_anomaly_map = {metric_name: {'anomalous_region': region}}
            
            # Analyze this region as if it were anomalous
            region_results = analyze_funnel_reasons(
                df_lost, df_block, config, dummy_anomaly_map,
                lost_columns=lost_columns, blocked_columns=blocked_columns
            )
            
            if metric_name in region_results:
                # Add region name to slide titles for clarity
                for slide_type, slide_data in region_results[metric_name]['slides'].items():
                    original_title = slide_data['slide_info']['title']
                    slide_data['slide_info']['title'] = f"{region} - {original_title}"
                
                all_results[metric_name][region] = region_results[metric_name]
    
    return all_results

def analyze_funnel_reasons_data(df_lost, df_block, *, region, stage, config, 
                              comparison_region=None, comparison_type='rest_of_world'):
    """Clean interface for funnel analysis. Returns data only.
    
    Args:
        df_lost, df_block: input DataFrames
        region, stage: focus region and stage
        config: configuration dict
        comparison_region: specific region for comparison (overrides comparison_type)
        comparison_type: 'rest_of_world' | 'global_average' | region name
        
    Returns:
        dict: Contains region_flow, comparison_flow, comparison_label, gap_metrics
    """
    df_norm = normalize_funnel(df_lost, df_block, config)
    
    region_flow = stage_flow(df_norm, df_lost, df_block, region=region, stage=stage, config=config)
    
    # Handle comparison - use comparison_region if provided, otherwise use comparison_type
    available_regions = df_lost[config['columns']['lost_territory']].unique()
    
    if comparison_region:
        # Specific region comparison 
        if comparison_region not in available_regions:
            raise ValueError(f"comparison_region '{comparison_region}' not found in data. Available: {available_regions}")
        comparison_flow = stage_flow(df_norm, df_lost, df_block, region=comparison_region, stage=stage, config=config)
        comparison_label = comparison_region
    elif comparison_type in available_regions:
        # Comparison type is a specific region name
        comparison_flow = stage_flow(df_norm, df_lost, df_block, region=comparison_type, stage=stage, config=config)
        comparison_label = comparison_type
    elif comparison_type == 'rest_of_world':
        # Rest of world comparison - aggregate all other regions
        rest_lost, rest_block = _create_aggregated_comparison_data(
            df_lost, df_block, focus_region=region, config=config, comparison_label='Rest of World'
        )
        rest_norm = normalize_funnel(rest_lost, rest_block, config)
        comparison_flow = stage_flow(rest_norm, rest_lost, rest_block, region='Rest of World', stage=stage, config=config)
        comparison_label = 'Rest of World'
    elif comparison_type == 'global_average':
        # Global average - aggregate all regions
        global_lost, global_block = _create_aggregated_comparison_data(
            df_lost, df_block, focus_region=None, config=config, comparison_label='Global'
        )
        global_norm = normalize_funnel(global_lost, global_block, config)
        comparison_flow = stage_flow(global_norm, global_lost, global_block, region='Global', stage=stage, config=config)
        comparison_label = 'Global'
    else:
        raise ValueError(f"comparison_type must be 'rest_of_world', 'global_average', or a valid region name from: {available_regions}")
    
    # Calculate gap metrics
    r_base = region_flow['base_amt']
    c_base = comparison_flow['base_amt']
    
    gap_metrics = {
        'lost_rate_region': region_flow['lost_amt'] / r_base if r_base > 0 else 0,
        'lost_rate_comparison': comparison_flow['lost_amt'] / c_base if c_base > 0 else 0,
        'blocked_rate_region': region_flow['block_amt'] / r_base if r_base > 0 else 0,
        'blocked_rate_comparison': comparison_flow['block_amt'] / c_base if c_base > 0 else 0,
        'advance_rate_region': region_flow['adv_amt'] / r_base if r_base > 0 else 0,
        'advance_rate_comparison': comparison_flow['adv_amt'] / c_base if c_base > 0 else 0,
    }
    
    gap_metrics.update({
        'lost_rate_diff_pp': (gap_metrics['lost_rate_region'] - gap_metrics['lost_rate_comparison']) * 100,
        'blocked_rate_diff_pp': (gap_metrics['blocked_rate_region'] - gap_metrics['blocked_rate_comparison']) * 100,
        'advance_rate_diff_pp': (gap_metrics['advance_rate_region'] - gap_metrics['advance_rate_comparison']) * 100,
    })
    
    return {
        'region_flow': region_flow,
        'comparison_flow': comparison_flow,
        'comparison_label': comparison_label,
        'gap_metrics': gap_metrics
    }

# =============================================================================
# SUMMARY TEXT BUILDERS - Standardized summary generation
# =============================================================================

def _format_reasons(reasons_list):
    """Format list of reasons into proper English text."""
    if not reasons_list:
        return "no specific reasons identified"
    if len(reasons_list) == 1:
        return reasons_list[0]
    if len(reasons_list) == 2:
        return f"{reasons_list[0]} and {reasons_list[1]}"
    return f"{', '.join(reasons_list[:-1])}, and {reasons_list[-1]}"

def build_single_stage_summary(flow, *, use='amt'):
    """Build summary text for single-stage funnel analysis."""
    base = (flow['base_amt'] if use == 'amt' else flow['base_n']) or 0
    lost = (flow['lost_amt'] if use == 'amt' else flow['lost_n'])
    blocked = (flow['block_amt'] if use == 'amt' else flow['block_n'])
    lost_pct = 0 if base == 0 else (lost / base) * 100
    blocked_pct = 0 if base == 0 else (blocked / base) * 100
    lost_reasons_text = _format_reasons(flow.get('lost_reasons_with_pct', []))
    blocked_reasons_text = _format_reasons(flow.get('blocked_reasons_with_pct', []))
    used = '\$PRC' if use == 'amt' else 'solutions'
    return (f"{lost_pct:.1f}% of {used} dropped-off from this stage mainly due to {lost_reasons_text}, "
            f"and among the currently active, {blocked_pct:.1f}% was blocked due to {blocked_reasons_text}. "
            )

def build_total_lost_summary(region_label, tl_dict, *, use='amt'):
    """Build summary text for total lost analysis."""
    st = tl_dict['stage_breakdown_df']
    base_primary = (tl_dict['base_amt'] if use == 'amt' else tl_dict['base_n']) or 0
    if st.empty or base_primary == 0:
        return f"{region_label} has no lost data available."
    
    # Stage summary using appropriate metric
    stage_summary = ", ".join([f"{row.lost_pct:.1f}% is from {row.stage}" for _, row in st.head(3).iterrows()])
    
    outer = tl_dict['outer']
    if outer.empty:
        reasons_summary = "no specific reasons identified"
    else:
        # Use appropriate column for grouping and percentage calculation
        val_col = 'val_amt' if use == 'amt' else 'val_n'
        top = (outer.groupby('reason', as_index=False)[val_col]
                    .sum().sort_values(val_col, ascending=False).head(3))
        reasons_summary = ", ".join([f"{r.reason} ({(getattr(r, val_col)/base_primary)*100:.1f}%)" 
                                   for r in top.itertuples()])
    
    metric_type = "\$PRC" if use == 'amt' else "solutions"
    return f"Of all lost {metric_type} in {region_label}, {stage_summary}. Top drivers: {reasons_summary}."

# =============================================================================
# SYNTHETIC DATA GENERATION - For testing
# =============================================================================

def create_funnel_synthetic_data():
    """Create synthetic funnel data for testing."""
    np.random.seed(42)
    regions = ["Global", "North America", "Europe", "Asia", "Latin America"]
    
    lost_data = []
    blocked_data = []
    
    regional_profiles = {
        'Global': {'perf_mult': 1.0, 'deal_size': 120000},
        'North America': {'perf_mult': 0.7, 'deal_size': 150000},
        'Europe': {'perf_mult': 1.0, 'deal_size': 120000},
        'Asia': {'perf_mult': 0.85, 'deal_size': 100000},
        'Latin America': {'perf_mult': 0.9, 'deal_size': 110000}
    }
    
    stages = ['Technical Demo', 'Business Proposal', 'Contract Negotiation']
    stage_issues = {
        'Technical Demo': {
            'lost': ['Poor Technical Fit', 'Feature Gaps', 'Performance Issues'],
            'blocked': ['Technical Validation', 'Security Review']
        },
        'Business Proposal': {
            'lost': ['Price Too High', 'Competitor Chosen', 'Timeline Issues'], 
            'blocked': ['Executive Approval', 'Budget Approval']
        },
        'Contract Negotiation': {
            'lost': ['Contract Terms', 'Legal Issues', 'Scope Changes'],
            'blocked': ['Contract Review', 'Compliance Check']
        }
    }
    
    for region in regions:
        profile = regional_profiles[region]
        
        for i, stage in enumerate(stages):
            base_pipeline = 300 - (i * 50)
            current_count = int(base_pipeline * (0.4 + np.random.normal(0, 0.05)))
            current_amount = current_count * profile['deal_size']
            
            lost_rate_base = 0.12 / profile['perf_mult']
            for reason in stage_issues[stage]['lost']:
                reason_rate = lost_rate_base * np.random.uniform(0.3, 1.8)
                lost_count = max(1, int(base_pipeline * reason_rate))
                lost_amount = lost_count * profile['deal_size'] * (1 + np.random.normal(0, 0.1))
                
                stage_total_count = current_count + lost_count
                stage_total_amount = current_amount + lost_amount
                
                lost_data.append({
                    'territory_l4_name': region,
                    'Stage Before Closed': stage,
                    'Type': 'Closed Lost',
                    'Closed Lost Reason': reason.split()[0],
                    'Closed Lost Sub Reason': reason,
                    '# of Solutions Lost': lost_count,
                    '$ PRC Lost': lost_amount,
                    '# of Solutions Currently in Stage': current_count,
                    '$ PRC Currently in Stage': current_amount,
                    'Total # Lost from Stage': lost_count,
                    'Total $ Lost from Stage': lost_amount,
                    'Total # of Solutions (Lost + Current)': stage_total_count,
                    'Total $ PRC (Lost + Current)': stage_total_amount,
                })
            
            block_rate_base = 0.06 * (2 - profile['perf_mult'])
            for reason in stage_issues[stage]['blocked']:
                reason_rate = block_rate_base * np.random.uniform(0.5, 1.5)
                blocked_count = max(1, int(base_pipeline * reason_rate))
                blocked_amount = blocked_count * profile['deal_size'] * (1 + np.random.normal(0, 0.1))
                
                blocked_data.append({
                    'territory_l4_name': region,
                    'Stage Being Blocked': stage,
                    'Blocked Sub Reason': reason,
                    '# of Solutions Blocked': blocked_count,
                    '$ PRC Blocked': blocked_amount,
                    '# of Solutions Currently in Stage': current_count,
                    '$ PRC Currently in Stage': current_amount,
                })
    
    return pd.DataFrame(lost_data), pd.DataFrame(blocked_data)

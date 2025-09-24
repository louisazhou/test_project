import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


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
    if primary_use == 'n':   # show $%
        return f" ({0 if base_amt==0 else num_amt/base_amt:.1%})"
    else:                    # show $ amount
        return f" ({_fmt_money_short(num_amt)})"

def _normalize(df_lost, df_block, config):
    """Normalize data using config column definitions."""
    cols = config['columns']
    
    # Use columns directly as defined in config
    lost = df_lost[[
        cols['lost_territory'], cols['lost_stage'], cols['lost_reason'], 
        cols['lost_count'], cols['lost_amount'], cols['lost_current_count'], 
        cols['lost_current_amount'], cols['lost_total_count'], cols['lost_total_amount']
    ]].copy()
    lost.columns = ['region', 'stage', 'reason', 'n', 'amt', 'curr_n', 'curr_amt', 'stage_total_n', 'stage_total_amt']
    lost['kind'] = 'Lost'

    block = df_block[[
        cols['blocked_territory'], cols['blocked_stage'], cols['blocked_reason'],
        cols['blocked_count'], cols['blocked_amount'], cols['blocked_current_count'], 
        cols['blocked_current_amount']
    ]].copy()
    block.columns = ['region', 'stage', 'reason', 'n', 'amt', 'curr_n', 'curr_amt']
    block['kind'] = 'Blocked'

    # Calculate stage origins
    st_l = (lost.groupby(['region','stage'], as_index=False)
                 .agg(stage_total_n=('stage_total_n','max'), curr_n=('curr_n','max'), lost_n=('n','sum'),
                      stage_total_amt=('stage_total_amt','max'), curr_amt=('curr_amt','max'), lost_amt=('amt','sum')))
    st_l['stage_origin_n'] = st_l['stage_total_n'].fillna(st_l['curr_n'] + st_l['lost_n'])
    st_l['stage_origin_amt'] = st_l['stage_total_amt'].fillna(st_l['curr_amt'] + st_l['lost_amt'])
    lost = lost.merge(st_l[['region','stage','stage_origin_n','stage_origin_amt']], on=['region','stage'], how='left')

    st_b = (block.groupby(['region','stage'], as_index=False)
                 .agg(curr_n=('curr_n','max'), block_n=('n','sum'), curr_amt=('curr_amt','max'), block_amt=('amt','sum')))
    st_b['stage_origin_n'] = st_b['curr_n'] + st_b['block_n']
    st_b['stage_origin_amt'] = st_b['curr_amt'] + st_b['block_amt']
    block = block.merge(st_b[['region','stage','stage_origin_n','stage_origin_amt']], on=['region','stage'], how='left')

    df = pd.concat([
        lost[['region','kind','stage','reason','n','amt','stage_origin_n','stage_origin_amt']],
        block[['region','kind','stage','reason','n','amt','stage_origin_n','stage_origin_amt']]
    ], ignore_index=True)

    df['rate_n'] = (df['n'] / df['stage_origin_n']).fillna(0.0)
    df['rate_amt'] = (df['amt'] / df['stage_origin_amt']).fillna(0.0)

    return df

def pareto_collapse(df_norm, *, coverage=0.8, top_k=3, min_rate=0.03, min_n=5, use='amt'):
    """Apply Pareto filtering."""
    use = 'amt' if use == 'amt' else 'n'
    rate_col = 'rate_amt' if use == 'amt' else 'rate_n'

    def _collapse(g):
        g = g.sort_values(use, ascending=False).reset_index(drop=True)
        core_idx = set(range(min(top_k, len(g))))
        total = g[use].sum()
        if total > 0:
            core_idx |= set(g.index[(g[use]/total).cumsum() <= coverage].tolist())
        keep_mask = g.index.isin(core_idx) | (g[rate_col] >= min_rate) | (g['n'] >= min_n)
        kept, other = g[keep_mask].copy(), g[~keep_mask].copy()
        if not other.empty:
            other_row = pd.DataFrame({
                'region':[g.at[0,'region']], 'kind':[g.at[0,'kind']], 'stage':[g.at[0,'stage']],
                'reason':['Other'], 'n':[other['n'].sum()], 'amt':[other['amt'].sum()],
                'stage_origin_n':[g.at[0,'stage_origin_n']], 'stage_origin_amt':[g.at[0,'stage_origin_amt']]
            })
            other_row['rate_n'] = other_row['n']/other_row['stage_origin_n']
            other_row['rate_amt'] = other_row['amt']/other_row['stage_origin_amt']
            kept = pd.concat([kept, other_row], ignore_index=True)
        return kept

    return (df_norm.groupby(['region','kind','stage'], group_keys=False).apply(_collapse)
                 .sort_values(['region','kind','stage',use], ascending=[True,True,True,False]))

def compute_stage_flow(df_lost, df_block, *, region, stage, config):
    """Compute funnel flow data using direct column access."""
    cols = config['columns']
    stage_transitions = config['stage_transitions']
    
    # Filter data using config column names directly
    Ls = df_lost[(df_lost[cols['lost_territory']] == region) & 
                 (df_lost[cols['lost_stage']] == stage)]
    Bs = df_block[(df_block[cols['blocked_territory']] == region) & 
                  (df_block[cols['blocked_stage']] == stage)]

    # Direct column access
    lost_n = Ls[cols['lost_count']].sum() if not Ls.empty else 0
    lost_amt = Ls[cols['lost_amount']].sum() if not Ls.empty else 0
    curr_n = Ls[cols['lost_current_count']].sum() if not Ls.empty else (Bs[cols['blocked_current_count']].sum() if not Bs.empty else 0)
    curr_amt = Ls[cols['lost_current_amount']].sum() if not Ls.empty else (Bs[cols['blocked_current_amount']].sum() if not Bs.empty else 0)
    block_n = Bs[cols['blocked_count']].sum() if not Bs.empty else 0
    block_amt = Bs[cols['blocked_amount']].sum() if not Bs.empty else 0

    # All stages after current stage (transitioned)
    current_stage_idx = list(stage_transitions.keys()).index(stage) if stage in stage_transitions else None
    if current_stage_idx is not None:
        future_stages = list(stage_transitions.keys())[current_stage_idx + 1:]
        # Add final stage if it exists
        final_stage = stage_transitions.get(list(stage_transitions.keys())[-1])
        if final_stage:
            future_stages.append(final_stage)
        
        adv_n = adv_amt = 0
        for future_stage in future_stages:
            Lfuture = df_lost[(df_lost[cols['lost_territory']] == region) & 
                             (df_lost[cols['lost_stage']] == future_stage)]
            adv_n += Lfuture[cols['lost_total_count']].sum() if not Lfuture.empty else 0
            adv_amt += Lfuture[cols['lost_total_amount']].sum() if not Lfuture.empty else 0
    else:
        adv_n, adv_amt = 0, 0

    in_unblocked_n = max(curr_n - block_n, 0)
    in_unblocked_amt = max(curr_amt - block_amt, 0)
    base_n = lost_n + block_n + in_unblocked_n + adv_n
    base_amt = lost_amt + block_amt + in_unblocked_amt + adv_amt

    # Get top reasons with percentages
    df_norm = _normalize(df_lost, df_block, config)
    region_data = df_norm[(df_norm['region'] == region) & (df_norm['stage'] == stage)]
    
    lost_reasons_with_pct = []
    blocked_reasons_with_pct = []
    
    if not region_data.empty:
        collapsed = pareto_collapse(region_data, coverage=0.8, top_k=3, use='amt')
        
        # Lost reasons with percentages
        lost_data = collapsed[collapsed['kind'] == 'Lost']
        for _, row in lost_data.iterrows():
            pct = (row['amt'] / base_amt * 100) if base_amt > 0 else 0
            lost_reasons_with_pct.append(f"{row['reason']} ({pct:.1f}%)")
        
        # Blocked reasons with percentages  
        blocked_data = collapsed[collapsed['kind'] == 'Blocked']
        for _, row in blocked_data.iterrows():
            pct = (row['amt'] / base_amt * 100) if base_amt > 0 else 0
            blocked_reasons_with_pct.append(f"{row['reason']} ({pct:.1f}%)")

    return {
        'lost_n': lost_n, 'lost_amt': lost_amt, 'block_n': block_n, 'block_amt': block_amt,
        'in_n': in_unblocked_n, 'in_amt': in_unblocked_amt, 'adv_n': adv_n, 'adv_amt': adv_amt,
        'base_n': base_n, 'base_amt': base_amt, 'lost_reasons_with_pct': lost_reasons_with_pct,
        'blocked_reasons_with_pct': blocked_reasons_with_pct, 'stage': stage, 'region': region
    }

def analyze_funnel_interactive(df_lost, df_block, *, region, stage, config, 
                              comparison_region=None, comparison_type='rest_of_world'):
    """Clean interface for funnel analysis. Returns data only."""
    
    region_flow = compute_stage_flow(df_lost, df_block, region=region, stage=stage, config=config)
    cols = config['columns']
    
    # Get available regions for validation
    available_regions = df_lost[config['columns']['lost_territory']].unique()
    
    if comparison_region:
        # Specific region comparison (like Oaxaca pattern)
        if comparison_region not in available_regions:
            raise ValueError(f"comparison_region '{comparison_region}' not found in data. Available: {available_regions}")
        comparison_flow = compute_stage_flow(df_lost, df_block, region=comparison_region, stage=stage, config=config)
        comparison_label = comparison_region
    elif comparison_type == 'rest_of_world':
        rest_lost = df_lost[df_lost[cols['lost_territory']] != region].copy()
        rest_block = df_block[df_block[cols['blocked_territory']] != region].copy()
        
        # Aggregate the rest of world data by reason before applying to compute_stage_flow
        # This prevents duplicate reasons from different regions
        if not rest_lost.empty:
            rest_lost_agg = rest_lost.groupby([
                cols['lost_stage'], cols['lost_reason']
            ]).agg({
                cols['lost_count']: 'sum',
                cols['lost_amount']: 'sum',
                cols['lost_current_count']: 'sum',
                cols['lost_current_amount']: 'sum',
                cols['lost_total_count']: 'sum',
                cols['lost_total_amount']: 'sum'
            }).reset_index()
            rest_lost_agg[cols['lost_territory']] = 'Rest of World'
            # Add other required columns
            for col in rest_lost.columns:
                if col not in rest_lost_agg.columns:
                    rest_lost_agg[col] = rest_lost[col].iloc[0] if not rest_lost.empty else ''
        else:
            rest_lost_agg = rest_lost.copy()
            rest_lost_agg[cols['lost_territory']] = 'Rest of World'
            
        if not rest_block.empty:
            rest_block_agg = rest_block.groupby([
                cols['blocked_stage'], cols['blocked_reason']
            ]).agg({
                cols['blocked_count']: 'sum',
                cols['blocked_amount']: 'sum',
                cols['blocked_current_count']: 'sum',
                cols['blocked_current_amount']: 'sum'
            }).reset_index()
            rest_block_agg[cols['blocked_territory']] = 'Rest of World'
            # Add other required columns
            for col in rest_block.columns:
                if col not in rest_block_agg.columns:
                    rest_block_agg[col] = rest_block[col].iloc[0] if not rest_block.empty else ''
        else:
            rest_block_agg = rest_block.copy()
            rest_block_agg[cols['blocked_territory']] = 'Rest of World'
        
        comparison_flow = compute_stage_flow(rest_lost_agg, rest_block_agg, region='Rest of World', stage=stage, config=config)
        comparison_label = 'Rest of World'
    elif comparison_type == 'global_average':
        # Apply same aggregation logic as rest_of_world
        global_lost = df_lost.copy()
        global_block = df_block.copy()
        
        # Aggregate the global data by reason 
        if not global_lost.empty:
            global_lost_agg = global_lost.groupby([
                cols['lost_stage'], cols['lost_reason']
            ]).agg({
                cols['lost_count']: 'sum',
                cols['lost_amount']: 'sum',
                cols['lost_current_count']: 'sum',
                cols['lost_current_amount']: 'sum',
                cols['lost_total_count']: 'sum',
                cols['lost_total_amount']: 'sum'
            }).reset_index()
            global_lost_agg[cols['lost_territory']] = 'Global'
            # Add other required columns
            for col in global_lost.columns:
                if col not in global_lost_agg.columns:
                    global_lost_agg[col] = global_lost[col].iloc[0] if not global_lost.empty else ''
        else:
            global_lost_agg = global_lost.copy()
            global_lost_agg[cols['lost_territory']] = 'Global'
            
        if not global_block.empty:
            global_block_agg = global_block.groupby([
                cols['blocked_stage'], cols['blocked_reason']
            ]).agg({
                cols['blocked_count']: 'sum',
                cols['blocked_amount']: 'sum',
                cols['blocked_current_count']: 'sum',
                cols['blocked_current_amount']: 'sum'
            }).reset_index()
            global_block_agg[cols['blocked_territory']] = 'Global'
            # Add other required columns
            for col in global_block.columns:
                if col not in global_block_agg.columns:
                    global_block_agg[col] = global_block[col].iloc[0] if not global_block.empty else ''
        else:
            global_block_agg = global_block.copy()
            global_block_agg[cols['blocked_territory']] = 'Global'
        
        comparison_flow = compute_stage_flow(global_lost_agg, global_block_agg, region='Global', stage=stage, config=config)
        comparison_label = 'Global'
    elif comparison_type in available_regions:
        # Handle region name as comparison_type (like Oaxaca pattern)
        comparison_flow = compute_stage_flow(df_lost, df_block, region=comparison_type, stage=stage, config=config)
        comparison_label = comparison_type
    else:
        raise ValueError(f"comparison_type must be 'rest_of_world', 'global_average', or a valid region name from: {available_regions}")
    
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
        'region_flow': region_flow, 'comparison_flow': comparison_flow,
        'comparison_label': comparison_label, 'gap_metrics': gap_metrics
    }

def plot_funnel_donut(flow_data, *, comparison_flow=None, comparison_label=None, use='amt', stage_transitions=None):
    """Plot funnel donut chart(s) from flow data."""
    
    def _plot_single_donut(ax, flow, label, stage_transitions):
        bn, ba = flow['base_n'], flow['base_amt']
        stage = flow['stage']
        
        # Get explicit next stage name
        next_stage = stage_transitions.get(stage, 'Final')
        next_label = f"→ {next_stage}" if next_stage and next_stage != 'Final' else "→ Closed Won"
        
        # Solid colors for outer ring (main categories) - the ones you specified
        OUTER_COLORS = {
            'in_stage': '#5DADE2',     # Blue 
            'advanced': '#2ecc71',     # Green  
            'blocked': '#FFC300',      # Orange
            'lost': '#e74c3c'          # Red
        }
        
        # Softer colors for inner ring (detailed reasons)
        INNER_COLORS = {
            'lost': '#e74c3c80',      # Red with transparency
            'blocked': '#FFC30080'    # Orange with transparency
        }
        
        # Outer ring: main categories
        outer_parts = [
            (next_label, flow['adv_n'], flow['adv_amt'], OUTER_COLORS['advanced']),
            ('Lost', flow['lost_n'], flow['lost_amt'], OUTER_COLORS['lost']),
            ('Blocked', flow['block_n'], flow['block_amt'], OUTER_COLORS['blocked']),
            (f"In {stage}", flow['in_n'], flow['in_amt'], OUTER_COLORS['in_stage']),
        ]
        
        outer_vals = [p[1] if use == 'n' else p[2] for p in outer_parts]
        base_primary = bn if use == 'n' else ba
        outer_vals = [max(0, float(v)) if pd.notna(v) else 0 for v in outer_vals]
        if sum(outer_vals) == 0:
            outer_vals = [1, 0, 0, 0]

        outer_labels = []
        for i, (name, n, amt, _) in enumerate(outer_parts):
            p = 0 if base_primary == 0 else outer_vals[i] / base_primary
            bracket = _bracket(use, n, amt, bn, ba)
            outer_labels.append(f"{name}\n{p:.1%}{bracket}")

        outer_colors = [p[3] for p in outer_parts]
        
        # Plot outer ring (main categories)
        if sum(outer_vals) > 0:
            ax.pie(outer_vals, radius=1.0, labels=outer_labels, colors=outer_colors, 
                  wedgeprops=dict(width=0.26), startangle=90, labeldistance=0.88)

        # Inner ring: detailed sub-reasons (ONLY the important ones from Pareto)
        inner_reasons = []
        inner_vals = []
        inner_colors = []
        
        # Only show reasons if they're significant (Pareto-filtered)
        if flow['lost_amt'] > 0 or flow['block_amt'] > 0:
            # Add lost reasons (keep the percentages for inner labels)
            if 'lost_reasons_with_pct' in flow and flow['lost_reasons_with_pct']:
                for reason_with_pct in flow['lost_reasons_with_pct']:
                    inner_reasons.append(reason_with_pct)  # Keep full label with percentage
                    
                    # Extract actual percentage value for sizing
                    if '(' in reason_with_pct:
                        pct_str = reason_with_pct.split('(')[1].split('%')[0]
                        try:
                            pct_val = float(pct_str) / 100 * (ba if use == 'amt' else bn)
                            inner_vals.append(pct_val)
                        except:
                            inner_vals.append(flow['lost_amt'] * 0.1)  # fallback 
                    else:
                        inner_vals.append(flow['lost_amt'] * 0.1)  # fallback
                    inner_colors.append(INNER_COLORS['lost'])
            
            # Add blocked reasons (keep the percentages for inner labels)
            if 'blocked_reasons_with_pct' in flow and flow['blocked_reasons_with_pct']:
                for reason_with_pct in flow['blocked_reasons_with_pct']:
                    inner_reasons.append(reason_with_pct)  # Keep full label with percentage
                    
                    # Extract actual percentage value for sizing
                    if '(' in reason_with_pct:
                        pct_str = reason_with_pct.split('(')[1].split('%')[0]
                        try:
                            pct_val = float(pct_str) / 100 * (ba if use == 'amt' else bn)
                            inner_vals.append(pct_val)
                        except:
                            inner_vals.append(flow['block_amt'] * 0.1)  # fallback
                    else:
                        inner_vals.append(flow['block_amt'] * 0.1)  # fallback
                    inner_colors.append(INNER_COLORS['blocked'])
        
        # Plot inner ring only if we have significant reasons to show
        if inner_vals and sum(inner_vals) > 0:
            ax.pie(inner_vals, radius=0.74, labels=inner_reasons, colors=inner_colors,
                  wedgeprops=dict(width=0.34), startangle=90, labeldistance=0.45,
                  textprops={'fontsize': 9})

        use_label = "# of Solutions" if use == 'n' else "$PRC"
        base_desc = f"funnel start: {_fmt_money_short(ba) if use == 'amt' else f'{int(bn):,}'}"
        title = f"{label}\n{stage} ({use_label}, {base_desc})"
        ax.set(aspect='equal', title=title)
        ax.title.set_position([0.5, 0.85])  # Move title down more to avoid overlap
        
        at_risk_primary = ((flow['lost_n'] + flow['block_n']) / bn if use == 'n' and bn > 0 else 
                          (flow['lost_amt'] + flow['block_amt']) / ba if ba > 0 else 0)
        ax.text(0, 0, f"{at_risk_primary:.1%}\nBlocked/Lost", 
                ha='center', va='center', fontsize=12, fontweight='bold')

    if comparison_flow:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))
        _plot_single_donut(ax1, flow_data, flow_data['region'], stage_transitions)
        _plot_single_donut(ax2, comparison_flow, comparison_label, stage_transitions)
        fig.suptitle(f'Funnel Analysis: {flow_data["region"]} vs {comparison_label} @ {flow_data["stage"]}', 
                    fontsize=14, y=0.98)  # Move suptitle higher
    else:
        fig, ax = plt.subplots(figsize=(7, 7))
        _plot_single_donut(ax, flow_data, flow_data['region'], stage_transitions)
    
    from matplotlib.patches import Patch
    LEGEND_COLORS = {'lost': '#e74c3c80', 'blocked': '#FFC30080'}  # Inner ring colors (soft)
    legend_handles = [
        Patch(color=LEGEND_COLORS['lost'], label='Lost sub-reasons (inner)'),
        Patch(color=LEGEND_COLORS['blocked'], label='Blocked sub-reasons (inner)')
    ]
    fig.legend(handles=legend_handles, loc='lower center', ncol=2, 
               bbox_to_anchor=(0.5, 0.02), frameon=False, fontsize=10)
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.1, top=0.88 if comparison_flow else 0.95)
    
    return fig

def analyze_funnel_depth(df_lost, df_block, config, metric_anomaly_map):
    """Depth analysis following standard pattern."""
    metrics_config = config.get('metrics', {})
    unified_results = {}
    
    for metric_name, metric_config in metrics_config.items():
        if not metric_anomaly_map or metric_name not in metric_anomaly_map:
            continue
            
        anomalous_region = metric_anomaly_map[metric_name].get('anomalous_region')
        if not anomalous_region:
            continue
        
        try:
            stage_list = metric_config['drop_off_from']
            
            # Handle both single stage and multi-stage metrics
            if isinstance(stage_list, list):
                # Multi-stage analysis - use consolidated function
                analysis_data = total_lost_analysis(df_lost, region=anomalous_region, config=config, use='amt')
                summary_text = analysis_data['summary_text']
                
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
                                        "function": plot_total_lost_comparison,
                                        "params": {
                                            "df_lost": df_lost,
                                            "region": anomalous_region,
                                            "config": config,
                                            "comparison_type": "rest_of_world",
                                            "use": "amt"
                                        },
                                        "caption": f"Left: {anomalous_region} total lost breakdown by stage. Right: Rest-of-world comparison. Inner ring shows stage distribution, outer ring shows top sub-reasons per stage."
                                    }
                                ],
                                'layout_type': "text_figure",
                                'total_hypotheses': 1
                            }
                        }
                    },
                    'payload': {
                        'analysis_data': analysis_data,
                        'stage_breakdown_df': analysis_data['stage_breakdown_df'],
                        'top_reasons_df': analysis_data['top_reasons_df'],
                        'stages': stage_list,
                        'anomalous_region': anomalous_region,
                        'total_hypotheses': 1
                    }
                }
                continue
            else:
                # Regular single-stage analysis
                stage = stage_list
                analysis = analyze_funnel_interactive(
                    df_lost, df_block, region=anomalous_region, stage=stage,
                    config=config, comparison_type='rest_of_world'
                )
            
            region_flow = analysis['region_flow']
            gap_metrics = analysis['gap_metrics']
            
            lost_pct = gap_metrics['lost_rate_region'] * 100
            blocked_pct = gap_metrics['blocked_rate_region'] * 100
            advance_pct = gap_metrics['advance_rate_region'] * 100
            
            def format_reasons(reasons_list):
                if not reasons_list:
                    return "no specific reasons identified"
                elif len(reasons_list) == 1:
                    return reasons_list[0]
                elif len(reasons_list) == 2:
                    return f"{reasons_list[0]} and {reasons_list[1]}"
                else:
                    return f"{', '.join(reasons_list[:-1])}, and {reasons_list[-1]}"
            
            lost_reasons_text = format_reasons(region_flow['lost_reasons_with_pct'])
            blocked_reasons_text = format_reasons(region_flow['blocked_reasons_with_pct'])
            
            summary_text = (
                f"{lost_pct:.1f}% was lost mainly due to {lost_reasons_text}, "
                f"and among the currently active, {blocked_pct:.1f}% was blocked due to {blocked_reasons_text}."
            )
            
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
                                    "params": {
                                        "flow_data": region_flow,
                                        "comparison_flow": analysis['comparison_flow'],
                                        "comparison_label": analysis['comparison_label'],
                                        "use": "amt",
                                        "stage_transitions": config['stage_transitions']
                                    },
                                    "caption": f"Left: {anomalous_region} funnel breakdown. Right: Rest-of-world comparison. Outer ring shows main funnel categories (solid colors), inner ring shows top sub-reasons (Pareto-filtered, soft colors)."
                                }
                            ],
                            'layout_type': "text_figure",
                            'total_hypotheses': 1
                        }
                    }
                },
                'payload': {
                    'analysis_data': analysis, 'stage': stage,
                    'anomalous_region': anomalous_region, 'total_hypotheses': 1
                }
            }
                
        except Exception as e:
            print(f"Error analyzing metric '{metric_name}': {e}")
            continue
    
    return unified_results

def total_lost_analysis(df_lost, *, region, config, use='amt', top_k=5, coverage=0.85, min_share=0.02):
    """Consolidated function for all total lost analysis and visualization data.
    
    Returns everything needed for summary text, visualization, and payload.
    
    Returns:
        dict: Contains all data structures needed for analysis and visualization
    """
    cols = config['columns']
    L = df_lost.loc[df_lost[cols['lost_territory']] == region, [
        cols['lost_stage'], cols['lost_reason'],
        cols['lost_count'], cols['lost_amount']
    ]].rename(columns={
        cols['lost_stage']:'stage', cols['lost_reason']:'reason',
        cols['lost_count']:'n', cols['lost_amount']:'amt'
    }).copy()

    if L.empty:
        return {
            # For summary text
            'stage_breakdown_df': pd.DataFrame(columns=['stage', 'lost_amount', 'lost_pct']),
            'top_reasons_df': pd.DataFrame(columns=['reason', 'lost_amount', 'lost_pct']), 
            'summary_text': f"{region} has no lost data available.",
            # For visualization
            'stage_totals': pd.DataFrame(columns=['stage','n','amt']),
            'outer': pd.DataFrame(columns=['stage','reason','val_n','val_amt']),
            'base_n': 0, 'base_amt': 0
        }

    # Basic totals
    base_n, base_amt = int(L['n'].sum()), float(L['amt'].sum())
    
    # Stage breakdown for summary (sorted by amount, descending)
    stage_breakdown = (L.groupby('stage', as_index=False)[['n','amt']].sum()
                      .sort_values('amt', ascending=False))
    stage_breakdown['lost_pct'] = (stage_breakdown['amt'] / base_amt * 100).round(1)
    stage_breakdown_df = stage_breakdown.rename(columns={'amt': 'lost_amount'}).copy()
    
    # Top reasons across all stages for summary (no stage grouping)
    top_reasons = (L.groupby('reason', as_index=False)[['n','amt']].sum()
                  .sort_values('amt', ascending=False).head(5))
    top_reasons['lost_pct'] = (top_reasons['amt'] / base_amt * 100).round(1)
    top_reasons_df = top_reasons.rename(columns={'amt': 'lost_amount'}).copy()
    
    # Outer ring data for visualization = Pareto-filtered reasons within each stage
    def _collapse(g):
        val_col = 'amt' if use=='amt' else 'n'
        g2 = (g.groupby('reason', as_index=False)[['n','amt']].sum()
                .rename(columns={'n':'val_n','amt':'val_amt'}))
        g2 = g2.sort_values('val_'+('amt' if use=='amt' else 'n'), ascending=False)
        keep = set(g2.head(top_k)['reason'])
        tot = g2['val_'+('amt' if use=='amt' else 'n')].sum()
        if tot > 0:
            keep |= set(g2.loc[g2['val_'+('amt' if use=='amt' else 'n')].cumsum()/tot <= coverage, 'reason'])
        denom = base_amt if use=='amt' else base_n
        if denom > 0:
            keep |= set(g2.loc[g2['val_'+('amt' if use=='amt' else 'n')]/denom >= min_share, 'reason'])
        kept, other = g2[g2['reason'].isin(keep)].copy(), g2[~g2['reason'].isin(keep)]
        if not other.empty:
            kept.loc[len(kept)] = {'reason':f"Other ({g.name})",
                                   'val_n':other['val_n'].sum(),'val_amt':other['val_amt'].sum()}
        kept.insert(0, 'stage', g.name)
        return kept

    outer = (L.groupby('stage', group_keys=False).apply(_collapse).reset_index(drop=True))
    
    # Create summary text
    stage_texts = []
    for _, row in stage_breakdown_df.head(3).iterrows():
        stage_texts.append(f"{row['lost_pct']:.1f}% in {row['stage']}")
    stage_summary = ", ".join(stage_texts)
    
    reason_texts = []
    for _, row in top_reasons_df.head(3).iterrows():
        reason_texts.append(f"{row['reason']} ({row['lost_pct']:.1f}%)")
    reason_summary = ", ".join(reason_texts)
    
    summary_text = (f"{region} lost {stage_summary}. "
                   f"Top drivers: {reason_summary}.")
    
    return {
        'stage_breakdown_df': stage_breakdown_df,
        'top_reasons_df': top_reasons_df, 
        'summary_text': summary_text,
        'outer': outer,
        'base_n': base_n, 
        'base_amt': base_amt
    }

def plot_total_lost_donut(total_lost, *, region, use='amt', stage_order=None, stage_color_map=None):
    """
    total_lost: output of total_lost_analysis(...)
    Inner = stages (share of ALL lost). Outer = top reasons within each stage.
    Labels follow your bracket rule (count→show $%; dollar→show $K/M/B).
    """

    
    st = total_lost['stage_breakdown_df'].copy()
    outer = total_lost['outer'].copy()
    bn, ba = total_lost['base_n'], total_lost['base_amt']
    if (use=='n' and bn==0) or (use=='amt' and ba==0):
        fig, ax = plt.subplots(figsize=(7,7)); ax.text(0.5,0.5,"No Lost data",ha='center'); return fig

    # Stage colors
    stages = stage_order or list(st.sort_values('lost_amount', ascending=False)['stage'])
    if stage_color_map is None:
        cmap = plt.get_cmap('tab20')
        stage_color_map = {s: cmap(i % cmap.N) for i, s in enumerate(stages)}

    # --- Inner ring - use stage_breakdown_df structure
    st['val'] = st['lost_amount']  # Always use amount for display
    # Use neutral colors that don't imply good/bad
    neutral_colors = ['#3498DB', '#4ECDC4', '#FFC300', '#FF9F43', '#AF7AC5']
    st['color'] = [neutral_colors[i % len(neutral_colors)] for i in range(len(st))]
    base_primary = ba  # Use base amount for percentages

    inner_labels = [f"{row.stage}\n{row.lost_pct:.1f}%" +
                    _bracket('amt', 0, row.lost_amount, bn, ba)
                    for _, row in st.iterrows()]
    fig, ax = plt.subplots(figsize=(7,7))
    ax.pie(st['val'].to_numpy(), radius=0.74, labels=inner_labels,
           colors=st['color'], wedgeprops=dict(width=0.34),
           startangle=90, labeldistance=0.45)

    # --- Outer ring (Stage — Reason) - follow regular funnel donut pattern
    if not outer.empty:
        outer['val_primary'] = outer['val_n'] if use=='n' else outer['val_amt']
        outer = outer.sort_values(['stage','val_primary'], ascending=[True,False])
        outer_vals = outer['val_primary'].to_numpy()
        
        # Smart labeling like regular funnel donut - only label significant slices
        outer_labels = []
        total_val = sum(outer_vals)
        for i, (_, row) in enumerate(outer.iterrows()):
            val = outer_vals[i]
            pct = val / total_val if total_val > 0 else 0
            
            # Only show label for slices > 3% or top 6 items
            if pct > 0.03 or i < 6:
                reason_short = row['reason'][:15] + '...' if len(row['reason']) > 15 else row['reason']
                outer_labels.append(f"{reason_short}\n{pct:.1%}")
            else:
                outer_labels.append('')  # No label for small slices
        
        # Use VERY different colors for outer ring (sub-reasons) - lighter/pastel
        outer_colors = []
        # Generate light/pastel versions of the inner ring colors dynamically
        for _, row in outer.iterrows():
            stage = row['stage']
            # Find which position this stage is in the stage totals
            stage_idx = st[st['stage'] == stage].index[0] if stage in st['stage'].values else 0
            # Get corresponding inner color and make it very light
            inner_color = neutral_colors[stage_idx % len(neutral_colors)]
            
            # Convert hex to very light version
            if inner_color.startswith('#'):
                # Extract RGB and make very light (closer to white)
                r = int(inner_color[1:3], 16)
                g = int(inner_color[3:5], 16) 
                b = int(inner_color[5:7], 16)
                # Make it 85% lighter (closer to white)
                light_r = int(r + (255 - r) * 0.85)
                light_g = int(g + (255 - g) * 0.85)
                light_b = int(b + (255 - b) * 0.85)
                light_color = f"#{light_r:02x}{light_g:02x}{light_b:02x}"
                outer_colors.append(light_color)
            else:
                outer_colors.append('#F5F5F5')  # Default light gray
        
        ax.pie(outer_vals, radius=1.0, labels=outer_labels, colors=outer_colors,
               wedgeprops=dict(width=0.26), startangle=90, labeldistance=0.88, 
               textprops={'fontsize':8})

    title_metric = "# of Solutions Lost" if use=='n' else "$PRC Lost"
    base_txt = f"{int(bn):,}" if use=='n' else _fmt_money_short(ba)
    ax.set(aspect='equal', title=f"{region} — Total Lost Breakdown ({title_metric}, total={base_txt})")
    # Better center text
    total_amt_text = _fmt_money_short(ba)
    ax.text(0, 0, f"Total Lost\n{total_amt_text}", ha='center', va='center', fontsize=12, fontweight='bold')
    
    # Add correct legend - only for stages that actually exist in the data
    legend_elements = []
    for i, stage in enumerate(st['stage']):
        color = neutral_colors[i % len(neutral_colors)]
        legend_elements.append(Patch(facecolor=color, label=stage))
    
    if legend_elements:
        ax.legend(handles=legend_elements, loc='center', bbox_to_anchor=(0.5, -0.15), 
                 ncol=2, frameon=False, fontsize=10)
    
    plt.tight_layout()
    return fig

def plot_total_lost_comparison(df_lost, *, region, config, comparison_region=None, comparison_type='rest_of_world', use='amt'):
    """Side-by-side total lost comparison using the same logic as funnel_donut_comparison."""
    cols = config['columns']
    available_regions = df_lost[cols['lost_territory']].unique()
    
    # Get focus region data
    region_breakdown = total_lost_analysis(df_lost, region=region, config=config, use=use)
    
    # Get comparison data using same logic as analyze_funnel_interactive
    if comparison_region:
        if comparison_region not in available_regions:
            raise ValueError(f"comparison_region '{comparison_region}' not found in data. Available: {available_regions}")
        comparison_breakdown = total_lost_analysis(df_lost, region=comparison_region, config=config, use=use)
        comparison_label = comparison_region
    elif comparison_type == 'rest_of_world':
        rest_lost = df_lost[df_lost[cols['lost_territory']] != region].copy()
        # Aggregate rest of world data
        if not rest_lost.empty:
            rest_lost_agg = rest_lost.groupby([
                cols['lost_stage'], cols['lost_reason']
            ]).agg({
                cols['lost_count']: 'sum',
                cols['lost_amount']: 'sum'
            }).reset_index()
            rest_lost_agg[cols['lost_territory']] = 'Rest of World'
            for col in rest_lost.columns:
                if col not in rest_lost_agg.columns:
                    rest_lost_agg[col] = rest_lost[col].iloc[0] if not rest_lost.empty else ''
        else:
            rest_lost_agg = rest_lost.copy()
            rest_lost_agg[cols['lost_territory']] = 'Rest of World'
        comparison_breakdown = total_lost_analysis(rest_lost_agg, region='Rest of World', config=config, use=use)
        comparison_label = 'Rest of World'
    elif comparison_type == 'global_average':
        global_lost = df_lost.copy()
        # Aggregate global data
        if not global_lost.empty:
            global_lost_agg = global_lost.groupby([
                cols['lost_stage'], cols['lost_reason']
            ]).agg({
                cols['lost_count']: 'sum',
                cols['lost_amount']: 'sum'
            }).reset_index()
            global_lost_agg[cols['lost_territory']] = 'Global'
            for col in global_lost.columns:
                if col not in global_lost_agg.columns:
                    global_lost_agg[col] = global_lost[col].iloc[0] if not global_lost.empty else ''
        else:
            global_lost_agg = global_lost.copy()
            global_lost_agg[cols['lost_territory']] = 'Global'
        comparison_breakdown = total_lost_analysis(global_lost_agg, region='Global', config=config, use=use)
        comparison_label = 'Global'
    elif comparison_type in available_regions:
        comparison_breakdown = total_lost_analysis(df_lost, region=comparison_type, config=config, use=use)
        comparison_label = comparison_type
    else:
        raise ValueError(f"comparison_type must be 'rest_of_world', 'global_average', or a valid region name from: {available_regions}")
    
    # Create side-by-side chart
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))
    
    # Use same stage order and colors for both sides
    stage_order = list(config['stage_transitions'].keys())
    cmap = plt.get_cmap('tab20')
    stage_color_map = {s: cmap(i % cmap.N) for i, s in enumerate(stage_order)}
    
    # Plot left side
    _plot_single_total_lost(ax1, region_breakdown, region, use, stage_order, stage_color_map)
    
    # Plot right side  
    _plot_single_total_lost(ax2, comparison_breakdown, comparison_label, use, stage_order, stage_color_map)
    
    fig.suptitle(f'Total Lost Analysis: {region} vs {comparison_label}', fontsize=14, y=0.98)
    
    # Add unified legend at bottom - using the correct inner ring colors
    legend_elements = []
    neutral_colors = ['#3498DB', '#4ECDC4', '#FFC300', '#FF9F43', '#AF7AC5']
    
    # Get stages from the actual data (both sides might have different stages)
    all_stages = set()
    if not region_breakdown['stage_breakdown_df'].empty:
        all_stages.update(region_breakdown['stage_breakdown_df']['stage'])
    if not comparison_breakdown['stage_breakdown_df'].empty:
        all_stages.update(comparison_breakdown['stage_breakdown_df']['stage'])
    
    for i, stage in enumerate(sorted(all_stages)):
        color = neutral_colors[i % len(neutral_colors)]
        legend_elements.append(Patch(facecolor=color, label=stage))
    
    if legend_elements:
        fig.legend(handles=legend_elements, loc='center', bbox_to_anchor=(0.5, 0.02), 
                  ncol=min(len(legend_elements), 3), frameon=False, fontsize=10)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.88, bottom=0.15)
    
    return fig

def _plot_single_total_lost(ax, total_lost, region, use, stage_order, stage_color_map):
    """Helper to plot single total lost donut on given axes."""

    
    st = total_lost['stage_breakdown_df'].copy()
    outer = total_lost['outer'].copy()
    bn, ba = total_lost['base_n'], total_lost['base_amt']
    
    if (use=='n' and bn==0) or (use=='amt' and ba==0):
        ax.text(0.5, 0.5, "No Lost data", ha='center', va='center')
        return

    # Inner ring (stages) - use same neutral colors  
    st['val'] = st['lost_amount']  # Always use amount
    neutral_colors = ['#3498DB', '#4ECDC4', '#FFC300', '#FF9F43', '#AF7AC5']
    st['color'] = [neutral_colors[i % len(neutral_colors)] for i in range(len(st))]
    base_primary = ba  # Use base amount

    inner_labels = [f"{row.stage}\n{row.lost_pct:.1f}%" +
                    _bracket('amt', 0, row.lost_amount, bn, ba)
                    for _, row in st.iterrows()]
    
    ax.pie(st['val'].to_numpy(), radius=0.74, labels=inner_labels,
           colors=st['color'], wedgeprops=dict(width=0.34),
           startangle=90, labeldistance=0.45)

    # Outer ring - same smart labeling as single chart
    if not outer.empty:
        outer['val_primary'] = outer['val_n'] if use=='n' else outer['val_amt']
        outer = outer.sort_values(['stage','val_primary'], ascending=[True,False])
        outer_vals = outer['val_primary'].to_numpy()
        
        # Smart labeling - only label significant slices
        outer_labels = []
        total_val = sum(outer_vals)
        for i, (_, row) in enumerate(outer.iterrows()):
            val = outer_vals[i]
            pct = val / total_val if total_val > 0 else 0
            
            # Only show label for slices > 3% or top 4 items (smaller for comparison)
            if pct > 0.05 or i < 4:
                reason_short = row['reason'][:12] + '...' if len(row['reason']) > 12 else row['reason']
                outer_labels.append(f"{reason_short}\n{pct:.1%}")
            else:
                outer_labels.append('')  # No label for small slices
        
        # Use dynamically generated light colors for outer ring  
        outer_colors = []
        for _, row in outer.iterrows():
            stage = row['stage']
            # Find which position this stage is in the stage totals
            stage_idx = st[st['stage'] == stage].index[0] if stage in st['stage'].values else 0
            # Get corresponding inner color and make it very light
            inner_color = neutral_colors[stage_idx % len(neutral_colors)]
            
            # Convert hex to very light version
            if inner_color.startswith('#'):
                # Extract RGB and make very light (closer to white)
                r = int(inner_color[1:3], 16)
                g = int(inner_color[3:5], 16) 
                b = int(inner_color[5:7], 16)
                # Make it 85% lighter (closer to white)
                light_r = int(r + (255 - r) * 0.85)
                light_g = int(g + (255 - g) * 0.85)
                light_b = int(b + (255 - b) * 0.85)
                light_color = f"#{light_r:02x}{light_g:02x}{light_b:02x}"
                outer_colors.append(light_color)
            else:
                outer_colors.append('#F5F5F5')  # Default light gray
        
        ax.pie(outer_vals, radius=1.0, labels=outer_labels, colors=outer_colors,
               wedgeprops=dict(width=0.26), startangle=90, labeldistance=0.88, 
               textprops={'fontsize':7})

    title_metric = "# of Solutions Lost" if use=='n' else "$PRC Lost"
    base_txt = f"{int(bn):,}" if use=='n' else _fmt_money_short(ba)
    ax.set(aspect='equal', title=f"{region}\n({title_metric}, total={base_txt})")
    ax.title.set_position([0.5, 0.85])
    
    # Better center text
    total_amt_text = _fmt_money_short(ba) if use=='amt' else f"{int(bn):,}"
    ax.text(0, 0, f"Total Lost\n{total_amt_text}", ha='center', va='center', fontsize=10, fontweight='bold')

def create_funnel_synthetic_data():
    """Create synthetic funnel data."""
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
# pyre-ignore-all-errors
"""
Oaxaca-Blinder Visualization and Storyboard Module

A comprehensive visualization toolkit for Oaxaca-Blinder decomposition results:
- Individual plot functions for different aspects (waterfall, contributions, etc.)
- Executive storyboard that automatically selects the most relevant visuals
- Plain-English captions tied to the narrative classification

Usage:
    from rca_package.viz_oaxaca import show_storyboard, plot_waterfall
    
    # Auto-select best visuals for the story
    show_storyboard(result, region="WEST", max_charts=2)
    
    # Or use individual plots
    fig, ax = plot_waterfall(result, "WEST")
    plt.show()
"""

from typing import List, Tuple, Dict, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --------- Utility Functions ---------

def _pp(x: float) -> float:
    """decimal → percentage points"""
    return float(x) * 100.0

def _get_region_rows(result, region: str) -> pd.DataFrame:
    rows = result.decomposition_df
    return rows[rows["region"] == region].copy()

def _get_region_totals(result, region: str) -> pd.Series:
    g = result.regional_gaps
    if region not in set(g["region"]):
        raise ValueError(f"Region '{region}' not found in result.regional_gaps.")
    return g[g["region"] == region].iloc[0]

def _common_mix_gap(rows: pd.DataFrame) -> float:
    """
    Compute common-mix rate gap (decimal).
    Expects columns: region_rate, rest_rate, region_mix_pct, rest_mix_pct
    Optional: region_trials, rest_trials
    """
    if {"region_trials","rest_trials"}.issubset(rows.columns):
        w = rows["region_trials"].fillna(0) + rows["rest_trials"].fillna(0)
    else:
        w = rows["region_mix_pct"] + rows["rest_mix_pct"]
    w = w / w.sum() if w.sum() else w
    diff = rows["region_rate"] - rows["rest_rate"]
    return float((w * diff).sum())

def _pooled_gap(rows: pd.DataFrame) -> float:
    """Headline pooled gap (decimal)."""
    return float(
        (rows["region_mix_pct"] * rows["region_rate"]).sum()
        - (rows["rest_mix_pct"] * rows["rest_rate"]).sum()
    )

def _category_label(s: pd.Series) -> str:
    if "category_clean" in s.index:
        return str(s["category_clean"])
    return str(s["category"])

# --------- 1) Waterfall Chart ---------

def plot_waterfall(result, region: str, title: Optional[str] = None):
    """
    Shows how the overall gap splits into Mix vs Execution (pp).
    """
    t = _get_region_totals(result, region)
    base = _pp(t["baseline_overall_rate"])
    mix = _pp(t["total_construct_gap"])
    exe = _pp(t["total_performance_gap"])
    final = _pp(t["region_overall_rate"])

    steps = ["Baseline rate", "Mix effect", "Execution effect", "Region rate"]
    # Waterfall positions
    y = [base, base + mix, base + mix + exe, final]
    deltas = [np.nan, mix, exe, np.nan]  # for bar heights

    fig, ax = plt.subplots(figsize=(8, 4.5))
    
    # Color scheme: baseline/region (blue), positive effects (green), negative effects (red)
    colors = ['steelblue', 'green' if mix >= 0 else 'red', 'green' if exe >= 0 else 'red', 'steelblue']
    
    # Baseline and Region bars (solid)
    ax.bar([0], [base], color=colors[0], alpha=0.7, label="Baseline rate")
    ax.bar([3], [final], color=colors[3], alpha=0.7, label="Region rate")

    # Mix + Execution as floating bars
    ax.bar([1], [mix], bottom=base, color=colors[1], alpha=0.7, label="Mix effect")
    ax.bar([2], [exe], bottom=base + mix, color=colors[2], alpha=0.7, label="Execution effect")

    # Value labels
    ax.text(0, base/2, f"{base:.1f}pp", ha="center", va="center", fontweight='bold')
    ax.text(1, base + mix/2, f"{mix:+.1f}pp", ha="center", va="center", fontweight='bold')
    ax.text(2, base + mix + exe/2, f"{exe:+.1f}pp", ha="center", va="center", fontweight='bold')
    ax.text(3, final/2, f"{final:.1f}pp", ha="center", va="center", fontweight='bold')

    # Connect with lines
    for i in range(3):
        ax.plot([i, i+1], [y[i], y[i]], 'k--', alpha=0.5, linewidth=1)
        ax.plot([i, i+1], [y[i+1], y[i+1]], 'k--', alpha=0.5, linewidth=1)

    ax.set_xticks(range(4), steps, rotation=0)
    ax.set_ylabel("Rate (percentage points)")
    ax.set_title(title or f"{region}: Decomposition Waterfall")
    ax.axhline(0, color='black', linewidth=1, alpha=0.3)
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig, ax

# --------- 2) Pooled vs Common-mix Comparison ---------

def plot_pooled_vs_common_mix(result, region: str, title: Optional[str] = None):
    """
    Compares headline (pooled) gap vs apples-to-apples (common-mix) gap (pp).
    """
    rows = _get_region_rows(result, region)
    pooled = _pp(_pooled_gap(rows))
    common = _pp(_common_mix_gap(rows))

    fig, ax = plt.subplots(figsize=(6, 4.5))
    
    colors = ['steelblue', 'orange']
    bars = ax.bar(["Headline gap", "Apples-to-apples"], [pooled, common], color=colors, alpha=0.7)
    
    for i, (bar, v) in enumerate(zip(bars, [pooled, common])):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + (0.5 if height >= 0 else -0.5),
                f"{v:+.1f}pp", ha="center", va="bottom" if height >= 0 else "top", fontweight='bold')
    
    ax.set_ylabel("Gap (percentage points)")
    ax.set_title(title or f"{region}: Headline vs Apples-to-Apples Gap")
    ax.axhline(0, color='black', linewidth=1, alpha=0.3)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig, ax

# --------- 3) Category Contributions Chart ---------

def plot_contributions_by_category(
    result,
    region: str,
    top_n: int = 10,
    sort_by: str = "abs_gap",  # "abs_gap" or "mix_delta"
    title: Optional[str] = None
):
    """
    Stacked bars per category: Mix (allocation) + Execution contributions (pp).
    Positive above 0, negative below 0. Sorted by |net gap| by default.
    """
    rows = _get_region_rows(result, region).copy()

    rows["mix_pp"] = rows["construct_gap_contribution"].astype(float) * 100
    rows["exe_pp"] = rows["performance_gap_contribution"].astype(float) * 100
    rows["net_pp"] = rows["net_gap"].astype(float) * 100
    rows["mix_delta_pp"] = (rows["region_mix_pct"] - rows["rest_mix_pct"]) * 100
    rows["label"] = rows.apply(_category_label, axis=1)

    if sort_by == "mix_delta":
        rows["_sort"] = rows["mix_delta_pp"].abs()
    else:
        rows["_sort"] = rows["net_pp"].abs()

    rows = rows.sort_values("_sort", ascending=False).head(top_n)
    rows = rows.iloc[::-1]  # so biggest is at the top visually if using barh

    fig, ax = plt.subplots(figsize=(10, max(4.5, 0.4 * len(rows))))
    
    # Stack: start with mix, then add execution on top (with sign respected)
    mix_bars = ax.barh(rows["label"], rows["mix_pp"], label="Mix effect (what we sell)", alpha=0.7, color='skyblue')
    exe_bars = ax.barh(rows["label"], rows["exe_pp"], left=rows["mix_pp"], label="Execution effect (how we sell)", alpha=0.7, color='lightcoral')

    # Annotate net on the right of each bar
    for i, (idx, row) in enumerate(rows.iterrows()):
        total_width = row["mix_pp"] + row["exe_pp"]
        ax.text(total_width + (0.2 if total_width >= 0 else -0.2), i, 
                f"{row['net_pp']:+.1f}pp", va="center", 
                ha="left" if total_width >= 0 else "right", fontweight='bold')

    ax.set_xlabel("Contribution to headline gap (percentage points)")
    ax.set_title(title or f"{region}: Top Category Drivers")
    ax.axvline(0, color='black', linewidth=1, alpha=0.3)
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3, axis='x')
    fig.tight_layout()
    return fig, ax

# --------- 4) Despite vs Due To Rates Chart ---------

def plot_despite_due_to_rates(
    result,
    region: str,
    top_n: int = 8,
    min_mix_diff_pp: float = 5.0,
    title: Optional[str] = None
):
    """
    For each category: two bars (Region vs Baseline) showing success rates (pp),
    sorted by exposure difference. Useful to tell the 'despite vs due to' story.
    """
    rows = _get_region_rows(result, region).copy()
    rows["rate_region_pp"] = rows["region_rate"] * 100
    rows["rate_base_pp"] = rows["rest_rate"] * 100
    rows["mix_diff_pp"] = (rows["region_mix_pct"] - rows["rest_mix_pct"]) * 100
    rows["label"] = rows.apply(_category_label, axis=1)

    # pick categories with the biggest |mix diff| first, otherwise by |net gap|
    candidates = rows[rows["mix_diff_pp"].abs() >= min_mix_diff_pp]
    if candidates.empty:
        candidates = rows
    rows = candidates.assign(_rank=candidates["mix_diff_pp"].abs()).sort_values("_rank", ascending=False).head(top_n)

    x = np.arange(len(rows))
    w = 0.35

    fig, ax = plt.subplots(figsize=(max(8, 1.0 * len(rows)), 5))
    
    bars1 = ax.bar(x - w/2, rows["rate_region_pp"], width=w, label="Our rate", alpha=0.7, color='steelblue')
    bars2 = ax.bar(x + w/2, rows["rate_base_pp"], width=w, label="Peers\' rate", alpha=0.7, color='orange')

    # annotate mix difference above bars
    for i, (idx, row) in enumerate(rows.iterrows()):
        max_height = max(row["rate_region_pp"], row["rate_base_pp"])
        ax.text(x[i], max_height + 1,
                f"mix Δ {row['mix_diff_pp']:+.0f}pp", ha="center", va="bottom", 
                fontsize=9, fontweight='bold')

    ax.set_xticks(x, rows["label"], rotation=45, ha="right")
    ax.set_ylabel("Success rate (percentage points)")
    ax.set_title(title or f"{region}: Rates by Category (with exposure deltas)")
    ax.set_ylim(0, max(100, (rows[["rate_region_pp","rate_base_pp"]].values.max() + 8)))
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3, axis='y')
    fig.tight_layout()
    return fig, ax

# --------- 5) Executive Summary Panel ---------

def plot_exec_summary_panel(result, region: str, title: Optional[str] = None):
    """
    A single panel with headline gap and a pie of |Mix| vs |Execution| shares.
    """
    t = _get_region_totals(result, region)
    gap_pp = _pp(t["total_net_gap"])
    r_pp = _pp(t["region_overall_rate"])
    b_pp = _pp(t["baseline_overall_rate"])
    mix_pp = abs(_pp(t["total_construct_gap"]))
    exe_pp = abs(_pp(t["total_performance_gap"]))
    parts = [mix_pp, exe_pp]
    labels = ["Mix effect", "Execution effect"]

    fig = plt.figure(figsize=(10, 5))
    gs = fig.add_gridspec(1, 2, width_ratios=[3, 2])
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])

    # headline text
    ax0.axis("off")
    verdict = "outperforms" if gap_pp > 0 else "underperforms" if gap_pp < 0 else "performs in line with"
    ax0.text(0.0, 0.8, f"{region} {verdict} peers", fontsize=16, weight="bold")
    ax0.text(0.0, 0.65, f"Gap: {gap_pp:+.1f} percentage points", fontsize=14)
    ax0.text(0.0, 0.5, f"Our rate: {r_pp:.1f}pp", fontsize=12)
    ax0.text(0.0, 0.4, f"Peers' rate: {b_pp:.1f}pp", fontsize=12)
    ax0.text(0.0, 0.2, "Share of gap explained (absolute):", fontsize=11, style='italic')

    # pie chart
    if sum(parts) > 0:
        colors = ['skyblue', 'lightcoral']
        wedges, texts, autotexts = ax1.pie(parts, labels=labels, autopct='%1.0f%%', 
                                          startangle=90, colors=colors, alpha=0.8)
        for autotext in autotexts:
            autotext.set_fontweight('bold')
    else:
        ax1.text(0.5, 0.5, "No significant\neffects", ha='center', va='center', transform=ax1.transAxes)
    
    ax1.set_title("Mix vs Execution", fontweight='bold')

    fig.suptitle(title or f"Executive Summary: {region}", fontsize=14, fontweight='bold')
    fig.tight_layout()
    return fig, (ax0, ax1)

# --------- Storyboard Composition ---------

# Import BusinessConclusion safely
try:
    from rca_package.oaxaca_blinder import BusinessConclusion
    _BC = BusinessConclusion
except Exception:
    class _BC:
        PERFORMANCE_DRIVEN = type("X", (), {"value": "performance_driven"})()
        COMPOSITION_DRIVEN = type("X", (), {"value": "composition_driven"})()
        SIMPSON_PARADOX = type("X", (), {"value": "simpson_paradox"})()
        MATHEMATICAL_CANCELLATION = type("X", (), {"value": "cancellation"})()
        NO_SIGNIFICANT_GAP = type("X", (), {"value": "no_gap"})()

def _as_str_root(root) -> str:
    return getattr(root, "value", str(root))

def _mix_exec_shares(totals: pd.Series) -> Dict[str, float]:
    mix = abs(float(totals["total_construct_gap"]))
    exe = abs(float(totals["total_performance_gap"]))
    denom = mix + exe if (mix + exe) > 0 else 1.0
    return {"mix_share": mix/denom, "exe_share": exe/denom}

def _disagree_share(rows: pd.DataFrame, pooled_gap: float) -> float:
    """Share of exposure that disagrees with pooled direction (used in Simpson caption)."""
    if {"region_trials","rest_trials"}.issubset(rows.columns):
        w = rows["region_trials"].fillna(0) + rows["rest_trials"].fillna(0)
    else:
        w = rows["region_mix_pct"] + rows["rest_mix_pct"]
    w = w / w.sum() if w.sum() else w
    diff = rows["region_rate"] - rows["rest_rate"]
    if pooled_gap == 0:
        return float(w[diff != 0].sum())
    return float(w[np.sign(diff) == -np.sign(pooled_gap)].sum())

def _top_driver_snippets(rows: pd.DataFrame, top_k: int = 2) -> str:
    """Return 'Product A (+1.2pp), Product B (–0.8pp)' style snippet by |net_pp|."""
    tmp = rows.copy()
    tmp["net_pp"] = tmp["net_gap"] * 100
    tmp["label"] = tmp.apply(_category_label, axis=1)
    picks = tmp.reindex(tmp["net_pp"].abs().sort_values(ascending=False).index).head(top_k)
    return ", ".join([f"{picks.iloc[i]['label']} ({picks.iloc[i]['net_pp']:+.1f}pp)" for i in range(len(picks))])

def compose_executive_storyboard(result, region: str, max_charts: int = 2) -> Dict:
    """
    Picks 1–2 charts that best support the region narrative, and builds plain-English captions.
    Returns:
      {
        "headline": <narrative string>,
        "exhibits": [(fig, ax, caption_str), ...]
      }
    """
    # 1) Pull narrative + core numbers
    decision = result.narrative_decisions[region]
    totals = _get_region_totals(result, region)
    base_pp = _pp(totals["baseline_overall_rate"])
    regn_pp = _pp(totals["region_overall_rate"])
    gap_pp  = _pp(totals["total_net_gap"])
    shares  = _mix_exec_shares(totals)
    rows    = _get_region_rows(result, region)

    exhibits: List[Tuple[plt.Figure, plt.Axes, str]] = []
    root = _as_str_root(decision.root_cause_type)

    # 2) Pick charts by root-cause type
    if root == _BC.SIMPSON_PARADOX.value:
        # A. Headline vs apples-to-apples (to show the flip)
        fig1, ax1 = plot_pooled_vs_common_mix(result, region)
        pooled = _pp(_pooled_gap(rows))
        common = _pp(_common_mix_gap(rows))
        share  = _disagree_share(rows, pooled/100.0)
        cap1 = (f"Headline gap {pooled:+.1f}pp vs apples-to-apples {common:+.1f}pp. "
                f"{share:.0%} of exposure pulls the opposite way — allocation explains the headline.")
        exhibits.append((fig1, ax1, cap1))

        # B. "Despite vs Due to" (rates + exposure deltas)
        if len(exhibits) < max_charts:
            fig2, ax2 = plot_despite_due_to_rates(result, region, top_n=8)
            cap2 = ("Bars show our success rate vs peers by category; labels show exposure deltas (mix Δ). "
                    "Despite better rates in several categories, the exposure mix drives the overall result.")
            exhibits.append((fig2, ax2, cap2))

    elif root == _BC.COMPOSITION_DRIVEN.value:
        # A. Waterfall (baseline → mix → execution → region)
        fig1, ax1 = plot_waterfall(result, region)
        cap1 = (f"From peers {base_pp:.1f}pp to us {regn_pp:.1f}pp. "
                f"Mix effect { _pp(totals['total_construct_gap']):+,.1f}pp; "
                f"Execution effect { _pp(totals['total_performance_gap']):+,.1f}pp. "
                f"~{shares['mix_share']:.0%} of the overall gap is explained by what we sell.")
        exhibits.append((fig1, ax1, cap1))

        # B. Top drivers stacked contribution (sort by allocation)
        if len(exhibits) < max_charts:
            fig2, ax2 = plot_contributions_by_category(result, region, top_n=8, sort_by="mix_delta")
            cap2 = ("Top categories by exposure misalignment. Positive bars help the gap; negative bars hurt. "
                    f"Biggest contributors: {_top_driver_snippets(rows, top_k=2)}.")
            exhibits.append((fig2, ax2, cap2))

    elif root == _BC.PERFORMANCE_DRIVEN.value:
        # A. Waterfall to establish how much is execution
        fig1, ax1 = plot_waterfall(result, region)
        cap1 = (f"From peers {base_pp:.1f}pp to us {regn_pp:.1f}pp. "
                f"Execution explains ~{shares['exe_share']:.0%} of the total gap.")
        exhibits.append((fig1, ax1, cap1))

        # B. Stacked contributions (sorted by absolute impact)
        if len(exhibits) < max_charts:
            fig2, ax2 = plot_contributions_by_category(result, region, top_n=8, sort_by="abs_gap")
            cap2 = ("Where execution bites most by category (stacked mix + execution). "
                    f"Top drivers: {_top_driver_snippets(rows, top_k=2)}.")
            exhibits.append((fig2, ax2, cap2))

    elif root == _BC.MATHEMATICAL_CANCELLATION.value:
        # Keep it light: show offsetting effects rather than every bar
        fig1, ax1 = plot_contributions_by_category(result, region, top_n=8, sort_by="abs_gap")
        cap1 = ("Offsetting category effects: large positives and negatives mostly cancel — "
                "portfolio result looks 'average' but hides opposing moves underneath.")
        exhibits.append((fig1, ax1, cap1))

        # Optional: add the exec summary instead of more bars
        if len(exhibits) < max_charts:
            fig2, (a0, a1) = plot_exec_summary_panel(result, region)
            cap2 = (f"Net gap {gap_pp:+.1f}pp (Our {regn_pp:.1f}pp vs Peers {base_pp:.1f}pp). "
                    "Cancellation means no single lever dominates.")
            exhibits.append((fig2, a0, cap2))  # a0 is fine for caption anchor

    else:  # NO_SIGNIFICANT_GAP
        # Show summary only (or nothing)
        fig1, (a0, a1) = plot_exec_summary_panel(result, region)
        cap1 = (f"Performance is in line: {regn_pp:.1f}pp vs {base_pp:.1f}pp (Δ {gap_pp:+.1f}pp).")
        exhibits.append((fig1, a0, cap1))

    # 3) Trim to requested count and return
    exhibits = exhibits[:max_charts]
    return {
        "headline": decision.narrative_text,  # already plain-English from your engine
        "exhibits": exhibits
    }

def show_storyboard(result, region: str, max_charts: int = 2):
    """Convenience: render the chosen story and print captions."""
    story = compose_executive_storyboard(result, region, max_charts=max_charts)
    print(f"\n💼 Executive headline: {story['headline']}")
    for i, (fig, ax, cap) in enumerate(story["exhibits"], 1):
        print(f"\nExhibit {i} — {cap}")
    plt.show()

# Convenience function for showing all key visuals
def show_comprehensive_analysis(result, region: str):
    """Show all major chart types for comprehensive analysis."""
    print(f"\n🔍 COMPREHENSIVE ANALYSIS: {region}")
    print("=" * 50)
    
    # Executive summary
    fig1, (ax1a, ax1b) = plot_exec_summary_panel(result, region)
    plt.show()
    
    # Waterfall
    fig2, ax2 = plot_waterfall(result, region)
    plt.show()
    
    # Contributions by category
    fig3, ax3 = plot_contributions_by_category(result, region, top_n=10)
    plt.show()
    
    # Pooled vs common mix (if applicable)
    try:
        fig4, ax4 = plot_pooled_vs_common_mix(result, region)
        plt.show()
    except:
        print("Note: Pooled vs common-mix comparison not applicable for this case")
    
    # Despite/due to rates
    try:
        fig5, ax5 = plot_despite_due_to_rates(result, region)
        plt.show()
    except:
        print("Note: Despite/due to analysis not applicable for this case")

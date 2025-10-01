"""
Slice-level Business Injection (after Market)
============================================

Purpose
- Insert up to K slice-level Business slides immediately after a metric's Market slide.
- Selection is purely Market/Depth-driven: pick the top-K slices by Depth "score" with no
  sign/heuristic filters. Each selected slice is then scored directionally by
  comparing the slice against its parent region (reference_row=region).

Typical Usage
- Call from your orchestration after you've aggregated module outputs into
  `unified_results` and before rendering slides, for example:

    unified_results = inject_slice_directional_after_depth(
        unified_results=unified_results,
        regional_df=df,                 # display-named regional table (index: regions)
        slice_df=slice_df,              # display-named slice table with 'slice' column
        config=config_scorer_dict,      # loaded from configs/config_scorer.yaml
        metric_anomaly_map=metric_anomaly_map,  # produced by anomaly detector
        k=2,
        only_metrics=None,              # optional whitelist
        verbose=False                   # set True for per-slice diagnostics
    )

What Triggers Injection
- A metric in `unified_results` has a 'Market' entry with a payload containing
  `summary_df` (full Market/Depth table) indexed by slice names.
- The metric exists in `metric_anomaly_map` to locate the anomalous region used
  as the parent reference for slice-level comparison.

How Selection Works (simple & deterministic)
- Read Market `summary_df` and sort by `score` desc; pick top-K slice names.
- Build a minimal pair DataFrame with two rows: `[region, slice]`, using the
  intersection of columns present in both the regional and slice rows.
- Compute the slice-vs-region delta directly (no anomaly thresholds); attach
  `higher_is_better` from config and use `reference_row=region`.
- Call `score_hypotheses_for_metrics` to produce a Business slide for each
  selected slice; insert those slides immediately after 'Market'.

Assumptions & Restrictions
- Display-named schema:
  - `regional_df` and `slice_df` must already be converted to display names via
    your YAML processor; column names for the metric and hypotheses must match.
  - `slice_df` must include a string column `slice` whose values match the slice
    names used by Depth (i.e., the index of `summary_df`).
- Column availability:
  - The metric column for a given metric must exist in both the region row and the
    slice row; otherwise that slice is skipped.
  - Hypothesis columns missing at slice grain are tolerated by the scorer (they
    are simply ignored in scoring/visualization).
- Module prerequisites:
- This module does not compute Market. It relies on `unified_results[metric]['slides']['Market']`
    and `payload['summary_df']` to exist.
  - Business slides require hypotheses to be defined for the metric in
    `config_scorer.yaml`. If none exist, a slice Directional slide may not render.
- Side effects:
  - `unified_results` is modified in place; slides are re-ordered to place slice
    Business slides immediately after 'Market'.
  - At most K slice slides are added per metric (default K=2).

Non-goals
- No sign logic (harmful vs helpful) and no highlight heuristics. Depth's score
  ranking is the single source of truth for which slices to show.
"""

from typing import Dict, Any, List, Optional
from collections import OrderedDict
import pandas as pd

from rca_package.hypothesis_scorer import score_hypotheses_for_metrics


def _pick_top_slices(depth_entry: Dict[str, Any], k: int = 2) -> List[str]:
    """Pick top-k slices by Market/Depth score (no sign or highlight logic)."""
    try:
        slide_info = depth_entry['slides']['Market']['slide_info']
        payload = depth_entry['payload']
        summary_df: pd.DataFrame = payload.get('summary_df')
        if summary_df is None or summary_df.empty:
            return []
        return (
            summary_df.sort_values('score', ascending=False)
            .head(k)
            .index.tolist()
        )
    except Exception:
        return []


def inject_slice_directional_after_depth(
    unified_results: Dict[str, Dict[str, Any]],
    regional_df: pd.DataFrame,
    slice_df: pd.DataFrame,
    config: Dict[str, Any],
    metric_anomaly_map: Dict[str, Dict[str, Any]],
    k: int = 2,
    only_metrics: Optional[List[str]] = None,
    verbose: bool = False,
) -> Dict[str, Dict[str, Any]]:
    """Insert up to k slice-level Business slides immediately after Market.

    Modifies and returns unified_results with new slides injected per metric.
    """
    for metric_name, metric_bundle in list(unified_results.items()):
        if only_metrics is not None and metric_name not in only_metrics:
            continue
        try:
            slides_dict = metric_bundle.get('slides', {})
            if 'Market' not in slides_dict:
                continue
            if metric_name not in metric_anomaly_map:
                continue
            region = metric_anomaly_map[metric_name]['anomalous_region']

            # Select top slices by score
            top_slices = _pick_top_slices(metric_bundle, k=k)
            if not top_slices:
                continue

            # Rebuild slides in desired order: copy existing, insert after Market
            new_slides = OrderedDict()
            inserted = False

            for key, slide_data in slides_dict.items():
                new_slides[key] = slide_data
                if key == 'Market' and not inserted:
                    for sl in top_slices[:k]:
                        try:
                            # Build paired df [region, slice] with display-named columns
                            if region not in regional_df.index:
                                continue
                            region_row = regional_df.loc[region]
                            # Collect ALL slices under this region (prefer 'region' column; fallback to 'parent_territory')
                            if 'region' in slice_df.columns:
                                group_rows = slice_df[slice_df['region'] == region].copy()
                            elif 'parent_territory' in slice_df.columns:
                                group_rows = slice_df[slice_df['parent_territory'] == region].copy()
                            else:
                                group_rows = slice_df[slice_df['slice'].str.startswith(f"{region}_", na=False)].copy()

                            if group_rows.empty:
                                continue
                            # Align columns intersection across region row and slice rows
                            slice_common_cols = [c for c in regional_df.columns if c in group_rows.columns]
                            if metric_name not in slice_common_cols:
                                continue
                            # Build combined df: region + all slices
                            combined = pd.concat(
                                [
                                    pd.DataFrame([region_row[slice_common_cols]], index=[region]),
                                    group_rows.set_index('slice')[slice_common_cols],
                                ],
                                axis=0
                            )

                            # Build anomaly vs region reference directly (always proceed)
                            if sl not in combined.index:
                                continue
                            metric_val = float(combined.loc[sl, metric_name])
                            ref_val = float(combined.loc[region, metric_name])
                            direction = 'higher' if metric_val > ref_val else 'lower'
                            # Magnitude formatting: pp for rates, % for others
                            name_l = metric_name.lower()
                            if ('pct' in name_l) or ('%' in metric_name) or ('rate' in name_l):
                                magnitude = f"{abs(metric_val - ref_val)*100:.1f}pp"
                            else:
                                magnitude = f"{(abs(metric_val - ref_val)/ref_val*100) if ref_val else 0:.1f}%"

                            try:
                                hib = (
                                    config
                                    .get('metrics', {})
                                    .get(metric_name, {})
                                    .get('higher_is_better', True)
                                )
                            except Exception:
                                hib = True
                            anomaly_info = {
                                'anomalous_region': sl,
                                'metric_val': metric_val,
                                'global_val': ref_val,
                                'direction': direction,
                                'magnitude': magnitude,
                                'higher_is_better': hib,
                            }

                            # Score hypotheses for the slice (reference_row=region)
                            slice_results = score_hypotheses_for_metrics(
                                regional_df=combined,
                                anomaly_map={metric_name: anomaly_info},
                                config=config,
                                reference_row=region,
                            )
                            # Extract slide and retitle
                            if metric_name in slice_results and 'Business' in slice_results[metric_name]['slides']:
                                sd = slice_results[metric_name]['slides']['Business']
                                # Adjust narrative to reference parent region instead of Global
                                try:
                                    txt = sd['slide_info'].get('template_text', '')
                                    if isinstance(txt, str) and txt:
                                        # Replace both 'Global' and 'global' mentions with the parent region name
                                        txt = txt.replace('Global', region).replace('global', region)
                                        sd['slide_info']['template_text'] = txt
                                    # Also adjust summary text if present
                                    if 'summary' in sd and 'summary_text' in sd['summary']:
                                        ssum = sd['summary']['summary_text']
                                        if isinstance(ssum, str) and ssum:
                                            ssum = ssum.replace('Global', region).replace('global', region)
                                            sd['summary']['summary_text'] = ssum
                                except Exception:
                                    pass
                                # Override slide metadata for injected pages: count as a single step
                                sd = {
                                    'summary': sd['summary'],
                                    'slide_info': {
                                        **sd['slide_info'],
                                        'title': f"{metric_name} - Business Ops. Difference ({sl})",
                                        'total_hypotheses': 1,
                                    },
                                }
                                new_slides[f"Business – {sl}"] = sd
                        except Exception as se:
                            if verbose:
                                print(f"Slice-level Directional injection failed for {metric_name} / {sl}: {se}")
                            continue
                    inserted = True

            unified_results[metric_name]['slides'] = new_slides
        except Exception as me:
            if verbose:
                print(f"Slice-level Directional injection failed for {metric_name}: {me}")
            continue

    return unified_results

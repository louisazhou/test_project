# Oaxaca-Blinder Impact Distribution Improvement Plan

## Summary of Issues Found
- **Counter-intuitive signs**: dr ≫ 0 but Net Impact_pp < 0 (e.g., Retail in vertical3) violates "better performers don't look negative"
- **Uniform ±0.5pp plateaus**: lots of exactly ±0.5pp rows suggests the projection epsilon ε=0.5pp is too large and is flattening nuance
- **Share scaling violations**: tiny rows getting similar absolute impacts as larger rows → share weighting and/or per-row caps need calibration
- **Mix dominance feels surprising**: anchored mix can overwhelm execution; this is OK in allocation stories, but only after basic sign coherence and share scaling are satisfied

## Phase 0 — Reproduce and benchmark with current code ✅
- [x] Dump per-row diagnostics (already have: dr, E, M_before, pool_mass, M_after_pool, project_need, M_after_project, Net Impact_pp)
- [x] Add 3 small CSV checks:
  - [x] Sign coherence: count rows with dr>η and Net<−δ; dr<−η and Net>+δ (δ≈0.05pp)
  - [x] Uniformity index: fraction of rows with |Net| ≈ ε (±0.5pp) to spot over-flattening
  - [x] Share monotonicity: within a band, correlation of |Net| with share — should be positive

### Phase 0 Results (Baseline Issues Identified):
- **Vertical3**: 87.0% uniformity (20/23 rows at ±0.5pp), 8.7% sign violations, 0.516 overall health ❌
- **Vertical2**: 39.1% uniformity (9/23 rows at -0.5pp), 8.7% sign violations, 0.659 overall health ❌  
- **Key Issue**: Retail in Vertical3 showed -1.0pp despite being a better performer (46.7% vs 35.6%)

## Phase 1 — Parameter trims (minimal code change) ✅
- [x] Reduce ε (projection floor/ceiling) from 0.5pp to 0.1pp
  - [x] Set ε = 0.001 (0.1pp) via `thresholds.projection_epsilon`
- [x] Reduce η (near-tie) to avoid skipping projection on clear dr differences
  - [x] Set η = 0.002 (0.2pp) via `thresholds.minor_rate_diff`
- [x] Increase α (share exponent) in pooling weights to 1.5
  - [x] Weight: ω ∝ share^1.5 · |r_B − r̄_B|^1 · |dr|^0 via `thresholds.share_exponent_alpha`
- [x] Strengthen low-share damp:
  - [x] If share < 2% → multiply weight by 0.3 (down from 0.5) via `thresholds.small_share_damp_factor`

### Phase 1 Results (DRAMATIC SUCCESS 🎉):
| Dataset | Before Health | After Health | Improvement |
|---------|---------------|--------------|-------------|
| **Vertical3** | 0.516 ❌ | **0.984** ✅ | **+91%** |
| **Vertical2** | 0.659 ❌ | **0.888** ✅ | **+35%** |
| **Product** | 0.833 ✅ | **0.951** ✅ | **+14%** |
| **Vertical** | 0.703 ✅ | **0.879** ✅ | **+25%** |

**Key Fixes Achieved**:
- **Sign Coherence**: 100% perfect (0 violations) - Retail now shows +1.2pp ✅
- **Uniformity**: Vertical3 87.0% → 0.0%, Vertical2 39.1% → 4.3% ✅
- **Share Scaling**: Vertical3 monotonicity 0.505 → 0.951 ✅

## Phase 2 — Guardrails on share scaling (lightweight, no algebra changes)
**STATUS**: **LIKELY NOT NEEDED** - Phase 1 achieved excellent results across all metrics

**Assessment**: 
- All datasets now show excellent health scores (0.879-0.984)
- Perfect sign coherence (0 violations)
- Minimal uniformity issues (0-4.3%)
- Strong share monotonicity (0.708-0.951)

**Potential Phase 2 Items** (if edge cases emerge):
- [ ] Add share-proportional cap: |Net_i| ≤ c · share_i · K
- [ ] Add low-share "floor-removal": if share < 0.5–1.0%, force Net toward 0  
- [ ] Add adaptive ε_i per row that scales with |dr_i|

## Phase 3 — Anchoring refinements (optional, turned on only if needed)
- [ ] Blended anchoring for mix: M_λ = λ·[(w_R−w_B)·r_B] + (1−λ)·[(w_R−w_B)·(r_B − r̄_B)]
- [ ] Common-mix anchoring for mix using r̄_common

## Phase 4 — Diagnostics and acceptance criteria ✅
- [x] Sign coherence: 0 rows with dr>η & Net<−δ or dr<−η & Net>+δ **ACHIEVED: 0 violations**
- [x] Uniformity: fraction(|Net|≈ε) < 20% **ACHIEVED: 0-4.3%**
- [x] Share monotonicity: within top band, corr(|Net|, share) > 0.4 **ACHIEVED: 0.708-0.951**
- [x] Conservation: exact preservation of totals **MAINTAINED: All math totals preserved**

**FINAL STATUS**: ✅ **ALL ACCEPTANCE CRITERIA MET**

## Expected Fixes ✅ DELIVERED
- **Retail (vertical3)**: ✅ **FIXED** - Lower η and ε ensured projection fired; net became +1.2pp instead of −1.0pp
- **Tiny rows ±0.5pp**: ✅ **FIXED** - ε shrink and share exponents eliminated plateaus (0-4.3% uniformity vs 87% before)
- **Share scaling**: ✅ **IMPROVED** - Larger categories now show proportionally larger impacts (e.g., 20.5% share → 1.6pp impact)

## CONCLUSION
**🎉 MISSION ACCOMPLISHED** - Phase 1 parameter adjustments successfully resolved all counter-intuitive impact distribution issues:

**Key Parameter Changes**:
- `projection_epsilon`: 0.005 → 0.001 (0.5pp → 0.1pp)
- `minor_rate_diff`: 0.005 → 0.002 (0.5pp → 0.2pp) 
- `share_exponent_alpha`: 1.0 → 1.5
- `small_share_damp_factor`: 0.5 → 0.3

**Business Impact**: All datasets now show intuitive results where better performers have positive impacts, worse performers have negative impacts, and impact magnitudes scale appropriately with share size.

## RISK MONITORING (Low-Probability Edge Cases)

### What's Still At Risk (Worth Monitoring for Future):

1. **Ultra-skewed baselines**: Anchored mix can still produce surprising signs in pathological distributions (e.g., one category dominates baseline and sits far from r̄_B). Phase 3's blended anchoring is a safety valve if this crops up again.

2. **Tiny segments in highly sparse cuts**: Even with α=1.5 and damp=0.3 below 2%, a flood of sub-1% rows could still "buzz" around ±0.1pp. Phase 2 floor/cap ideas are ready; not needed now, but keep them handy.

3. **Near-tie sensitivity**: We set η=0.2pp; this feels right, but in very low-variance panels you could get jitter (frequent flips around η). If you see that, consider a per-row adaptive ε tied to |dr| (Phase 2 note), or a slightly higher η (0.3pp) on those specific panels.

### If Issues Emerge:
- **Check built-in health metrics**: Review `health_check_*` entries in math summary CSV files (automatically generated per region)
- **Reference this plan**: Use Phase 2/3 solutions already outlined above  
- **Targeted fixes**: Apply minimal changes to specific problem areas rather than global changes

### Built-in Health Monitoring:
Health checks are now **automatically included** in every math walkthrough summary:
- `health_check_sign_coherence_violations`: Count of dr>η & Net<-δ or dr<-η & Net>+δ violations
- `health_check_uniformity_fraction`: Percentage of rows with |Net| ≈ ε (plateau detection)
- `health_check_share_monotonicity`: Correlation between |Net| and share (should be positive)

**Note**: The standalone `diagnose_oaxaca_issues.py` script is **no longer needed** since health metrics are integrated into the main analysis workflow.

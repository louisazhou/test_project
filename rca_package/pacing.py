import matplotlib.pyplot as plt
import pandas as pd

_C_BLUE  = "#3b82f6"   # Current (still in stage)
_C_LIGHT = "#bfdbfe"   # Transitioned to next
_C_GRAY  = "#9ca3af"   # Lost / not recoverable

def plot_funnel_row(row: pd.Series, use: str = "PRC"):
    """
    Supports two schemas by ds:
      - ds >= 2025-07-01:
          Early → Pitched/Committed → Actioned → Partially Adopted → Won
          Each (except Won) has: Current %, ->Next %, ->Closed Lost % (Partially: Current% + ->Won% = 100).
      - ds <  2025-07-01:
          Early → Pitched/Committed → Closed Won (terminal), plus '-> Actioned %' as a branch out of Pitched.
          Only two groups available & sum to 100:
            * Early: Current %, ->Pitched/Committed %, ->Closed Lost %
            * Pitched: Current %, ->Actioned %, ->Closed Won %
    Returns: matplotlib Figure
    """
    # ----- helpers -----
    def _fmt_amount(v, mode, total):
        val = int(round((v/100.0) * total))
        return f"${val:,.0f}" if mode == "prc" else f"{val:,.0f}"

    def _total_and_fmt(row, use):
        mode = use.lower()
        if mode == "prc":
            total = row.get("total PRC", row.get("total_PRC"))
            if total is None: raise KeyError("Need 'total PRC' (or 'total_PRC') for use='PRC'.")
            title_tail = f"Total PRC ${int(round(total)):,.0f}"
        elif mode == "solution":
            total = row["Total Initiatives"]
            title_tail = f"Total Solutions {int(round(total)):,.0f}"
        else:
            raise ValueError("use must be 'PRC' or 'Solution'")
        return mode, float(total), title_tail

    ds = pd.to_datetime(row.get("ds", "1970-01-01"))
    mode, TOTAL, title_tail = _total_and_fmt(row, use)
    territory = row.get("territory", "")

    if ds >= pd.Timestamp("2025-07-01"):
        # ===== NEWER SCHEMA =====
        stages = ["Early Stage", "Pitched/Committed", "Actioned", "Partially Adopted", "Won"]
        to_next = {
            "Early Stage":           float(row["Early Stage -> Pitched/Committed %"]),
            "Pitched/Committed":     float(row["Pitched/Committed -> Actioned %"]),
            "Actioned":              float(row["Actioned -> Partially Adopted %"]),
            "Partially Adopted":     float(row["Partially Adopted -> Won%"]),
            "Won":                   0.0,
        }
        current_within = {
            "Early Stage":           float(row["Early Stage Current %"]),
            "Pitched/Committed":     float(row["Pitched/Committed Current %"]),
            "Actioned":              float(row["Actioned Current %"]),
            "Partially Adopted":     float(row["Partially Adopted Current %"]),
            "Won":                   100.0,
        }
        closed_lost = {
            "Early Stage":           float(row["Early Stage -> Closed Lost %"]),
            "Pitched/Committed":     float(row["Pitched/Committed -> Closed Lost %"]),
            "Actioned":              float(row["Actioned -> Closed Lost %"]),
            "Partially Adopted":     float(row.get("Partially Adopted -> Closed Lost %", 0.0)),
            "Won":                   0.0,
        }
        # validate per-stage sums
        for s in stages[:-1]:
            tot = current_within[s] + to_next[s] + closed_lost[s]
            if abs(tot - 100.0) > 1e-6:
                raise ValueError(f"Stage '{s}' must sum to 100; got {tot:.2f}.")

        # base: % of origin that reached each stage
        base = [100.0]
        base.append(base[-1] * to_next["Early Stage"] / 100.0)
        base.append(base[-1] * to_next["Pitched/Committed"] / 100.0)
        base.append(base[-1] * to_next["Actioned"] / 100.0)
        base.append(base[-1] * to_next["Partially Adopted"] / 100.0)

        seg_cur  = [b * (current_within[s] / 100.0) for s, b in zip(stages, base)]
        seg_tran = [b * (to_next[s]       / 100.0) for s, b in zip(stages, base)]
        seg_lost = [b * (closed_lost[s]   / 100.0) for s, b in zip(stages, base)]

    else:
        # ===== OLDER SCHEMA =====
        # Stages: Early → Pitched/Committed → Closed Won (terminal).
        # Pitched has a three-way split: Current / ->Actioned / ->Closed Won (sum to 100).
        stages = ["Early Stage", "Pitched/Committed", "Closed Won"]

        # to_next defined only for transitions that are part of the funnel flow:
        to_next = {
            "Early Stage":       float(row["Early Stage -> Pitched/Committed %"]),
            "Pitched/Committed": float(row["Pitched/Committed -> Actioned %"]),  # branch to Actioned
            "Closed Won":        0.0,
        }
        # current-within for the two groups we actually have:
        current_within = {
            "Early Stage":       float(row["Early Stage Current %"]),
            "Pitched/Committed": float(row["Pitched/Committed Current %"]),
            "Closed Won":        100.0,  # terminal
        }
        # closed lost where defined:
        closed_lost = {
            "Early Stage":       float(row["Early Stage -> Closed Lost %"]),
            "Pitched/Committed": 0.0,  # no "Closed Lost" here in old schema
            "Closed Won":        0.0,
        }
        # "Won" portion comes directly off Pitched/Committed
        pct_closed_won = float(row["Pitched/Committed -> Closed Won %"])

        # validate sums
        tot_early  = current_within["Early Stage"] + to_next["Early Stage"] + closed_lost["Early Stage"]
        tot_pitched= current_within["Pitched/Committed"] + to_next["Pitched/Committed"] + pct_closed_won
        if abs(tot_early - 100.0) > 1e-6:
            raise ValueError(f"'Early Stage' must sum to 100; got {tot_early:.2f}.")
        if abs(tot_pitched - 100.0) > 1e-6:
            raise ValueError(f"'Pitched/Committed' must sum to 100; got {tot_pitched:.2f}.")

        # base (reached):
        base_early   = 100.0
        base_pitched = base_early * to_next["Early Stage"] / 100.0
        base_closed_won = base_pitched * pct_closed_won / 100.0   # terminal
        base = [base_early, base_pitched, base_closed_won]

        # segments:
        # Early: current / transitioned(to pitched) / lost
        seg_cur_early  = base_early * (current_within["Early Stage"] / 100.0)
        seg_trn_early  = base_early * (to_next["Early Stage"]       / 100.0)
        seg_lost_early = base_early * (closed_lost["Early Stage"]   / 100.0)

        # Pitched: current / transitioned(to Actioned) / won (terminal branch)
        seg_cur_pitch  = base_pitched * (current_within["Pitched/Committed"] / 100.0)
        seg_trn_pitch  = base_pitched * (to_next["Pitched/Committed"]        / 100.0)
        seg_won_pitch  = base_pitched * (pct_closed_won                      / 100.0)

        # Closed Won stage: treat as terminal stock = 100% current of its base
        seg_cur_cwon   = base_closed_won
        seg_trn_cwon   = 0.0
        seg_lost_cwon  = 0.0

        seg_cur  = [seg_cur_early, seg_cur_pitch, seg_cur_cwon]
        seg_tran = [seg_trn_early, seg_trn_pitch, seg_trn_cwon]
        # "lost" at pitched is 0 in old schema; use gray for Early's lost only.
        seg_lost = [seg_lost_early, 0.0, 0.0]

    # ----- Plot (common) -----
    fig, ax = plt.subplots(figsize=(10, 6))
    max_w = max(base)
    left_offsets = [(max_w - w) / 2 for w in base]

    for i, s in enumerate(stages):
        y = len(stages) - 1 - i
        left = left_offsets[i]
        bw, tw, gw = seg_cur[i], seg_tran[i], seg_lost[i]

        # draw stacked: current (blue), transitioned (light), lost (gray)
        if bw > 0: ax.barh(y, bw, left=left, color=_C_BLUE,  edgecolor="none")
        if tw > 0: ax.barh(y, tw, left=left + bw, color=_C_LIGHT, edgecolor="none")
        if gw > 0: ax.barh(y, gw, left=left + bw + tw, color=_C_GRAY,  edgecolor="none")

        # annotate on segments (% + integer amount from origin)
        if bw > 0:
            ax.text(left + bw/2, y, f"{bw:.1f}%\n{_fmt_amount(bw, mode, TOTAL)}",
                    ha="center", va="center", color="white", fontsize=9, fontweight="bold")
        if tw > 0:
            # display the within-stage transition % on light segment
            if ds >= pd.Timestamp("2025-07-01"):
                within = to_next[s]
            else:
                within = to_next[s] if s in to_next else 0.0
            ax.text(left + bw + tw/2, y, f"{within:.0f}%\n{_fmt_amount(tw, mode, TOTAL)}",
                    ha="center", va="center", color="black", fontsize=9)
        if gw > 0:
            # show the within-stage lost% (or none if not defined)
            lost_pct = (gw / base[i] * 100.0) if base[i] > 0 else 0.0
            ax.text(left + bw + tw + gw/2, y, f"{lost_pct:.0f}%\n{_fmt_amount(gw, mode, TOTAL)}",
                    ha="center", va="center", color="black", fontsize=9)

        # right-side: reached share and amount (base of stage)
        ax.text(max_w + 4, y, f"Reached: {base[i]:.1f}% • {_fmt_amount(base[i], mode, TOTAL)}",
                va="center", fontsize=9)

    # axes/legend/title
    ax.set_yticks(range(len(stages)))
    ax.set_yticklabels(stages[::-1])
    ax.set_xticks([]); ax.grid(False)
    for sp in ax.spines.values(): sp.set_visible(False)

    title_left = " • ".join([t for t in [str(row.get("ds", "")), territory, title_tail] if t])
    ax.set_title(title_left, fontsize=12, loc="left", pad=12)

    # legend bottom, 2 columns
    fig.legend(
        handles=[
            plt.Line2D([0], [0], color=_C_BLUE,  lw=10, label="Current (not transitioned yet)"),
            plt.Line2D([0], [0], color=_C_LIGHT, lw=10, label="Transitioned to next"),
            plt.Line2D([0], [0], color=_C_GRAY,  lw=10, label="Lost (not recoverable)"),
        ],
        loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.06)
    )
    plt.tight_layout(); plt.subplots_adjust(bottom=0.18)
    return fig
import pandas as pd
import matplotlib.pyplot as plt

_C_BLUE  = "#3b82f6"   # Current
_C_LIGHT = "#bfdbfe"   # Transitioned / Terminal won
_C_GRAY  = "#9ca3af"   # Lost
MIN_VIS_PCT = 1.0

def plot_funnel_row(row: pd.Series, use: str = "PRC"):
    def _fmt_amount(v, mode, total):
        amt = int(round((v/100.0) * total))
        # Convert to BMK suffix (B, M, K) with 2 decimal rounding
        def _bmk(n: int) -> str:
            absn = abs(n)
            if absn >= 1_000_000_000:
                num = round(n / 1_000_000_000, 2)
                suf = "B"
            elif absn >= 1_000_000:
                num = round(n / 1_000_000, 2)
                suf = "M"
            elif absn >= 1_000:
                num = round(n / 1_000, 2)
                suf = "K"
            else:
                num = round(n, 2)
                suf = ""
            return f"{num:.2f}{suf}"
        s = _bmk(amt)
        return f"${s}" if mode == "prc" else s

    def _total_and_fmt(row, use):
        mode = use.lower()
        if mode == "prc":
            total = row.get("Total PRC $")
            if total is None:
                raise KeyError("Need 'Total PRC $' for use='PRC'.")
            title_tail = f"Total PRC ${int(round(total)):,.0f}"
        elif mode == "solution":
            total = row["Total Initiatives"]
            title_tail = f"Total Solutions {int(round(total)):,.0f}"
        else:
            raise ValueError("use must be 'PRC' or 'Solution'")
        return mode, float(total), title_tail

    # Date handling
    date_val = row.get("Date")
    date_str = ""
    if date_val is not None and pd.notnull(date_val):
        try:
            date_str = pd.to_datetime(date_val).strftime('%Y-%m-%d')
        except Exception:
            date_str = str(date_val)
    ds = pd.to_datetime(date_val if date_val is not None else "1970-01-01")
    mode, TOTAL, title_tail = _total_and_fmt(row, use)
    territory = row.get("territory", "")

    won_key = "Partially Adopted -> Won %" if "Partially Adopted -> Won %" in row.index else "Partially Adopted -> Won%"
    pc_closed_won_key = "Pitched/Committed -> Closed Won %" if "Pitched/Committed -> Closed Won %" in row.index else "Pitched/Committed -> Closed Won%"

    if ds >= pd.Timestamp("2025-07-01"):
        stages = ["Early Stage", "Pitched/Committed", "Actioned", "Partially Adopted", "Won"]
        to_next = {
            "Early Stage":           float(row.get("Early Stage -> Pitched/Committed %", 0) or 0),
            "Pitched/Committed":     float(row.get("Pitched/Committed -> Actioned %", 0) or 0),
            "Actioned":              float(row.get("Actioned -> Partially Adopted %", 0) or 0),
            "Partially Adopted":     float(row.get(won_key, 0) or 0),
            "Won":                   0.0,
        }
        current_within = {
            "Early Stage":           float(row.get("Early Stage Current %", 0) or 0),
            "Pitched/Committed":     float(row.get("Pitched/Committed Current %", 0) or 0),
            "Actioned":              float(row.get("Actioned Current %", 0) or 0),
            "Partially Adopted":     float(row.get("Partially Adopted Current %", 0) or 0),
            "Won":                   100.0,
        }
        closed_lost = {
            "Early Stage":           float(row.get("Early Stage -> Closed Lost %", 0) or 0),
            "Pitched/Committed":     float(row.get("Pitched/Committed -> Closed Lost %", 0) or 0),
            "Actioned":              float(row.get("Actioned -> Closed Lost %", 0) or 0),
            "Partially Adopted":     float(row.get("Partially Adopted -> Closed Lost %", 0.0) or 0.0),
            "Won":                   0.0,
        }
        # tolerance 99-101
        for s in stages[:-1]:
            tot = current_within[s] + to_next[s] + closed_lost[s]
            if tot < 99 or tot > 101:
                raise ValueError(f"Stage '{s}' must sum to ~100 (99-101 allowed); got {tot:.2f}.")

        # base
        base = [100.0]
        base.append(base[-1] * to_next["Early Stage"] / 100.0)
        base.append(base[-1] * to_next["Pitched/Committed"] / 100.0)
        base.append(base[-1] * to_next["Actioned"] / 100.0)
        base.append(base[-1] * to_next["Partially Adopted"] / 100.0)

        # segments (% of origin)
        seg_cur  = [b * (current_within[s] / 100.0) for s, b in zip(stages, base)]
        seg_tran = [b * (to_next[s]       / 100.0) for s, b in zip(stages, base)]
        seg_lost = [b * (closed_lost[s]   / 100.0) for s, b in zip(stages, base)]

        # concise side text
        side_group_text = {
            "Early Stage":       f"Curr {current_within['Early Stage']:.2f}% | →Pitched {to_next['Early Stage']:.2f}% | Lost {closed_lost['Early Stage']:.2f}%",
            "Pitched/Committed": f"Curr {current_within['Pitched/Committed']:.2f}% | →Actioned {to_next['Pitched/Committed']:.2f}% | Lost {closed_lost['Pitched/Committed']:.2f}%",
            "Actioned":          f"Curr {current_within['Actioned']:.2f}% | →Part. Adopted {to_next['Actioned']:.2f}% | Lost {closed_lost['Actioned']:.2f}%",
            "Partially Adopted": f"Curr {current_within['Partially Adopted']:.2f}% | →Won {to_next['Partially Adopted']:.2f}%",
            "Won":               "Terminal"
        }

        # labels for 'Next' segment text
        next_label = {
            "Early Stage": "→Pitched",
            "Pitched/Committed": "→Actioned",
            "Actioned": "→Part. Adopted",
            "Partially Adopted": "→Won",
            "Won": ""
        }

    else:
        # OLD schema
        stages = ["Early Stage", "Pitched/Committed", "Closed Won"]

        early_to_pitched = float(row.get("Early Stage -> Pitched/Committed %", 0) or 0)
        pitched_current  = float(row.get("Pitched/Committed Current %", 0) or 0)
        pitched_closed_won = float(row.get(pc_closed_won_key, 0.0) or 0.0)
        pitched_closed_lost = float(row.get("Pitched/Committed -> Closed Lost %", 0.0) or 0.0)
        early_current = float(row.get("Early Stage Current %", 0) or 0)
        early_closed_lost = float(row.get("Early Stage -> Closed Lost %", 0) or 0)

        # validation
        tot_early   = early_current + early_to_pitched + early_closed_lost
        tot_pitched = pitched_current + pitched_closed_lost + pitched_closed_won
        if tot_early < 99 or tot_early > 101:
            raise ValueError(f"'Early Stage' must sum to ~100 (99-101 allowed); got {tot_early:.2f}.")
        if tot_pitched < 99 or tot_pitched > 101:
            raise ValueError(f"'Pitched/Committed' must sum to ~100 (99-101 allowed); got {tot_pitched:.2f}.")

        # base
        base_early   = 100.0
        base_pitched = base_early * early_to_pitched / 100.0
        base_won     = base_pitched * pitched_closed_won / 100.0
        base = [base_early, base_pitched, base_won]

        # segments
        seg_cur  = [
            base_early   * (early_current / 100.0),
            base_pitched * (pitched_current / 100.0),
            base_won     * 1.0  # terminal stock
        ]
        seg_tran = [
            base_early   * (early_to_pitched / 100.0),   # transition to pitched
            base_pitched * (pitched_closed_won / 100.0), # transition to terminal won
            0.0
        ]
        seg_lost = [
            base_early   * (early_closed_lost / 100.0),
            base_pitched * (pitched_closed_lost / 100.0),
            0.0
        ]

        # concise side text
        side_group_text = {
            "Early Stage":       f"Curr {early_current:.2f}% | →Pitched {early_to_pitched:.2f}% | Lost {early_closed_lost:.2f}%",
            "Pitched/Committed": f"Curr {pitched_current:.2f}% | →Won {pitched_closed_won:.2f}% | Lost {pitched_closed_lost:.2f}%",
            "Closed Won":        "Terminal"
        }
        next_label = {
            "Early Stage": "→Pitched",
            "Pitched/Committed": "→Won",
            "Closed Won": ""
        }

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    max_w = max(base)
    left_offsets = [(max_w - w) / 2 for w in base]

    for i, s in enumerate(stages):
        y = len(stages) - 1 - i
        left = left_offsets[i]
        bw, tw, gw = seg_cur[i], seg_tran[i], seg_lost[i]

        # display widths (min visible)
        dbw = MIN_VIS_PCT if (bw > 0 and bw < MIN_VIS_PCT) else bw
        dtw = MIN_VIS_PCT if (tw > 0 and tw < MIN_VIS_PCT) else tw
        dgw = MIN_VIS_PCT if (gw > 0 and gw < MIN_VIS_PCT) else gw

        if bw > 0: ax.barh(y, dbw, left=left, color=_C_BLUE,  edgecolor="none")
        if tw > 0: ax.barh(y, dtw, left=left + dbw, color=_C_LIGHT, edgecolor="none")
        if gw > 0: ax.barh(y, dgw, left=left + dbw + dtw, color=_C_GRAY,  edgecolor="none")

        # Numeric annotations on bars (Current, Transitioned, Lost)
        if bw > 0:
            txt = f"{bw:.1f}%\n{_fmt_amount(bw, mode, TOTAL)}"
            if dbw < 2.0:
                ax.text(left + dbw + 0.5, y, txt, ha="left", va="center", color="white", fontsize=9, fontweight="bold")
            else:
                ax.text(left + dbw/2, y, txt, ha="center", va="center", color="white", fontsize=9, fontweight="bold")
        if tw > 0:
            txt = f"{tw:.1f}%\n{_fmt_amount(tw, mode, TOTAL)}"
            if dtw < 2.0:
                ax.text(left + dbw + dtw + 0.5, y, txt, ha="left", va="center", color="black", fontsize=9)
            else:
                ax.text(left + dbw + dtw/2, y, txt, ha="center", va="center", color="black", fontsize=9)
        if gw > 0:
            txt = f"{gw:.1f}%\n{_fmt_amount(gw, mode, TOTAL)}"
            if dgw < 2.0:
                ax.text(left + dbw + dtw + dgw + 0.5, y, txt, ha="left", va="center", color="black", fontsize=9)
            else:
                ax.text(left + dbw + dtw + dgw/2, y, txt, ha="center", va="center", color="black", fontsize=9)

        # right-side annotations, compact two lines
        right_text = f"Reached {base[i]:.1f}% • {_fmt_amount(base[i], mode, TOTAL)}\n{side_group_text.get(s, '')}"
        ax.text(max_w + 4, y, right_text, va="center", fontsize=9)

    ax.set_yticks(range(len(stages)))
    ax.set_yticklabels(stages[::-1])
    ax.set_xticks([]); ax.grid(False)
    for sp in ax.spines.values(): sp.set_visible(False)

    title_left = " • ".join([t for t in [date_str, territory, title_tail] if t])
    ax.set_title(title_left, fontsize=12, loc="left", pad=12)

    fig.legend(
        handles=[
            plt.Line2D([0], [0], color=_C_BLUE,  lw=10, label="Current (not transitioned yet)"),
            plt.Line2D([0], [0], color=_C_LIGHT, lw=10, label="Transitioned (to next/terminal)"),
            plt.Line2D([0], [0], color=_C_GRAY,  lw=10, label="Lost (not recoverable)"),
        ],
        loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.06)
    )
    plt.tight_layout(); plt.subplots_adjust(bottom=0.18)
    return fig
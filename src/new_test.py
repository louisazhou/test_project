import yaml

with open("../config/closed_lost_reason_tags.yaml") as f:
    reason_yaml = yaml.safe_load(f)


mock_reason_data = [
    {"region": "AM-APAC", "reason": "meta_trust_issues", "opportunity_lost": 1_500_000, "pct_of_total": 0.15},
    {"region": "AM-APAC", "reason": "initiative_duplication", "opportunity_lost": 1_000_000, "pct_of_total": 0.10},
    {"region": "AM-APAC", "reason": "Advertiser not response", "opportunity_lost": 1_100_000, "pct_of_total": 0.11},
    {"region": "Global", "reason": "meta_trust_issues", "opportunity_lost": 5_000_000, "pct_of_total": 0.05},
    {"region": "Global", "reason": "initiative_duplication", "opportunity_lost": 4_000_000, "pct_of_total": 0.06},
    {"region": "Global", "reason": "Advertiser not response", "opportunity_lost": 3_500_000, "pct_of_total": 0.04}
]
df = pd.DataFrame(mock_reason_data)

def compute_overindex_ratio(df: pd.DataFrame, reason_yaml: dict, region_column: str, baseline_region: str = "Global") -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Compute overindex ratio and return two DataFrames: one with all data and one with only new columns."""
    total_opportunity = df.groupby(region_column)["opportunity_lost"].sum().to_dict()
    pct_pivot = df.pivot(index="reason", columns=region_column, values="pct_of_total").reset_index()
    val_pivot = df.pivot(index="reason", columns=region_column, values="opportunity_lost").reset_index()
    merged = pd.merge(pct_pivot, val_pivot, on="reason", suffixes=("_pct", "_val"))
    
    merged.reset_index(drop=True, inplace=True)
    
    # Create a DataFrame to store only the new columns
    new_columns_df = pd.DataFrame()

    # Process each region except the baseline_region
    for focus_region in df[region_column].unique():
        if focus_region == baseline_region:
            continue
        
        delta_pct = merged[f"{focus_region}_pct"] - merged[f"{baseline_region}_pct"]
        overindex_ratio = merged[f"{focus_region}_pct"] / merged[f"{baseline_region}_pct"]
        expected_loss_at_global_pct = total_opportunity[focus_region] * merged[f"{baseline_region}_pct"]
        excess_loss_dollar = merged[f"{focus_region}_val"] - expected_loss_at_global_pct
        
        # Create a temporary DataFrame for new columns
        temp_df = pd.DataFrame({
            "reason": merged["reason"],
            "delta_pct": delta_pct,
            "overindex_ratio": overindex_ratio,
            "expected_loss_at_global_pct": expected_loss_at_global_pct,
            "excess_loss_$": excess_loss_dollar,
            "category": merged["reason"].map(lambda r: reason_yaml.get(r, {}).get("category", "unknown"))
        })
        
        # Determine top-2 excess_loss_$ for recommended_action
        temp_df["recommended_action"] = ""  # Initialize as empty
        top_2_indices = temp_df["excess_loss_$"].nlargest(2).index
        temp_df.loc[top_2_indices, "recommended_action"] = temp_df.loc[top_2_indices, "reason"].map(
            lambda r: reason_yaml.get(r, {}).get("potential_action", "review manually")
        )
        
        # Append to new_columns_df
        temp_df.set_index(pd.Index([focus_region] * len(temp_df)), inplace=True)
        new_columns_df = pd.concat([new_columns_df, temp_df], axis=0)

    return merged, new_columns_df

merged_df, new_columns_df = compute_overindex_ratio(df, reason_yaml, region_column="region")


def fill_insight_template(merged_df: pd.DataFrame, new_columns_df: pd.DataFrame, focus_region: str, threshold_pct_diff: float, min_excess_loss_dollar: float) -> str:
    """Fill the insight template for closed-lost reasons overindexing in a region."""
    insights = []
    
    # Filter new_columns_df for the focus_region
    region_specific_df = new_columns_df.loc[focus_region]
    
    for _, row in region_specific_df.iterrows():
        if row["delta_pct"] > threshold_pct_diff and row["excess_loss_$"] > min_excess_loss_dollar:
            reason = row["reason"]
            region_pct = merged_df.loc[merged_df["reason"] == reason, f"{focus_region}_pct"].values[0] * 100
            global_pct = merged_df.loc[merged_df["reason"] == reason, "Global_pct"].values[0] * 100
            pct_diff = row["delta_pct"] * 100
            excess_dollar = row["excess_loss_$"]
            mapped_category = row["category"]
            mapped_action = row["recommended_action"]
            
            insight = f"""
            In {focus_region}, the reason "{reason}" accounts for {region_pct:.1f}% of closed-lost vs {global_pct:.1f}% globally (+{pct_diff:.1f}pp), 
            resulting in ${excess_dollar:,.0f} more in opportunity lost than expected. 
            This suggests {mapped_category}. Action: {mapped_action}.
            """
            insights.append(insight.strip())
    
    return "\n\n".join(insights)

threshold_pct_diff = 0.05
min_excess_loss_dollar = 100000
print(fill_insight_template(merged_df, new_columns_df, "AM-APAC", threshold_pct_diff, min_excess_loss_dollar))


def plot_top_overindexed_reasons(new_columns_df: pd.DataFrame, focus_region: str, top_n: int = 5):
    """Plot the top over-indexed closed lost reasons for a given region."""
    # Filter the DataFrame for the focus region
    region_specific_df = new_columns_df.loc[focus_region]
    
    # Sort and select the top N reasons by excess loss
    top_df = region_specific_df.sort_values("excess_loss_$", ascending=True).tail(top_n)
    
    # Plot
    plt.figure(figsize=(10, 6))
    bars = plt.barh(top_df["reason"], top_df["excess_loss_$"], color="steelblue")
    plt.xlabel("Excess Loss vs Global Pattern ($)")
    plt.title(f"Top Over-Indexed Closed Lost Reasons in {focus_region} (Dollar Impact)")
    
    # Annotate bars with the excess loss value
    for bar in bars:
        width = bar.get_width()
        plt.text(width + 10000, bar.get_y() + bar.get_height() / 2, f"${width:,.0f}", va='center')
    
    plt.tight_layout()
    plt.grid(axis='x', linestyle='--', alpha=0.6)
    plt.show()

plot_top_overindexed_reasons(new_columns_df, "AM-APAC", top_n=5)
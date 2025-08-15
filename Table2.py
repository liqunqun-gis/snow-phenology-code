import pandas as pd
import numpy as np
import statsmodels.api as sm
from pathlib import Path

# Change this to your actual file path or name:
INPUT_XLSX = r"F:\snow_shuju\site_data\SOD_SED_SCD\annual_variation\Table2.xlsx"

# Variable mapping (your codes → column names in the file)
VAR_MAP = {
    "T_A": "Annual_Avg_Temp",
    "T_S": "Snow_Season_Avg_Temp",
    "T_M": "Melt_Season_Avg_Temp",
    "S_A": "Snowfall",
    "S_S": "Snow_Season_Snowfall",
    "S_M": "Melt_Season_Snowfall",
    "R_A": "Rainfall",
    "R_S": "Snow_Season_Rainfall",
    "R_M": "Melt_Season_Rainfall",
}

# Desired row order
ROW_ORDER = [
    "All Stations",
    "Stable Region",
    "Unstable Region",
    "Before 1990",
    "After 1990",
]

def ols_slope_and_p(years: pd.Series, values: pd.Series):

    df = pd.DataFrame({"Year": years, "Y": values}).dropna()
    if len(df) < 3:
        # Not enough data to regress
        return np.nan, np.nan
    X = sm.add_constant(df["Year"].values)  # intercept + Year
    y = df["Y"].values
    model = sm.OLS(y, X, missing="drop").fit()
    slope = model.params[1]         # coefficient on Year
    p_value = model.pvalues[1]      # p-value for the slope
    return slope, p_value


def signif_mark(p):
    if pd.isna(p):
        return ""
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


def fmt_slope_with_signif(slope, p):
    if pd.isna(slope):
        return ""
    return f"{slope:.2f}{signif_mark(p)}"

path = Path(INPUT_XLSX)
if not path.exists():
    raise FileNotFoundError(f"Cannot find input file: {path.resolve()}")

df = pd.read_excel(path)

# Ensure needed columns exist
required_cols = {"Year", "SCD"} | set(VAR_MAP.values())
missing = required_cols - set(df.columns)
if missing:
    raise ValueError(f"Missing required columns in input file: {missing}")

# Coerce numeric columns (in case of stray non-numeric values)
df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
df["SCD"] = pd.to_numeric(df["SCD"], errors="coerce")
for col in VAR_MAP.values():
    df[col] = pd.to_numeric(df[col], errors="coerce")

stable_mask = df["SCD"] > 60
unstable_mask = ~stable_mask

before_1990_mask = df["Year"] < 1990
after_1990_mask  = df["Year"] >= 1990

groups = {
    "All Stations": df,
    "Stable Region": df[stable_mask],
    "Unstable Region": df[unstable_mask],
    "Before 1990": df[before_1990_mask],
    "After 1990": df[after_1990_mask],
}

results = pd.DataFrame(index=ROW_ORDER, columns=list(VAR_MAP.keys()), dtype=object)

for group_name, gdf in groups.items():
    for var_code, col_name in VAR_MAP.items():
        slope, p = ols_slope_and_p(gdf["Year"], gdf[col_name])
        results.loc[group_name, var_code] = fmt_slope_with_signif(slope, p)

# Reindex to the desired order (handles any missing groups gracefully)
results = results.reindex(ROW_ORDER)

print("\nTable 2: Trend analysis (linear regression) of meteorological variables")
print("(Values are slopes per year; *: p < 0.05, **: p < 0.01.)\n")
print(results)

# Save to files (Excel + CSV)
results.to_excel(r"F:\snow_shuju\site_data\SOD_SED_SCD\annual_variation\table2_results.xlsx", merge_cells=False)

print("\nSaved:")
print(" - table2_results.xlsx")
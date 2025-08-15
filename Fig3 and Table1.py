import os
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import FuncFormatter
import matplotlib.ticker as mticker

def extract_start_year(year_range):
    start_year = int(year_range.split('-')[0])
    return start_year

def calculate_trend(data, param_col):
    data['Year'] = data['YearRange'].apply(extract_start_year)
    years = data['Year']
    param_values = data[param_col].dropna()

    if len(param_values) > 1 and np.std(param_values) > 0:
        X = sm.add_constant(years.loc[param_values.index])
        model = sm.OLS(param_values, X).fit()
        return model.params[1] if len(model.params) > 1 else np.nan
    else:
        return np.nan

def process_station_folder(folder_path, columns_to_analyze):
    trends_all_stations = []

    for filename in os.listdir(folder_path):
        if filename.endswith('.xlsx'):
            file_path = os.path.join(folder_path, filename)
            station_data = pd.read_excel(file_path)
            station_trends = {}

            station_trends['Station'] = filename.split('.')[0]

            for column in columns_to_analyze:
                if column in station_data.columns:
                    station_trends[column] = calculate_trend(station_data, column)

            trends_all_stations.append(station_trends)

    return pd.DataFrame(trends_all_stations)

data_folder_path = "F:\\snow_shuju\\site_data\\SOD_SED_SCD\\Meteorological data\\Fig3_5_6_8 and Table2"

columns_to_analyze = [
    'Annual_Avg_Temp', 'Snow_Season_Avg_Temp', 'Melt_Season_Avg_Temp',
    'Snowfall', 'Snow_Season_Snowfall', 'Melt_Season_Snowfall',
    'Rainfall', 'Snow_Season_Rainfall', 'Melt_Season_Rainfall',
    'SCD', 'SOD_Days', 'SED_Days'
]

trends = process_station_folder(data_folder_path, columns_to_analyze)

def add_regression_info(ax, x, y):
    correlation_matrix = np.corrcoef(x, y)
    r_value = correlation_matrix[0, 1]

    X = sm.add_constant(x)
    model = sm.OLS(y, X).fit()

    slope = model.params[1]
    p_value = model.pvalues[1]

    if p_value < 0.001:
        p_display = r'P < 0.001'
    elif p_value < 0.01:
        p_display = r'P < 0.01'
    elif p_value < 0.05:
        p_display = r'P < 0.05'
    else:
        p_display = r'P = %.3f' % p_value

    ax.text(0.95, 0.95, r'$R = %.2f$, %s' % (r_value, p_display),
            transform=ax.transAxes, fontsize=20, va='top', ha='right',
            bbox=dict(boxstyle='round,pad=0.3', edgecolor='none', facecolor='white', alpha=0.8))
    return slope, p_value, r_value


def custom_formatter(value, pos):
    if value == int(value):
        return f'{int(value)}'
    elif abs(value) < 0.1:
        return f'{value:.2f}'
    else:
        return f'{value:.1f}'

subplot_labels = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)', '(g)', '(h)', '(i)']

datasets = [
    (trends, 'Annual_Avg_Temp', 'SCD', 'Trend\ in\ T_A\ (\degree C\ year^{-1})', 'Trend\ in\ SCD\ (day\ year^{-1})'),
    (trends, 'Snowfall', 'SCD', 'Trend\ in\ S_{A}\ (mm\ year^{-1})', 'Trend\ in\ SCD\ (day\ year^{-1})'),
    (trends, 'Rainfall', 'SCD', 'Trend\ in\ R_A\ (mm\ year^{-1})', 'Trend\ in\ SCD\ (day\ year^{-1})'),
    (trends, 'Snow_Season_Avg_Temp', 'SOD_Days', 'Trend\ in\ T_{S}\ (\degree C\ year^{-1})', 'Trend\ in\ SOD\ (day\ year^{-1})'),
    (trends, 'Snow_Season_Snowfall', 'SOD_Days', 'Trend\ in\ S_{S}\ (mm\ year^{-1})', 'Trend\ in\ SOD\ (day\ year^{-1})'),
    (trends, 'Snow_Season_Rainfall', 'SOD_Days', 'Trend\ in\ R_{S}\ (mm\ year^{-1})', 'Trend\ in\ SOD\ (day\ year^{-1})'),
    (trends, 'Melt_Season_Avg_Temp', 'SED_Days', 'Trend\ in\ T_M\ (\degree C\ year^{-1})', 'Trend\ in\ SED\ (day\ year^{-1})'),
    (trends, 'Melt_Season_Snowfall', 'SED_Days', 'Trend\ in\ S_M\ (mm\ year^{-1})', 'Trend\ in\ SED\ (day\ year^{-1})'),
    (trends, 'Melt_Season_Rainfall', 'SED_Days', 'Trend\ in\ R_M\ (mm\ year^{-1})', 'Trend\ in\ SED\ (day\ year^{-1})')
]

def _compute_row_ylim(datasets):

    row_groups = [(0, 1, 2), (3, 4, 5), (6, 7, 8)]
    row_ylim = {}

    for ridx, group in enumerate(row_groups):
        vals = []
        for idx in group:
            data, _, y_col, _, _ = datasets[idx]
            if y_col in data.columns:
                v = data[y_col].dropna().values
                if v.size > 0:
                    vals.append(v)

        if vals:
            allv = np.concatenate(vals)
            ymin, ymax = np.nanmin(allv), np.nanmax(allv)
            pad = 0.05 * max(1e-9, (ymax - ymin))
            if np.isclose(ymin, ymax):
                ymin -= 0.5
                ymax += 0.5
            row_ylim[ridx] = (ymin - pad, ymax + pad)
        else:
            row_ylim[ridx] = (None, None)

    return row_ylim

color = "#1f77b4"

row_ylim = _compute_row_ylim(datasets)

fig, axes = plt.subplots(3, 3, figsize=(18, 15))
axes = axes.flatten()

for idx, (ax, (data, x_col, y_col, xlabel, ylabel)) in enumerate(zip(axes, datasets)):
    if x_col in data.columns and y_col in data.columns:
        x = data[x_col].dropna()
        y = data[y_col].dropna()
        x, y = x.align(y, join='inner')

        if len(x) > 1 and len(y) > 1:
            sns.regplot(
                x=x, y=y, ax=ax,
                scatter_kws={"s": 60, "color": color},
                line_kws={"color": color, "lw": 2},
                ci=95
            )

            ax.set_xlabel(r'$\mathrm{%s}$' % xlabel, fontsize=18)
            if idx % 3 == 0:
                ax.set_ylabel(r'$\mathrm{%s}$' % ylabel, fontsize=18)
            else:
                ax.set_ylabel('')
                ax.set_yticklabels([])

            ax.tick_params(axis='both', which='major', labelsize=20)
            ax.xaxis.set_major_formatter(mticker.FuncFormatter(custom_formatter))
            ax.yaxis.set_major_formatter(mticker.FuncFormatter(custom_formatter))

            # R、P
            slope, p_value, r_value = add_regression_info(ax, x, y)

            print(f"{subplot_labels[idx]}  [{y_col} ~ {x_col}]  slope={slope:.6f}, p={p_value:.4g}, R={r_value:.3f}")

        else:
            ax.text(0.5, 0.5, 'Not enough data', ha='center', va='center', fontsize=18)

    ax.text(0.05, 0.95, subplot_labels[idx], transform=ax.transAxes,
            fontsize=24, va='top', ha='left')

    row_id = idx // 3  # 0: a–c, 1: d–f, 2: g–i
    ymin, ymax = row_ylim.get(row_id, (None, None))
    if ymin is not None and ymax is not None and np.isfinite([ymin, ymax]).all():
        ax.set_ylim(ymin, ymax)

plt.tight_layout()
plt.savefig("F:\\snow_shuju\\site_data\\site_map\\Sensitivity\\Fig3.png", dpi=300)
plt.show()
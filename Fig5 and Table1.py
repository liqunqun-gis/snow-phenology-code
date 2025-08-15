import os
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import FuncFormatter
import matplotlib.ticker as mticker

plt.rcParams["font.family"] = "Times New Roman"

def extract_start_year(year_range):
    start_year = int(year_range.split('-')[0])
    return start_year

def calculate_trend(data, param_col, start_year=None, end_year=None):
    data = data.copy()
    data['Year'] = data['YearRange'].apply(extract_start_year)
    if start_year is not None:
        data = data[data['Year'] >= start_year]
    if end_year is not None:
        data = data[data['Year'] <= end_year]

    years = data['Year']
    param_values = data[param_col].dropna()

    if len(param_values) > 1 and np.std(param_values) > 0:
        X = sm.add_constant(years.loc[param_values.index])
        model = sm.OLS(param_values, X).fit()
        return model.params[1] if len(model.params) > 1 else np.nan
    else:
        return np.nan

def custom_formatter(value, pos):
    if value == int(value):
        return f'{int(value)}'
    elif abs(value) < 0.1:
        return f'{value:.2f}'
    else:
        return f'{value:.1f}'

def calculate_mean_scd(data):
    if 'SCD' in data.columns:
        return data['SCD'].mean()
    else:
        return np.nan

def process_station_folder(folder_path, columns_to_analyze, stability_threshold=60, start_year=None, end_year=None):
    trends_stable, trends_unstable = [], []

    for filename in os.listdir(folder_path):
        if filename.endswith('.xlsx'):
            file_path = os.path.join(folder_path, filename)
            print(f"Processing file: {file_path}")
            station_data = pd.read_excel(file_path)
            station_trends = {}
            station_trends['Station'] = filename.split('.')[0]

            mean_scd = calculate_mean_scd(station_data)

            if not np.isnan(mean_scd):
                for column in columns_to_analyze:
                    if column in station_data.columns:
                        station_trends[column] = calculate_trend(station_data, column, start_year, end_year)

                if mean_scd > stability_threshold:
                    trends_stable.append(station_trends)
                else:
                    trends_unstable.append(station_trends)

    return pd.DataFrame(trends_stable), pd.DataFrame(trends_unstable)

def add_regression_info(ax, x, y, *, category, x_name, y_name, panel_label, color, position='top'):

    correlation_matrix = np.corrcoef(x, y)
    r_value = correlation_matrix[0, 1]

    X = sm.add_constant(x)
    model = sm.OLS(y, X).fit()

    slope = model.params[1]
    p_value = model.pvalues[1]

    print(f"{panel_label} [{category}] {y_name} ~ {x_name}  |  slope={slope:.6f}, R={r_value:.3f}, p={p_value:.4g}")

    if p_value < 0.001:
        p_display = r'P < 0.001'
    elif p_value < 0.01:
        p_display = r'P < 0.01'
    elif p_value < 0.05:
        p_display = r'P < 0.05'
    else:
        p_display = r'P = %.3f' % p_value

    vertical_position = 0.95 if position == 'top' else 0.85
    ax.text(0.95, vertical_position, r'$R = %.2f$, %s' % (r_value, p_display),
            transform=ax.transAxes, fontsize=20, color=color, va='top', ha='right',
            bbox=dict(boxstyle='round,pad=0.3', edgecolor='none', facecolor='white', alpha=0.8),
            fontweight='bold')
    return slope

def _compute_y_limits_by_row(datasets_stable, datasets_unstable):
    row_groups = [(0, 1, 2), (3, 4, 5), (6, 7, 8)]
    row_ylim = {}

    for ridx, group in enumerate(row_groups):
        y_vals = []
        for idx in group:
            data_s, _, y_col_s, _, _ = datasets_stable[idx]
            data_u, _, y_col_u, _, _ = datasets_unstable[idx]

            if y_col_s in data_s.columns:
                y_vals.append(data_s[y_col_s].dropna().values)
            if y_col_u in data_u.columns:
                y_vals.append(data_u[y_col_u].dropna().values)

        if len(y_vals) == 0 or all(len(v) == 0 for v in y_vals):
            row_ylim[ridx] = (None, None)
        else:
            y_all = np.concatenate([v for v in y_vals if len(v) > 0])
            ymin, ymax = np.nanmin(y_all), np.nanmax(y_all)
            pad = 0.05 * max(1e-9, (ymax - ymin))
            row_ylim[ridx] = (ymin - pad, ymax + pad)

    return row_ylim

def generate_combined_scatter_plots(datasets_stable, datasets_unstable, output_path):
    sns.set_palette(sns.color_palette("muted"))
    stable_color = sns.color_palette("muted", 2)[0]
    unstable_color = sns.color_palette("muted", 2)[1]

    row_ylim = _compute_y_limits_by_row(datasets_stable, datasets_unstable)

    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
    axes = axes.flatten()
    subplot_labels = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)', '(g)', '(h)', '(i)']

    for idx, ((data_stable, x_col_s, y_col_s, xlabel, ylabel),
              (data_unstable, x_col_u, y_col_u, _, _)) in enumerate(
        zip(datasets_stable, datasets_unstable)):

        ax = axes[idx]

        for side in ['left', 'bottom', 'right', 'top']:
            ax.spines[side].set_color('black')
            ax.spines[side].set_linewidth(1.5)

        if x_col_s in data_stable.columns and y_col_s in data_stable.columns:
            x_s = data_stable[x_col_s].dropna()
            y_s = data_stable[y_col_s].dropna()
            x_s, y_s = x_s.align(y_s, join='inner')
            if len(x_s) > 1 and len(y_s) > 1:
                ax.scatter(x_s, y_s, s=70, color=stable_color, marker='o')
                sns.regplot(x=x_s, y=y_s, ax=ax, scatter=False,
                            line_kws={"color": stable_color, "lw": 5}, ci=None)
                add_regression_info(
                    ax, x_s, y_s,
                    category='Stable',
                    x_name=x_col_s,
                    y_name=y_col_s,
                    panel_label=subplot_labels[idx],
                    color=stable_color,
                    position='top'
                )

        if x_col_u in data_unstable.columns and y_col_u in data_unstable.columns:
            x_u = data_unstable[x_col_u].dropna()
            y_u = data_unstable[y_col_u].dropna()
            x_u, y_u = x_u.align(y_u, join='inner')
            if len(x_u) > 1 and len(y_u) > 1:
                ax.scatter(x_u, y_u, s=80, color=unstable_color, marker='+')
                sns.regplot(x=x_u, y=y_u, ax=ax, scatter=False,
                            line_kws={"color": unstable_color, "lw": 5}, ci=None)
                add_regression_info(
                    ax, x_u, y_u,
                    category='Unstable',
                    x_name=x_col_u,
                    y_name=y_col_u,
                    panel_label=subplot_labels[idx],
                    color=unstable_color,
                    position='bottom'
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

        ax.text(0.05, 0.95, subplot_labels[idx], transform=ax.transAxes,
                fontsize=24, va='top', ha='left')

        row_id = idx // 3  # 0: a–c, 1: d–f, 2: g–i
        ymin, ymax = row_ylim.get(row_id, (None, None))
        if ymin is not None and ymax is not None and np.isfinite([ymin, ymax]).all():
            if np.isclose(ymin, ymax):
                ymin -= 0.5
                ymax += 0.5
            ax.set_ylim(ymin, ymax)

    plt.tight_layout()
    plt.savefig(output_path, dpi=600, bbox_inches='tight')
    plt.show()

data_folder_path = "F:\\snow_shuju\\site_data\\SOD_SED_SCD\\Meteorological data\\Fig3_5_6_8 and Table2"

columns_to_analyze = [
    'Annual_Avg_Temp', 'Snow_Season_Avg_Temp', 'Melt_Season_Avg_Temp',
    'Snowfall', 'Snow_Season_Snowfall', 'Melt_Season_Snowfall',
    'Rainfall', 'Snow_Season_Rainfall', 'Melt_Season_Rainfall',
    'SCD', 'SOD_Days', 'SED_Days'
]

trends_stable, trends_unstable = process_station_folder(data_folder_path, columns_to_analyze)

datasets_stable = [
    (trends_stable, 'Annual_Avg_Temp', 'SCD', 'Trend\\ in\\ T_A\\ (\\degree C\\ year^{-1})', 'Trend\\ in\\ SCD\\ (day\\ year^{-1})'),
    (trends_stable, 'Snowfall', 'SCD', 'Trend\\ in\\ S_{A}\\ (mm\\ year^{-1})', 'Trend\\ in\\ SCD\\ (day\\ year^{-1})'),
    (trends_stable, 'Rainfall', 'SCD', 'Trend\\ in\\ R_A\\ (mm\\ year^{-1})', 'Trend\\ in\\ SCD\\ (day\\ year^{-1})'),
    (trends_stable, 'Snow_Season_Avg_Temp', 'SOD_Days', 'Trend\\ in\\ T_{S}\\ (\\degree C\\ year^{-1})', 'Trend\\ in\\ SOD\\ (day\\ year^{-1})'),
    (trends_stable, 'Snow_Season_Snowfall', 'SOD_Days', 'Trend\\ in\\ S_{S}\\ (mm\\ year^{-1})', 'Trend\\ in\\ SOD\\ (day\\ year^{-1})'),
    (trends_stable, 'Snow_Season_Rainfall', 'SOD_Days', 'Trend\\ in\\ R_{S}\\ (mm\\ year^{-1})', 'Trend\\ in\\ SOD\\ (day\\ year^{-1})'),
    (trends_stable, 'Melt_Season_Avg_Temp', 'SED_Days', 'Trend\\ in\\ T_M\\ (\\degree C\\ year^{-1})', 'Trend\\ in\\ SED\\ (day\\ year^{-1})'),
    (trends_stable, 'Melt_Season_Snowfall', 'SED_Days', 'Trend\\ in\\ S_M\\ (mm\\ year^{-1})', 'Trend\\ in\\ SED\\ (day\\ year^{-1})'),
    (trends_stable, 'Melt_Season_Rainfall', 'SED_Days', 'Trend\\ in\\ R_M\\ (mm\\ year^{-1})', 'Trend\\ in\\ SED\\ (day\\ year^{-1})')
]

datasets_unstable = [
    (trends_unstable, 'Annual_Avg_Temp', 'SCD', 'Trend\\ in\\ T_A\\ (\\degree C\\ year^{-1})', 'Trend\\ in\\ SCD\\ (day\\ year^{-1})'),
    (trends_unstable, 'Snowfall', 'SCD', 'Trend\\ in\\ S_{A}\\ (mm\\ year^{-1})', 'Trend\\ in\\ SCD\\ (day\\ year^{-1})'),
    (trends_unstable, 'Rainfall', 'SCD', 'Trend\\ in\\ R_A\\ (mm\\ year^{-1})', 'Trend\\ in\\ SCD\\ (day\\ year^{-1})'),
    (trends_unstable, 'Snow_Season_Avg_Temp', 'SOD_Days', 'Trend\\ in\\ T_{S}\\ (\\degree C\\ year^{-1})', 'Trend\\ in\\ SOD\\ (day\\ year^{-1})'),
    (trends_unstable, 'Snow_Season_Snowfall', 'SOD_Days', 'Trend\\ in\\ S_{S}\\ (mm\\ year^{-1})', 'Trend\\ in\\ SOD\\ (day\\ year^{-1})'),
    (trends_unstable, 'Snow_Season_Rainfall', 'SOD_Days', 'Trend\\ in\\ R_{S}\\ (mm\\ year^{-1})', 'Trend\\ in\\ SOD\\ (day\\ year^{-1})'),
    (trends_unstable, 'Melt_Season_Avg_Temp', 'SED_Days', 'Trend\\ in\\ T_M\\ (\\degree C\\ year^{-1})', 'Trend\\ in\\ SED\\ (day\\ year^{-1})'),
    (trends_unstable, 'Melt_Season_Snowfall', 'SED_Days', 'Trend\\ in\\ S_M\\ (mm\\ year^{-1})', 'Trend\\ in\\ SED\\ (day\\ year^{-1})'),
    (trends_unstable, 'Melt_Season_Rainfall', 'SED_Days', 'Trend\\ in\\ R_M\\ (mm\\ year^{-1})', 'Trend\\ in\\ SED\\ (day\\ year^{-1})')
]

generate_combined_scatter_plots(
    datasets_stable, datasets_unstable,
    "F:\\snow_shuju\\site_data\\site_map\\Sensitivity\\Fig5.png"
)
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

def process_station_folder(folder_path, columns_to_analyze, start_year=None, end_year=None):
    trends_all_stations = []

    for filename in os.listdir(folder_path):
        if filename.endswith('.xlsx'):
            file_path = os.path.join(folder_path, filename)
            station_data = pd.read_excel(file_path)
            station_trends = {}

            station_trends['Station'] = filename.split('.')[0]

            for column in columns_to_analyze:
                if column in station_data.columns:
                    station_trends[column] = calculate_trend(station_data, column, start_year, end_year)

            trends_all_stations.append(station_trends)

    return pd.DataFrame(trends_all_stations)

def add_regression_info(ax, x, y, *, period, x_name, y_name, panel_label, color, position='top'):

    correlation_matrix = np.corrcoef(x, y)
    r_value = correlation_matrix[0, 1]

    X = sm.add_constant(x)
    model = sm.OLS(y, X).fit()

    slope = model.params[1]
    p_value = model.pvalues[1]

    print(f"{panel_label} [{period}] {y_name} ~ {x_name}  |  slope={slope:.6f}, R={r_value:.3f}, p={p_value:.4g}")

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

def custom_formatter(value, pos):
    if value == int(value):
        return f'{int(value)}'
    elif abs(value) < 0.1:
        return f'{value:.2f}'
    else:
        return f'{value:.1f}'

def _compute_row_ylim_for_periods(datasets_pre, datasets_post):

    row_groups = [(0, 1, 2), (3, 4, 5), (6, 7, 8)]
    row_ylim = {}

    for ridx, group in enumerate(row_groups):
        y_vals = []
        for idx in group:

            data_pre, _, y_col_pre, _, _ = datasets_pre[idx]
            data_post, _, y_col_post, _, _ = datasets_post[idx]

            if y_col_pre in data_pre.columns:
                y_vals.append(data_pre[y_col_pre].dropna().values)
            if y_col_post in data_post.columns:
                y_vals.append(data_post[y_col_post].dropna().values)

        if len(y_vals) == 0 or all(len(v) == 0 for v in y_vals):
            row_ylim[ridx] = (None, None)
        else:
            y_all = np.concatenate([v for v in y_vals if len(v) > 0])
            ymin, ymax = np.nanmin(y_all), np.nanmax(y_all)
            pad = 0.05 * max(1e-9, (ymax - ymin))

            if np.isclose(ymin, ymax):
                ymin -= 0.5
                ymax += 0.5
            row_ylim[ridx] = (ymin - pad, ymax + pad)

    return row_ylim


def generate_combined_scatter_plots(datasets_pre_1990, datasets_post_1990, output_path):
    sns.set_palette(sns.color_palette("muted"))
    color_pre_1990 = sns.color_palette("muted", 2)[0]
    color_post_1990 = sns.color_palette("muted", 2)[1]

    row_ylim = _compute_row_ylim_for_periods(datasets_pre_1990, datasets_post_1990)

    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
    axes = axes.flatten()
    subplot_labels = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)', '(g)', '(h)', '(i)']

    for idx, ((data_pre, x_col_pre, y_col_pre, xlabel, ylabel),
              (data_post, x_col_post, y_col_post, _, _)) in enumerate(
        zip(datasets_pre_1990, datasets_post_1990)):

        ax = axes[idx]
        for side in ['left', 'bottom', 'right', 'top']:
            ax.spines[side].set_color('black')
            ax.spines[side].set_linewidth(1.5)

        if x_col_pre in data_pre.columns and y_col_pre in data_pre.columns:
            x_pre = data_pre[x_col_pre].dropna()
            y_pre = data_pre[y_col_pre].dropna()
            x_pre, y_pre = x_pre.align(y_pre, join='inner')
            if len(x_pre) > 1 and len(y_pre) > 1:
                ax.scatter(x_pre, y_pre, s=70, color=color_pre_1990, marker='o')
                sns.regplot(x=x_pre, y=y_pre, ax=ax, scatter=False,
                            line_kws={"color": color_pre_1990, "lw": 5}, ci=None)
                add_regression_info(
                    ax, x_pre, y_pre,
                    period='Pre-1990',
                    x_name=x_col_pre,
                    y_name=y_col_pre,
                    panel_label=subplot_labels[idx],
                    color=color_pre_1990,
                    position='top'
                )

        if x_col_post in data_post.columns and y_col_post in data_post.columns:
            x_post = data_post[x_col_post].dropna()
            y_post = data_post[y_col_post].dropna()
            x_post, y_post = x_post.align(y_post, join='inner')
            if len(x_post) > 1 and len(y_post) > 1:
                ax.scatter(x_post, y_post, s=80, color=color_post_1990, marker='+')
                sns.regplot(x=x_post, y=y_post, ax=ax, scatter=False,
                            line_kws={"color": color_post_1990, "lw": 5}, ci=None)
                add_regression_info(
                    ax, x_post, y_post,
                    period='Post-1990',
                    x_name=x_col_post,
                    y_name=y_col_post,
                    panel_label=subplot_labels[idx],
                    color=color_post_1990,
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

        row_id = idx // 3
        ymin, ymax = row_ylim.get(row_id, (None, None))
        if ymin is not None and ymax is not None and np.isfinite([ymin, ymax]).all():
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

trends_pre_1990 = process_station_folder(data_folder_path, columns_to_analyze, end_year=1989)

trends_post_1990 = process_station_folder(data_folder_path, columns_to_analyze, start_year=1990)

datasets_pre_1990 = [
    (trends_pre_1990, 'Annual_Avg_Temp', 'SCD', 'Trend\ in\ T_A\ (\degree C\ year^{-1})', 'Trend\ in\ SCD\ (day\ year^{-1})'),
    (trends_pre_1990, 'Snowfall', 'SCD', 'Trend\ in\ S_{A}\ (mm\ year^{-1})', 'Trend\ in\ SCD\ (day\ year^{-1})'),
    (trends_pre_1990, 'Rainfall', 'SCD', 'Trend\ in\ R_A\ (mm\ year^{-1})', 'Trend\ in\ SCD\ (day\ year^{-1})'),
    (trends_pre_1990, 'Snow_Season_Avg_Temp', 'SOD_Days', 'Trend\ in\ T_{S}\ (\degree C\ year^{-1})', 'Trend\ in\ SOD\ (day\ year^{-1})'),
    (trends_pre_1990, 'Snow_Season_Snowfall', 'SOD_Days', 'Trend\ in\ S_{S}\ (mm\ year^{-1})', 'Trend\ in\ SOD\ (day\ year^{-1})'),
    (trends_pre_1990, 'Snow_Season_Rainfall', 'SOD_Days', 'Trend\ in\ R_{S}\ (mm\ year^{-1})', 'Trend\ in\ SOD\ (day\ year^{-1})'),
    (trends_pre_1990, 'Melt_Season_Avg_Temp', 'SED_Days', 'Trend\ in\ T_M\ (\degree C\ year^{-1})', 'Trend\ in\ SED\ (day\ year^{-1})'),
    (trends_pre_1990, 'Melt_Season_Snowfall', 'SED_Days', 'Trend\ in\ S_M\ (mm\ year^{-1})', 'Trend\ in\ SED\ (day\ year^{-1})'),
    (trends_pre_1990, 'Melt_Season_Rainfall', 'SED_Days', 'Trend\ in\ R_M\ (mm\ year^{-1})', 'Trend\ in\ SED\ (day\ year^{-1})')
]

datasets_post_1990 = [
    (trends_post_1990, 'Annual_Avg_Temp', 'SCD', 'Trend\ in\ T_A\ (\degree C\ year^{-1})', 'Trend\ in\ SCD\ (day\ year^{-1})'),
    (trends_post_1990, 'Snowfall', 'SCD', 'Trend\ in\ S_{A}\ (mm\ year^{-1})', 'Trend\ in\ SCD\ (day\ year^{-1})'),
    (trends_post_1990, 'Rainfall', 'SCD', 'Trend\ in\ R_A\ (mm\ year^{-1})', 'Trend\ in\ SCD\ (day\ year^{-1})'),
    (trends_post_1990, 'Snow_Season_Avg_Temp', 'SOD_Days', 'Trend\ in\ T_{S}\ (\degree C\ year^{-1})', 'Trend\ in\ SOD\ (day\ year^{-1})'),
    (trends_post_1990, 'Snow_Season_Snowfall', 'SOD_Days', 'Trend\ in\ S_{S}\ (mm\ year^{-1})', 'Trend\ in\ SOD\ (day\ year^{-1})'),
    (trends_post_1990, 'Snow_Season_Rainfall', 'SOD_Days', 'Trend\ in\ R_{S}\ (mm\ year^{-1})', 'Trend\ in\ SOD\ (day\ year^{-1})'),
    (trends_post_1990, 'Melt_Season_Avg_Temp', 'SED_Days', 'Trend\ in\ T_M\ (\degree C\ year^{-1})', 'Trend\ in\ SED\ (day\ year^{-1})'),
    (trends_post_1990, 'Melt_Season_Snowfall', 'SED_Days', 'Trend\ in\ S_M\ (mm\ year^{-1})', 'Trend\ in\ SED\ (day\ year^{-1})'),
    (trends_post_1990, 'Melt_Season_Rainfall', 'SED_Days', 'Trend\ in\ R_M\ (mm\ year^{-1})', 'Trend\ in\ SED\ (day\ year^{-1})')
]

generate_combined_scatter_plots(
    datasets_pre_1990, datasets_post_1990,
    "F:\\snow_shuju\\site_data\\site_map\\Sensitivity\\Fig8.png")
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.patches as mpatches
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.lines import Line2D

plt.rcParams['font.family'] = 'Times New Roman'

def add_black_white_scalebar(ax, length_km=3000, height=4, linewidth=8):

    inset_ax = inset_axes(ax, width="32%", height="3%", loc='lower right', borderpad=1,
                           bbox_to_anchor=(-0.77, -0.08, 1, 1), bbox_transform=ax.transAxes)
    inset_ax.set_axis_off()
    half_length = length_km / 2
    inset_ax.plot([0, length_km], [0.2, 0.2], color='white', lw=linewidth+4, alpha=0.5, zorder=0)
    inset_ax.plot([0, half_length], [0, 0], color='black', lw=linewidth, zorder=1)
    inset_ax.plot([half_length, length_km], [0, 0], color='white', lw=linewidth, zorder=1)
    inset_ax.plot([half_length, length_km], [0, 0], color='black', lw=linewidth+1, zorder=0)
    inset_ax.text(0, 0.3, '0', ha='center', va='bottom', fontsize=16, zorder=2)
    inset_ax.text(half_length, 0.3, f'{int(half_length)}', ha='center', va='bottom', fontsize=16, zorder=2)
    inset_ax.text(length_km, 0.3, f'{int(length_km)}', ha='center', va='bottom', fontsize=16, zorder=2)
    inset_ax.text(length_km + (length_km * 0.1), -0.02, 'Km', ha='left', fontsize=16)

def add_latitude_labels(ax):

    for lat in [30, 60]:
        ax.text(-30, lat, f'{lat}°N', transform=ccrs.Geodetic(), fontsize=16, ha='center', va='top')

scd_before_1990_path = r"F:\snow_shuju\site_data\SOD_SED_SCD\1990_trend\MK_Sen_SCD_Before_1990_All_Stations.xlsx"
scd_after_1990_path  = r"F:\snow_shuju\site_data\SOD_SED_SCD\1990_trend\MK_Sen_SCD_After_1990_All_Stations.xlsx"

scd_before_df = pd.read_excel(scd_before_1990_path)
scd_after_df  = pd.read_excel(scd_after_1990_path)

df_map = pd.merge(
    scd_before_df[['Station', 'Latitude', 'Longitude', 'SCD_Average']],
    scd_after_df[['Station', 'SCD_Average']],
    on='Station',
    suffixes=('_Before', '_After')
)

def determine_change(row):

    before_stable = row['SCD_Average_Before'] > 60
    after_stable  = row['SCD_Average_After'] > 60
    if before_stable and after_stable:
        return 'Stable → Stable'
    elif before_stable and not after_stable:
        return 'Stable → Unstable'
    elif not before_stable and after_stable:
        return 'Unstable → Stable'
    else:
        return 'Unstable → Unstable'

df_map['Change_Type'] = df_map.apply(determine_change, axis=1)

def read_station_data(file_path):
    return pd.read_excel(file_path)

def categorize_z_values(z_value):
    if z_value < -1.96:
        return 'Significant Decrease'
    elif -1.96 <= z_value < 0:
        return 'Slight Decrease'
    elif z_value == 0:
        return 'No Significant Change'
    elif 0 < z_value <= 1.96:
        return 'Slight Increase'
    else:
        return 'Significant Increase'

def count_trend_types(station_data):
    station_data['Z_Category'] = station_data['Z'].apply(categorize_z_values)
    return station_data['Z_Category'].value_counts().reindex(
        ['Significant Decrease', 'Slight Decrease', 'No Significant Change', 'Slight Increase', 'Significant Increase'],
        fill_value=0
    )

before_1990_scd_path = r"F:\snow_shuju\site_data\SOD_SED_SCD\1990_trend\MK_Sen_SCD_Before_1990_All_Stations.xlsx"
after_1990_scd_path  = r"F:\snow_shuju\site_data\SOD_SED_SCD\1990_trend\MK_Sen_SCD_After_1990_All_Stations.xlsx"
before_1990_sed_path = r"F:\snow_shuju\site_data\SOD_SED_SCD\1990_trend\MK_Sen_SED_Days_Before_1990_All_Stations.xlsx"
after_1990_sed_path  = r"F:\snow_shuju\site_data\SOD_SED_SCD\1990_trend\MK_Sen_SED_Days_After_1990_All_Stations.xlsx"
before_1990_sod_path = r"F:\snow_shuju\site_data\SOD_SED_SCD\1990_trend\MK_Sen_SOD_Days_Before_1990_All_Stations.xlsx"
after_1990_sod_path  = r"F:\snow_shuju\site_data\SOD_SED_SCD\1990_trend\MK_Sen_SOD_Days_After_1990_All_Stations.xlsx"

scd_before = read_station_data(before_1990_scd_path)
scd_after  = read_station_data(after_1990_scd_path)
sed_before = read_station_data(before_1990_sed_path)
sed_after  = read_station_data(after_1990_sed_path)
sod_before = read_station_data(before_1990_sod_path)
sod_after  = read_station_data(after_1990_sod_path)

before_means = [
    sod_before['Sen_Slope'].mean(),
    sed_before['Sen_Slope'].mean(),
    scd_before['Sen_Slope'].mean()
]
after_means = [
    sod_after['Sen_Slope'].mean(),
    sed_after['Sen_Slope'].mean(),
    scd_after['Sen_Slope'].mean()
]

sod_trend_before = count_trend_types(sod_before)
sod_trend_after  = count_trend_types(sod_after)
sed_trend_before = count_trend_types(sed_before)
sed_trend_after  = count_trend_types(sed_after)
scd_trend_before = count_trend_types(scd_before)
scd_trend_after  = count_trend_types(scd_after)

parameters = ['SOD', 'SED', 'SCD']
categories = ['Significant Decrease', 'Slight Decrease', 'No Significant Change', 'Slight Increase', 'Significant Increase']

freq_colors = [
    (0.35, 0.22, 0.95, 1.0),
    (0.48, 1.00, 0.69, 1.0),
    (0.998, 1.00, 0.62, 1.0),
    (1.00, 0.36, 0.00, 1.0),
    (0.93, 0.00, 0.38, 1.0)
]

fig = plt.figure(figsize=(15, 11))

change_colors_new = {
    'Stable → Stable': freq_colors[0],
    'Unstable → Unstable': freq_colors[4],
    'Stable → Unstable': freq_colors[1],
    'Unstable → Stable': freq_colors[2]
}

ax1 = fig.add_subplot(2, 2, 1, projection=ccrs.Orthographic(0, 90))
ax1.set_global()
ax1.add_feature(cfeature.OCEAN, facecolor='white')
ax1.add_feature(cfeature.LAND, facecolor='whitesmoke')
ax1.add_feature(cfeature.COASTLINE)
ax1.add_feature(cfeature.BORDERS, linestyle=':')
gl = ax1.gridlines(draw_labels=True, color='gray', alpha=0.5, linestyle='--')
gl.top_labels = False
gl.right_labels = False
gl.xlabel_style = {'size': 16}
gl.ylabel_style = {'size': 16}
add_latitude_labels(ax1)

for change_type in ['Stable → Stable', 'Unstable → Unstable']:
    group = df_map[df_map['Change_Type'] == change_type]
    ax1.scatter(group['Longitude'], group['Latitude'],
                color=change_colors_new[change_type],
                alpha=0.7, s=60, transform=ccrs.PlateCarree())

for change_type in ['Stable → Unstable', 'Unstable → Stable']:
    group = df_map[df_map['Change_Type'] == change_type]
    ax1.scatter(group['Longitude'], group['Latitude'],
                color=change_colors_new[change_type],
                edgecolor='black', s=80, transform=ccrs.PlateCarree())
add_black_white_scalebar(ax1, length_km=3000)
ax1.text(-0.1, 0.98, '(a)', transform=ax1.transAxes, fontsize=20, va='top', ha='right')

ax2 = fig.add_subplot(2, 2, 2)
bar_width = 0.3
x = np.arange(len(parameters))
ax2.axhline(y=0, color='black', linewidth=1)
ax2.bar(x - bar_width/2, before_means, bar_width,
        label='Before 1990', color='white', edgecolor='black')
ax2.bar(x + bar_width/2, after_means, bar_width,
        label='After 1990', color='white', edgecolor='black', hatch='/')
ax2.set_xticks(x)
ax2.set_xticklabels(parameters, fontsize=16)
ax2.set_ylabel('(day year$^{-1}$)', fontsize=18)
ax2.tick_params(axis='both', labelsize=16)
ax2.text(0.07, 0.97, '(b)', transform=ax2.transAxes, fontsize=20, va='top', ha='right')
legend_handles_b, legend_labels_b = ax2.get_legend_handles_labels()

ax3 = fig.add_subplot(2, 2, 4)
bar_width = 0.5
for param_idx, param in enumerate(parameters):
    if param == 'SOD':
        trend_before = sod_trend_before
        trend_after  = sod_trend_after
    elif param == 'SED':
        trend_before = sed_trend_before
        trend_after  = sed_trend_after
    elif param == 'SCD':
        trend_before = scd_trend_before
        trend_after  = scd_trend_after
    total_before = trend_before.sum() if trend_before.sum() > 0 else 1
    total_after  = trend_after.sum()  if trend_after.sum()  > 0 else 1
    for cat_idx, category in enumerate(categories):
        before_freq = trend_before[category] / total_before
        after_freq  = trend_after[category]  / total_after
        if before_freq > 0:
            ax3.bar(param_idx * (len(categories) + 1) + cat_idx,
                    before_freq, bar_width, alpha=0.7, color=freq_colors[cat_idx])
        if after_freq > 0:
            ax3.bar(param_idx * (len(categories) + 1) + cat_idx + bar_width,
                    after_freq, bar_width, alpha=0.7, color=freq_colors[cat_idx],
                    edgecolor='black', hatch='/')
ax3.set_xticks([param_idx * (len(categories) + 1) + len(categories)/2 for param_idx in range(len(parameters))])
ax3.set_xticklabels(parameters, fontsize=16)
ax3.set_ylabel('Frequency', fontsize=18)
ax3.tick_params(axis='both', labelsize=16)
ax3.text(0.07, 0.96, '(c)', transform=ax3.transAxes, fontsize=20, va='top', ha='right')

ax4 = fig.add_subplot(2, 2, 3)
ax4.axis('off')

legend_handles_a = []
for ch in ['Stable → Stable', 'Unstable → Unstable', 'Stable → Unstable', 'Unstable → Stable']:
    if ch in ['Stable → Unstable', 'Unstable → Stable']:
        lh = Line2D([0], [0], marker='o', color='w',
                    markerfacecolor=change_colors_new[ch],
                    markersize=10, markeredgecolor='black',
                    label=ch)
    else:
        lh = Line2D([0], [0], marker='o', color='w',
                    markerfacecolor=change_colors_new[ch],
                    markersize=10, markeredgecolor='none', alpha=0.8,
                    label=ch)
    legend_handles_a.append(lh)

legend_patches_b = [mpatches.Patch(facecolor='white', edgecolor='black', label='Before 1990'),
                    mpatches.Patch(facecolor='white', edgecolor='black', hatch='/', label='After 1990')]

legend_patches_c = [mpatches.Patch(color=freq_colors[i], alpha=0.8, label=categories[i])
                    for i in range(len(categories))]
all_legend_handles = legend_handles_a + legend_patches_b + legend_patches_c
ax4.legend(handles=all_legend_handles,
           loc='center', fontsize=18, title='Legends', title_fontsize=18, frameon=False)

plt.tight_layout()
save_path = r"F:\snow_shuju\site_data\site_map\trend_distribution\Fig7.png"
plt.savefig(save_path, dpi=600)
plt.show()
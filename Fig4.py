import os
import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import Point
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.patches as mpatches
from matplotlib import colors as mcolors
from matplotlib.colors import BoundaryNorm

plt.rcParams['font.family'] = 'Times New Roman'

def increase_saturation_brightness(color, factor=1.5):
    h, s, v = mcolors.rgb_to_hsv(color[:3])
    s = min(1, s * factor)
    v = min(1, v * factor)
    return mcolors.hsv_to_rgb([h, s, v])

spectral_r = plt.cm.Spectral_r(np.linspace(0, 1, 256))
bright_spectral_reversed = [increase_saturation_brightness(c) for c in spectral_r]
bright_spectral_reversed_cmap = mcolors.ListedColormap(bright_spectral_reversed)

bins = [-0.4, -0.2, -0.01, 0.01, 0.2, 0.4]
bin_centers = [(bins[i] + bins[i + 1]) / 2 for i in range(len(bins) - 1)]
norm = BoundaryNorm(bins, ncolors=256, clip=True)
colors = [bright_spectral_reversed_cmap(norm(center) / 255) for center in bin_centers]

scd_info_path = "F:\\snow_shuju\\site_data\\SOD_SED_SCD\\1960-2023_trend\\MK_Sen_SCD_All_Stations.xlsx"
sod_info_path = "F:\\snow_shuju\\site_data\\SOD_SED_SCD\\1960-2023_trend\\MK_Sen_SOD_Days_All_Stations.xlsx"
sed_info_path = "F:\\snow_shuju\\site_data\\SOD_SED_SCD\\1960-2023_trend\\MK_Sen_SED_Days_All_Stations.xlsx"
output_path = "F:\\snow_shuju\\site_data\\site_map\\trend_distribution"
if not os.path.exists(output_path):
    os.makedirs(output_path)

def read_station_data(file_path):
    return pd.read_excel(file_path)

def process_data(station_info):
    geometry = [Point(xy) for xy in zip(station_info['Longitude'], station_info['Latitude'])]
    gdf = gpd.GeoDataFrame(station_info, geometry=geometry)
    return gdf

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

def count_trend_types_by_stability(gdf):
    gdf['Z_Category'] = gdf['Z'].apply(categorize_z_values)
    stability_categories = gdf['Stability'].unique()
    trend_counts = {}
    for stability in stability_categories:
        stability_gdf = gdf[gdf['Stability'] == stability]
        counts = stability_gdf['Z_Category'].value_counts().reindex(
            ['Significant Decrease', 'Slight Decrease', 'No Significant Change', 'Slight Increase', 'Significant Increase'],
            fill_value=0
        )
        trend_counts[stability] = counts
    return trend_counts

def add_black_white_scalebar(ax, length_km=3000, height=4, linewidth=8):
    inset_ax = inset_axes(ax, width="32%", height="3%", loc='lower right', borderpad=1,
                          bbox_to_anchor=(-0.77, -0.08, 1, 1), bbox_transform=ax.transAxes)
    inset_ax.set_axis_off()
    half_length = length_km / 2
    inset_ax.plot([0, length_km], [0.2, 0.2], color='white', lw=linewidth + 4, alpha=0.5, zorder=0)
    inset_ax.plot([0, half_length], [0, 0], color='black', lw=linewidth, zorder=1)
    inset_ax.plot([half_length, length_km], [0, 0], color='white', lw=linewidth, zorder=1)
    inset_ax.plot([half_length, length_km], [0, 0], color='black', lw=linewidth + 1, zorder=0)
    inset_ax.text(0, 0.3, '0', ha='center', va='bottom', fontsize=16, zorder=2)
    inset_ax.text(half_length, 0.3, f'{int(half_length)}', ha='center', va='bottom', fontsize=16, zorder=2)
    inset_ax.text(length_km, 0.3, f'{int(length_km)}', ha='center', va='bottom', fontsize=16, zorder=2)
    inset_ax.text(length_km + (length_km * 0.1), -0.02, 'Km', ha='left', fontsize=16)

def add_latitude_labels(ax):
    for lat in [30, 60]:
        ax.text(-30, lat, f'{lat}°N', transform=ccrs.Geodetic(), fontsize=16, ha='center', va='top')

scd_gdf = process_data(read_station_data(scd_info_path))
sod_gdf = process_data(read_station_data(sod_info_path))
sed_gdf = process_data(read_station_data(sed_info_path))

scd_trend_counts = count_trend_types_by_stability(scd_gdf)
sod_trend_counts = count_trend_types_by_stability(sod_gdf)
sed_trend_counts = count_trend_types_by_stability(sed_gdf)

parameters = ['SOD', 'SED', 'SCD']
categories = ['Significant Decrease', 'Slight Decrease', 'No Significant Change', 'Slight Increase', 'Significant Increase']

x = np.arange(3)
labels = ['SOD', 'SED', 'SCD']

stable_means = [sod_gdf[sod_gdf['Stability'] == 'Stable Snow Region']['Sen_Slope'].mean(),
                sed_gdf[sed_gdf['Stability'] == 'Stable Snow Region']['Sen_Slope'].mean(),
                scd_gdf[scd_gdf['Stability'] == 'Stable Snow Region']['Sen_Slope'].mean()]

unstable_means = [sod_gdf[sod_gdf['Stability'] == 'Unstable Snow Region']['Sen_Slope'].mean(),
                  sed_gdf[sed_gdf['Stability'] == 'Unstable Snow Region']['Sen_Slope'].mean(),
                  scd_gdf[scd_gdf['Stability'] == 'Unstable Snow Region']['Sen_Slope'].mean()]

fig = plt.figure(figsize=(15, 11))
ax1 = fig.add_subplot(2, 2, 1, projection=ccrs.Orthographic(0, 90))
ax1.set_global()
ax1.add_feature(cfeature.OCEAN, facecolor='white')
ax1.add_feature(cfeature.LAND, facecolor='whitesmoke')
ax1.add_feature(cfeature.COASTLINE)
ax1.add_feature(cfeature.BORDERS, linestyle=':')

gridlines = ax1.gridlines(draw_labels=True, color='gray', alpha=0.5, linestyle='--')
gridlines.top_labels = False
gridlines.right_labels = False
gridlines.xlabel_style = {'size': 16}
gridlines.ylabel_style = {'size': 16}
add_latitude_labels(ax1)

stable_color = colors[0]
unstable_color = colors[4]
stable_points = scd_gdf[scd_gdf['Stability'] == 'Stable Snow Region']
unstable_points = scd_gdf[scd_gdf['Stability'] == 'Unstable Snow Region']
ax1.scatter(stable_points['Longitude'], stable_points['Latitude'], color=stable_color, label='Stable Snow Region', transform=ccrs.Geodetic())
ax1.scatter(unstable_points['Longitude'], unstable_points['Latitude'], color=unstable_color, label='Unstable Snow Region', transform=ccrs.Geodetic())
add_black_white_scalebar(ax1, length_km=3000)
ax1.text(-0.1, 0.98, '(a)', transform=ax1.transAxes, fontsize=20, va='top', ha='right')
legend_handles_1, legend_labels_1 = ax1.get_legend_handles_labels()

ax2 = fig.add_subplot(2, 2, 2)
bar_width = 0.3
ax2.axhline(y=0, color='black', linewidth=1)
ax2.bar(x - bar_width / 2, stable_means, bar_width, label='Stable Snow Region', color='white', edgecolor='black')
ax2.bar(x + bar_width / 2, unstable_means, bar_width, label='Unstable Snow Region', color='white', edgecolor='black', hatch='/')
ax2.set_xticks(x)
ax2.set_xticklabels(labels, fontsize=16)
ax2.set_ylabel('(day year$^{-1}$)', fontsize=18)
ax2.tick_params(axis='both', labelsize=16)
ax2.text(0.07, 0.97, '(b)', transform=ax2.transAxes, fontsize=20, va='top', ha='right')

ax3 = fig.add_subplot(2, 2, 4)
bar_width = 0.5
hatches = '/'
for param_idx, param in enumerate(parameters):
    trend_counts = scd_trend_counts if param == 'SCD' else (sod_trend_counts if param == 'SOD' else sed_trend_counts)
    for cat_idx, category in enumerate(categories):
        stable_count = trend_counts['Stable Snow Region'][category]
        unstable_count = trend_counts['Unstable Snow Region'][category]
        stable_freq = stable_count / trend_counts['Stable Snow Region'].sum() if stable_count > 0 else 0
        unstable_freq = unstable_count / trend_counts['Unstable Snow Region'].sum() if unstable_count > 0 else 0
        if stable_freq > 0:
            ax3.bar(param_idx * (len(categories) + 1) + cat_idx, stable_freq, bar_width, color=colors[cat_idx], alpha=0.7)
        if unstable_freq > 0:
            ax3.bar(param_idx * (len(categories) + 1) + cat_idx + bar_width, unstable_freq, bar_width,
                    color=colors[cat_idx], edgecolor='black', hatch=hatches, alpha=0.7)
ax3.set_xticks([param_idx * (len(categories) + 1) + len(categories) / 2 for param_idx in range(len(parameters))])
ax3.set_xticklabels(parameters, fontsize=16)
ax3.set_ylabel('Frequency', fontsize=18)
ax3.tick_params(axis='both', labelsize=16)
ax3.text(0.07, 0.96, '(c)', transform=ax3.transAxes, fontsize=20, va='top', ha='right')

ax4 = fig.add_subplot(2, 2, 3)
ax4.axis('off')
legend_patches_3 = [mpatches.Patch(color=colors[i], alpha=0.8, label=categories[i]) for i in range(len(categories))]
legend_patches_3 += [mpatches.Patch(facecolor='white', edgecolor='black', label='Stable Area Region'),
                     mpatches.Patch(facecolor='white', edgecolor='black', hatch=hatches, label='Unstable Area Region')]
ax4.legend(handles=legend_handles_1 + legend_patches_3, loc='center', fontsize=18, title='Legends', title_fontsize=18, frameon=False)

plt.tight_layout()
plt.savefig(os.path.join(output_path, 'Fig4.png'), dpi=600)
plt.show()
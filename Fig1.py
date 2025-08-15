import os
import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd
import matplotlib.colors as mcolors
from matplotlib.ticker import ScalarFormatter, FuncFormatter
from shapely.geometry import Point
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
from matplotlib.colors import BoundaryNorm
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import matplotlib.ticker as mticker

plt.rcParams["font.family"] = "Times New Roman"

def read_station_data(file_path, value_column):
    station_data = pd.read_excel(file_path)
    average_value = station_data[value_column].mean()
    latitude = station_data['LATITUDE'].iloc[0]
    longitude = station_data['LONGITUDE'].iloc[0]
    return average_value, latitude, longitude

def process_data(folder_path, value_column):
    data_list = []
    for file in os.listdir(folder_path):
        if file.endswith('.xlsx') and not file.startswith('~$'):
            file_path = os.path.join(folder_path, file)
            station_name = os.path.splitext(file)[0]
            average_value, latitude, longitude = read_station_data(file_path, value_column)
            data_list.append({
                'Station': station_name,
                'Latitude': latitude,
                'Longitude': longitude,
                f'average_{value_column}': average_value
            })

    df = pd.DataFrame(data_list)
    geometry = [Point(xy) for xy in zip(df['Longitude'], df['Latitude'])]
    gdf = gpd.GeoDataFrame(df, geometry=geometry)
    return gdf

def get_custom_bins(data, num_bins=6):
    data_min = int(np.floor(data.min()))
    data_max = int(np.ceil(data.max()))
    bins = np.linspace(data_min, data_max, num_bins + 1)
    return bins

def increase_saturation_brightness(color, factor=1.5):
    h, s, v = mcolors.rgb_to_hsv(color[:3])
    s = min(1, s * factor)
    v = min(1, v * factor)
    return mcolors.hsv_to_rgb([h, s, v])

def custom_formatter(x, pos):
    if x == 0:
        return '0'
    else:
        return f'{x:.1f}'

def add_black_white_scalebar(ax, length_km=3000, height=4, linewidth=8):

    inset_ax = inset_axes(ax, width="35%", height="5%", loc='lower left', borderpad=1, bbox_to_anchor=(-0.24, -0.08, 1, 1) #(x, y, width, height)
                          , bbox_transform=ax.transAxes)

    inset_ax.set_axis_off()

    half_length = length_km / 2

    inset_ax.plot([0, half_length], [0, 0], color='black', lw=linewidth)

    inset_ax.plot([half_length, length_km], [0, 0], color='white', lw=linewidth, zorder=1)
    inset_ax.plot([half_length, length_km], [0, 0], color='black', lw=linewidth + 1, zorder=0)

    inset_ax.text(0, 0.1, '0', ha='center', va='bottom', fontsize=16)
    inset_ax.text(half_length, 0.1, f'{int(half_length)}', ha='center', va='bottom', fontsize=16)
    inset_ax.text(length_km, 0.1, f'{int(length_km)}', ha='center', va='bottom', fontsize=16)

    inset_ax.text(length_km + (length_km * 0.1), -0.02, 'Km', ha='left', fontsize=16)

def add_latitude_labels(ax):
    latitudes = [30, 60]
    for lat in latitudes:
        ax.text(-30, lat, f'{lat}°N', transform=ccrs.Geodetic(), fontsize=20, ha='center', va='top')

data_folder_path = "F:\\snow_shuju\\site_data\\SOD_SED_SCD\\Fig1"
output_path = "F:\\snow_shuju\\site_data\\site_map\\spatial_distribution"

if not os.path.exists(output_path):
    os.makedirs(output_path)

scd_gdf = process_data(data_folder_path, 'SCD')
sod_gdf = process_data(data_folder_path, 'SOD_Days')
sed_gdf = process_data(data_folder_path, 'SED_Days')

scd_bins = get_custom_bins(scd_gdf['average_SCD'])
sod_bins = get_custom_bins(sod_gdf['average_SOD_Days'])
sed_bins = get_custom_bins(sed_gdf['average_SED_Days'])

bright_spectral = plt.cm.Spectral(np.linspace(0, 1, 256))
bright_spectral = [increase_saturation_brightness(c) for c in bright_spectral]
bright_spectral_cmap = mcolors.ListedColormap(bright_spectral)

bright_spectral_reversed = plt.cm.Spectral_r(np.linspace(0, 1, 256))
bright_spectral_reversed = [increase_saturation_brightness(c) for c in bright_spectral_reversed]
bright_spectral_reversed_cmap = mcolors.ListedColormap(bright_spectral_reversed)

fig, axes = plt.subplots(3, 1, figsize=(12, 18), subplot_kw={'projection': ccrs.Orthographic(0, 90)})

labels = ['(a) SOD', '(b) SED', '(c) SCD']

def plot_histogram_with_inset(ax, gdf, column, cmap, norm, bins, total_stations):

    inset_ax = inset_axes(ax, width="40%", height="46%", loc='lower left', bbox_to_anchor=(-0.53, 0.26, 1, 1),
                          bbox_transform=ax.transAxes)

    # inset_ax = inset_axes(ax, width="40%", height="25%", loc='upper center', bbox_to_anchor=(0, 0.09, 1, 1),#(x, y, width, height)
    #                       bbox_transform=ax.transAxes)

    inset_ax.patch.set_alpha(0)  # Set the background to be transparent
    n, hist_bins, patches = inset_ax.hist(gdf[column], bins=bins, edgecolor='black', rwidth=0.8)

    for patch in patches:
        patch.set_edgecolor('black')
        patch.set_linewidth(1)


    for spine in inset_ax.spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(1)

    # Calculate and display frequencies
    frequencies = n / total_stations
    for patch, frequency in zip(patches, frequencies):
        patch.set_height(max(frequency, 0.1))
    for patch, left_bin_edge, right_bin_edge in zip(patches, hist_bins[:-1], hist_bins[1:]):
        color = cmap(norm((left_bin_edge + right_bin_edge) / 2))
        patch.set_facecolor(color)

    inset_ax.set_ylabel('Frequency', fontsize=20)
    inset_ax.tick_params(axis='y', labelsize=20)  # Adjust the font size as needed

    inset_ax.set_ylim(0, max(frequencies) * 1.2)  # Set y-axis limits with some margin

    # Adjust y-axis ticks to reflect frequencies
    inset_ax.yaxis.set_major_formatter(ScalarFormatter())
    inset_ax.yaxis.set_major_formatter(FuncFormatter(custom_formatter))

    # Remove the frequency values displayed as text
    inset_ax.xaxis.set_visible(False)  # Hide x-axis

for i, (ax, gdf, column, title, cmap, unit, bins) in enumerate(zip(
        axes,
        [sod_gdf, sed_gdf, scd_gdf],
        ['average_SOD_Days', 'average_SED_Days', 'average_SCD'],
        ['Spatial Distribution of Average SOD',
         'Spatial Distribution of Average SED', 'Spatial Distribution of Average SCD'],
        [bright_spectral_cmap, bright_spectral_reversed_cmap, bright_spectral_reversed_cmap],
        ['(DOY)', '(DOY)', '(days)'],
        [sod_bins, sed_bins, scd_bins])):

    ax.set_global()
    ax.add_feature(cfeature.OCEAN, facecolor='white')
    ax.add_feature(cfeature.LAND, facecolor='whitesmoke')
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS, linestyle=':')

    gl = ax.gridlines(draw_labels=True, color='gray', alpha=0.5, linestyle='--')
    gl.ylocator = mticker.FixedLocator([30, 60])
    gl.top_labels = True
    gl.right_labels = True
    gl.xlabel_style = {'size': 20}
    gl.ylabel_style = {'size': 20}

    ax.patch.set_edgecolor('black')
    ax.patch.set_linewidth(4)

    norm = BoundaryNorm(bins, ncolors=256, clip=True)
    sc = ax.scatter(gdf['Longitude'], gdf['Latitude'], c=gdf[column], cmap=cmap, norm=norm, s=90,
                    transform=ccrs.Geodetic())

    cbar = plt.colorbar(sc, ax=ax, orientation='vertical', pad=0.07, aspect=20, shrink=0.94)
    cbar.set_label(unit, fontsize=20, labelpad=-90)
    cbar.set_ticks(bins[1:-1])
    cbar.ax.tick_params(labelsize=20)
    labels_cbar = [str(int(b)) for b in bins[1:-1]]
    cbar.ax.set_yticklabels(labels_cbar)

    add_black_white_scalebar(ax, length_km=3000)

    add_latitude_labels(ax)

    # Inset histogram with transparent background and adjusted Y-axis limits
    plot_histogram_with_inset(ax, gdf, column, cmap, norm, bins, len(gdf))

    ax.text(-0.55, 1.02, labels[i], transform=ax.transAxes, fontsize=22, va='top', ha='left')

plt.tight_layout()
plt.subplots_adjust(top=0.95)
plt.savefig(os.path.join(output_path, 'Fig1.png'), dpi=300)
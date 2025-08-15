import bbox
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import linregress
from matplotlib.ticker import MultipleLocator, FixedLocator
from matplotlib.lines import Line2D

plt.rcParams["font.family"] = "Times New Roman"

file_path = "F:\\snow_shuju\\site_data\\SOD_SED_SCD\\annual_variation\\Fig2.xlsx"
df = pd.read_excel(file_path)

adjusted_colors = {
    'SCD': '#FF69B4',
    'SOD': '#32CD32',
    'SED': '#1E90FF'
}

subsets = ['All', 'Stable', 'Unstable']
subplot_labels = ['(a)', '(b)', '(c)']
legend_locations = ['upper left', 'center', 'upper left']
legend_bbox = [(0.07, 0.98), (0.50, 0.56), (0.07, 0.98)]

fig, axes = plt.subplots(1, 3, figsize=(19.8, 5.5), sharey=False)

for i, (subset, label, legend_loc, bbox) in enumerate(zip(subsets, subplot_labels, legend_locations, legend_bbox)):
    ax = axes[i]
    sub_df = df[df['Stability'] == subset]

    years = sub_df['Year']
    years_centered = years - 1960
    scd = sub_df['SCD']
    sod = sub_df['SOD_Days']
    sed = sub_df['SED_Days']

    scd_reg = linregress(years_centered, scd)
    sod_reg = linregress(years_centered, sod)
    sed_reg = linregress(years_centered, sed)

    ax.plot(years, scd, marker='^', markersize=5, label='SCD', color=adjusted_colors['SCD'])
    ax.plot(years, sod, marker='*', markersize=7, label='SOD', color=adjusted_colors['SOD'])
    ax.plot(years, sed, marker='o', markersize=5, label='SED', color=adjusted_colors['SED'])

    ax.plot(years, scd_reg.slope * years_centered + scd_reg.intercept, '-', color=adjusted_colors['SCD'])
    ax.plot(years, sod_reg.slope * years_centered + sod_reg.intercept, '-', color=adjusted_colors['SOD'])
    ax.plot(years, sed_reg.slope * years_centered + sed_reg.intercept, '-', color=adjusted_colors['SED'])

    ax.text(0.03, 0.95, label, transform=ax.transAxes,
            fontsize=18, verticalalignment='top')

    ax.set_xlabel("Year", fontsize=18)
    if i == 0:
        ax.set_ylabel("Days for SCD and DOY for SOD, SED", fontsize=18)

    ax.xaxis.set_major_locator(MultipleLocator(10))
    ax.xaxis.set_minor_locator(MultipleLocator(5))
    ax.yaxis.set_minor_locator(MultipleLocator(25))

    minor_x_ticks = ax.get_xticks(minor=True)
    ax.xaxis.set_minor_locator(FixedLocator(minor_x_ticks[:-1]))

    ax.tick_params(axis='both', which='major', length=7, width=1.5, direction='in', labelsize=17)
    ax.tick_params(axis='both', which='minor', length=4, width=1, direction='in')
    ax.xaxis.set_ticks_position('both')
    ax.yaxis.set_ticks_position('both')

    if subset == 'All':
        ax.set_ylim(25, 368)
    elif subset == 'Stable':
        ax.set_ylim(80, 286)
    elif subset == 'Unstable':
        ax.set_ylim(-15, 470)

    def format_p(p):
        if p < 0.001:
            return "P < 0.001"
        elif p < 0.01:
            return "P < 0.01"
        elif p < 0.05:
            return "P < 0.05"
        else:
            return f"P = {p:.2f}"

    legend_texts = [
        f"SCD  y={scd_reg.slope:.2f}x+{scd_reg.intercept:.2f}, {format_p(scd_reg.pvalue)}",
        f"SOD  y={sod_reg.slope:.2f}x+{sod_reg.intercept:.2f}, {format_p(sod_reg.pvalue)}",
        f"SED  y={sed_reg.slope:.2f}x+{sed_reg.intercept:.2f}, {format_p(sed_reg.pvalue)}"
    ]

    legend_lines = [
        Line2D([0], [0], color=adjusted_colors['SCD'], marker='^', linestyle='-', markersize=5),
        Line2D([0], [0], color=adjusted_colors['SOD'], marker='*', linestyle='-', markersize=7),
        Line2D([0], [0], color=adjusted_colors['SED'], marker='o', linestyle='-', markersize=5)
    ]

    ax.legend(legend_lines, legend_texts, loc=legend_loc, bbox_to_anchor=bbox, fontsize=18, frameon=False)

plt.tight_layout()
output_path = "F:\\snow_shuju\\site_data\\SOD_SED_SCD\\annual_variation\\Fig2.png"
plt.savefig(output_path, dpi=600)
plt.show()
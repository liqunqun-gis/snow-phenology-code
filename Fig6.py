import os, re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.path as mpath
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

DATA_DIR = r"F:\snow_shuju\site_data\SOD_SED_SCD\Meteorological data\Fig3_5_6_8 and Table2"
OUT_PNG  = r"F:\snow_shuju\site_data\SOD_SED_SCD\change_point\Fig6.png"
ALPHA    = 0.05
VARIABLES = [
    ("SCD", "SCD (days)"),
    ("SOD_Days", "SOD (days)"),
    ("SED_Days", "SED (days)"),
    ("Annual_Avg_Temp", "Annual mean temperature (°C)")
]

YEAR_BINS  = [1950, 1975, 1980, 1985, 1990, 1995, 2000, 2010, 2100]
BIN_LABELS = ["<1970", "1970–74", "1975–79", "1980–84", "1985–89",
              "1990–94", "1995–99", "≥2000"]
BIN_COLORS = [
    "#3B2DB8",  # deep violet
    "#5942F2",  # bright violet
    "#7AEFAF",  # mint
    "#FEEF9E",  # light yellow
    "#FF8A33",  # orange
    "#ED005F",  # magenta
    "#9A004A",  # deep magenta
    "#009B9E"   # teal
]


plt.rcParams["font.family"] = "Times New Roman"

def extract_year_from_range(s):
    if pd.isna(s): return np.nan
    m = re.match(r"(\d{4})", str(s))
    return int(m.group(1)) if m else np.nan

def pettitt_test(x):
    x = np.asarray(pd.to_numeric(x, errors="coerce"), dtype=float)
    x = x[~np.isnan(x)]
    n = len(x)
    if n < 3:
        return np.nan, np.nan, np.zeros(n)
    A = np.sign(x[:, None] - x[None, :])
    U = np.array([A[:t+1, t+1:].sum() for t in range(n-1)] + [0])
    K = np.abs(U).max()
    t0 = int(np.abs(U).argmax())
    p  = 2 * np.exp((-6.0 * K**2) / (n**3 + n**2))
    return t0, p, U

def classify_year_bin(year):
    for lo, hi, lab in zip(YEAR_BINS[:-1], YEAR_BINS[1:], BIN_LABELS):
        if lo <= year < hi: return lab
    return "NA"

def circle_boundary(ax):
    theta = np.linspace(0, 2*np.pi, 200)
    center, radius = [0.5, 0.5], 0.5
    verts = np.vstack([np.sin(theta), np.cos(theta)]).T
    ax.set_boundary(mpath.Path(verts*radius + center), transform=ax.transAxes)

def add_black_white_scalebar(ax, length_km=1500, height=4, linewidth=8):
    inset_ax = inset_axes(ax, width="32%", height="3%", loc='lower right', borderpad=1,
                           bbox_to_anchor=(-0.77, -0.08, 1, 1), bbox_transform=ax.transAxes)
    inset_ax.set_axis_off()
    half_length = length_km / 2
    inset_ax.plot([0, length_km], [0.2, 0.2], color='white', lw=linewidth+4, alpha=0.5, zorder=0)
    inset_ax.plot([0, half_length], [0, 0], color='black', lw=linewidth, zorder=1)
    inset_ax.plot([half_length, length_km], [0, 0], color='white', lw=linewidth, zorder=1)
    inset_ax.plot([half_length, length_km], [0, 0], color='black', lw=linewidth+1, zorder=0)
    inset_ax.text(0, 0.3, '0', ha='center', va='bottom', fontsize=14, zorder=2)
    inset_ax.text(half_length, 0.3, f'{int(half_length)}', ha='center', va='bottom', fontsize=14, zorder=2)
    inset_ax.text(length_km, 0.3, f'{int(length_km)}', ha='center', va='bottom', fontsize=14, zorder=2)
    inset_ax.text(length_km + (length_km * 0.1), -0.02, 'Km', ha='left', fontsize=14)

def add_latitude_labels(ax):
    for lat in [30, 60]:
        ax.text(-30, lat, f'{lat}°N', transform=ccrs.Geodetic(), fontsize=14, ha='center', va='top')

def compute_change_points_for_var(varname):
    rows = []
    for fn in os.listdir(DATA_DIR):
        if not fn.lower().endswith(".xlsx"):
            continue
        path = os.path.join(DATA_DIR, fn)
        try:
            df = pd.read_excel(path)
        except Exception:
            continue
        if varname not in df.columns or "YearRange" not in df.columns:
            continue
        years = df["YearRange"].apply(extract_year_from_range)
        series = pd.to_numeric(df[varname], errors="coerce")
        mask = (~years.isna()) & (~series.isna())
        years = years[mask].astype(int).to_numpy()
        values = series[mask].to_numpy()
        if len(values) < 10:
            continue
        t0, pval, _ = pettitt_test(values)
        if np.isnan(t0):
            continue
        change_year = int(years[int(t0)])
        idx0 = mask[mask].index[0]
        lat = float(df.get("LATITUDE", pd.NA).iloc[idx0])
        lon = float(df.get("LONGITUDE", pd.NA).iloc[idx0])
        rows.append({
            "station": os.path.splitext(fn)[0],
            "lon": lon, "lat": lat,
            "change_year": change_year,
            "p_value": pval
        })
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["significant"] = out["p_value"] < ALPHA
    out["year_bin"] = out["change_year"].apply(classify_year_bin)
    return out

def plot_panel(ax, df):
    ax.set_global()
    ax.add_feature(cfeature.OCEAN, facecolor='white')
    ax.add_feature(cfeature.LAND,  facecolor='whitesmoke')
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS, linestyle=':')

    gl = ax.gridlines(draw_labels=True, color='gray', alpha=0.5, linestyle='--')
    gl.top_labels   = False
    gl.right_labels = False
    gl.xlabel_style = {'size': 16}
    gl.ylabel_style = {'size': 16}

    add_latitude_labels(ax)

    for lab, col in zip(BIN_LABELS, BIN_COLORS):
        sub = df[(df["year_bin"] == lab) & (~df["significant"])]
        if not sub.empty:
            ax.scatter(sub["lon"], sub["lat"], s=50, marker='o',
                       facecolors=col, edgecolors='none',
                       transform=ccrs.PlateCarree(), zorder=3)

    for lab, col in zip(BIN_LABELS, BIN_COLORS):
        sub = df[(df["year_bin"] == lab) & (df["significant"])]
        if not sub.empty:
            ax.scatter(sub["lon"], sub["lat"], s=50, marker='o',
                       facecolors=col, edgecolors='black', linewidths=0.8,
                       transform=ccrs.PlateCarree(), zorder=4)

    add_black_white_scalebar(ax, length_km=3000)

def build_shared_legend(fig):
    patches = [Patch(facecolor=col, edgecolor='k', linewidth=0.5, label=lab)
               for lab, col in zip(BIN_LABELS, BIN_COLORS)]
    fig.legend(
        handles=patches,
        loc="lower center",
        ncol=4,
        frameon=True,
        fontsize=18,
        title="Change-point year",
        title_fontsize=18,
        bbox_to_anchor=(0.5, -0.09)
    )


fig = plt.figure(figsize=(12, 10))
proj_map = ccrs.Orthographic(0, 90)

axes = []
gs = fig.add_gridspec(2, 2, wspace=0.03, hspace=0.08, left=0.08, right=0.98, top=0.98, bottom=0.08)
for i in range(2):
    for j in range(2):
        ax = fig.add_subplot(gs[i, j], projection=proj_map)
        axes.append(ax)

for ax, (var, _) in zip(axes, VARIABLES):
    chg = compute_change_points_for_var(var)
    if chg.empty:
        ax.axis("off")
    else:
        plot_panel(ax, chg)

build_shared_legend(fig)
labels = ["(a)SCD", "(b)SOD", "(c)SED", "(d)Temperature"]
for ax, lab in zip(axes, labels):
    ax.text(-0.15, 0.98, lab, transform=ax.transAxes, ha="left", va="top", fontsize=18)

plt.savefig(OUT_PNG, dpi=300, bbox_inches="tight")
plt.show()
print("Saved ->", OUT_PNG)
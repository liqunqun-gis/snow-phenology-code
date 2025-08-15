Code components:
Snow Phenology Calculation.py
Extracts snow phenology metrics from GHCN snow depth records (after preprocessing for missing dates). Outputs 372 valid stations.

Process meteorological data.py
Complements missing temperature and precipitation records using ERA5 reanalysis data. Produces 200-site dataset (Combined_Results) for subsequent analyses and plotting.

Fig1.py
Generates Figure 1 from extracted snow phenology results.

Table2.py
Merges snow phenology and meteorological variables for 200 stations into Table2.xlsx, used in Table 2.

Fig2.py
Produces Figure 2 based on summary statistics from Table2.xlsx.

Fig3 and Table1.py, Fig5 and Table1.py, Fig8 and Table1.py
Create snow sensitivity plots for all stations (Fig 3), snow regions (Fig 5), and time periods (Fig 8). Also compute slope and p-values for Table 1.

Table1.py
Contains three script blocks used to compute temporal trends for all stations, time periods, and snow regions. These trends are reported in Table 1 and support Figures 4 and 7.

Fig4.py and Fig7.py
Use outputs from Table1.py to generate Figures 4 and 7 respectively.

All scripts are structured and annotated for transparency and reproducibility. Required input files are referenced by filename in each script.

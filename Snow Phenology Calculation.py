import os
import pandas as pd
from collections import Counter
import datetime
import calendar
import glob

# Set folder paths
source_folder = "F:\\snow_data\\site_data\\site_SNWD_DATE_data\\"  # Folder containing CSV files
output_folder = "F:\\snow_data\\site_data\\output1\\SOD_SED_SCD\\results\\"  # Folder to save results

# Ensure output folder exists
os.makedirs(output_folder, exist_ok=True)

# Get all CSV files
files = glob.glob(source_folder + "*.csv")

# Process each file
for file in files:
    print(f"Processing file: {file}")
    df = pd.read_csv(file)
    df['DATE'] = pd.to_datetime(df['DATE'])
    df['year'] = df['DATE'].dt.year

    # Extract station info
    station_info = df[['LATITUDE', 'LONGITUDE', 'ELEVATION']].drop_duplicates().iloc[0]

    start_snow_months = []
    end_snow_months = []

    start_year = max(df['year'].min(), 1960)  # Start year no earlier than 1960
    end_year = min(df['year'].max(), 2023)    # End year no later than 2023

    valid_hydro_years_data = []

    for year in range(start_year, end_year + 1):
        hydro_year_start = datetime.datetime(year, 8, 1)
        hydro_year_end = datetime.datetime(year + 1, 7, 31)
        df_year = df[(df['DATE'] >= hydro_year_start) & (df['DATE'] <= hydro_year_end)].copy()

        df_year['SNWD'] = df_year['SNWD'].replace('NA', pd.NA)
        df_year['SNWD'] = pd.to_numeric(df_year['SNWD'], errors='coerce')

        # Filter for days with snow and record start/end months
        snow_days = df_year.dropna(subset=['SNWD'])[df_year['SNWD'] > 0]['DATE']
        if not snow_days.empty:
            start_snow_months.append(snow_days.dt.month.iloc[0])
            end_snow_months.append(snow_days.dt.month.iloc[-1])

    if not start_snow_months:
        print(f"No start snow months data found for the file: {file}. Skipping this file.")
        continue

    # Determine most common snow start and end months
    most_common_start_month = Counter(start_snow_months).most_common(1)[0][0]
    most_common_end_month = Counter(end_snow_months).most_common(1)[0][0]

    hydro_year_group = 1
    for year in range(start_year, end_year):
        hydro_year_start = datetime.datetime(year, 8, 1)
        hydro_year_end = datetime.datetime(year + 1, 7, 31)

        snow_range_start = datetime.datetime(year, most_common_start_month, 1) if most_common_start_month >= 8 else datetime.datetime(year + 1, most_common_start_month, 1)
        snow_range_end = datetime.datetime(year, most_common_end_month, calendar.monthrange(year, most_common_end_month)[1]) if most_common_end_month >= 8 else datetime.datetime(year + 1, most_common_end_month, calendar.monthrange(year + 1, most_common_end_month)[1])

        df_year = df[(df['DATE'] >= hydro_year_start) & (df['DATE'] <= hydro_year_end)].copy()

        snow_date_range = df_year[(df_year['DATE'] >= snow_range_start) & (df_year['DATE'] <= snow_range_end)]

        # Calculate NA ratio
        total_days = len(snow_date_range)
        na_count = snow_date_range['SNWD'].isna().sum()

        # Skip if NA values are 20% or more
        if total_days == 0 or na_count / total_days >= 0.2:
            continue

        # Process each hydrological year’s data
        snow_cover_days = df_year['SNWD'].gt(0).sum()

        # Calculate SOD and SED
        snow_bool = df_year['SNWD'] > 0
        rolling_snow = snow_bool.rolling(5).sum()
        sod_index = rolling_snow[rolling_snow == 5].first_valid_index()
        sod = df_year.loc[sod_index, 'DATE'] - pd.Timedelta(days=4) if sod_index else pd.NaT
        sed_index = rolling_snow[rolling_snow == 5].last_valid_index()
        sed = df_year.loc[sed_index, 'DATE'] if sed_index else pd.NaT

        # Handle cases where SCD < 5 or no continuous 5-day snow event is found
        if snow_cover_days < 5 or sod is pd.NaT or sed is pd.NaT:
            sod = hydro_year_end    # Set to last day of the hydrological year
            sed = hydro_year_start  # Set to first day of the hydrological year

        valid_hydro_years_data.append({
            'HydroYearGroup': hydro_year_group,
            'YearRange': f"{year}-{year + 1}",
            'SCD': snow_cover_days,
            'SnowMonthRange': f"{snow_range_start.strftime('%Y-%m-%d')} to {snow_range_end.strftime('%Y-%m-%d')}",
            'NA_Count': na_count,
            'SOD_Date': sod.strftime('%Y-%m-%d') if pd.notnull(sod) else 'NaN',
            'SED_Date': sed.strftime('%Y-%m-%d') if pd.notnull(sed) else 'NaN',
            'SOD_Days': ((sod - hydro_year_start).days + 1) if pd.notnull(sod) else (366 if hydro_year_end.year % 4 == 0 else 365),
            'SED_Days': ((sed - hydro_year_start).days + 1) if pd.notnull(sed) else 1
        })
        hydro_year_group += 1

    if not valid_hydro_years_data:
        print(f"No valid data found in file: {file}. Skipping this file.")
        continue

    # Check year continuity and range
    results_df = pd.DataFrame(valid_hydro_years_data)
    start_years = results_df['YearRange'].apply(lambda x: int(x.split('-')[0]))
    end_years = results_df['YearRange'].apply(lambda x: int(x.split('-')[1]))
    is_continuous = all(end_years.iloc[i] == start_years.iloc[i + 1] for i in range(len(start_years) - 1))

    if start_years.min() == 1960 and end_years.max() == 2023 and is_continuous:
        # Extra filter: keep only stations where SCD is not all zeros
        if results_df['SCD'].sum() > 0:
            # Add station info to results
            for key in ['LATITUDE', 'LONGITUDE', 'ELEVATION']:
                results_df[key] = station_info[key]

            # Save results
            excel_file_path = os.path.join(output_folder, os.path.basename(file).replace('.csv', '.xlsx'))
            results_df.to_excel(excel_file_path, index=False)
            print(f"Saved results to {excel_file_path}")
        else:
            print(f"File {os.path.basename(file)} has all SCD values as 0, skipping save.")
    else:
        print(f"File {os.path.basename(file)} does not meet the year range or continuity criteria.")

print("Processing complete.")

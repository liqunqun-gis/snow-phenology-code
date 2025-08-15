import pandas as pd
import glob
import os

# Define folder paths
source_folder = "F:\\snow_data\\site_data\\site_SNWD_DATE_data"
result_folder = "F:\\snow_data\\site_data\\SOD_SED_SCD\\results"
output_folder = "F:\\snow_data\\site_data\\SOD_SED_SCD\\Meteorological data\\Combined_Results"
supplementary_temp_folder = "F:\\snow_data\\site_data\\SOD_SED_SCD\\Meteorological data\\ERA5_T_daily_C"
supplementary_precip_folder = "F:\\snow_data\\site_data\\SOD_SED_SCD\\Meteorological data\\ERA5_P_mm"

# Missing data threshold
missing_threshold = 0.2  # 20%

# To record missing columns and excluded site information
missing_columns_info = []
excluded_sites_info = []

# Read and organize all supplementary temperature data
supplementary_data_temp = {}
for file in glob.glob(f'{supplementary_temp_folder}/*.csv'):
    site_name = os.path.basename(file).split('.')[0]
    df_supplementary = pd.read_csv(file, parse_dates=['date'], low_memory=False)
    df_supplementary['temperature'].replace(-9999, pd.NA, inplace=True)  # Replace -9999 with NaN
    df_supplementary['temperature'] = df_supplementary['temperature'] / 10  # Unit conversion
    df_supplementary.rename(columns={'date': 'DATE', 'temperature': 'Daily_Avg_Temp'}, inplace=True)
    supplementary_data_temp[site_name] = df_supplementary

# Read and organize all supplementary precipitation data
supplementary_data_precip = {}
for file in glob.glob(f'{supplementary_precip_folder}/*.csv'):
    site_name = os.path.basename(file).split('.')[0]
    df_supplementary = pd.read_csv(file, parse_dates=['date'], low_memory=False)
    df_supplementary['precipitation'].replace(-9999, pd.NA, inplace=True)  # Replace -9999 with NaN
    df_supplementary['precipitation'] = df_supplementary['precipitation'] / 10  # Unit conversion
    df_supplementary.rename(columns={'date': 'DATE', 'precipitation': 'Supplementary_Precip'}, inplace=True)
    supplementary_data_precip[site_name] = df_supplementary

# Loop through each file in the results folder
for result_file in glob.glob(f'{result_folder}/*.xlsx'):
    # Extract site name from file name
    site_name = os.path.basename(result_file).split('.')[0]

    try:
        source_file = glob.glob(f'{source_folder}/{site_name}.csv')[0]
        df_source = pd.read_csv(source_file, parse_dates=['DATE'], low_memory=False)
    except IndexError:
        print(f"Data file for site {site_name} does not exist.")
        continue

    try:
        df_result = pd.read_excel(result_file)
    except Exception as e:
        print(f"Unable to read result file {result_file}: {e}")
        continue

    # Check if temperature and precipitation data exist
    has_temp_data = any(col in df_source.columns for col in ['TMAX', 'TAVG', 'TMIN', 'TAXN'])
    has_precip_data = 'PRCP' in df_source.columns

    # If no temperature data column and no supplementary data, skip precipitation calculation
    if not has_temp_data and site_name not in supplementary_data_temp:
        print(f"Site {site_name} has no temperature data and no supplementary data, skipping precipitation calculation.")
        continue

    # If no temperature data column but supplementary data exists, use supplementary data
    if not has_temp_data and site_name in supplementary_data_temp:
        df_supplementary = supplementary_data_temp[site_name]
        if df_supplementary['Daily_Avg_Temp'].notna().all():
            df_source = df_supplementary.copy()
            has_temp_data = True
        else:
            print(f"Supplementary temperature data for site {site_name} is incomplete, skipping this site.")
            continue

    # Unit conversions
    if has_precip_data and 'PRCP' in df_source.columns:
        df_source['PRCP'] = df_source['PRCP'] / 10
    if 'TMAX' in df_source.columns:
        df_source['TMAX'] = df_source['TMAX'] / 10
    if 'TMIN' in df_source.columns:
        df_source['TMIN'] = df_source['TMIN'] / 10
    if 'TAVG' in df_source.columns:
        df_source['TAVG'] = df_source['TAVG'] / 10

    if has_temp_data:
        # Calculate daily average temperature
        if 'Daily_Avg_Temp' not in df_source.columns:
            df_source['Daily_Avg_Temp'] = df_source.apply(
                lambda row:
                (row['TMAX'] + row['TMIN']) / 2 if pd.notna(row.get('TMAX')) and pd.notna(row.get('TMIN'))
                else row['TAVG'] if pd.notna(row.get('TAVG'))
                else row['TAXN'] / 10 if pd.notna(row.get('TAXN'))
                else None,
                axis=1
            )

        # Merge supplementary data if available
        if site_name in supplementary_data_temp:
            df_supplementary = supplementary_data_temp[site_name]
            df_source = pd.merge(df_source, df_supplementary, on='DATE', how='left', suffixes=('', '_supp'))
            df_source['Daily_Avg_Temp'].fillna(df_source['Daily_Avg_Temp_supp'], inplace=True)
            df_source.drop(columns=['Daily_Avg_Temp_supp'], inplace=True)

    # Merge supplementary precipitation data if available
    if has_precip_data and site_name in supplementary_data_precip:
        df_supplementary_precip = supplementary_data_precip[site_name]
        df_source = pd.merge(df_source, df_supplementary_precip, on='DATE', how='left', suffixes=('', '_supp'))
        df_source['PRCP'].fillna(df_source['Supplementary_Precip'], inplace=True)
        df_source.drop(columns=['Supplementary_Precip'], inplace=True)

    if has_precip_data and 'PRCP' in df_source.columns:
        # Differentiate precipitation type by daily average temperature
        df_source['Snowfall'] = df_source.apply(
            lambda row: row['PRCP'] if row['Daily_Avg_Temp'] < 1 else row['PRCP'] / 2 if row['Daily_Avg_Temp'] == 1 else 0,
            axis=1)
        df_source['Rainfall'] = df_source.apply(
            lambda row: row['PRCP'] if row['Daily_Avg_Temp'] > 1 else row['PRCP'] / 2 if row['Daily_Avg_Temp'] == 1 else 0,
            axis=1)

    # Initialize new columns
    for col in ['Annual_Avg_Temp', 'Melt_Season_Avg_Temp', 'Snow_Season_Avg_Temp',
                'Snowfall', 'Rainfall', 'Melt_Season_Snowfall', 'Melt_Season_Rainfall',
                'Snow_Season_Snowfall', 'Snow_Season_Rainfall']:
        df_result[col] = None

    save_data = True

    # Process each year range
    for i, row in df_result.iterrows():
        start_year, end_year = map(int, row['YearRange'].split('-'))

        start_date = f"{start_year}-08-01"
        end_date = f"{end_year}-07-31"

        melt_start_date = f"{end_year}-02-01"
        melt_end_date = f"{end_year}-07-31"
        snow_start_date = f"{start_year}-08-01"
        snow_end_date = f"{end_year}-01-31"

        year_data = df_source[(df_source['DATE'] >= start_date) & (df_source['DATE'] <= end_date)]
        melt_season_data = year_data[(year_data['DATE'] >= melt_start_date) & (year_data['DATE'] <= melt_end_date)]
        snow_season_data = year_data[(year_data['DATE'] >= snow_start_date) & (year_data['DATE'] <= snow_end_date)]

        if has_temp_data:
            temp_missing_ratio_total = year_data[['Daily_Avg_Temp']].isna().mean().max()
            temp_missing_ratio_melt = melt_season_data[['Daily_Avg_Temp']].isna().mean().max()
            temp_missing_ratio_snow = snow_season_data[['Daily_Avg_Temp']].isna().mean().max()

            if temp_missing_ratio_total < missing_threshold:
                df_result.at[i, 'Annual_Avg_Temp'] = year_data['Daily_Avg_Temp'].mean()
            else:
                save_data = False
                print(f"Site {site_name}, year range {start_year}-{end_year}: annual temperature missing data too high.")

            if temp_missing_ratio_melt < missing_threshold:
                df_result.at[i, 'Melt_Season_Avg_Temp'] = melt_season_data['Daily_Avg_Temp'].mean()
            else:
                save_data = False
                print(f"Site {site_name}, year range {start_year}-{end_year}: melt season temperature missing data too high.")

            if temp_missing_ratio_snow < missing_threshold:
                df_result.at[i, 'Snow_Season_Avg_Temp'] = snow_season_data['Daily_Avg_Temp'].mean()
            else:
                save_data = False
                print(f"Site {site_name}, year range {start_year}-{end_year}: snow season temperature missing data too high.")

        if has_precip_data and 'PRCP' in df_source.columns:
            precip_missing_ratio_total = year_data[['Snowfall', 'Rainfall']].isna().mean().max()
            precip_missing_ratio_melt = melt_season_data[['Snowfall', 'Rainfall']].isna().mean().max()
            precip_missing_ratio_snow = snow_season_data[['Snowfall', 'Rainfall']].isna().mean().max()

            if precip_missing_ratio_total < missing_threshold:
                df_result.at[i, 'Snowfall'] = year_data['Snowfall'].sum()
                df_result.at[i, 'Rainfall'] = year_data['Rainfall'].sum()
            else:
                save_data = False
                print(f"Site {site_name}, year range {start_year}-{end_year}: annual precipitation missing data too high.")

            if precip_missing_ratio_melt < missing_threshold:
                df_result.at[i, 'Melt_Season_Snowfall'] = melt_season_data['Snowfall'].sum()
                df_result.at[i, 'Melt_Season_Rainfall'] = melt_season_data['Rainfall'].sum()
            else:
                save_data = False
                print(f"Site {site_name}, year range {start_year}-{end_year}: melt season precipitation missing data too high.")

            if precip_missing_ratio_snow < missing_threshold:
                df_result.at[i, 'Snow_Season_Snowfall'] = snow_season_data['Snowfall'].sum()
                df_result.at[i, 'Snow_Season_Rainfall'] = snow_season_data['Rainfall'].sum()
            else:
                save_data = False
                print(f"Site {site_name}, year range {start_year}-{end_year}: snow season precipitation missing data too high.")

    if save_data:
        output_path = os.path.join(output_folder, f'Processed_{site_name}.xlsx')
        df_result.to_excel(output_path, index=False)
    else:
        print(f"Data for site {site_name} not saved due to high missing data ratio in some year ranges.")

# Output missing column and excluded site info
for site, cols in missing_columns_info:
    print(f"Site {site} missing columns: {', '.join(cols)}")

for site, reason in excluded_sites_info:
    print(f"Site {site} excluded: {reason}")

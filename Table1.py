"""
1_all station
"""
import pandas as pd
import numpy as np
import scipy.stats as stats
import os
import matplotlib.pyplot as plt


# Define Sen's Slope Estimation function
def sen_slope_estimation(data):
    n = len(data)
    slopes = []
    for i in range(n - 1):
        for j in range(i + 1, n):
            slope = (data.iloc[j] - data.iloc[i]) / (j - i)
            slopes.append(slope)
    return np.median(slopes)


# Define the function to calculate S statistic and Z score
def calculate_s_and_z(data):
    n = len(data)
    s = 0
    for i in range(n - 1):
        for j in range(i + 1, n):
            s += np.sign(data[j] - data[i])

    var_s = (n * (n - 1) * (2 * n + 5)) / 18

    if s > 0:
        z = (s - 1) / np.sqrt(var_s)
    elif s < 0:
        z = (s + 1) / np.sqrt(var_s)
    else:
        z = 0

    return s, z


# Set significance level
alpha = 0.05

# List of columns to analyze
columns_to_analyze = ['SCD', 'SOD_Days', 'SED_Days']

# Loop through all xlsx files in the folder
folder_path = "F:\\snow_shuju\\site_data\\SOD_SED_SCD\\results"
output_folder = "F:\\snow_shuju\\site_data\\SOD_SED_SCD\\1960-2023_trend"

# Create a dictionary to store results for each parameter
all_results = {column: [] for column in columns_to_analyze}

# Create dictionary to store multi-year average values for each station
average_values = {column: {} for column in columns_to_analyze}

for file in os.listdir(folder_path):
    if file.endswith('.xlsx') and not file.startswith('~$'):
        file_path = os.path.join(folder_path, file)

        # Use the filename as the station name
        station_name = os.path.splitext(file)[0]
        data = pd.read_excel(file_path)

        # Extract latitude, longitude, and elevation information
        latitude = data['LATITUDE'].iloc[0]
        longitude = data['LONGITUDE'].iloc[0]
        elevation = data['ELEVATION'].iloc[0]

        # Calculate multi-year average for each station
        for column in columns_to_analyze:
            if column in data.columns:
                average_value = data[column].mean()
                average_values[column][station_name] = average_value

        for column in columns_to_analyze:
            if column in data.columns:
                # Mann-Kendall trend test
                tau, p_value = stats.kendalltau(data['HydroYearGroup'], data[column])

                # Sen's slope estimation
                sen_slope = sen_slope_estimation(data[column])

                # Calculate S statistic and Z score
                s, z = calculate_s_and_z(data[column])

                # Check if the result is significant
                significant = 'Yes' if p_value < alpha else 'No'

                # Append the results to the list
                trend = 'increasing' if tau > 0 else 'decreasing' if tau < 0 else 'no trend'
                scd_avg = average_values['SCD'][station_name]  # Get the SCD average value for the station
                stability = 'Stable Snow Area' if scd_avg > 60 else 'Unstable Snow Area'
                all_results[column].append(
                    [station_name, latitude, longitude, elevation, tau, trend, p_value, sen_slope, significant, s, z,
                     scd_avg, stability])

# Summarize statistics and output results
overall_trend = {}
z_type_counts = {}
sen_type_counts = {}

for column, results in all_results.items():
    results_df = pd.DataFrame(results, columns=['Station', 'Latitude', 'Longitude', 'Elevation', 'MK_Tau', 'MK_Trend',
                                                'MK_P_Value', 'Sen_Slope', 'Significant', 'S', 'Z', 'SCD_Average',
                                                'Stability'])

    # Overall trend statistics
    trend_summary = results_df['MK_Trend'].value_counts(normalize=True) * 100
    overall_trend[column] = trend_summary

    # Z value type statistics
    z_bins = [-float('inf'), -1.96, 0, 1.96, float('inf')]
    z_labels = ['Significantly Decreased', 'Slightly Decreased', 'Slightly Increased', 'Significantly Increased']
    results_df['Z_Type'] = pd.cut(
        results_df['Z'],
        bins=z_bins,
        labels=z_labels
    )
    # Add "Not Significant" category and handle Z = 0 case
    results_df['Z_Type'] = results_df['Z_Type'].cat.add_categories('Not Significant')
    results_df.loc[results_df['Z'] == 0, 'Z_Type'] = 'Not Significant'

    z_type_summary = results_df['Z_Type'].value_counts(normalize=True) * 100
    z_type_counts[column] = z_type_summary

    # Sen's Slope type statistics
    slope_bins = [-float('inf'), -4, -2, 0, 2, 4, float('inf')]
    slope_labels = ['<-4 d/a', '-4 to -2 d/a', '-2 to 0 d/a', '0 to 2 d/a', '2 to 4 d/a', '>4 d/a']
    results_df['Sen_Slope_Type'] = pd.cut(
        results_df['Sen_Slope'],
        bins=slope_bins,
        labels=slope_labels
    )
    # Add '0' category and handle Sen's Slope = 0 case
    results_df['Sen_Slope_Type'] = results_df['Sen_Slope_Type'].cat.add_categories('0')
    results_df.loc[results_df['Sen_Slope'] == 0, 'Sen_Slope_Type'] = '0'

    sen_type_summary = results_df['Sen_Slope_Type'].value_counts(normalize=True) * 100
    sen_type_counts[column] = sen_type_summary

    # Average change per year
    avg_change = results_df['Sen_Slope'].mean()

    # Output overall trend
    decreasing_percentage = trend_summary.get('decreasing', 0)
    increasing_percentage = trend_summary.get('increasing', 0)
    no_trend_percentage = trend_summary.get('no trend', 0)
    significant_percentage = results_df['Significant'].value_counts(normalize=True).get('Yes', 0) * 100

    print(
        f"{decreasing_percentage:.2f}% of the stations show a decrease in {column}, {significant_percentage:.2f}% show significant change; "
        f"{increasing_percentage:.2f}% show an increase in {column}, {significant_percentage:.2f}% show significant change; "
        f"{no_trend_percentage:.2f}% show no trend in {column}.")

    # Output Z value type distribution
    z_type_output = ", ".join([f"{perc:.2f}% of the stations show {z_type}" for z_type, perc in z_type_summary.items()])
    print(f"{z_type_output}.")

    # Output Sen's Slope type distribution
    sen_type_output = ", ".join(
        [f"{perc:.2f}% of the stations have changes in the range of {slope_label}" for slope_label, perc in sen_type_summary.items()])
    print(f"{sen_type_output}.")

    # Output average change
    print(f"The average annual change for {column} is approximately {avg_change:.2f} days.\n")

    # Save results to Excel
    output_file_path = os.path.join(output_folder, f'MK_Sen_{column}_All_Stations.xlsx')
    results_df.to_excel(output_file_path, index=False)

# Generate and save trend statistics plot
for column, trend_summary in overall_trend.items():
    plt.figure(figsize=(10, 6))
    trend_summary.plot(kind='bar', color='skyblue')
    plt.title(f'{column} Overall Trend')
    plt.xlabel('Trend')
    plt.ylabel('Percentage (%)')
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, f'{column}_Overall_Trend.png'))
    plt.close()

# Generate and save Z value type statistics plot
for column, z_type_summary in z_type_counts.items():
    plt.figure(figsize=(10, 6))
    z_type_summary.plot(kind='bar', color='lightgreen')
    plt.title(f'{column} Z Value Type Distribution')
    plt.xlabel('Z Type')
    plt.ylabel('Percentage (%)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, f'{column}_Z_Type_Distribution.png'))
    plt.close()

# Generate and save Sen's Slope type statistics plot
for column, sen_type_summary in sen_type_counts.items():
    plt.figure(figsize=(10, 6))
    sen_type_summary.plot(kind='bar', color='orange')
    plt.title(f'{column} Sen Slope Type Distribution')
    plt.xlabel('Sen Slope Type')
    plt.ylabel('Percentage (%)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, f'{column}_Sen_Slope_Type_Distribution.png'))
    plt.close()


"""
2_pre1990_post1990
"""
import pandas as pd
import numpy as np
import scipy.stats as stats
import os
import matplotlib.pyplot as plt

# Define Sen's Slope Estimation function
def sen_slope_estimation(data):
    n = len(data)
    slopes = []
    for i in range(n - 1):
        for j in range(i + 1, n):
            slope = (data.iloc[j] - data.iloc[i]) / (j - i)
            slopes.append(slope)
    return np.median(slopes)

# Define the function to calculate S statistic and Z score
def calculate_s_and_z(data):
    n = len(data)
    s = 0
    for i in range(n - 1):
        for j in range(i + 1, n):
            s += np.sign(data.iloc[j] - data.iloc[i])

    var_s = (n * (n - 1) * (2 * n + 5)) / 18

    if s > 0:
        z = (s - 1) / np.sqrt(var_s)
    elif s < 0:
        z = (s + 1) / np.sqrt(var_s)
    else:
        z = 0

    return s, z

# Set significance level
alpha = 0.05

# List of columns to analyze
columns_to_analyze = ['SCD', 'SOD_Days', 'SED_Days']

# Loop through all xlsx files in the folder
folder_path = "F:\\snow_shuju\\site_data\\SOD_SED_SCD\\results"
output_folder = "F:\\snow_shuju\\site_data\\SOD_SED_SCD\\1990_trend"

# Create dictionaries to store results before and after 1990 for each parameter
all_results_before_1990 = {column: [] for column in columns_to_analyze}
all_results_after_1990 = {column: [] for column in columns_to_analyze}

for file in os.listdir(folder_path):
    if file.endswith('.xlsx') and not file.startswith('~$'):
        file_path = os.path.join(folder_path, file)

        # Use the filename as the station name
        station_name = os.path.splitext(file)[0]
        data = pd.read_excel(file_path)

        # Extract latitude, longitude, and elevation information
        latitude = data['LATITUDE'].iloc[0]
        longitude = data['LONGITUDE'].iloc[0]
        elevation = data['ELEVATION'].iloc[0]

        # Extract the start year from the "YearRange" column
        data['StartYear'] = data['YearRange'].apply(lambda x: int(x.split('-')[0]))

        # Split the data into before and after 1990
        data_before_1990 = data[data['StartYear'] < 1990]
        data_after_1990 = data[data['StartYear'] >= 1990]

        for column in columns_to_analyze:
            if column in data.columns:
                # Calculate trends for data before 1990
                if not data_before_1990.empty:
                    # Mann-Kendall trend test
                    tau, p_value = stats.kendalltau(data_before_1990['StartYear'], data_before_1990[column])

                    # Sen's slope estimation
                    sen_slope = sen_slope_estimation(data_before_1990[column])

                    # Calculate S statistic and Z score
                    s, z = calculate_s_and_z(data_before_1990[column])

                    # Check if the result is significant
                    significant = 'Yes' if p_value < alpha else 'No'

                    # Append the results
                    trend = 'increasing' if tau > 0 else 'decreasing' if tau < 0 else 'no trend'
                    scd_avg = data_before_1990['SCD'].mean() if 'SCD' in data_before_1990.columns else np.nan
                    stability = 'Stable Snow Area' if scd_avg > 60 else 'Unstable Snow Area'
                    all_results_before_1990[column].append(
                        [station_name, latitude, longitude, elevation, tau, trend, p_value, sen_slope, significant, s, z,
                         scd_avg, stability])

                # Calculate trends for data after 1990
                if not data_after_1990.empty:
                    # Mann-Kendall trend test
                    tau, p_value = stats.kendalltau(data_after_1990['StartYear'], data_after_1990[column])

                    # Sen's slope estimation
                    sen_slope = sen_slope_estimation(data_after_1990[column])

                    # Calculate S statistic and Z score
                    s, z = calculate_s_and_z(data_after_1990[column])

                    # Check if the result is significant
                    significant = 'Yes' if p_value < alpha else 'No'

                    # Append the results
                    trend = 'increasing' if tau > 0 else 'decreasing' if tau < 0 else 'no trend'
                    scd_avg = data_after_1990['SCD'].mean() if 'SCD' in data_after_1990.columns else np.nan
                    stability = 'Stable Snow Area' if scd_avg > 60 else 'Unstable Snow Area'
                    all_results_after_1990[column].append(
                        [station_name, latitude, longitude, elevation, tau, trend, p_value, sen_slope, significant, s, z,
                         scd_avg, stability])

# Summarize and output the results
def summarize_and_output(results_dict, time_period):
    for column, results in results_dict.items():
        results_df = pd.DataFrame(results, columns=['Station', 'Latitude', 'Longitude', 'Elevation', 'MK_Tau', 'MK_Trend',
                                                    'MK_P_Value', 'Sen_Slope', 'Significant', 'S', 'Z', 'SCD_Average',
                                                    'Stability'])

        # Overall trend statistics
        trend_summary = results_df['MK_Trend'].value_counts(normalize=True) * 100

        # Z value type statistics
        z_bins = [-float('inf'), -1.96, 0, 1.96, float('inf')]
        z_labels = ['Significantly Decreased', 'Slightly Decreased', 'Slightly Increased', 'Significantly Increased']
        results_df['Z_Type'] = pd.cut(
            results_df['Z'],
            bins=z_bins,
            labels=z_labels
        )
        # Add "Not Significant" category and handle Z = 0 case
        results_df['Z_Type'] = results_df['Z_Type'].cat.add_categories('Not Significant')
        results_df.loc[results_df['Z'] == 0, 'Z_Type'] = 'Not Significant'

        z_type_summary = results_df['Z_Type'].value_counts(normalize=True) * 100

        # Sen's Slope type statistics
        slope_bins = [-float('inf'), -4, -2, 0, 2, 4, float('inf')]
        slope_labels = ['<-4 d/a', '-4 to -2 d/a', '-2 to 0 d/a', '0 to 2 d/a', '2 to 4 d/a', '>4 d/a']
        results_df['Sen_Slope_Type'] = pd.cut(
            results_df['Sen_Slope'],
            bins=slope_bins,
            labels=slope_labels
        )
        # Add '0' category and handle Sen's Slope = 0 case
        results_df['Sen_Slope_Type'] = results_df['Sen_Slope_Type'].cat.add_categories('0')
        results_df.loc[results_df['Sen_Slope'] == 0, 'Sen_Slope_Type'] = '0'

        sen_type_summary = results_df['Sen_Slope_Type'].value_counts(normalize=True) * 100

        # Average change per year
        avg_change = results_df['Sen_Slope'].mean()

        # Output overall trend
        decreasing_percentage = trend_summary.get('decreasing', 0)
        increasing_percentage = trend_summary.get('increasing', 0)
        no_trend_percentage = trend_summary.get('no trend', 0)
        significant_percentage = results_df['Significant'].value_counts(normalize=True).get('Yes', 0) * 100

        print(f"{time_period} - {column} analysis results:")
        print(
            f"{decreasing_percentage:.2f}% of the stations show a decrease in {column}, {significant_percentage:.2f}% show significant change; "
            f"{increasing_percentage:.2f}% show an increase in {column}, {significant_percentage:.2f}% show significant change; "
            f"{no_trend_percentage:.2f}% show no trend in {column}.")

        # Output Z value type distribution
        z_type_output = ", ".join([f"{perc:.2f}% of the stations show {z_type}" for z_type, perc in z_type_summary.items()])
        print(f"{z_type_output}.")

        # Output Sen's Slope type distribution
        sen_type_output = ", ".join(
            [f"{perc:.2f}% of the stations have changes in the range of {slope_label}" for slope_label, perc in sen_type_summary.items()])
        print(f"{sen_type_output}.")

        # Output average change
        print(f"The average annual change for {column} is approximately {avg_change:.2f} days.\n")

        # Save results to Excel
        output_file_path = os.path.join(output_folder, f'MK_Sen_{column}_{time_period}_All_Stations.xlsx')
        results_df.to_excel(output_file_path, index=False)

# Define the function to generate trend charts
def plot_trend_chart(results_dict, time_period):
    for column, results in results_dict.items():
        results_df = pd.DataFrame(results, columns=['Station', 'Latitude', 'Longitude', 'Elevation', 'MK_Tau', 'MK_Trend',
                                                    'MK_P_Value', 'Sen_Slope', 'Significant', 'S', 'Z', 'SCD_Average',
                                                    'Stability'])

        # Check if 'Z' column exists before processing 'Z_Type'
        if 'Z' in results_df.columns:
            # Generate Z value type statistics chart
            z_bins = [-float('inf'), -1.96, 0, 1.96, float('inf')]
            z_labels = ['Significantly Decreased', 'Slightly Decreased', 'Slightly Increased', 'Significantly Increased']
            results_df['Z_Type'] = pd.cut(
                results_df['Z'],
                bins=z_bins,
                labels=z_labels
            )
            # Add "Not Significant" category and handle Z = 0 case
            results_df['Z_Type'] = results_df['Z_Type'].cat.add_categories('Not Significant')
            results_df.loc[results_df['Z'] == 0, 'Z_Type'] = 'Not Significant'

            # Generate the Z value type statistics plot
            plt.figure(figsize=(10, 6))
            results_df['Z_Type'].value_counts(normalize=True).plot(kind='bar', color='lightgreen')
            plt.title(f'{column} Z Value Type Distribution ({time_period})')
            plt.xlabel('Z Type')
            plt.ylabel('Percentage (%)')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(os.path.join(output_folder, f'{column}_Z_Type_{time_period}.png'))
            plt.close()
        else:
            print(f"Warning: 'Z' column is missing for {column} in {time_period} data. Skipping Z-Type chart.")

        # Check if 'Sen_Slope' column exists before processing 'Sen_Slope_Type'
        if 'Sen_Slope' in results_df.columns:
            # Create 'Sen_Slope_Type' based on 'Sen_Slope' values
            slope_bins = [-float('inf'), -4, -2, 0, 2, 4, float('inf')]
            slope_labels = ['<-4 d/a', '-4 to -2 d/a', '-2 to 0 d/a', '0 to 2 d/a', '2 to 4 d/a', '>4 d/a']
            results_df['Sen_Slope_Type'] = pd.cut(
                results_df['Sen_Slope'],
                bins=slope_bins,
                labels=slope_labels
            )
            # Add '0' category and handle Sen's Slope = 0 case
            results_df['Sen_Slope_Type'] = results_df['Sen_Slope_Type'].cat.add_categories('0')
            results_df.loc[results_df['Sen_Slope'] == 0, 'Sen_Slope_Type'] = '0'

            # Generate the Sen's Slope type statistics plot
            plt.figure(figsize=(10, 6))
            results_df['Sen_Slope_Type'].value_counts(normalize=True).plot(kind='bar', color='orange')
            plt.title(f'{column} Sen Slope Type Distribution ({time_period})')
            plt.xlabel('Sen Slope Type')
            plt.ylabel('Percentage (%)')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(os.path.join(output_folder, f'{column}_Sen_Slope_Type_{time_period}.png'))
            plt.close()
        else:
            print(f"Warning: 'Sen_Slope' column is missing for {column} in {time_period} data. Skipping Sen Slope chart.")


        # Generate Sen's Slope type statistics chart
        plt.figure(figsize=(10, 6))
        results_df['Sen_Slope_Type'].value_counts(normalize=True).plot(kind='bar', color='orange')
        plt.title(f'{column} Sen Slope Type Distribution ({time_period})')
        plt.xlabel('Sen Slope Type')
        plt.ylabel('Percentage (%)')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, f'{column}_Sen_Slope_Type_{time_period}.png'))
        plt.close()

# Generate trend charts before and after 1990
plot_trend_chart(all_results_before_1990, 'Before_1990')
plot_trend_chart(all_results_after_1990, 'After_1990')

# Output the results before and after 1990
summarize_and_output(all_results_before_1990, 'Before_1990')
summarize_and_output(all_results_after_1990, 'After_1990')


"""
3_Stable_Unstable
"""
import pandas as pd
import numpy as np
import scipy.stats as stats
import os
import matplotlib.pyplot as plt

# Define Sen's Slope Estimation function
def sen_slope_estimation(data):
    n = len(data)
    slopes = []
    for i in range(n - 1):
        for j in range(i + 1, n):
            slope = (data.iloc[j] - data.iloc[i]) / (j - i)
            slopes.append(slope)
    return np.median(slopes)

# Define the function to calculate S statistic and Z score
def calculate_s_and_z(data):
    n = len(data)
    s = 0
    for i in range(n - 1):
        for j in range(i + 1, n):
            s += np.sign(data.iloc[j] - data.iloc[i])

    var_s = (n * (n - 1) * (2 * n + 5)) / 18

    if s > 0:
        z = (s - 1) / np.sqrt(var_s)
    elif s < 0:
        z = (s + 1) / np.sqrt(var_s)
    else:
        z = 0

    return s, z

# Set significance level
alpha = 0.05

# List of columns to analyze
columns_to_analyze = ['SCD', 'SOD_Days', 'SED_Days']

# Loop through all xlsx files in the folder
folder_path = "F:\\snow_shuju\\site_data\\SOD_SED_SCD\\results"
output_folder = "F:\\snow_shuju\\site_data\\SOD_SED_SCD\\snow_area_trend"

# Create dictionaries to store results for Stable and Unstable Snow Areas
all_results_stable_snow_area = {column: [] for column in columns_to_analyze}
all_results_unstable_snow_area = {column: [] for column in columns_to_analyze}

for file in os.listdir(folder_path):
    if file.endswith('.xlsx') and not file.startswith('~$'):
        file_path = os.path.join(folder_path, file)

        # Use the filename as the station name
        station_name = os.path.splitext(file)[0]
        data = pd.read_excel(file_path)

        # Extract latitude, longitude, and elevation information
        latitude = data['LATITUDE'].iloc[0]
        longitude = data['LONGITUDE'].iloc[0]
        elevation = data['ELEVATION'].iloc[0]

        # Extract the start year from the "YearRange" column
        data['StartYear'] = data['YearRange'].apply(lambda x: int(x.split('-')[0]))

        # Calculate the average SCD (Snow Cover Duration)
        scd_avg = data['SCD'].mean() if 'SCD' in data.columns else np.nan

        # Categorize the station based on SCD average
        if scd_avg > 60:
            stability = 'Stable Snow Area'
            result_dict = all_results_stable_snow_area
        else:
            stability = 'Unstable Snow Area'
            result_dict = all_results_unstable_snow_area

        for column in columns_to_analyze:
            if column in data.columns:
                # Mann-Kendall trend test
                tau, p_value = stats.kendalltau(data['StartYear'], data[column])

                # Sen's slope estimation
                sen_slope = sen_slope_estimation(data[column])

                # Calculate S statistic and Z score
                s, z = calculate_s_and_z(data[column])

                # Check if the result is significant
                significant = 'Yes' if p_value < alpha else 'No'

                # Append the results
                trend = 'increasing' if tau > 0 else 'decreasing' if tau < 0 else 'no trend'
                result_dict[column].append(
                    [station_name, latitude, longitude, elevation, tau, trend, p_value, sen_slope, significant, s, z,
                     scd_avg, stability])

# Summarize and output the results
def summarize_and_output(results_dict, stability_label):
    for column, results in results_dict.items():
        results_df = pd.DataFrame(results, columns=['Station', 'Latitude', 'Longitude', 'Elevation', 'MK_Tau', 'MK_Trend',
                                                    'MK_P_Value', 'Sen_Slope', 'Significant', 'S', 'Z', 'SCD_Average',
                                                    'Stability'])

        # Overall trend statistics
        trend_summary = results_df['MK_Trend'].value_counts(normalize=True) * 100

        # Z value type statistics
        z_bins = [-float('inf'), -1.96, 0, 1.96, float('inf')]
        z_labels = ['Significantly Decreased', 'Slightly Decreased', 'Slightly Increased', 'Significantly Increased']
        results_df['Z_Type'] = pd.cut(
            results_df['Z'],
            bins=z_bins,
            labels=z_labels
        )
        # Add "Not Significant" category and handle Z = 0 case
        results_df['Z_Type'] = results_df['Z_Type'].cat.add_categories('Not Significant')
        results_df.loc[results_df['Z'] == 0, 'Z_Type'] = 'Not Significant'

        z_type_summary = results_df['Z_Type'].value_counts(normalize=True) * 100

        # Sen's Slope type statistics
        slope_bins = [-float('inf'), -4, -2, 0, 2, 4, float('inf')]
        slope_labels = ['<-4 d/a', '-4 to -2 d/a', '-2 to 0 d/a', '0 to 2 d/a', '2 to 4 d/a', '>4 d/a']
        results_df['Sen_Slope_Type'] = pd.cut(
            results_df['Sen_Slope'],
            bins=slope_bins,
            labels=slope_labels
        )
        # Add '0' category and handle Sen's Slope = 0 case
        results_df['Sen_Slope_Type'] = results_df['Sen_Slope_Type'].cat.add_categories('0')
        results_df.loc[results_df['Sen_Slope'] == 0, 'Sen_Slope_Type'] = '0'

        sen_type_summary = results_df['Sen_Slope_Type'].value_counts(normalize=True) * 100

        # Average change per year
        avg_change = results_df['Sen_Slope'].mean()

        # Output overall trend
        decreasing_percentage = trend_summary.get('decreasing', 0)
        increasing_percentage = trend_summary.get('increasing', 0)
        no_trend_percentage = trend_summary.get('no trend', 0)
        significant_percentage = results_df['Significant'].value_counts(normalize=True).get('Yes', 0) * 100

        print(f"{stability_label} - {column} analysis results:")
        print(
            f"{decreasing_percentage:.2f}% of the stations show a decrease in {column}, {significant_percentage:.2f}% show significant change; "
            f"{increasing_percentage:.2f}% show an increase in {column}, {significant_percentage:.2f}% show significant change; "
            f"{no_trend_percentage:.2f}% show no trend in {column}.")

        # Output Z value type distribution
        z_type_output = ", ".join([f"{perc:.2f}% of the stations show {z_type}" for z_type, perc in z_type_summary.items()])
        print(f"{z_type_output}.")

        # Output Sen's Slope type distribution
        sen_type_output = ", ".join(
            [f"{perc:.2f}% of the stations have changes in the range of {slope_label}" for slope_label, perc in sen_type_summary.items()])
        print(f"{sen_type_output}.")

        # Output average change
        print(f"The average annual change for {column} is approximately {avg_change:.2f} days.\n")

        # Save results to Excel
        output_file_path = os.path.join(output_folder, f'MK_Sen_{column}_{stability_label}_All_Stations.xlsx')
        results_df.to_excel(output_file_path, index=False)

# Define the function to generate trend charts
def plot_trend_chart(results_dict, stability_label):
    for column, results in results_dict.items():
        results_df = pd.DataFrame(results, columns=['Station', 'Latitude', 'Longitude', 'Elevation', 'MK_Tau', 'MK_Trend',
                                                    'MK_P_Value', 'Sen_Slope', 'Significant', 'S', 'Z', 'SCD_Average',
                                                    'Stability'])

        # Check if 'Z' column exists before processing 'Z_Type'
        if 'Z' in results_df.columns:
            # Generate Z value type statistics chart
            z_bins = [-float('inf'), -1.96, 0, 1.96, float('inf')]
            z_labels = ['Significantly Decreased', 'Slightly Decreased', 'Slightly Increased', 'Significantly Increased']
            results_df['Z_Type'] = pd.cut(
                results_df['Z'],
                bins=z_bins,
                labels=z_labels
            )
            # Add "Not Significant" category and handle Z = 0 case
            results_df['Z_Type'] = results_df['Z_Type'].cat.add_categories('Not Significant')
            results_df.loc[results_df['Z'] == 0, 'Z_Type'] = 'Not Significant'

            # Generate the Z value type statistics plot
            plt.figure(figsize=(10, 6))
            results_df['Z_Type'].value_counts(normalize=True).plot(kind='bar', color='lightgreen')
            plt.title(f'{column} Z Value Type Distribution ({stability_label})')
            plt.xlabel('Z Type')
            plt.ylabel('Percentage (%)')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(os.path.join(output_folder, f'{column}_Z_Type_{stability_label}.png'))
            plt.close()

        # Check if 'Sen_Slope' column exists before processing 'Sen_Slope_Type'
        if 'Sen_Slope' in results_df.columns:
            # Create 'Sen_Slope_Type' based on 'Sen_Slope' values
            slope_bins = [-float('inf'), -4, -2, 0, 2, 4, float('inf')]
            slope_labels = ['<-4 d/a', '-4 to -2 d/a', '-2 to 0 d/a', '0 to 2 d/a', '2 to 4 d/a', '>4 d/a']
            results_df['Sen_Slope_Type'] = pd.cut(
                results_df['Sen_Slope'],
                bins=slope_bins,
                labels=slope_labels
            )
            # Add '0' category and handle Sen's Slope = 0 case
            results_df['Sen_Slope_Type'] = results_df['Sen_Slope_Type'].cat.add_categories('0')
            results_df.loc[results_df['Sen_Slope'] == 0, 'Sen_Slope_Type'] = '0'

            # Generate the Sen's Slope type statistics plot
            plt.figure(figsize=(10, 6))
            results_df['Sen_Slope_Type'].value_counts(normalize=True).plot(kind='bar', color='orange')
            plt.title(f'{column} Sen Slope Type Distribution ({stability_label})')
            plt.xlabel('Sen Slope Type')
            plt.ylabel('Percentage (%)')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(os.path.join(output_folder, f'{column}_Sen_Slope_Type_{stability_label}.png'))
            plt.close()

# Generate trend charts for stable and unstable snow areas
plot_trend_chart(all_results_stable_snow_area, 'Stable Snow Area')
plot_trend_chart(all_results_unstable_snow_area, 'Unstable Snow Area')

# Output the results for stable and unstable snow areas
summarize_and_output(all_results_stable_snow_area, 'Stable Snow Area')
summarize_and_output(all_results_unstable_snow_area, 'Unstable Snow Area')

import pandas as pd
import numpy as np
import os


def annual_avg_on_density_surfaces(input_df: pd.DataFrame, output_file_name: str,
                                   density_surfaces: list, station: str, input_file_names: str):
    # Take annual average of oxygen data on potential density anomaly surfaces

    df = input_df
    # df columns are: Profile number, Time, Profile is interpolated,
    # Depth [m]	Oxygen [umol/kg], Potential density anomaly [kg/m]
    # Potential density anomaly bin [kg/m]

    # Convert time to pandas datetime
    df['Datetime'] = pd.to_datetime(df.loc[:, 'Time'])
    years_available = np.sort(np.unique(df.loc[:, 'Datetime'].dt.year))
    obs_per_year = np.zeros((len(years_available), len(density_surfaces)),
                            dtype='int32')

    # Initialize dataframe to hold annual averages
    density_column = 'Potential density anomaly bin [kg/m]'
    df_avg = pd.DataFrame(columns=['Year', density_column, 'Average oxygen [umol/kg]'])

    # Take the average for each year and density level
    for i in range(len(years_available)):
        for j in range(len(density_surfaces)):
            indexer = np.where(
                (df.loc[:, 'Datetime'].dt.year == years_available[i]) &
                (df.loc[:, density_column] == density_surfaces[j])
            )[0]
            # Append a new row to the end of the initialized df
            if len(indexer) > 0:
                df_avg.loc[len(df_avg)] = [
                    years_available[i],
                    density_surfaces[j],
                    np.nanmean(df.loc[indexer, 'Oxygen [umol/kg]'].to_numpy(float))
                ]
            # Update summary statistics
            obs_per_year[i, j] = len(indexer)

    # Save summary statistics
    summary_file = os.path.join(
        os.path.dirname(output_file_name),
        '{}_averaging_summary_statistics.txt'.format(station))
    with open(summary_file, 'w') as txtfile:
        txtfile.write('Input file: ' + input_file_names + '\n')
        txtfile.write('Output file: ' + output_file_name + '\n')
        txtfile.write('Number of observations:\n')
        txtfile.write('Year')
        for d in density_surfaces:
            txtfile.write(', {} kg/m3'.format(d))
        for i in range(len(years_available)):
            txtfile.write('\n{}'.format(years_available[i]))
            for j in range(len(obs_per_year[i])):
                txtfile.write(', {}'.format(obs_per_year[i, j]))

    df_avg.to_csv(output_file_name, index=False)
    return


# for each station: P4 and P26, LB08
stn = 'P4'
# station_name = stn
# station_name = 'OSP'

# parent_dir = 'C:\\Users\\HourstonH\\Documents\\charles\\' \
#              'line_P_data_products\\csv\\has_osd_ctd_flags\\'
# parent_dir = 'C:\\Users\\HourstonH\\Documents\\charles\\' \
#              'bottom_oxygen\\'

# parent_dir = 'D:\\lineP\\processing\\'

parent_dir = ('C:\\Users\\hourstonh\\Documents\\charles\\line_P_data_products\\'
              'update_jan2024_sopo\\csv_data\\')

o2_bin_dir = '10_bin_o2_to_select_densities'
o2_bin_file = os.path.join(parent_dir, o2_bin_dir, '{}_data.csv'.format(stn))
# o2_bin_file = os.path.join(parent_dir, o2_bin_dir,
#                            '{}_ctd_data_qc.csv'.format(stn))

avg_dir = '11_annual_avg_on_dens_surfaces'
avg_file = os.path.join(parent_dir, avg_dir, os.path.basename(o2_bin_file))

densities = [26.5, 26.7, 26.9]

# Pre-2022 data
pre_2022_file = f'D:\\lineP\\processing\\10_bin_o2_to_select_densities\\{stn}_data.csv'

df_merged = pd.read_csv(o2_bin_file)
df_merged = pd.concat((pd.read_csv(pre_2022_file), df_merged))
df_merged.reset_index(drop=True, inplace=True)

annual_avg_on_density_surfaces(df_merged, avg_file, densities, stn,
                               input_file_names=' & '.join([o2_bin_file, pre_2022_file]))

# ------------------------Bill's data------------------------
"""
import glob

bill_dir = 'C:\\Users\\HourstonH\\Documents\\charles\\' \
           'line_P_data_products\\bill_crawford\\masked\\'

station_name = '26'  # 26 4849
input_files = glob.glob(bill_dir + f'*{station_name}*masked.csv')
input_files.sort()

output_df_filename = bill_dir + f'CrawfordPena Line P 1950-2015 {station_name} oxy annual avg.csv'

dfout = pd.DataFrame(columns=['Year', 'Potential density anomaly bin [kg/m]',
                              'Average oxygen [umol/kg]'])
for f in input_files:
    dfin = pd.read_csv(f)
    # Remove almost-all nan lines
    print(len(dfin))
    dfin.dropna(axis='index', how='all', subset=['Date'], inplace=True)
    print(len(dfin))
    obs_years = [int(d) for d in dfin.loc[:, 'Date']]
    years_available = np.sort(np.unique(obs_years))
    sigma_theta = np.round(dfin.loc[0, 'Sigma_Theta (from CT and AS)'], 1)
    print(sigma_theta)
    # Find the name of the oxygen column
    ox_umol_colname = None
    for colname in ['Ox (umol/kg) ', 'Ox (mmol/kg) ', 'O2 (umol/kg) ']:
        # I think the mmol is just a typo in Bill's file
        if colname in dfin.columns:
            ox_umol_colname = colname
    if ox_umol_colname is None:
        print('Error: oxygen (umol/kg) column not found in input dataset')
    is_close_mask = dfin.loc[:, 'is_close_to_station']
    for y in years_available:
        year_mask = obs_years == y
        # Take average
        avg_ox = np.nanmean(dfin.loc[year_mask & is_close_mask, ox_umol_colname])
        # Add to output df
        dfout.loc[len(dfout), :] = [y, sigma_theta, avg_ox]

# Save dfout
dfout.to_csv(output_df_filename, index=False)
"""

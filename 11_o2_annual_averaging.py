import pandas as pd
import numpy as np
import os


def annual_avg_on_density_surfaces(input_file_name: str, output_file_name: str,
                                   density_surfaces: list, station: str):
    df = pd.read_csv(input_file_name)
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
    df_avg = pd.DataFrame(
        columns=['Year', density_column,
                 'Average oxygen [umol/kg]'])

    # Take the average for each year and density level
    for i in range(len(years_available)):
        for j in range(len(density_surfaces)):
            indexer = np.where(
                (df.loc[:, 'Datetime'].dt.year == years_available[i]) &
                (df.loc[:, density_column] == density_surfaces[j])
            )[0]
            # Append a new row to the end of the initialized df
            df_avg.loc[len(df_avg)] = [
                years_available[i], density_surfaces[j], np.nanmean(
                    df.loc[indexer,
                           'Oxygen [umol/kg]'].to_numpy(
                        float))
            ]
            # Update summary statistics
            obs_per_year[i, j] = len(indexer)

    # Save summary statistics
    summary_file = os.path.join(
        os.path.dirname(output_file_name),
        '{}_averaging_summary_statistics.txt'.format(station))
    with open(summary_file, 'w') as txtfile:
        txtfile.write('Input file: ' + input_file_name + '\n')
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
stn = 'P26'
# station_name = stn
station_name = 'OSP'

parent_dir = 'C:\\Users\\HourstonH\\Documents\\charles\\' \
             'line_P_data_products\\csv\\has_osd_ctd_flags\\'
# parent_dir = 'C:\\Users\\HourstonH\\Documents\\charles\\' \
#              'bottom_oxygen\\'

o2_bin_dir = '10N_bin_o2_to_select_densities'
o2_bin_file = os.path.join(parent_dir, o2_bin_dir,
                           '{}_data.csv'.format(stn))

avg_dir = '11N_annual_avg_on_dens_surfaces'
avg_file = os.path.join(parent_dir, avg_dir,
                        '{}_data.csv'.format(stn))

densities = [26.5, 26.7, 26.9]

annual_avg_on_density_surfaces(o2_bin_file, avg_file,
                               densities, stn)

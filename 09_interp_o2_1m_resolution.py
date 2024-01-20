import pandas as pd
import numpy as np
import gsw
import os
from scipy.interpolate import interp1d
import convert
from tqdm import trange
from helpers import get_profile_st_en_idx


def interp_1m_resolution(in_df_name: str, out_df_name: str,
                         oxy_unit: str, station: str):
    """
    Linearly interpolate bottle oxygen to 1m vertical resolution
    if not already at that resolution
    :param in_df_name:
    :param station:
    :param out_df_name:
    :param oxy_unit: oxygen unit, "mL/L" or "umol/kg"
    :return:
    """
    in_df = pd.read_csv(in_df_name)

    # Check if oxy unit is in umol/kg
    if oxy_unit == 'mL/L':
        pressure = gsw.p_from_z(
            -in_df.loc[:, 'Depth [m]'].to_numpy(float),
            in_df.loc[:, 'Latitude [deg N]'].to_numpy(float))
        in_df['Oxygen [umol/kg]'] = convert.ml_l_to_umol_kg(
            in_df.loc[:, 'Oxygen [mL/L]'].to_numpy(float),
            in_df.loc[:, 'Longitude [deg E]'].to_numpy(float),
            in_df.loc[:, 'Latitude [deg N]'].to_numpy(float),
            in_df.loc[:, 'Temperature [C]'].to_numpy(float),
            in_df.loc[:, 'Salinity [PSS-78]'].to_numpy(float),
            pressure, in_df_name)[0]

    # Get profile start and end indices
    profile_start_idx, profile_end_idx = get_profile_st_en_idx(
        in_df.loc[:, 'Profile number'])

    # Initialize dataframe to hold interpolated data
    # Need time, density, oxygen value
    out_df_columns = ['Profile number', 'Latitude [deg N]',
                      'Longitude [deg E]', 'Time',
                      'Profile is interpolated',
                      'Depth [m]', 'Oxygen [umol/kg]',
                      'Potential density anomaly [kg/m]', 'File']
    out_df = pd.DataFrame(columns=out_df_columns)

    # Iterate through all the profiles
    counter_profiles_all_nan = 0
    for i in trange(len(profile_start_idx)):
        st = profile_start_idx[i]
        en = profile_end_idx[i]

        # Evaluate conditions for no interpolation

        # Profile only contains one measurement
        profile_has_len_1 = st == en

        # Profile is high resolution with measurements spaced less than 1.5 m apart in the water column
        profile_is_high_res = np.all(
            np.diff(in_df.loc[st:en, 'Depth [m]']) < 1.5
        ) if not profile_has_len_1 else False

        # Profile is too low resolution with measurements spaced more than 0.2 sigma-theta units apart
        profile_spaced_too_far = np.all(np.diff(
            in_df.loc[st:en, 'Potential density anomaly [kg/m]']) > 0.2
        ) if not profile_has_len_1 else False

        # Oxygen measurements are all nans
        if all(np.isnan(in_df.loc[st:en, 'Oxygen [umol/kg]'].to_numpy())):
            # Skip the profile
            counter_profiles_all_nan += 1
            continue
        elif profile_has_len_1 or profile_is_high_res or profile_spaced_too_far:
            # print('Not interpolating')
            # Do not interpolate and just pass the observed level data
            # to the output df
            observed_profile_len = len(in_df.loc[st:en, 'Profile number'])
            dict_add = {
                'Profile number': in_df.loc[st:en, 'Profile number'],
                'Latitude [deg N]': in_df.loc[st:en, 'Latitude [deg N]'],
                'Longitude [deg E]': in_df.loc[st:en, 'Longitude [deg E]'],
                'Time': in_df.loc[st:en, 'Time'],
                'Depth [m]': in_df.loc[st:en, 'Depth [m]'],
                'Profile is interpolated': np.zeros(observed_profile_len),
                'Oxygen [umol/kg]': in_df.loc[st:en, 'Oxygen [umol/kg]'],
                'Potential density anomaly [kg/m]':
                    in_df.loc[st:en, 'Potential density anomaly [kg/m]'],
                'File': in_df.loc[st:en, 'File']
            }
            # for key, value in dict_add.items():
            #     print(key, len(value))
            # print()

            out_df = pd.concat((out_df, pd.DataFrame(dict_add)))
        else:
            # print('Doing linear interpolation')
            # Do linear interpolation to 1m vertical resolution
            # interpolation is not done to standard depths
            interp_oxy_fn = interp1d(
                in_df.loc[st:en, 'Depth [m]'].to_numpy(float),
                in_df.loc[st:en, 'Oxygen [umol/kg]'].to_numpy(float),
                kind='linear',
                bounds_error=False,
                fill_value=np.nan
            )
            interp_density_fn = interp1d(
                in_df.loc[st:en, 'Depth [m]'].to_numpy(float),
                in_df.loc[st:en, 'Potential density anomaly [kg/m]'].to_numpy(float),
                kind='linear',
                bounds_error=False,
                fill_value=np.nan
            )
            # +1 since numpy range not inclusive of end, but pandas range is inclusive
            depth_1m_freq = np.arange(
                np.nanmin(in_df.loc[st:en, 'Depth [m]'].to_numpy(float)),
                np.nanmax(in_df.loc[st:en, 'Depth [m]'].to_numpy(float)) + 1)

            interpolated_profile_len = len(depth_1m_freq)

            oxy_interpolated = interp_oxy_fn(depth_1m_freq)

            density_interpolated = interp_density_fn(depth_1m_freq)

            # Dictionary of values to add to the output df
            dict_add = {
                'Profile number': np.repeat(in_df.loc[st, 'Profile number'],
                                            interpolated_profile_len),
                'Latitude [deg N]': np.repeat(in_df.loc[st, 'Latitude [deg N]'],
                                              interpolated_profile_len),
                'Longitude [deg E]': np.repeat(in_df.loc[st, 'Longitude [deg E]'],
                                               interpolated_profile_len),
                'Time': np.repeat(in_df.loc[st, 'Time'], interpolated_profile_len),
                'Depth [m]': depth_1m_freq,
                'Profile is interpolated': np.ones(interpolated_profile_len),
                'Oxygen [umol/kg]': oxy_interpolated,
                'Potential density anomaly [kg/m]': density_interpolated,
                'File': np.repeat(in_df.loc[st, 'File'],
                                  interpolated_profile_len)
            }
            out_df = pd.concat((out_df, pd.DataFrame(dict_add)))

    # Save the summary statistics
    out_df.reset_index(drop=True, inplace=True)
    out_profile_start_idx = get_profile_st_en_idx(
        out_df.loc[:, 'Profile number'])[0]
    num_interpolated_profiles = int(sum(
        out_df.loc[out_profile_start_idx, 'Profile is interpolated']))
    num_not_interpolated_profiles = int(sum(
        out_df.loc[out_profile_start_idx, 'Profile is interpolated'] == 0))
    summary_stats_file = os.path.join(
        os.path.dirname(out_df_name),
        '{}_interpolation_summary_statistics.txt'.format(station))

    with open(summary_stats_file, 'w') as txtfile:
        txtfile.write('Input file: ' + in_df_name + '\n')
        txtfile.write('Output file: ' + out_df_name + '\n')
        txtfile.write(
            'Number of profiles in: {}\n'.format(len(profile_start_idx)))
        txtfile.write(
            'Number of observations in: {}\n'.format(len(in_df)))
        txtfile.write(
            'Number of profiles out: {}\n'.format(
                len(np.unique(out_df.loc[:, 'Profile number'],
                              return_index=True)[1])))
        txtfile.write(
            'Number of observations out: {}\n'.format(len(out_df)))
        txtfile.write(
            'Number of interpolated profiles: {}\n'.format(
                num_interpolated_profiles))
        txtfile.write(
            'Number of profiles not interpolated: {}\n'.format(
                num_not_interpolated_profiles))
        txtfile.write('Number of profiles containing all nans: {}'.format(
            counter_profiles_all_nan))

    # Save the df
    out_df.to_csv(out_df_name, index=False)
    return


# ----------Interpolate O2 to 1m resolution-----------

# for each station: P4 and P26, LB08
stn = 'P26'
station_name = stn
# station_name = 'OSP'

# parent_dir = 'C:\\Users\\HourstonH\\Documents\\charles\\' \
#              'line_P_data_products\\csv\\has_osd_ctd_flags\\'
# parent_dir = 'C:\\Users\\HourstonH\\Documents\\charles\\' \
#              'bottom_oxygen\\'

parent_dir = 'D:\\lineP\\processing\\'
# parent_dir = ('C:\\Users\\hourstonh\\Documents\\charles\\line_P_data_products\\'
#               'update_jan2024_sopo\\csv_data\\')

density_dir = '08_potential_density_anomalies\\'

density_file = os.path.join(parent_dir, density_dir,
                            '{}_data.csv'.format(stn))
# density_file = os.path.join(parent_dir, density_dir,
#                             '{}_ctd_data_qc.csv'.format(stn))

o2_interp_dir = '09_interpolate_o2_to_1m_res'
o2_interp_file = os.path.join(parent_dir, o2_interp_dir,
                              os.path.basename(density_file))

oxygen_unit = 'umol/kg'  # mL/L umol/kg

interp_1m_resolution(density_file, o2_interp_file, oxygen_unit, stn)

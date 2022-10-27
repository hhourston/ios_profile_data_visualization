import gsw
import pandas as pd
import numpy as np
from gradient_check import vvd_gradient_check
from tqdm import trange
import os
from haversine import haversine
import matplotlib.pyplot as plt


def oxy_ml_l_to_umol_kg(var_df):

    oxygen_umol_per_ml = 44.661
    metre_cube_per_litre = 0.001

    # mask_not_99 = var_df.loc[:, 'Oxygen [umol/kg]'].to_numpy() != -99

    # Calculate pressure
    # Calculate absolute salinity
    # Calculate conservative temperature
    # Calculate density
    # Convert oxygen from ml/l to umol/kg
    pressure_dbar = gsw.p_from_z(
        -var_df.loc[:, 'Depth [m]'].to_numpy(),
        var_df.loc[:, 'Latitude [deg N]'].to_numpy())
    salinity_SA = gsw.SA_from_SP(
        var_df.loc[:, 'Salinity [PSS-78]'].to_numpy(),
        pressure_dbar,
        var_df.loc[:, 'Longitude [deg E]'].to_numpy(),
        var_df.loc[:, 'Latitude [deg N]'].to_numpy())
    temperature_CT = gsw.CT_from_t(
        salinity_SA, var_df.loc[:, 'Temperature [C]'].to_numpy(),
        pressure_dbar)
    density = gsw.rho(salinity_SA, temperature_CT, pressure_dbar)

    # oxygen_umol = np.repeat(-99, len(var_df.loc[:, 'Oxygen [mL/L]']))
    # oxygen_umol[mask_not_99] = [
    #     o / d * oxygen_umol_per_ml/metre_cube_per_litre
    #     for o, d in zip(
    #         var_df.loc[mask_not_99, 'Oxygen [mL/L]'].to_numpy(),
    #         density[mask_not_99])]
    oxygen_umol = [
        o / d * oxygen_umol_per_ml / metre_cube_per_litre
        for o, d in zip(
            var_df.loc[:, 'Oxygen [mL/L]'].to_numpy(float),
            density)]

    return np.array(oxygen_umol)


def range_check(depth, var_data, range_df):
    # Initialize range mask
    range_mask = np.repeat(True, len(depth))
    # True is good, False is failing
    # This check also masks out any bad fill values of -99

    for i in trange(len(depth)):  # len(df) 10
        # Want to find the last depth in the range_df that the i-th depth is
        # greater than?
        # cond = np.where(range_df.loc['Depth_m'] > df.loc[i, 'Depth_m'])[0]

        for j in range(len(range_df)):
            # depth_cond = range_df.loc[j, 'Depth_min'] <= var_df.loc[
            #     i, 'Depth [m]'] <= range_df.loc[j, 'Depth_max']
            # range_cond = range_df.loc[j, 'Coast_N_Pacific_min'] <= var_df.loc[
            #     i, var] <= range_df.loc[j, 'Coast_N_Pacific_max']

            depth_cond = range_df.loc[
                             j, 'Depth_min'] <= depth[i] <= range_df.loc[
                             j, 'Depth_max']
            range_cond = range_df.loc[
                             j, 'Coast_N_Pacific_min'
                         ] <= var_data[i] <= range_df.loc[
                j, 'Coast_N_Pacific_max']

            if depth_cond and not range_cond:
                # Flag the df row if value is not within accepted range
                range_mask[i] = False

    return range_mask


def depth_inv_check(var_df):
    nobs = len(var_df)

    # Initialize mask for depth inversion and copy check
    depth_inv_copy_mask = np.repeat(True, nobs)

    # Profile start indices
    prof_start_ind = np.unique(var_df.loc[:, 'Profile number'],
                               return_index=True)[1]
    # Profile end indices
    prof_end_ind = np.concatenate((prof_start_ind[1:], [nobs]))

    # Iterate through all of the profiles
    for i in range(len(prof_start_ind)):
        # Get profile data;
        # np.arange not inclusive of end which we want here
        indices = np.arange(prof_start_ind[i], prof_end_ind[i])

        # Take first-order difference on the depths
        profile_depth_diffs = np.diff(var_df.loc[indices, 'Depth [m]'])

        # TODO check for upcasts? Otherwise any are all masked out

        profile_depth_mask = np.repeat(True, len(indices))
        profile_depth_mask[1:] = profile_depth_diffs > 0

        depth_inv_copy_mask[indices] = profile_depth_mask

    return depth_inv_copy_mask


def plot_after_coord_checks(station, inFilePath, outPNGpath):
    ctd_df = pd.read_csv(inFilePath)

    # Lat/lon checks
    # Median robust to outliers compared to mean
    median_lat = np.median(ctd_df.loc[:, 'Latitude [deg N]'])
    median_lon = np.median(ctd_df.loc[:, 'Longitude [deg E]'])

    print('Median {} lon and lat: {}, {}'.format(station, median_lon,
                                                 median_lat))

    print('Min and max {} lat: {}, {}'.format(
        station, np.nanmin(ctd_df.loc[:, 'Latitude [deg N]']),
        np.nanmax(ctd_df.loc[:, 'Latitude [deg N]'])))

    print('Min and max {} lon: {}, {}'.format(
        station, np.nanmin(ctd_df.loc[:, 'Longitude [deg E]']),
        np.nanmax(ctd_df.loc[:, 'Longitude [deg E]'])))

    latlon_mask = (ctd_df.loc[:, 'Latitude [deg N]'] > median_lat - 0.1) & \
                  (ctd_df.loc[:, 'Latitude [deg N]'] < median_lat + 0.1) & \
                  (ctd_df.loc[:, 'Longitude [deg E]'] > median_lon - 0.1) & \
                  (ctd_df.loc[:, 'Longitude [deg E]'] < median_lon + 0.1)

    # Apply the mask
    ctd_df_out = ctd_df.loc[latlon_mask, :]

    # Reset the index
    ctd_df_out.reset_index(drop=True, inplace=True)

    # Make the plot

    # Convert time to pandas datetime
    ctd_df['Datetime'] = pd.to_datetime(ctd_df.loc[:, 'Time'])

    fig, ax = plt.subplots()
    ax.scatter(ctd_df.loc[:, 'Datetime'], ctd_df.loc[:, 'Depth [m]'], s=4)
    plt.gca().invert_yaxis()

    # Add text about bottom depth
    # By default, this is in data coordinates.
    text_xloc, text_yloc = [0.95, 0.01]
    # Transform the coordinates from data to plot coordinates
    # max_depth >= common maximum depth
    max_depth = np.round(np.nanmax(ctd_df.loc[:, 'Depth [m]']), 2)
    ax.text(text_xloc, text_yloc,
            '{} bottom depth = {}m'.format(station, max_depth),
            verticalalignment='bottom', horizontalalignment='right',
            transform=ax.transAxes, fontsize='large')

    ax.set_ylabel('Depth [m]')
    ax.set_title('{} CTD Depth vs Time, only lat/lon check'.format(station))
    plt.tight_layout()
    plt.savefig(outPNGpath)
    plt.close()

    return


def main(station, inFilePath, outFilePath, coord_check='planar', station_coords=None):
    # coord_check: 'planar' or 'haversine', must provide station_coords (lat, lon) for haversine
    ctd_df = pd.read_csv(inFilePath)
    oxygen_column = ctd_df.columns[
        ['Oxygen' in colname for colname in ctd_df.columns]][0]
    oxygen_unit = oxygen_column.split('[')[1][:-1]

    # Lat/lon checks
    # Median robust to outliers compared to mean
    median_lat = np.median(ctd_df.loc[:, 'Latitude [deg N]'])
    median_lon = np.median(ctd_df.loc[:, 'Longitude [deg E]'])

    print('Median {} lon and lat: {}, {}'.format(station, median_lon,
                                                 median_lat))

    print('Min and max {} lat: {}, {}'.format(
        station, np.nanmin(ctd_df.loc[:, 'Latitude [deg N]']),
        np.nanmax(ctd_df.loc[:, 'Latitude [deg N]'])))

    print('Min and max {} lon: {}, {}'.format(
        station, np.nanmin(ctd_df.loc[:, 'Longitude [deg E]']),
        np.nanmax(ctd_df.loc[:, 'Longitude [deg E]'])))

    if coord_check == 'planar':
        # Set maximum variation limit from median
        # 2022-09-06 reduced from 0.1 to 0.075
        limit = 0.075 if station_coords is None else station_coords
        latlon_mask = (ctd_df.loc[:, 'Latitude [deg N]'] > median_lat - limit) & \
                      (ctd_df.loc[:, 'Latitude [deg N]'] < median_lat + limit) & \
                      (ctd_df.loc[:, 'Longitude [deg E]'] > median_lon - limit) & \
                      (ctd_df.loc[:, 'Longitude [deg E]'] < median_lon + limit)

        # Apply the mask
        ctd_df_out = ctd_df.loc[latlon_mask, :]
    elif coord_check == 'haversine':
        km_to_decimal_degrees = 1/111
        # The limit used in Cummins & Ross (2020)
        limit = 24 * km_to_decimal_degrees
        distances = np.array([haversine((lat_i, lon_i), station_coords)
                              for lat_i, lon_i in zip(ctd_df.loc[:, 'Latitude [deg N]'],
                                                      ctd_df.loc[:, 'Longitude [deg E]'])])
        latlon_mask = distances <= limit
        ctd_df_out = ctd_df.loc[latlon_mask, :]
    else:
        print(f'coord_check method {coord_check} is invalid')

    # Reset the index
    ctd_df_out.reset_index(drop=True, inplace=True)

    # ------------------------Data checks from NEP climatology------------------------

    # -----Depth checks-----

    # Mask out depths out of range (above water or below 10,000m)
    depth_lim_mask = (ctd_df_out.loc[:, 'Depth [m]'] > 0) & \
                     (ctd_df_out.loc[:, 'Depth [m]'] < 1e4)

    # Apply the masks
    ctd_df_out = ctd_df_out.loc[depth_lim_mask, :]

    # Reset the index
    ctd_df_out.reset_index(drop=True, inplace=True)

    # Mask out depth inversions and copies
    depth_inv_mask = depth_inv_check(ctd_df_out)

    # Apply the mask
    ctd_df_out = ctd_df_out.loc[depth_inv_mask, :]

    # Reset the index
    ctd_df_out.reset_index(drop=True, inplace=True)

    # -----Range checks-----

    # Mask out values outside acceptable ranges for each variable
    # Use preset ranges from WOD
    range_file_T = 'C:\\Users\\HourstonH\\Documents\\climatology\\' \
                   'wod18_users_manual_tables\\wod18_ranges_TEMP_Coast_N_Pac.csv'
    range_file_S = 'C:\\Users\\HourstonH\\Documents\\climatology\\' \
                   'wod18_users_manual_tables\\wod18_ranges_PSAL_Coast_N_Pac.csv'
    range_file_O = 'C:\\Users\\HourstonH\\Documents\\climatology\\' \
                   'wod18_users_manual_tables\\wod18_ranges_DOXY_Coast_N_Pac.csv'

    range_T_df = pd.read_csv(range_file_T)
    range_S_df = pd.read_csv(range_file_S)
    range_O_df = pd.read_csv(range_file_O)

    # Make sure O ranges are in the right units for comparing to WOA18
    if oxygen_unit == 'mL/L':
        o_umol_arr = oxy_ml_l_to_umol_kg(ctd_df_out)
    else:
        o_umol_arr = ctd_df_out.loc[:, oxygen_column].to_numpy(dtype=float)

    T_range_mask = range_check(
        ctd_df_out.loc[:, 'Depth [m]'].to_numpy(dtype=float),
        ctd_df_out.loc[:, 'Temperature [C]'].to_numpy(dtype=float),
        range_T_df)
    S_range_mask = range_check(
        ctd_df_out.loc[:, 'Depth [m]'].to_numpy(dtype=float),
        ctd_df_out.loc[:, 'Salinity [PSS-78]'].to_numpy(dtype=float),
        range_S_df)
    O_range_mask = range_check(
        ctd_df_out.loc[:, 'Depth [m]'].to_numpy(dtype=float),
        o_umol_arr,
        range_O_df)

    ctd_df_out.loc[~T_range_mask, 'Temperature [C]'] = np.nan
    ctd_df_out.loc[~S_range_mask, 'Salinity [PSS-78]'] = np.nan
    ctd_df_out.loc[~O_range_mask, oxygen_column] = np.nan

    # -----Gradient checks-----

    gradient_file = 'C:\\Users\\HourstonH\\Documents\\climatology\\' \
                    'wod18_users_manual_tables\\' \
                    'wod18_max_gradient_inversion.csv'

    gradient_df = pd.read_csv(gradient_file, index_col='Variable')

    T_gradient_mask = vvd_gradient_check(
        ctd_df_out.loc[:, 'Profile number'].to_numpy(dtype=int),
        ctd_df_out.loc[:, 'Depth [m]'].to_numpy(dtype=float),
        ctd_df_out.loc[:, 'Temperature [C]'].to_numpy(dtype=float),
        gradient_df, 'Temperature')
    S_gradient_mask = vvd_gradient_check(
        ctd_df_out.loc[:, 'Profile number'].to_numpy(dtype=int),
        ctd_df_out.loc[:, 'Depth [m]'].to_numpy(dtype=float),
        ctd_df_out.loc[:, 'Salinity [PSS-78]'].to_numpy(dtype=float),
        gradient_df, 'Salinity')
    O_gradient_mask = vvd_gradient_check(
        ctd_df_out.loc[:, 'Profile number'].to_numpy(dtype=int),
        ctd_df_out.loc[:, 'Depth [m]'].to_numpy(dtype=float),
        o_umol_arr,
        gradient_df, 'Oxygen')

    ctd_df_out.loc[~T_gradient_mask, 'Temperature [C]'] = np.nan
    ctd_df_out.loc[~S_gradient_mask, 'Salinity [PSS-78]'] = np.nan
    ctd_df_out.loc[~O_gradient_mask, oxygen_column] = np.nan

    # -----Apply masks-----

    # Print summary statistics
    summary_statistics_file = os.path.join(
        os.path.dirname(outFilePath),
        '{}_QC_summary_statistics.txt'.format(station))
    with open(summary_statistics_file, 'w') as txtfile:
        txtfile.write('Source file: {}\n'.format(inFilePath))
        txtfile.write('Output file: {}\n'.format(outFilePath))
        txtfile.write(
            'Number of input observations: {}\n'.format(len(ctd_df)))
        txtfile.write(
            'Number of obs passing lat/lon check: {}\n'.format(sum(latlon_mask)))
        txtfile.write(
            'Number of obs passing depth limits check: {}\n'.format(sum(depth_lim_mask)))
        txtfile.write(
            'Number of obs passing depth inversion/copy check: {}\n'.format(sum(depth_inv_mask)))
        txtfile.write(
            'Number of T obs passing range check: {}\n'.format(sum(T_range_mask)))
        txtfile.write(
            'Number of S obs passing range check: {}\n'.format(sum(S_range_mask)))
        txtfile.write(
            'Number of O obs passing range check: {}\n'.format(sum(O_range_mask)))
        txtfile.write(
            'Number of T obs passing gradient check: {}\n'.format(sum(T_gradient_mask)))
        txtfile.write(
            'Number of S obs passing gradient check: {}\n'.format(sum(S_gradient_mask)))
        txtfile.write(
            'Number of O obs passing gradient check: {}'.format(sum(O_gradient_mask)))

    # # Combine masks with logical "and"
    # merged_mask = latlon_mask & depth_lim_mask & depth_inv_mask
    #
    # T_mask = T_range_mask & T_gradient_mask
    # S_mask = S_range_mask & S_gradient_mask
    # O_mask = O_range_mask & O_gradient_mask
    #
    # # Apply the masks to the dataframe of observations
    # ctd_df_out = ctd_df
    # ctd_df_out.loc[~T_mask, 'Temperature [C]'] = np.nan
    # ctd_df_out.loc[~S_mask, 'Salinity [PSS-78]'] = np.nan
    # ctd_df_out.loc[~O_mask, 'Oxygen [umol/kg]'] = np.nan
    # ctd_df_out = ctd_df_out.loc[merged_mask, :]

    # Export the QC'ed dataframe of observations to a csv file
    ctd_df_out.to_csv(outFilePath, index=False)

    return


# ------------------------------OSP upper ocean T---------------------------------
parent_dir = 'C:\\Users\\HourstonH\\Documents\\charles\\' \
             'our_warming_ocean\\osp_sst\\csv\\'

sampling_station = 'P26'
p26_coords = (50, -145)  # Cummins and Ross (2020)

data_file_path = os.path.join(
    parent_dir, '02_merge\\{}_data.csv'.format(sampling_station))

output_file_path = os.path.join(
    parent_dir, '03_QC', os.path.basename(data_file_path))

main(sampling_station, data_file_path, output_file_path,
     coord_check='haversine', station_coords=p26_coords)

# ---------------------------------SSI stations-----------------------------------
# # 'SI01'  # '59'  # '42'  # 'GEO1'  # 'LBP3'  # 'LB08'  # 'P1'
# # P4 P26
# sampling_station = 'P26'
# # data_types = 'ctd'
# # data_types = 'CTD_BOT_CHE_OSD'
# parent_dir = 'C:\\Users\\HourstonH\\Documents\\charles\\' \
#              'line_P_data_products\\csv\\has_osd_ctd_flags\\'
#
# data_file_path = os.path.join(
#     parent_dir, '02_merge\\{}_data.csv'.format(sampling_station))
#
# output_file_path = os.path.join(
#     parent_dir, '03_QC', os.path.basename(data_file_path))
#
# main(sampling_station, data_file_path, output_file_path)

# for s in ['59', '42', 'GEO1', 'LBP3', 'LB08', 'P1']:
#     data_file_path = 'C:\\Users\\HourstonH\\Documents\\ctd_visualization\\' \
#                      'csv\\{}_{}_data.csv'.format(s, data_types)
#     output_file_dir = os.path.dirname(data_file_path)
#     output_file_path = os.path.join(
#         output_file_dir, os.path.basename(data_file_path).replace('.csv', '_qc.csv'))
#
#     main(s, data_file_path, output_file_path)

# --------------------------------------------------------------------
# # Testing stations for missing data at depth
# # LBP3 LB08
# sampling_station = 'LBP3'
# data_types = 'ctd'
# LB_file = 'C:\\Users\\HourstonH\\Documents\\ctd_visualization\\csv\\' \
#           '{}_{}_data.csv'.format(sampling_station, data_types)
# output_file_dir = 'C:\\Users\\HourstonH\\Documents\\ctd_visualization\\csv\\'
# png_path = os.path.join(output_file_dir, '{}_{}_depth_vs_time.png'.format(
#     sampling_station, data_types))
# plot_after_coord_checks(sampling_station, LB_file, png_path)

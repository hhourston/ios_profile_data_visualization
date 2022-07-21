import gsw
import pandas as pd
import numpy as np
from gradient_check import vvd_gradient_check
from tqdm import trange


def oxy_ml_l_to_umol_kg(var_df):

    oxygen_umol_per_ml = 44.661
    metre_cube_per_litre = 0.001

    # mask_not_99 = var_df.loc[:, 'Oxygen [mL/L]'].to_numpy() != -99

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
            var_df.loc[:, 'Oxygen [mL/L]'].to_numpy(),
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


def main(station):
    ctd_infile = 'C:\\Users\\HourstonH\\Documents\\ctd_visualization\\' \
                 'csv\\{}_ctd_data.csv'.format(station)

    ctd_df = pd.read_csv(ctd_infile)

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

    # ------------------------Data checks from NEP climatology------------------------

    # -----Depth checks-----

    # Mask out depths out of range (above water or below 10,000m)
    depth_lim_mask = (ctd_df_out.loc[:, 'Depth [m]'] > 0) | \
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
    range_file_T = 'C:\\Users\\HourstonH\\Documents\\NEP_climatology\\' \
                   'literature\\WOA docs\\wod18_users_manual_tables\\' \
                   'wod18_ranges_TEMP_Coast_N_Pac.csv'
    range_file_S = 'C:\\Users\\HourstonH\\Documents\\NEP_climatology\\' \
                   'literature\\WOA docs\\wod18_users_manual_tables\\' \
                   'wod18_ranges_PSAL_Coast_N_Pac.csv'
    range_file_O = 'C:\\Users\\HourstonH\\Documents\\NEP_climatology\\' \
                   'literature\\WOA docs\\wod18_users_manual_tables\\' \
                   'wod18_ranges_DOXY_Coast_N_Pac.csv'

    range_T_df = pd.read_csv(range_file_T)
    range_S_df = pd.read_csv(range_file_S)
    range_O_df = pd.read_csv(range_file_O)

    # Make sure O ranges are in the right units for comparing to WOA18
    o_umol = oxy_ml_l_to_umol_kg(ctd_df_out)

    T_range_mask = range_check(
        ctd_df_out.loc[:, 'Depth [m]'].to_numpy(),
        ctd_df_out.loc[:, 'Temperature [C]'].to_numpy(), range_T_df)
    S_range_mask = range_check(
        ctd_df_out.loc[:, 'Depth [m]'].to_numpy(),
        ctd_df_out.loc[:, 'Salinity [PSS-78]'].to_numpy(), range_S_df)
    O_range_mask = range_check(
        ctd_df_out.loc[:, 'Depth [m]'].to_numpy(),
        o_umol, range_O_df)

    ctd_df_out.loc[~T_range_mask, 'Temperature [C]'] = np.nan
    ctd_df_out.loc[~S_range_mask, 'Salinity [PSS-78]'] = np.nan
    ctd_df_out.loc[~O_range_mask, 'Oxygen [mL/L]'] = np.nan

    # -----Gradient checks-----

    gradient_file = 'C:\\Users\\HourstonH\\Documents\\NEP_climatology\\' \
                    'literature\\WOA docs\\wod18_users_manual_tables\\' \
                    'wod18_max_gradient_inversion.csv'

    gradient_df = pd.read_csv(gradient_file, index_col='Variable')

    T_gradient_mask = vvd_gradient_check(
        ctd_df_out.loc[:, 'Profile number'].to_numpy(),
        ctd_df_out.loc[:, 'Depth [m]'].to_numpy(),
        ctd_df_out.loc[:, 'Temperature [C]'].to_numpy(),
        gradient_df, 'Temperature')
    S_gradient_mask = vvd_gradient_check(
        ctd_df_out.loc[:, 'Profile number'].to_numpy(),
        ctd_df_out.loc[:, 'Depth [m]'].to_numpy(),
        ctd_df_out.loc[:, 'Salinity [PSS-78]'].to_numpy(),
        gradient_df, 'Salinity')
    O_gradient_mask = vvd_gradient_check(
        ctd_df_out.loc[:, 'Profile number'].to_numpy(),
        ctd_df_out.loc[:, 'Depth [m]'].to_numpy(),
        o_umol, gradient_df, 'Oxygen')

    ctd_df_out.loc[~T_gradient_mask, 'Temperature [C]'] = np.nan
    ctd_df_out.loc[~S_gradient_mask, 'Salinity [PSS-78]'] = np.nan
    ctd_df_out.loc[~O_gradient_mask, 'Oxygen [mL/L]'] = np.nan

    # -----Apply masks-----

    # Print summary statistics
    print('Number of input observations:', len(ctd_df))
    print('Number of obs passing lat/lon check:', sum(latlon_mask))
    print('Number of obs passing depth limits check:', sum(depth_lim_mask))
    print('Number of obs passing depth inversion/copy check:', sum(depth_inv_mask))
    print('Number of T obs passing range check:', sum(T_range_mask))
    print('Number of S obs passing range check:', sum(S_range_mask))
    print('Number of O obs passing range check:', sum(O_range_mask))
    print('Number of T obs passing gradient check:', sum(T_gradient_mask))
    print('Number of S obs passing gradient check:', sum(S_gradient_mask))
    print('Number of O obs passing gradient check:', sum(O_gradient_mask))

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
    # ctd_df_out.loc[~O_mask, 'Oxygen [mL/L]'] = np.nan
    # ctd_df_out = ctd_df_out.loc[merged_mask, :]

    # Export the QC'ed dataframe of observations to a csv file
    df_out_name = 'C:\\Users\\HourstonH\\Documents\\ctd_visualization\\' \
                  'csv\\{}_ctd_data_qc.csv'.format(station)

    ctd_df_out.to_csv(df_out_name, index=False)

    return df_out_name


ctd_station = 'LB08'  # 'SI01'  # '59'  # '42'  # 'GEO1'  # 'LBP3'  # 'LB08'  # 'P1'

main(ctd_station)

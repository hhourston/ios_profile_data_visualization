import pandas as pd
import numpy as np
import os
from tqdm import trange
from helpers import get_profile_st_en_idx


def bin_o2_to_select_densities(in_df_name: str, out_df_name: str,
                               select_densities, station: str):
    in_df = pd.read_csv(in_df_name)
    # Criteria for binning to select densities
    # Potential density must be within close_criteria of the select density
    # for the oxygen value to be binned to the select density
    close_criteria = 0.005
    # Only bin one value per select density per profile

    in_df['Potential density anomaly bin [kg/m]'] = np.repeat(
        np.nan, len(in_df))
    # in_df['Potential density anomaly bin is dupl'] = np.zeros(len(in_df))

    # for d in select_densities:
    #     # Create mask
    #     msk = abs(in_df.loc[:, 'Potential density anomaly [kg/m]'] - d
    #               ) < close_criteria
    #     in_df.loc[msk, 'Potential density anomaly bin [kg/m]'] = d
    #
    #     # msk_prof_nums = in_df.loc[msk, 'Profile number']
    #     # for i in range(len(msk_prof_nums) - 1):

    # Remove bin duplicates in any
    # profiles, keeping the observation with the closest density to the
    # select density
    profile_start_idx, profile_end_idx = get_profile_st_en_idx(
        in_df.loc[:, 'Profile number'])

    # counter_density_duplicates = 0
    for i in trange(len(profile_start_idx)):
        st, en = profile_start_idx[i], profile_end_idx[i]
        # If there are binned density duplicates in the profile
        # Find the observation with the closest observed or interpolated
        # density to the binned density
        # And set to nan the other duplicates
        for d in select_densities:
            abs_density_diffs = abs(
                in_df.loc[st:en, 'Potential density anomaly [kg/m]'] - d
            )
            if np.nanmin(abs_density_diffs) < close_criteria:
                # argmin returns the index of the minimum value
                in_df.loc[np.nanargmin(abs_density_diffs) + st,
                          'Potential density anomaly bin [kg/m]'] = d
            # subsetter = np.where(
            #     in_df.loc[st:en, 'Potential density anomaly bin [kg/m]'
            #               ] == d
            # )[0] + st
            # if len(subsetter) > 1:
            #     density_differences = in_df.loc[
            #         subsetter, 'Potential density anomaly [kg/m]']
            #     # Order is from zero (the smallest difference)
            #     # to n, where n is the number of bin matches in the profile
            #     density_diffs_order = np.array([
            #         x for _, x in
            #         zip(sorted(density_differences), np.arange(3))
            #     ])
            #     # Update potential density bin value to no bin
            #     in_df.loc[subsetter[density_diffs_order > 0],
            #               'Potential density anomaly bin'
            #               ] = np.nan
            #     # Update counter
            #     counter_density_duplicates += sum(density_diffs_order > 0)

    out_df = in_df.dropna(
        axis='index', subset=['Potential density anomaly bin [kg/m]'])

    # Save summary statistics
    summary_statistics_file = os.path.join(
        os.path.dirname(out_df_name),
        '{}_density_bin_check_summary.txt'.format(station))

    output_profile_start_idx = get_profile_st_en_idx(
        out_df.loc[:, 'Profile number'])[0]

    with open(summary_statistics_file, 'w') as txtfile:
        txtfile.write('Input file: ' + in_df_name + '\n')
        txtfile.write('Output file: ' + out_df_name + '\n')
        txtfile.write('Number of input profiles: ' +
                      str(len(profile_start_idx)) + '\n')
        txtfile.write('Number of output profiles: ' +
                      str(len(output_profile_start_idx)))
        # txtfile.write('Number of discarded density bin duplicates: ' +
        #               str(counter_density_duplicates))
        # txtfile.write('Number of obs binned by density: {}'.format())

    out_df.to_csv(out_df_name, index=False)

    return out_df


# ------------------------------------------------------------------
# Bin the 1m resolution oxygen data to the select densities

# for each station: P4 and P26, LB08
stn = 'P4'
station_name = stn
# station_name = 'OSP'
parent_dir = 'C:\\Users\\HourstonH\\Documents\\charles\\' \
             'line_P_data_products\\csv\\has_osd_ctd_flags\\'
# parent_dir = 'C:\\Users\\HourstonH\\Documents\\charles\\' \
#              'bottom_oxygen\\'

o2_interp_dir = '09N_interpolate_o2_to_1m_res'
o2_interp_file = os.path.join(parent_dir, o2_interp_dir,
                              '{}_data'.format(stn))

potential_densities = np.array([26.5, 26.7, 26.9])
o2_bin_dir = '10N_bin_o2_to_select_densities'
o2_bin_file = os.path.join(parent_dir, o2_bin_dir,
                           '{}_data'.format(stn))
print(o2_interp_file)
print(o2_bin_file)

dfout = bin_o2_to_select_densities(o2_interp_file, o2_bin_file,
                                   potential_densities, stn)

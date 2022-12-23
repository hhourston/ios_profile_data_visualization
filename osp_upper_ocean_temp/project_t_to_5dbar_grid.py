import pandas as pd
import numpy as np
import os
from gsw import p_from_z
from scipy.interpolate import interp1d
from datetime import datetime
from tqdm import trange

# Something is broken in here... need to fix next time
# Also make a steps document for this project to share with Lu

pressure_below_OMZ = 1950  # dbar
pressure_inside_OMZ = (500, 1500)  # dbar
# Define upper oxycline as above 500 dbar

parent_dir = 'C:\\Users\\HourstonH\\Documents\\charles\\our_warming_ocean\\' \
             'osp_upper_ocean_temp\\processing\\'
input_file = os.path.join(parent_dir, '04_inexact_duplicate_check', 'P26_data.csv')
output_file = os.path.join(parent_dir, '05_5dbar_vertical_grid',
                           os.path.basename(input_file))

indf = pd.read_csv(input_file)

prof_st_idx = np.unique(indf.loc[:, 'Profile number'], return_index=True)[1]
prof_en_idx = np.concatenate((prof_st_idx[1:] - 1, np.array([len(indf) - 1])))

print('Number of profiles:', len(prof_st_idx))

count_profiles_used = 0

# Initialize output gridded dataframe
outdf = pd.DataFrame()
outdf_columns = indf.columns[:4].to_list() + ['Gridded pressure [dbar]',
                                              'Gridded temperature [C]',
                                              'Gridded salinity [PSS-78]',
                                              'Gridded oxygen [umol/kg]',
                                              'Time_dt']

for i in trange(len(prof_st_idx)):
    d = indf.loc[prof_st_idx[i]:prof_en_idx[i], 'Depth [m]'].to_numpy()
    lat = indf.loc[prof_st_idx[i], 'Latitude [deg N]']
    lon = indf.loc[prof_st_idx[i], 'Longitude [deg E]']
    t = indf.loc[prof_st_idx[i]:prof_en_idx[i], 'Temperature [C]'].to_numpy()
    s = indf.loc[prof_st_idx[i]:prof_en_idx[i], 'Salinity [PSS-78]'].to_numpy()
    o = indf.loc[prof_st_idx[i]:prof_en_idx[i], 'Oxygen [umol/kg]'].to_numpy()
    p = p_from_z(-d, lat)

    # Set up conditions for linear interpolation
    if len(d) <= 1:
        # Skip this cast and proceed to the next one
        continue

    # Set up pressure criteria
    below_OMZ_criteria = any(p > pressure_below_OMZ)
    in_OMZ_criteria = sum((p >= pressure_inside_OMZ[0]) & (p <= pressure_inside_OMZ[1])
                          ) >= 3
    # Need at least 100 dbar resolution in upper oxycline
    # Use prepend to return same size result as input
    oxycline_criteria = sum((p < 500) & (np.diff(p, prepend=p[0]) < 100)) == sum(p < 500)

    if below_OMZ_criteria and in_OMZ_criteria and oxycline_criteria:
        # Set up 5 dbar grid according to range of p
        p_grid_5dbar = np.arange(0, np.nanmax(p) + 5, 5)
        p_grid_5dbar = p_grid_5dbar[(p_grid_5dbar >= np.nanmin(p)) &
                                    (p_grid_5dbar <= np.nanmax(p))]
        # Do linear interpolation to 5dbar grid
        interp_fn_t = interp1d(p, t, kind='linear')
        interp_fn_s = interp1d(p, s, kind='linear')
        interp_fn_o = interp1d(p, o, kind='linear')
        t_gridded = interp_fn_t(p_grid_5dbar)
        s_gridded = interp_fn_s(p_grid_5dbar)
        o_gridded = interp_fn_o(p_grid_5dbar)

        # Append results to output dataframe
        dict_to_add = {
            'Profile number': np.repeat(indf.loc[prof_st_idx, 'Profile number'],
                                        len(p_grid_5dbar)),
            'Latitude [deg N]': np.repeat(indf.loc[prof_st_idx, 'Latitude [deg N]'],
                                          len(p_grid_5dbar)),
            'Longitude [deg E]': np.repeat(indf.loc[prof_st_idx, 'Longitude [deg E]'],
                                           len(p_grid_5dbar)),
            'Time': np.repeat(indf.loc[prof_st_idx, 'Time'], len(p_grid_5dbar)),
            'Time_dt': np.repeat(indf.loc[prof_st_idx, 'Time_dt'], len(p_grid_5dbar)),
            'Gridded pressure [dbar]': p_grid_5dbar,
            'Gridded temperature [C]': t_gridded,
            'Gridded salinity [PSS-78]': s_gridded,
            'Gridded oxygen [umol/kg]': o_gridded
        }
        outdf = pd.concat((outdf, pd.DataFrame(dict_to_add)))

        # Update count
        count_profiles_used += 1

# Save output dataframe
outdf.to_csv(output_file, index=False)

# Save summary statistics to a txt file
summary_stats_file = os.path.join(os.path.dirname(output_file),
                                  'summary_statistics.txt')
with open(summary_stats_file, 'w') as f:
    f.write(datetime.today().strftime('%Y-%m-%d %H:%M:%S') + '\n')
    f.write('Input file: ' + input_file + '\n')
    f.write('Output file: ' + output_file + '\n')
    f.write('Input number of profiles: ' + str(len(prof_st_idx)) + '\n')
    f.write('Output number of profiles: ' + str(count_profiles_used))

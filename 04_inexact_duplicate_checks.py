import pandas as pd
import numpy as np
from tqdm import trange
import os

# Need to do inexact duplicate checking if there are bottle
# and CTD data from the same time/location
# Keep the CTD data since it's higher resolution


def coords_are_close(lat1, lon1, lat2, lon2, criteria):
    # bounding box criteria for observation coordinates being close together
    return abs(lat1 - lat2) < criteria and abs(lon1 - lon2) < criteria


def time_is_close(t1: pd.Timestamp, t2: pd.Timestamp, criteria):
    # Take difference
    # Check if difference is less than criteria
    return abs(t2 - t1) < pd.to_timedelta(criteria)


# P4 P26
sampling_station = 'P4'
# data_types = 'ctd'
# data_types = 'CTD_BOT_CHE_OSD'

# input_dir = 'C:\\Users\\HourstonH\\Documents\\charles\\' \
#              'line_P_data_products\\csv\\has_osd_ctd_flags\\03_QC\\'

# input_dir = 'C:\\Users\\HourstonH\\Documents\\charles\\' \
#              'our_warming_ocean\\osp_sst\\csv\\03_QC\\'

# data_file_path = 'C:\\Users\\HourstonH\\Documents\\charles\\' \
#                  'line_P_data_products\\csv\\03_QC\\' \
#                  '{}_{}_data.csv'.format(sampling_station, data_types)

# input_file_path = os.path.join(
#     input_dir, '{}_data.csv'.format(sampling_station))

# output_file_path = 'C:\\Users\\HourstonH\\Documents\\charles\\' \
#                    'line_P_data_products\\csv\\' \
#                    '{}_{}_data_idc.csv'.format(sampling_station, data_types)

input_dir = 'D:\\lineP\\csv_data\\03_merge\\'

input_file_path = os.path.join(input_dir, '{}_data.csv'.format(sampling_station))

output_file_path = input_file_path.replace(
    '03_merge', '04_inexact_duplicate_check')

# Iterate through all the profile numbers
# If there is a match of time/lat/lon, keep the profile
# that has more measurements in it (corresponding to CTD)

dfin = pd.read_csv(input_file_path)

profile_start_idx = np.unique(dfin.loc[:, 'Profile number'],
                              return_index=True)[1]
profile_end_idx = np.concatenate(
    (profile_start_idx[1:] - 1, np.array([len(dfin)])))

# Convert time to pandas.datetime format
dfin['Time_dt'] = pd.to_datetime(dfin.Time)

# Add column containing flags
dfin['Inexact_duplicate_flag'] = np.zeros(len(dfin), dtype='int32')

for i in trange(len(profile_start_idx)):
    start_idx_i = profile_start_idx[i]
    end_idx_i = profile_end_idx[i]
    time_i = dfin.loc[start_idx_i, 'Time_dt']
    lat_i = dfin.loc[start_idx_i, 'Latitude [deg N]']
    lon_i = dfin.loc[start_idx_i, 'Longitude [deg E]']
    # Iterate through the rest of the profiles
    for j in range(i + 1, len(profile_start_idx)):
        start_idx_j = profile_start_idx[j]
        end_idx_j = profile_end_idx[j]
        # Check if profile has already been flagged in an earlier iteration
        # If so then skip it
        if dfin.loc[start_idx_j, 'Inexact_duplicate_flag'] == 1:
            continue
        time_j = dfin.loc[start_idx_j, 'Time_dt']
        lat_j = dfin.loc[start_idx_j, 'Latitude [deg N]']
        lon_j = dfin.loc[start_idx_j, 'Longitude [deg E]']
        # Check if profiles are empty or contain only nans?
        # Compare the two profiles selected in time and space
        if coords_are_close(
                lat_i, lon_i, lat_j, lon_j, 0.5  # 0.2
        ) and time_is_close(time_i, time_j, '1 hour'):
            # Check which profile is longer
            if len(dfin.loc[start_idx_i:end_idx_i]
                   ) >= len(dfin.loc[start_idx_j:end_idx_j]):
                dfin.loc[start_idx_j:end_idx_j, 'Inexact_duplicate_flag'] = 1
            else:
                dfin.loc[start_idx_i:end_idx_i, 'Inexact_duplicate_flag'] = 1

# Print summary statistics to text file
summary_statistics_file = os.path.join(
    os.path.dirname(output_file_path),
    '{}_inexact_duplicate_check_summary_statistics.txt'.format(
        sampling_station))
with open(summary_statistics_file, 'a') as txtfile:
    txtfile.write('Source file: {}\n'.format(input_file_path))
    txtfile.write('Output file: {}\n'.format(output_file_path))
    txtfile.write(
        'Number of profiles in: {}\n'.format(
            len(profile_start_idx)))
    txtfile.write(
        'Number of profiles out: {}\n\n'.format(
            sum(dfin.loc[profile_start_idx,
                         'Inexact_duplicate_flag'] == 0)))

# Apply the inexact duplicate flag
msk = dfin.loc[:, 'Inexact_duplicate_flag'] == 0
dfout = dfin.loc[msk, :]

# Remove the inexact duplicate flag column without SettingWithCopyWarning
dfout = dfout.drop(columns='Inexact_duplicate_flag')

# Save the dataframe to csv
dfout.to_csv(output_file_path, index=False)

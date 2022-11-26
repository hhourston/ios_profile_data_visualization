import pandas as pd
import os
import numpy as np

# input_file = 'C:\\Users\\HourstonH\\Documents\\charles\\' \
#              'line_P_data_products\\csv\\01_convert\\' \
#              'P4_NODC_OSD_data.csv'

# input_file = 'C:\\Users\\HourstonH\\Documents\\charles\\' \
#              'line_P_data_products\\csv\\01_convert\\' \
#              'P26_NODC_OSD_GLD_PFL_data.csv'

# P4 P26
stn = 'P4'
# parent_dir = 'C:\\Users\\HourstonH\\Documents\\charles\\' \
#              'line_P_data_products\\csv\\has_osd_ctd_flags\\'

# parent_dir = 'C:\\Users\\HourstonH\\Documents\\charles\\' \
#              'our_warming_ocean\\osp_sst\\csv\\'
parent_dir = 'D:\\lineP\\csv_data\\'

input_file = os.path.join(
    parent_dir, '01_convert\\{}_NODC_OSD_CTD_data.csv'.format(stn))

output_dir = os.path.join(parent_dir, '01b_apply_nodc_flags\\')

output_file = os.path.join(output_dir,
                           os.path.basename(input_file))

df = pd.read_csv(input_file)

# Create masks of bad values
msk_T = df.loc[:, 'Temperature profile flag'] != 0
msk_S = df.loc[:, 'Salinity profile flag'] != 0
msk_O = df.loc[:, 'Oxygen profile flag'] != 0

# Print number of bad values for each variable
print(sum(msk_T))
print(sum(msk_S))
print(sum(msk_O))
print()

df.loc[msk_T, 'Temperature [C]'] = np.nan
df.loc[msk_S, 'Salinity [PSS-78]'] = np.nan
df.loc[msk_O, 'Oxygen [umol/kg]'] = np.nan

# Iterate through all the profiles and delete any profiles
# having all nan T and S and O

prof_start_idx = np.unique(df.loc[:, 'Profile number'],
                           return_index=True)[1]
prof_end_idx = np.concatenate((prof_start_idx[1:] - 1,
                               np.array([len(df)])))

print(len(df))
for i in range(len(prof_start_idx)):
    st = prof_start_idx[i]
    en = prof_end_idx[i]
    # Change or to and, to be less strict
    if (np.all(
            pd.isna(
                df.loc[
                    st:en, 'Temperature [C]'])) &
        np.all(
            pd.isna(
                df.loc[
                    st:en, 'Salinity [PSS-78]'])) &
        np.all(
            pd.isna(
                df.loc[st:en, 'Oxygen [umol/kg]']))
        ):
        # print(df.loc[st:en, :])
        df.drop(index=np.arange(st, en + 1), inplace=True)
print(len(df))

df.to_csv(output_file, index=False)

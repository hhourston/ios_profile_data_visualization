import pandas as pd
import os
import glob

# Merge the nodc and wp csv files before qc checks
# Take nodc files from folder where nodc flags were applied
# and take wp csv files from the "convert" folder before

stn = 'P4'  # P26 P4
nodc_dtypes = 'OSD_CTD'  # OSD_GLD_PFL
# data_types = 'CTD_BOT_CHE_OSD'

"""
# parent_dir = 'C:\\Users\\HourstonH\\Documents\\charles\\' \
#              'line_P_data_products\\csv\\has_osd_ctd_flags\\'
parent_dir = 'C:\\Users\\HourstonH\\Documents\\charles\\' \
             'our_warming_ocean\\osp_sst\\csv\\'
nodc_file = os.path.join(parent_dir, '01b_apply_nodc_flags',
                         '{}_NODC_{}_data.csv'.format(stn, nodc_dtypes))
wp_dir = 'C:\\Users\\HourstonH\\Documents\\charles\\' \
         'line_P_data_products\\csv\\has_osd_ctd_flags\\'
wp_file = os.path.join(wp_dir, '01_convert',
                       '{}_CTD_BOT_CHE_data.csv'.format(stn))
"""
parent_dir = 'D:\\lineP\\csv_data\\'
input_files = glob.glob(parent_dir + '02_QC\\{}*.csv'.format(stn))
input_files.sort()

# output_folder = os.path.join(parent_dir, '02_merge')
output_folder = os.path.join(parent_dir, '03_merge')
output_file = os.path.join(output_folder,
                           '{}_data.csv'.format(stn))

# Initialize output dataframe
output_df = pd.DataFrame()

# Counter for preventing duplicate profile numbers
start_idx_adjustment = 0
for f in input_files:
    dfin = pd.read_csv(f)
    if 'NODC' in os.path.basename(f):
        dfin.drop(columns=['Temperature profile flag',
                           'Salinity profile flag',
                           'Oxygen profile flag'], inplace=True)
    # Adjust the profile numbers so that none are repeated
    dfin.loc[:, 'Profile number'] += start_idx_adjustment
    output_df = pd.concat((output_df, dfin))
    # Add 1 because profile numbering starts at zero
    start_idx_adjustment += dfin.loc[len(dfin) - 1, 'Profile number'] + 1

output_df.to_csv(output_file, index=False)

# nodc_df = pd.read_csv(nodc_file)
# nodc_df.drop(columns=['Temperature profile flag',
#                       'Salinity profile flag',
#                       'Oxygen profile flag'], inplace=True)
# wp_df = pd.read_csv(wp_file)
# wp_max_prof_ind = wp_df.loc[len(wp_df)-1, 'Profile number']
#
# nodc_df.loc[:, 'Profile number'] += wp_max_prof_ind + 1
#
# output_df = pd.concat((wp_df, nodc_df))
# output_df.to_csv(output_file, index=False)

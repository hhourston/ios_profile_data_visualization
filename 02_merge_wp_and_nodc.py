import pandas as pd
import os

# Merge the nodc and wp csv files before qc checks
# Take nodc files from folder where nodc flags were applied
# and take wp csv files from the "convert" folder before

stn = 'P26' #P26 P4
# data_types = 'CTD_BOT_CHE_OSD'
# nodc_file = 'C:\\Users\\HourstonH\\Documents\\charles\\' \
#             'line_P_data_products\\csv\\01b_apply_nodc_flags\\' \
#             '{}_NODC_OSD_data.csv'.format(stn)
nodc_file = 'C:\\Users\\HourstonH\\Documents\\charles\\' \
            'line_P_data_products\\csv\\01b_apply_nodc_flags\\' \
            '{}_NODC_OSD_GLD_PFL_data.csv'.format(stn)
wp_file = 'C:\\Users\\HourstonH\\Documents\\charles\\' \
          'line_P_data_products\\csv\\01_convert\\' \
          '{}_CTD_BOT_CHE_data.csv'.format(stn)
output_folder = 'C:\\Users\\HourstonH\\Documents\\charles\\' \
                'line_P_data_products\\csv\\02_merge\\'
output_file = os.path.join(output_folder,
                           '{}_data.csv'.format(stn))

nodc_df = pd.read_csv(nodc_file)
nodc_df.drop(columns=['Temperature profile flag',
                      'Salinity profile flag',
                      'Oxygen profile flag'], inplace=True)
wp_df = pd.read_csv(wp_file)
output_df = pd.concat((wp_df, nodc_df))
output_df.to_csv(output_file, index=False)

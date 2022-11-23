import os
import pandas as pd

parent_dir = 'C:\\Users\\HourstonH\\Documents\\charles\\' \
             'line_P_data_products\\csv\\has_osd_ctd_flags\\'

# for each station: P4 and P26, LB08
stn = 'P26'

average_file = os.path.join(
    parent_dir,
    '11N_annual_avg_on_dens_surfaces\\{}_data.csv'.format(stn))

dfin = pd.read_csv(average_file)

# dfin['Datetime'] = pd.to_datetime(dfin.loc[:, 'Time'])
# mask = dfin.loc[:, 'Datetime'].year >= 2010

mask = dfin.loc[:, 'Year'] >= 2010

dfout = dfin.loc[mask, :]

dfout_file_name = os.path.join(
    os.path.dirname(average_file),
    '{}_data_2010_onward.csv'.format(stn))

dfout.to_csv(dfout_file_name, index=False)

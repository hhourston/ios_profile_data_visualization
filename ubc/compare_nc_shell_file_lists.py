import os
import pandas as pd

# Compare wget nc and shell file download lists
# to see what nc files are missing

# P4 P26
station = 'P4'
charles_dir = 'C:\\Users\\HourstonH\\Documents\\charles\\' \
              'line_P_data_products\\'
shell_csv = os.path.join(
    charles_dir, 'wget_file_download_list_{}.csv'.format(station))
nc_csv = os.path.join(
    charles_dir, 'wget_netcdf_file_download_list_{}.csv'.format(station))

# Squeeze reads csv into pandas Series instead of DataFrame if only 1 column
shell_arr = pd.read_csv(shell_csv).squeeze("columns").to_numpy()
nc_arr = pd.read_csv(nc_csv).squeeze("columns").to_numpy()

shell_li = [os.path.basename(p) for p in shell_arr]
nc_li = [os.path.basename(p)[:-3] for p in nc_arr]

shell_extras = [name for name in shell_li if name not in nc_li]
print(shell_extras)

# P4 missing netCDF files
# ['2020-005-0082.che', '2013-001-0074.che']
import pandas as pd
import numpy as np

# Bin the ctd data into bins for every meter
# with pandas cut() function
# Required bins: [0, 0.5), [0.5, 1.5), [1.5, 2.5), ...
# Square bracket is inclusive, round bracket is not

station = '42'  # 'SI01'  # '59'  # '42'  # 'GEO1'  # 'LBP3'  # 'LB08'  # 'P1'
f = 'C:\\Users\\HourstonH\\Documents\\ctd_visualization\\csv\\' \
    '{}_ctd_data_qc.csv'.format(station)
df_in = pd.read_csv(f)

# Round the maximum depth value to the closest whole number
# Returns a float not an integer
max_depth_bin = np.round(np.max(df_in.loc[:, 'Depth [m]']), decimals=0)

depth_bins = np.concatenate((np.array([0]),
                             np.arange(0.5, max_depth_bin + 1, 1)))

bin_labels = [str(x) for x in depth_bins[1:] - 0.5]

df_in['Depth bin [m]'] = pd.cut(df_in['Depth [m]'], bins=depth_bins,
                                right=False, labels=bin_labels)

# Export the dataframe to csv
df_out_name = f.replace('qc', 'binned')
print(df_out_name)

df_in.to_csv(df_out_name, index=False)

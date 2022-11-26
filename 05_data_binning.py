import pandas as pd
import numpy as np

# Bin the ctd data into bins for every meter
# with pandas cut() function
# Required bins: [0, 0.5), [0.5, 1.5), [1.5, 2.5), ...
# Square bracket is inclusive, round bracket is not


def main(fin, fout):
    """
    Bin the input data to 1m size vertical bins in the water column
    :param fin: absolute path of input data file
    :param fout: absolute path of output data file
    :return: nothing
    """
    # df_out_name = f.replace('.csv', '_no_qc_binned.csv')

    df_in = pd.read_csv(fin)

    # Round the maximum depth value to the closest whole number
    # Returns a float not an integer
    max_depth_bin = np.round(np.max(df_in.loc[:, 'Depth [m]']), decimals=0)

    depth_bins = np.concatenate((np.array([0]),
                                 np.arange(0.5, max_depth_bin + 1, 1)))

    bin_labels = [str(x) for x in depth_bins[1:] - 0.5]

    # Do the binning
    df_in['Depth bin [m]'] = pd.cut(df_in['Depth [m]'], bins=depth_bins,
                                    right=False, labels=bin_labels)

    # Export the dataframe to csv
    print(fout)

    df_in.to_csv(fout, index=False)
    return


# ctd_station = '42'  # 'SI01'  # '59'  # '42'  # 'GEO1'  # 'LBP3'  # 'LB08'  # 'P1'

# ctd_stations = ['42', '59', '42', 'GEO1', 'LBP3', 'LB08', 'P1']
# ctd_stations = ['GEO1', 'LBP3', 'LB08']
stations = ['P4', 'P26']

# for s in ctd_stations:
#     file_name = 'C:\\Users\\HourstonH\\Documents\\' \
#                 'ctd_visualization\\csv\\' \
#                 '{}_ctd_data_qc.csv'.format(s)
#     # f = 'C:\\Users\\HourstonH\\Documents\\ctd_visualization\\
#     #     'csv\\{}_ctd_data.csv'.format(station)
#     df_out_name = file_name.replace('qc', 'binned')
#     main(file_name)
# parent_dir = 'C:\\Users\\HourstonH\\Documents\\charles\\' \
#              'our_warming_ocean\\osp_sst\\csv\\'
parent_dir = 'D:\\lineP\\csv_data\\'

for s in stations[:1]:
    file_name = parent_dir + '04_inexact_duplicate_check\\' \
                '{}_data.csv'.format(s)
    output_file = parent_dir + '05_data_binning\\{}_data.csv'.format(s)
    main(file_name, output_file)

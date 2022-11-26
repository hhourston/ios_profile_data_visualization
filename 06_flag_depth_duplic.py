import pandas as pd
import numpy as np

# Flag or remove bin depth duplicates in each profile
# if they occur


def main(fin, fout):
    """
    Flag bin depth duplicates in any casts
    :param fin: absolute path of input data file
    :param fout: absolute path of output data file
    :return: nothing
    """
    df_in = pd.read_csv(fin)

    # Profile start indices
    prof_start_ind = np.unique(df_in.loc[:, 'Profile number'],
                               return_index=True)[1]

    # Profile end indices
    nobs = len(df_in)
    prof_end_ind = np.concatenate((prof_start_ind[1:], [nobs]))

    # Initialize column for holding duplicate binned depth mask
    df_in['Unique binned depth mask'] = np.repeat(True, len(df_in))

    # Iterate through all of the profiles
    for i in range(len(prof_start_ind)):

        # Get profile data; np.arange not inclusive of end which we want here
        indices = np.arange(prof_start_ind[i], prof_end_ind[i])

        # Check for binned depth duplicates
        # Keep the row with the closer observed depth to the binned depth
        # Flag or remove the other

        depth_binned = df_in.loc[indices, 'Depth bin [m]']
        # depth_observed = df_in.loc[indices, 'Depth [m]']

        # Keep first occurrence here
        df_in.loc[
            indices[1:], 'Unique binned depth mask'] = np.diff(depth_binned) > 0

        # # Didn't work properly...
        # df_in.loc[
        #     indices, 'Unique binned depth mask'] = depth_binned.duplicated(
        #     keep='first')

    # Print summary statistics
    print('Number of observations in:', len(df_in))
    print('Number of observations out:',
          sum(df_in.loc[:, 'Unique binned depth mask']))

    # Apply mask
    # mask_duplicates = df_in.loc[:, 'Duplicate binned depth mask']
    # df_out = df_in.loc[mask_duplicates]
    df_out = df_in

    # Export dataframe to csv
    df_out.to_csv(fout, index=False)

    return


# ctd_station = '42'  # 'SI01'  # '59'  # '42'  # 'GEO1'  # 'LBP3'  # 'LB08'  # 'P1'

# ctd_stations = ['42', '59', '42', 'GEO1', 'LBP3', 'LB08', 'P1']
#
# for s in ctd_stations:
#     file_name = 'C:\\Users\\HourstonH\\Documents\\' \
#                 'ctd_visualization\\csv\\' \
#                 '{}_ctd_data_binned.csv'.format(s)
#     df_out_name = file_name.replace('binned', 'binned_depth_dupl')
#
#     main(file_name, df_out_name)

stations = ['P4', 'P26']
# parent_dir = 'C:\\Users\\HourstonH\\Documents\\charles\\' \
#              'our_warming_ocean\\osp_sst\\csv\\'
parent_dir = 'D:\\lineP\\csv_data\\'
for s in stations[:]:
    print(s)
    file_name = parent_dir + '05_data_binning\\{}_data.csv'.format(s)
    output_file = parent_dir + '06_flag_depth_duplicates\\{}_data.csv'.format(s)
    main(file_name, output_file)

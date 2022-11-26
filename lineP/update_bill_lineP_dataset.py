import pandas as pd
import os
import glob
import numpy as np


def do_update(bill_file_265: str, bill_file_267: str, bill_file_269: str,
              hana_file: str, nrows_to_drop=None):
    """
    Update Bill's station p4 and p26 csv datasets with data from IOS
    and NOAA that has been pre-gathered and formatted correctly
    :param bill_file_265: absolute file name
    :param bill_file_267: absolute file name
    :param bill_file_269: absolute file name
    :param hana_file: path to csv dataset from IOS and NOAA that has been
    gathered and formatted correctly
    :param nrows_to_drop: number of rows to drop from the end of each of
    [bill_file_265, bill_file_267, bill_file_269]
    :return: nothing
    """

    input_file_list = [bill_file_265, bill_file_267, bill_file_269]
    densities = [26.5, 26.7, 26.9]

    hana_df = pd.read_csv(hana_file)
    for f, d in zip(input_file_list, densities):
        print(os.path.basename(f))
        dfin = pd.read_csv(f)

        # if 'Ox (umol/kg) ' in dfin.columns:
        #     o2_colname = 'Ox (umol/kg) '
        # elif 'O2 (umol/kg) ' in dfin.columns:
        #     o2_colname = 'O2 (umol/kg) '
        # else:
        #     print('Oxygen data in umol/kg not found')

        o2_colname = dfin.columns[8]
        o2_avg_colname = dfin.columns[15]
        print(o2_colname, o2_avg_colname)

        dfin.dropna(axis=0, how='all',
                    subset=[o2_colname, 'Date',
                            'Sigma_Theta (from CT and AS)'],
                    inplace=True)
        dfin.reset_index(drop=True, inplace=True)

        if nrows_to_drop is not None:
            # drop the last few rows
            dfin.drop(dfin.index[-nrows_to_drop:], inplace=True)
            print('Dropped', nrows_to_drop, 'rows from the dataframe tail')
            dfin.loc[:, 'Date'] = pd.to_numeric(dfin['Date']).values

        dfin.loc[:, o2_colname] = pd.to_numeric(dfin[o2_colname]).values

        # Append to bottom of dataframe
        mask = ((hana_df['Year'] > dfin['Date'].max()) &
                (hana_df['Potential density anomaly bin [kg/m]'] == d))
        print(sum(mask), 'observations to add')

        new_df_dims = (sum(mask), len(dfin.columns))
        df_to_add = pd.DataFrame(
            np.repeat(np.nan, new_df_dims[0] * new_df_dims[1]).reshape(new_df_dims),
            columns=dfin.columns
        )
        df_to_add.loc[:, 'Longitude'] = hana_df.loc[mask, 'Longitude [deg E]'].values
        df_to_add.loc[:, 'Latitude'] = hana_df.loc[mask, 'Latitude [deg N]'].values
        df_to_add.loc[:, 'Depth'] = hana_df.loc[mask, 'Depth [m]'].values
        df_to_add.loc[:, 'Sigma_Theta (from CT and AS)'] = hana_df.loc[
            mask, 'Potential density anomaly [kg/m]'].values
        df_to_add.loc[:, o2_colname] = hana_df.loc[mask, 'Oxygen [umol/kg]'].values
        df_to_add.loc[:, 'Date'] = hana_df.loc[mask, 'Year'].values
        df_to_add.loc[:, 'File'] = hana_df.loc[mask, 'File'].values
        if 'Day of Year' in dfin.columns:
            df_to_add.loc[:, 'Day of Year'] = hana_df.loc[mask, 'Day of year'].values

        dfout = pd.concat((dfin, df_to_add))
        dfout.reset_index(drop=True, inplace=True)

        # Update or compute the annual averages
        min_added_year = int(np.nanmin(df_to_add.loc[:, 'Date']))
        max_added_year = int(np.nanmax(df_to_add.loc[:, 'Date']))

        print('Years to compute new average:', min_added_year, 'to',
              max_added_year)
        for y in range(min_added_year, max_added_year + 1):
            mask = dfout['Date'].astype(int) == y
            # Find index of first observation in the year
            dfout.loc[dfout.index[mask][0], o2_avg_colname
                      ] = dfout.loc[mask, o2_colname].mean()
            # print('Computed mean for', d)

        # Save the df
        dfout_filename = f.replace('.csv', '_Nov2022.csv')
        dfout.to_csv(dfout_filename, index=False)
        print('df saved to', os.path.basename(dfout_filename))
    return


parent_dir = 'C:\\Users\\HourstonH\\Documents\\charles\\' \
             'line_P_data_products\\bill_crawford\\'

d_drive = 'D:\\lineP\\csv_data\\10_bin_o2_to_select_densities\\'

bill_p4_files = glob.glob(parent_dir + 'CrawfordPena Line P*4849*.csv')
bill_p26_files = glob.glob(parent_dir + 'CrawfordPena Line P*26*.csv')
bill_p4_files.sort()
bill_p26_files.sort()

hana_p4_file = os.path.join(d_drive, 'P4_data.csv')
hana_p26_file = os.path.join(d_drive, 'P26_data.csv')

p26_nrows_to_drop = 2

do_update(*bill_p4_files, hana_p4_file)

do_update(*bill_p26_files, hana_p26_file, p26_nrows_to_drop)

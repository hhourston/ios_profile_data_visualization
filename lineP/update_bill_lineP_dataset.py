import pandas as pd
import os
import glob
import numpy as np

parent_dir = 'C:\\Users\\HourstonH\\Documents\\charles\\' \
             'line_P_data_products\\bill_crawford\\'

d_drive = 'D:\\lineP\\csv_data\\10_bin_o2_to_select_densities\\'

bill_p4_files = glob.glob(parent_dir + 'CrawfordPena Line P*4849*.csv')
bill_p26_files = glob.glob(parent_dir + 'CrawfordPena Line P*26*.csv')
bill_p4_files.sort()
bill_p26_files.sort()

hana_p4_file = os.path.join(d_drive, 'P4_data.csv')
hana_p26_file = os.path.join(d_drive, 'P26_data.csv')

hana_p4_df = pd.read_csv(hana_p4_file)
hana_p26_df = pd.read_csv(hana_p26_file)

densities = [26.5, 26.7, 26.9]

for f, d in zip(bill_p4_files, densities):
    print(os.path.basename(f))
    dfin = pd.read_csv(f)
    dfin.dropna(axis=0, how='all', inplace=True)
    dfin.reset_index(drop=True, inplace=True)
    # print(dfin.columns)

    # Append to bottom of dataframe
    mask = ((hana_p4_df['Year'] > dfin['Date'].max()) &
            (hana_p4_df['Potential density anomaly bin [kg/m]'] == d))
    print(sum(mask), 'observations to add')

    new_df_dims = (sum(mask), len(dfin.columns))
    df_to_add = pd.DataFrame(
        np.repeat(np.nan, new_df_dims[0] * new_df_dims[1]).reshape(new_df_dims),
        columns=dfin.columns
    )
    df_to_add.loc[:, 'Longitude'] = hana_p4_df.loc[mask, 'Longitude [deg E]'].values
    df_to_add.loc[:, 'Latitude'] = hana_p4_df.loc[mask, 'Latitude [deg N]'].values
    df_to_add.loc[:, 'Depth'] = hana_p4_df.loc[mask, 'Depth [m]'].values
    df_to_add.loc[:, 'Sigma_Theta (from CT and AS)'] = hana_p4_df.loc[
        mask, 'Potential density anomaly [kg/m]'].values
    df_to_add.loc[:, 'Ox (umol/kg) '] = hana_p4_df.loc[mask, 'Oxygen [umol/kg]'].values
    df_to_add.loc[:, 'Date'] = hana_p4_df.loc[mask, 'Year'].values
    df_to_add.loc[:, 'File'] = hana_p4_df.loc[mask, 'File'].values
    if 'Day of Year' in dfin.columns:
        df_to_add.loc[:, 'Day of Year'] = hana_p4_df.loc[mask, 'Day of year'].values

    dfout = pd.concat((dfin, df_to_add))

    # Save the df
    dfout_filename = f.replace('.csv', '_Nov2022.csv')
    dfout.to_csv(dfout_filename, index=False)

# repeat for p26
for f, d in zip(bill_p26_files, densities):
    print(os.path.basename(f))
    dfin = pd.read_csv(f)
    dfin.dropna(axis=0, how='all', inplace=True)
    dfin.reset_index(drop=True, inplace=True)
    # drop the last few rows
    nrows_to_drop = 2
    dfin.drop(dfin.index[-nrows_to_drop:], inplace=True)
    # print(dfin.columns)

    # Append to bottom of dataframe
    mask = ((hana_p26_df['Year'] > dfin['Date'].astype(float).max()) &
            (hana_p26_df['Potential density anomaly bin [kg/m]'] == d))
    print(sum(mask), 'observations to add')

    new_df_dims = (sum(mask), len(dfin.columns))
    df_to_add = pd.DataFrame(
        np.repeat(np.nan, new_df_dims[0] * new_df_dims[1]).reshape(new_df_dims),
        columns=dfin.columns
    )
    df_to_add.loc[:, 'Longitude'] = hana_p26_df.loc[mask, 'Longitude [deg E]'].values
    df_to_add.loc[:, 'Latitude'] = hana_p26_df.loc[mask, 'Latitude [deg N]'].values
    df_to_add.loc[:, 'Depth'] = hana_p26_df.loc[mask, 'Depth [m]'].values
    df_to_add.loc[:, 'Sigma_Theta (from CT and AS)'] = hana_p26_df.loc[
        mask, 'Potential density anomaly [kg/m]'].values
    df_to_add.loc[:, 'Ox (umol/kg) '] = hana_p26_df.loc[mask, 'Oxygen [umol/kg]'].values
    df_to_add.loc[:, 'Date'] = hana_p26_df.loc[mask, 'Year'].values
    df_to_add.loc[:, 'File'] = hana_p26_df.loc[mask, 'File'].values
    if 'Day of Year' in dfin.columns:
        df_to_add.loc[:, 'Day of Year'] = hana_p26_df.loc[mask, 'Day of year'].values

    dfout = pd.concat((dfin, df_to_add))

    # Save the df
    dfout_filename = f.replace('.csv', '_Nov2022.csv')
    dfout.to_csv(dfout_filename, index=False)

import pandas as pd
import numpy as np
import os
import xarray as xr

# Compare inexact matches of WP and NODC data from UBC 1951-1954

wp_dir = 'C:\\Users\\HourstonH\\Documents\\charles\\' \
         'ubc_historical_cruises\\ios_wp_ubc_1951_1954\\'
wp_file = os.path.join(wp_dir, 'wp_ubc_1951_1954_data.csv')

nodc_dir = 'C:\\Users\\HourstonH\\Documents\\charles\\' \
           'ubc_historical_cruises\\'
nodc_file = os.path.join(nodc_dir, 'ocldb1660928666.22434_OSD.nc')

# Comparison file
comp_file = os.path.join(nodc_dir, 'nodc_wp_comparison.csv')

wp_df = pd.read_csv(wp_file)
nodc_ds = xr.open_dataset(nodc_file)
comp_df = pd.read_csv(comp_file)

# Get nodc profile start and end indices
# Depth (z) increases with time (assume only downcasts)
nodc_start_idx = np.concatenate(
    (np.array([0]), np.where(np.diff(nodc_ds.z.data) < 0)[0] + 1))

nodc_end_idx = np.concatenate(
    (nodc_start_idx[1:], np.array([len(nodc_ds.z.data)])))

wp_start_idx = np.unique(
    wp_df.loc[:, 'Profile number'], return_index=True)[1]

# Pandas indexing is inclusive of end!
wp_end_idx = np.concatenate((wp_start_idx[1:]-1, np.array([len(wp_df)])))

print(wp_df.loc[wp_start_idx[0]:wp_end_idx[0], 'Depth'])

comp_df['Confirmed match'] = np.repeat('N', len(comp_df))

for i in range(len(comp_df)): #len(comp_df)
    nodc_idx = comp_df.loc[i, 'Indices from NODC']
    # Find WP index based on the source .UBC file matching
    wp_idx = np.where(
        wp_df.loc[wp_start_idx, 'Source file name'] ==
        os.path.basename(comp_df.loc[i, 'WP url']))[0]
    print('NODC index', nodc_idx)
    print('WP index', wp_idx)
    # wp_idx = comp_df.loc[i, 'Indices from WP']
    # print(comp_df.loc[i, 'WP url'])
    # print(wp_df.loc[wp_start_idx[wp_idx], 'Source file name'])
    # Locate the data in the dataset or dataframe
    nodc_z = nodc_ds.z.data[
             nodc_start_idx[nodc_idx]:nodc_end_idx[nodc_idx]]
    wp_z = wp_df.loc[
           wp_start_idx[wp_idx[0]]:wp_end_idx[wp_idx[0]], 'Depth'].to_numpy()
    # print(nodc_ds.time.data[nodc_idx], nodc_z,
    #       nodc_ds.Temperature.data[nodc_start_idx[nodc_idx]:nodc_end_idx[nodc_idx]],
    #       wp_df.loc[wp_start_idx[wp_idx[0]], 'Time'],
    #       wp_z,
    #       wp_df.loc[wp_start_idx[wp_idx[0]]:wp_end_idx[wp_idx[0]], 'Temperature [C]'].to_numpy(), sep='\n')
    # # If it's a match, then the TSO values would be the same
    # BUT not necessarily the depth and time values

    # Try TSO separately
    nodc_T = nodc_ds.Temperature.data[
             nodc_start_idx[nodc_idx]:nodc_end_idx[nodc_idx]].astype('float32')
    wp_T = wp_df.loc[
           wp_start_idx[wp_idx[0]]:wp_end_idx[wp_idx[0]], 'Temperature [C]'
           ].to_numpy().astype('float32')
    # nodc_T = nodc_ds.Salinity.data[
    #          nodc_start_idx[nodc_idx]:nodc_end_idx[nodc_idx]].astype('float32')
    # wp_T = wp_df.loc[
    #        wp_start_idx[wp_idx[0]]:wp_end_idx[wp_idx[0]], 'Salinity [PSU]'
    #        ].to_numpy().astype('float32')
    # TODO nodc oxygen in umol/kg, would have to convert
    # nodc_T = nodc_ds.Oxygen.data[
    #          nodc_start_idx[nodc_idx]:nodc_end_idx[nodc_idx]].astype('float32')
    # wp_T = wp_df.loc[
    #        wp_start_idx[wp_idx[0]]:wp_end_idx[wp_idx[0]], 'Oxygen [mL/L]'
    #        ].to_numpy().astype('float32')
    # TODO use np.allclose() with a tolerance? or np.equal()
    # What absolute and relative tolerances to use?
    if len(wp_T) == len(nodc_T):
        if np.allclose(wp_T, nodc_T, rtol=1e-03, atol=1e-03, equal_nan=True):
            comp_df.loc[i, 'Confirmed match'] = 'Y'
    else:
        # Check if all elements in smaller array are in the bigger array
        bigger = wp_T if len(wp_T) > len(nodc_T) else nodc_T
        smaller = wp_T if len(wp_T) < len(nodc_T) else nodc_T
        if all([a in bigger for a in smaller]):
            comp_df.loc[i, 'Confirmed match'] = 'Y'

print(len(comp_df))
print(comp_df.loc[:, 'Confirmed match'])
print(sum(comp_df.loc[:, 'Confirmed match'] == 'Y'))

msk = comp_df.loc[:, 'Confirmed match'] == 'Y'

comp_df.to_csv(comp_file.replace('.csv', '_check_profiles.csv'), index=False)

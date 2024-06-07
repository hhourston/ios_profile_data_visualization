import os
import glob
import gsw
import numpy as np
import xarray as xr
import pandas as pd
import convert


def get_var(ds: xr.Dataset, attr_names):
    # Search for all the available salinity and temperature variables
    # More than one code is used
    for attr in attr_names:
        if hasattr(ds, attr):
            return getattr(ds, attr)

    return None


def get_temperature_var(ds: xr.Dataset, cast_length: int):
    # Find temperature data in ds
    temperature_names = [
        "TEMPRTN1",
        "TEMPST01",
        "TEMPPR01",
        "TEMPPR03",
        "TEMPS901",
        "TEMPS601"
    ]
    # Convert between temperature standards as well?
    try:
        return get_var(ds, temperature_names).data
    except AttributeError:
        print('Warning: temperature data not found')
        # return np.repeat(-99, len(ds.depth.data))
        return np.repeat(np.nan, cast_length)


def get_salinity_var(ds: xr.Dataset, cast_length: int):
    # Find salinity data in ds
    salinity_names = [
        "PSLTZZ01",
        "ODSDM021",
        "SSALST01",
        "PSALST01",
        "PSALBST1",
        "sea_water_practical_salinity"
    ]
    # Convert units? PPT to PSS-78?
    sal_variable = get_var(ds, salinity_names)
    if sal_variable is not None:
        salinity, salinity_computed = convert.convert_salinity(
            sal_variable, sal_variable.units, 'ctd_logger.txt')
        return salinity.data
    else:
        # Oxygen data not present in netCDF file
        print('Warning: salinity data not found')
        # return np.repeat(-99, len(ds.depth.data))
        return np.repeat(np.nan, cast_length)


def get_pressure_var(ds: xr.Dataset, depth_data, latitude):
    # Find pressure data in ds
    pressure_names = ['PRESPR01', 'sea_water_pressure']

    try:
        return get_var(ds, pressure_names).data
    except AttributeError:
        # Compute pressure from depth using gsw toolbox
        return gsw.p_from_z(-depth_data, latitude)


def get_depth_var(ds: xr.Dataset):
    # Find depth data in ds
    depth_names = ['depth', 'depth_nominal']
    try:
        return get_var(ds, depth_names).data
    except AttributeError:
        print('Warning: depth data not found')


def get_oxygen_var(ds: xr.Dataset, temp_data, sal_data, depth_data,
                   filename, cast_length: int, required_unit='mL/L'):
    # Find oxygen data in ds
    # DOXYZZ01: has mL/L units; DOXMZZ01: has umol/kg units
    oxygen_names = ["DOXYZZ01", "DOXMZZ01"]

    if required_unit.lower() == 'umol/kg':
        # Reverse the order of the list
        oxygen_names = oxygen_names[::-1]
    oxy_variable = get_var(ds, oxygen_names)

    if oxy_variable is not None:
        # Find pressure data
        pres_data = get_pressure_var(ds, depth_data, ds.latitude.data)
        if required_unit.lower() == 'ml/l':
            oxygen, oxygen_computed, density_assumed = convert.convert_oxygen(
                oxy_variable, oxy_variable.units, ds.longitude.data,
                ds.latitude.data, temp_data, sal_data, pres_data,
                'ctd_logger.txt')
            return oxygen.data
        elif required_unit == 'umol/kg':
            # Added option to convert oxygen to umol/kg from mL/L
            if oxy_variable.units.lower() in ["umol/kg", "mmol/m", "mmol/m**3"]:
                return oxy_variable.data
            elif oxy_variable.units.lower() == 'ml/l':
                # Convert mL/L to umol/kg
                oxygen, density_assumed = convert.ml_l_to_umol_kg(
                    oxy_variable, ds.longitude.data, ds.latitude.data,
                    temp_data, sal_data, pres_data, filename)
                return oxygen.data
            else:
                # If in % for example, convert to ml/l then to umol/kg
                oxygen_ml, oxygen_computed, density_assumed = convert.convert_oxygen(
                    oxy_variable, oxy_variable.units, ds.longitude.data,
                    ds.latitude.data, temp_data, sal_data, pres_data,
                    'ctd_logger.txt')
                oxygen_umol, density_assumed = convert.ml_l_to_umol_kg(
                    oxygen_ml, ds.longitude.data, ds.latitude.data,
                    temp_data, sal_data, pres_data, filename)
                return oxygen_umol.data
    else:
        # Oxygen data not present in netCDF file
        print('Warning: oxygen data not found')
        return np.repeat(np.nan, cast_length)


def main_ios(nc_list: list, out_file_name: str, oxy_unit: str):
    """
    Format input netCDF data for later data preparation steps and plotting
    Export data in csv format
    oxy_unit: umol/kg or mL/L
    """

    # Initialize output dataframe
    df_out = pd.DataFrame()

    # Iterate through all netcdf file names in the input list
    for i, f in enumerate(nc_list):  # [139:140]
        print(os.path.basename(f))
        # Grab time, depth, TEMPS901, PSALST01
        ncdata = xr.open_dataset(f)

        # Get depth data
        depth_var = get_depth_var(ncdata)

        nobs_in_cast = len(depth_var)

        profile_number = np.repeat(i, nobs_in_cast)

        # Need to include lat/lon in order to check later
        lat_array = np.repeat(ncdata.latitude.data, nobs_in_cast)
        lon_array = np.repeat(ncdata.longitude.data, nobs_in_cast)

        # Need to convert time to string for csv files
        time_array = np.repeat(ncdata.time.data.astype('str'),
                               nobs_in_cast)

        # Convert temperature and salinity data as needed
        temp_var = get_temperature_var(ncdata, nobs_in_cast)
        sal_var = get_salinity_var(ncdata, nobs_in_cast)
        oxy_var = get_oxygen_var(ncdata, temp_var, sal_var, depth_var,
                                 os.path.basename(f), nobs_in_cast, oxy_unit)
        file_array = np.repeat(os.path.basename(f), nobs_in_cast)
        cast_type_array = np.repeat(os.path.basename(f)[-6:-3].upper(),
                                    nobs_in_cast)

        # Create dataframe to append to the output dataframe
        df_add = pd.DataFrame(
            np.array([profile_number, lat_array, lon_array, time_array,
                      depth_var, temp_var, sal_var, oxy_var, file_array,
                      cast_type_array],
                     dtype='object'
                     ).transpose(),
            columns=['Profile number', 'Latitude [deg N]', 'Longitude [deg E]',
                     'Time', 'Depth [m]', 'Temperature [C]',
                     'Salinity [PSS-78]', 'Oxygen [{}]'.format(oxy_unit), 'File',
                     'Cast type'])
        # PSS-78 and PSU taken as equivalent

        df_out = pd.concat([df_out, df_add])
        df_out.reset_index(drop=True, inplace=True)

    # print(len(df_out))
    # print(sum(df_out.loc[:, 'Oxygen [{}]'.format(oxy_unit)] != np.nan))

    # ADD CTD IF ONLY CTD
    df_out.to_csv(out_file_name, index=False)
    return


def main_nodc(nc_list, out_file_name):
    # Format input netCDF data for later data preparation steps and plotting
    dtype_list = []

    # Initialize dataframe to hold data
    df_nodc = pd.DataFrame(
        columns=['Profile number', 'Latitude [deg N]', 'Longitude [deg E]',
                 'Time', 'Depth [m]', 'Temperature [C]',
                 'Temperature profile flag', 'Salinity [PSS-78]',
                 'Salinity profile flag', 'Oxygen [umol/kg]',
                 'Oxygen profile flag', 'File', 'Cast type'])

    profile_number = 0
    # Iterate through each nc file
    for f in nc_list:
        print(os.path.basename(f))
        file_dtype = f[-6:-3]
        print(file_dtype)
        dtype_list.append(file_dtype)

        nodc_ds = xr.open_dataset(f)

        # Get profile start and end indices
        nodc_start_idx = np.concatenate(
            (np.array([0]), np.where(np.diff(nodc_ds.z.data) < 0)[0] + 1))

        nodc_end_idx = np.concatenate(
            (nodc_start_idx[1:], np.array([len(nodc_ds.z.data)])))

        # Iterate through each cast
        for i in range(len(nodc_start_idx)):
            st = nodc_start_idx[i]
            en = nodc_end_idx[i]
            prof_len = en - st
            dct_add = {
                'Profile number': np.repeat(profile_number, prof_len),
                'Latitude [deg N]': np.repeat(nodc_ds.lat.data[i], prof_len),
                'Longitude [deg E]': np.repeat(nodc_ds.lon.data[i], prof_len),
                'Time': np.repeat(nodc_ds.time.data[i], prof_len),
                'Depth [m]': nodc_ds.z.data[st:en],
                'Temperature [C]': nodc_ds.Temperature.data[st:en],
                'Temperature profile flag': np.repeat(
                    nodc_ds.Temperature_WODprofileflag.data[i], prof_len),
                'Salinity [PSS-78]': nodc_ds.Salinity.data[st:en],
                'Salinity profile flag': np.repeat(
                    nodc_ds.Salinity_WODprofileflag.data[i], prof_len),
                'Oxygen [umol/kg]': nodc_ds.Oxygen.data[st:en],
                'Oxygen profile flag': np.repeat(
                    nodc_ds.Oxygen_WODprofileflag.data[i], prof_len),
                'File': np.repeat(os.path.basename(f), prof_len),
                'Cast type': np.repeat(file_dtype, prof_len)
            }
            df_nodc = pd.concat((df_nodc, pd.DataFrame(dct_add)))

            profile_number += 1

    # Save the df
    df_nodc.to_csv(out_file_name, index=False)

    return


"""
# ----------------------------------------------------------------------------
# Ask James Hannah about searching files quickly by station on osd data archive

# ctd_file = 'C:\\Users\\HourstonH\\Documents\\ctd_visualization\\2002-001-0002.ctd.nc'
# ncdata = xr.open_dataset(ctd_file)

# 'SI01'  # '59'  # '42'  # 'GEO1'  # 'LBP3'  # 'LB08'  # 'P1'
# P4 P26
station = '42'
data_types = 'ctd'
# data_types = 'CTD_BOT_CHE'

oxygen_unit = 'mL/L'  # umol/kg 'mL/L' IMPORTANT!

# input_dir = 'C:\\Users\\HourstonH\\Documents\\charles\\' \
#             'line_P_data_products\\' + station + '\\water_properties'
# output_dir = 'C:\\Users\\HourstonH\\Documents\\charles\\' \
#              'line_P_data_products\\csv\\01_convert'

# Copied from James Hannah ios-inlets
# https://github.com/cyborgsphinx/ios-inlets/blob/main/inlets.py#L132

# Depth, range, gradient checks as in NEP climatology?
# Need to put all nc data in a csv table to make this easier
# as in the climatology project?

for s in ['59', '42', 'GEO1', 'LBP3', 'LB08', 'P1']:
    input_dir = 'C:\\Users\\HourstonH\\Documents\\ctd_visualization\\' \
                '{}\\'.format(s)
    output_dir = 'C:\\Users\\HourstonH\\Documents\\ctd_visualization\\' \
                 'csv\\'
    output_fname = os.path.join(
        output_dir, '{}_{}_data.csv'.format(s, data_types))

    data_flist = glob.glob(input_dir + '\\*.nc')
    data_flist.sort()
    print(len(data_flist))

    main_ios(data_flist, output_fname, oxygen_unit)
"""
# ----------------------------------------------------------------------

# CS09 data
stn = 'CS09'
raw_data_dir = 'C:\\Users\\hourstonh\\Documents\\charles\\more_oxygen_projects\\'
output_dir = raw_data_dir + 'CS09_01_convert\\'
raw_wp_files = glob.glob(raw_data_dir + 'CS09\\' + '*.nc')
data_types = 'CTD_BOT_CHE'
main_ios(nc_list=raw_wp_files, out_file_name=output_dir + f'{stn}_{data_types}_data.csv',
         oxy_unit='mL/L')

"""
# NODC data
# 'P26' P4
stn = 'P26'
# data_type = 'OSD'

# raw_data_dir = 'D:\\lineP\\{}_raw_data\\'.format(stn)
parent_dir = 'C:\\Users\\hourstonh\\Documents\\charles\\line_P_data_products\\update_jan2024\\'
raw_data_dir = parent_dir + f'{stn}_raw_data\\'

# raw_nodc_files = glob.glob(raw_data_dir + 'wodselect\\*.nc')
raw_wp_files = glob.glob(raw_data_dir + '*.nc')
raw_wp_files.sort()

output_dir = parent_dir + 'csv_data\\01_convert\\'
data_types = 'CTD_CHE'

main_ios(nc_list=raw_wp_files, out_file_name=output_dir + f'{stn}_{data_types}_data.csv',
         oxy_unit='umol/kg')

"""

"""
# nodc_file = 'C:\\Users\\HourstonH\\Documents\\charles\\' \
#             'line_P_data_products\\{}\\wodselect\\' \
#             'ocldb1661989089.32104_{}.nc'.format(stn, data_type)

# nodc_file = 'C:\\Users\\HourstonH\\Documents\\charles\\' \
#             'line_P_data_products\\{}\\wodselect\\' \
#             'ocldb1662746296.725_{}.nc'.format(stn, data_type)

# p26_parent_dir = 'C:\\Users\\HourstonH\\Documents\\charles\\' \
#                  'line_P_data_products\\{}\\wodselect\\'.format(stn)
# p26_files = [
#     os.path.join(p26_parent_dir, 'ocldb1661989089.32104_OSD.nc'),
#     os.path.join(p26_parent_dir, 'already_in_ios_archive',
#                  'ocldb1663010426.31871_CTD.nc')]

p26_parent_dir = 'C:\\Users\\HourstonH\\Documents\\charles\\' \
                 'our_warming_ocean\\osp_sst\\raw\\'
p26_files = glob.glob(p26_parent_dir + '*.nc')
p26_files.sort()
output_dir = 'C:\\Users\\HourstonH\\Documents\\charles\\our_warming_ocean\\osp_sst\\csv\\'

# p4_parent_dir = 'C:\\Users\\HourstonH\\Documents\\charles\\' \
#                 'line_P_data_products\\{}\\wodselect\\'.format(stn)
# p4_files = [
#     os.path.join(p4_parent_dir, 'ocldb1662746296.725_OSD.nc'),
#     os.path.join(p4_parent_dir, 'already_in_ios_archive',
#                  'ocldb1662746296.725_CTD.nc')]

# nodc_dir = 'C:\\Users\\HourstonH\\Documents\\charles\\' \
#            'line_P_data_products\\{}\\wodselect\\'.format(stn)
# nodc_flist = glob.glob(nodc_dir + '*.nc', recursive=False)
# nodc_flist.sort()
# output_dir = 'C:\\Users\\HourstonH\\Documents\\charles\\' \
#              'line_P_data_products\\csv\\has_osd_ctd_flags\\' \
#              '01_convert\\'
"""

# # main_nodc(raw_nodc_files, output_dir + '{}_NODC_OSD_CTD_data.csv'.format(stn))
# idx_of_failure = raw_wp_files.index(
#     'D:\\lineP\\P4_raw_data\\water_properties\\2002-038-0025.bot.nc')
#
# idx2_of_failure = raw_wp_files.index(
#     'D:\\lineP\\P4_raw_data\\water_properties\\2016-040-0025.bot.nc')
#
# main_ios(raw_wp_files[:idx_of_failure],
#          output_dir + '{}_WP_CTD_BOT_CHE_data_1933_2002.csv'.format(stn),
#          'umol/kg')
#
# # main_ios(raw_wp_files[idx_of_failure:idx2_of_failure],
# #          output_dir + '{}_WP_CTD_BOT_CHE_data_2002_2016.csv'.format(stn),
# #          'umol/kg')
#
# # main_ios(raw_wp_files[idx2_of_failure:],
# #          output_dir + '{}_WP_CTD_BOT_CHE_data_2016_2022.csv'.format(stn),
# #          'umol/kg')

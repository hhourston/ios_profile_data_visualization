import os
import glob
import numpy as np
import xarray as xr
import pandas as pd
import convert


def get_var(ds, attr_names):
    # Search for all the available salinity and temperature variables
    # More than one code is used
    for attr in attr_names:
        if hasattr(ds, attr):
            return getattr(ds, attr)

    return None


def get_temperature_var(ds):
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
        return np.repeat(np.nan, len(ds.depth.data))


def get_salinity_var(ds):
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
        return np.repeat(np.nan, len(ds.depth.data))


def get_pressure_var(ds):
    pressure_names = ['PRESPR01', 'sea_water_pressure']
    return get_var(ds, pressure_names)


def ml_l_to_umol_kg(oxy_ml_l, longitude, latitude, temperature_C,
                    salinity_SP, pressure_dbar, filename):
    # Convert oxygen mL/L units to umol/kg units
    oxygen_umol_per_ml = 44.661
    metre_cube_per_litre = 0.001
    density, assumed_density = convert.calculate_density(
        len(oxy_ml_l),
        temperature_C,
        salinity_SP,
        pressure_dbar,
        longitude,
        latitude,
        filename,
    )
    return (
        np.fromiter(
            (
                o * oxygen_umol_per_ml / (rho * metre_cube_per_litre)
                for o, rho in zip(oxy_ml_l, density)
            ),
            dtype=float,
            count=len(oxy_ml_l),
        ),
        assumed_density
    )


def get_oxygen_var(ds, temp_data, sal_data, filename, required_unit='mL/L'):
    # ds: xarray dataset
    # DOXYZZ01: has mL/L units; DOXMZZ01: has umol/kg units
    oxygen_names = ["DOXYZZ01", "DOXMZZ01"]
    if required_unit == 'umol/kg':
        # Reverse the order of the list
        oxygen_names = oxygen_names[::-1]
    oxy_variable = get_var(ds, oxygen_names)

    if oxy_variable is not None:
        # Find pressure data
        pres_data = get_pressure_var(ds)
        if required_unit == 'mL/L':
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
                oxygen, density_assumed = ml_l_to_umol_kg(
                    oxy_variable, ds.longitude.data, ds.latitude.data,
                    temp_data, sal_data, pres_data, filename)
                return oxygen.data
            else:
                # If in % for example, convert to ml/l then to umol/kg
                oxygen_ml, oxygen_computed, density_assumed = convert.convert_oxygen(
                    oxy_variable, oxy_variable.units, ds.longitude.data,
                    ds.latitude.data, temp_data, sal_data, pres_data,
                    'ctd_logger.txt')
                oxygen_umol, density_assumed = ml_l_to_umol_kg(
                    oxygen_ml, ds.longitude.data, ds.latitude.data,
                    temp_data, sal_data, pres_data, filename)
                return oxygen_umol.data
    else:
        # Oxygen data not present in netCDF file
        print('Warning: oxygen data not found')
        return np.repeat(np.nan, len(temp_data))


def get_fluorescence_var(ds):
    # Fluorescence not in netCDF files only shell files
    fluorescence_names = []
    return


# ----------------------------------------------------------------------------
# Ask James Hannah about searching files quickly by station on osd data archive

# ctd_file = 'C:\\Users\\HourstonH\\Documents\\ctd_visualization\\2002-001-0002.ctd.nc'
# ncdata = xr.open_dataset(ctd_file)

# 'SI01'  # '59'  # '42'  # 'GEO1'  # 'LBP3'  # 'LB08'  # 'P1'
# P4 P26
station = 'P4'
# data_types = 'ctd'
data_types = 'CTD_BOT_CHE'

# ctd_dir = 'C:\\Users\\HourstonH\\Documents\\ctd_visualization\\' \
#           '{}\\'.format(station)
# output_dir = 'C:\\Users\\HourstonH\\Documents\\ctd_visualization\\' \
#              'csv\\'
input_dir = 'C:\\Users\\HourstonH\\Documents\\charles\\' \
            'line_P_data_products\\' + station + '\\water_properties'
output_dir = 'C:\\Users\\HourstonH\\Documents\\charles\\' \
             'line_P_data_products\\csv\\01_convert'
data_flist = glob.glob(input_dir + '\\*.nc')
data_flist.sort()
print(len(data_flist))

# Copied from James Hannah ios-inlets
# https://github.com/cyborgsphinx/ios-inlets/blob/main/inlets.py#L132

# Depth, range, gradient checks as in NEP climatology?
# Need to put all nc data in a csv table to make this easier
# as in the climatology project?
df_out = pd.DataFrame()

oxy_unit = 'umol/kg'  # 'mL/L'

for i, f in enumerate(data_flist):  # [139:140]
    print(os.path.basename(f))
    # Grab time, depth, TEMPS901, PSALST01
    ncdata = xr.open_dataset(f)

    nobs_in_cast = len(ncdata.depth.data)

    profile_number = np.repeat(i, nobs_in_cast)

    # Need to include lat/lon in order to check later
    lat_array = np.repeat(ncdata.latitude.data, nobs_in_cast)
    lon_array = np.repeat(ncdata.longitude.data, nobs_in_cast)

    # Need to convert time to string for csv files
    time_array = np.repeat(ncdata.time.data.astype('str'),
                           nobs_in_cast)

    # Convert temperature and salinity data as needed
    temp_var = get_temperature_var(ncdata)
    sal_var = get_salinity_var(ncdata)
    oxy_var = get_oxygen_var(ncdata, temp_var, sal_var,
                             os.path.basename(f), oxy_unit)

    df_add = pd.DataFrame(
        np.array([profile_number, lat_array, lon_array, time_array,
                  ncdata.depth.data, temp_var, sal_var, oxy_var],
                 dtype='object'
                 ).transpose(),
        columns=['Profile number', 'Latitude [deg N]', 'Longitude [deg E]',
                 'Time', 'Depth [m]', 'Temperature [C]',
                 'Salinity [PSS-78]', 'Oxygen [{}]'.format(oxy_unit)])
    # PSS-78 and PSU taken as equivalent

    df_out = pd.concat([df_out, df_add])
    df_out.reset_index(drop=True, inplace=True)

# print(len(df_out))
# print(sum(df_out.loc[:, 'Oxygen [{}]'.format(oxy_unit)] != np.nan))

# ADD CTD IF ONLY CTD
df_name = os.path.join(output_dir, '{}_{}_data.csv'.format(
    station, data_types))
df_out.to_csv(df_name, index=False)

# ----------------------------------------------------------------------

# # NODC data
#
# nodc_file = 'C:\\Users\\HourstonH\\Documents\\charles\\' \
#             'line_P_data_products\\P26\\wodselect\\' \
#             'ocldb1661989089.32104_OSD.nc'
#
# nodc_ds = xr.open_dataset(nodc_file)
#
# nodc_start_idx = np.concatenate(
#     (np.array([0]), np.where(np.diff(nodc_ds.z.data) < 0)[0] + 1))
#
# nodc_end_idx = np.concatenate(
#     (nodc_start_idx[1:], np.array([len(nodc_ds.z.data)])))
#
# df_nodc = pd.DataFrame(
#     columns=['Profile number', 'Latitude [deg N]', 'Longitude [deg E]',
#              'Time', 'Depth [m]', 'Temperature [C]',
#              'Salinity [PSS-78]', 'Oxygen [umol/kg]'])
#
# for i in range(len(nodc_start_idx)):
#     st = nodc_start_idx[i]
#     en = nodc_end_idx[i]
#     prof_len = en - st
#     dct_add = {
#         'Profile number': np.repeat(i, prof_len),
#         'Latitude [deg N]': np.repeat(nodc_ds.lat.data[i], prof_len),
#         'Longitude [deg E]': np.repeat(nodc_ds.lon.data[i], prof_len),
#         'Time': np.repeat(nodc_ds.time.data[i], prof_len),
#         'Depth [m]': nodc_ds.z.data[st:en],
#         'Temperature [C]': nodc_ds.Temperature.data[st:en],
#         'Salinity [PSS-78]': nodc_ds.Salinity.data[st:en],
#         'Oxygen [umol/kg]': nodc_ds.Oxygen.data[st:en]
#     }
#     df_nodc = pd.concat((df_nodc, pd.DataFrame(dct_add)))
#
# # Save the df
# nodc_output = os.path.join(os.path.dirname(
#     nodc_file), 'P26_NODC_OSD_data.csv')
#
# df_nodc.to_csv(nodc_output, index=False)

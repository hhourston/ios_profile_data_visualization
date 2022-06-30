import matplotlib.pyplot as plt
import os
import glob
import numpy as np
import xarray as xr
import pandas as pd
import convert

# Ask James Hannah about searching files quickly by station on osd data archive

# ctd_file = 'C:\\Users\\HourstonH\\Documents\\ctd_visualization\\2002-001-0002.ctd.nc'
# ncdata = xr.open_dataset(ctd_file)

ctd_dir = 'C:\\Users\\HourstonH\\Documents\\ctd_visualization\\p1_ctd_files\\'
ctd_flist = glob.glob(ctd_dir + '*.nc')
ctd_flist.sort()

# Copied from James Hannah ios-inlets
# https://github.com/cyborgsphinx/ios-inlets/blob/main/inlets.py#L132


def get_var(ds, attr_names):
    # Search for all the available salinity and temperature variables
    # More than one code is used
    for attr in attr_names:
        if hasattr(ds, attr):
            return getattr(ds, attr)


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
    return get_var(ds, temperature_names)


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

    salinity, salinity_computed = convert.convert_salinity(
        sal_variable, sal_variable.units, 'ctd_logger.txt')

    return salinity


# Depth, range, gradient checks as in NEP climatology?
# Need to put all nc data in a csv table to make this easier
# as in the climatology project?
df_ctd = pd.DataFrame()

for i, f in enumerate(ctd_flist):
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

    df_add = pd.DataFrame(
        np.array([profile_number, lat_array, lon_array, time_array,
                  ncdata.depth.data, temp_var.data, sal_var.data]).transpose(),
        columns=['Profile number', 'Latitude [deg N]', 'Longitude [deg E]',
                 'Time', 'Depth [m]', 'Temperature [C]', 'Salinity [PSS-78]'])

    df_ctd = pd.concat([df_ctd, df_add])

df_name = 'C:\\Users\\HourstonH\\Documents\\ctd_visualization\\P1_ctd_data.csv'
df_ctd.to_csv(df_name, index=False)


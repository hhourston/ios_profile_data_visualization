import glob
import os
import ios_shell.shell as ios
import datetime
import gsw
import numpy as np
import pandas as pd
import convert
# import ios_shell.sections as sections


# Use James Hannah's code
# https://github.com/IOS-OSD-DPG/ios-shell/blob/main/tests/shell_test.py
# inlet.add_data_from_shell()
# https://github.com/cyborgsphinx/ios-inlets/blob/5a6e7395edd5327e2f8c578b92046124b2d309d4/inlets.py#L613

EXCEPTIONALLY_BIG = 9.9e36


def find_column(source, name: str, *units: str) -> int:
    # Author: James Hannah
    name_lower = name.lower()
    potentials = [line for line in source if name_lower in line.name.lower()]
    units_lower = [unit.lower() for unit in units]
    with_units = [line for line in potentials if line.units.lower() in units_lower]
    if len(with_units) > 0:
        # pick first line with matching units
        return with_units[0].no - 1
    elif len(potentials) > 0:
        # pick first line with matching name
        return potentials[0].no - 1
    else:
        return -1


def get_length(arr):
    # Author: James Hannah
    if arr is None:
        return 0
    return arr.size if hasattr(arr, "size") else len(arr)


def reinsert_nan(data, placeholder, length=None):
    # Author: James Hannah
    if length is None:
        length = get_length(data)
    return np.fromiter(
        (np.nan if x == placeholder or x > EXCEPTIONALLY_BIG else x for x in data),
        float,
        count=length,
    )


def to_float(source):
    # Author: James Hannah
    if isinstance(source, float) or isinstance(source, int):
        return source
    elif isinstance(source, bytes):
        return (
            np.nan
            if source.strip() in [b"' '", b"n/a", b""]
            else float(source.strip().decode("utf-8"))
        )
    elif isinstance(source, str):
        return (
            np.nan if source.strip() in ["' '", "n/a", ""] else float(source.strip())
        )
    else:
        raise ValueError(f"to_float called on {source}")


def get_pad_value(info, index):
    if index < 0 or info is None or len(info) == 0:
        return None
    return to_float(info[index].pad)


def extract_data(source, index, replace):
    # Author: James Hannah
    if index < 0:
        return None
    return reinsert_nan(
        (to_float(row[index]) for row in source),
        replace,
        length=len(source),
    )


def has_quality(value_index, names):
    quality_index = value_index + 1
    return quality_index < len(names) and (
        names[quality_index].startswith("Quality")
        or names[quality_index].startswith("Flag")
    )


def add_data_from_shell(data, profile_number, source_file_name):
    # Author: James Hannah
    channels = data.file.channels
    channel_details = data.file.channel_details
    names = [channel.name for channel in channels]
    units = [channel.units for channel in channels]

    longitude, latitude = data.location.longitude, data.location.latitude

    date_idx = find_column(channels, "Date")
    if date_idx < 0:
        # time not included in data, just use start date
        time = np.full(len(data.data), data.get_time())
    else:
        time_idx = find_column(channels, "Time")
        if time_idx < 0:
            # only date included in data
            time = [d[date_idx] for d in data.data]
        else:
            dates = [d[date_idx] for d in data.data]
            times = [d[time_idx] for d in data.data]
            time = [
                datetime.datetime.combine(d, t, tzinfo=data.get_time().tzinfo)
                for d, t in zip(dates, times)
            ]

    depth_idx = find_column(channels, "Depth", "m", "metre")
    depth_pad = get_pad_value(channel_details, depth_idx)
    if depth_pad is None or np.isnan(depth_pad):
        depth_pad = -99
    depth_data = extract_data(data.data, depth_idx, depth_pad)

    temperature_idx = find_column(channels, "Temperature", "C", "'deg C'")
    temperature_pad = get_pad_value(channel_details, temperature_idx)
    if temperature_pad is None or np.isnan(temperature_pad):
        temperature_pad = -99
    temperature_data = extract_data(data.data, temperature_idx, temperature_pad)
    temperature_quality = [0] * get_length(temperature_data)
    if has_quality(temperature_idx, names):
        temperature_quality = extract_data(data.data, temperature_idx + 1, None)

    salinity_idx = find_column(channels, "Salinity", "PSU", "PSS-78")
    salinity_pad = get_pad_value(channel_details, salinity_idx)
    if salinity_pad is None or np.isnan(salinity_pad):
        salinity_pad = -99
    salinity_data = extract_data(data.data, salinity_idx, salinity_pad)
    salinity_quality = [0] * get_length(salinity_data)
    if has_quality(salinity_idx, names):
        salinity_quality = extract_data(data.data, salinity_idx + 1, None)

    oxygen_idx = find_column(channels, "Oxygen", "mL/L")
    oxygen_pad = get_pad_value(channel_details, oxygen_idx)
    if oxygen_pad is None or np.isnan(oxygen_pad):
        oxygen_pad = -99
    oxygen_data = extract_data(data.data, oxygen_idx, oxygen_pad)
    oxygen_quality = [0] * get_length(oxygen_data)
    if has_quality(oxygen_idx, names):
        oxygen_quality = extract_data(data.data, oxygen_idx + 1, None)

    pressure_idx = find_column(channels, "Pressure", "dbar", "decibar")
    pressure_pad = get_pad_value(channel_details, pressure_idx)
    if pressure_pad is None or np.isnan(pressure_pad):
        pressure_pad = -99
    pressure_data = extract_data(data.data, pressure_idx, pressure_pad)

    if (
            depth_data is None
            and data.instrument is not None
            and not np.isnan(data.instrument.depth)
    ):
        depth_data = np.full(1, float(data.instrument.raw["depth"]))
    elif depth_data is None:
        if pressure_data is not None:
            depth_data = gsw.z_from_p(pressure_data, latitude) * -1
        else:
            # logging.warning(
            #     f"{data.filename} does not have depth or pressure data. Skipping"
            # )
            print(f"{data.filename} does not have depth or pressure data. Skipping"
            )
            return
    elif pressure_data is None:
        # depth_data is not None in this case
        pressure_data = gsw.p_from_z(depth_data * -1, latitude)

    salinity_data, salinity_computed = convert.convert_salinity(
        salinity_data, units[salinity_idx].strip(), data.filename
    )

    oxygen_data, oxygen_computed, oxygen_assumed_density = convert.convert_oxygen(
        oxygen_data,
        units[oxygen_idx].strip(),
        longitude,
        latitude,
        temperature_data,
        salinity_data,
        pressure_data
        if pressure_data is not None
        else gsw.p_from_z(depth_data * -1, latitude),
        data.filename,
    )

    # Initialize dataframe to add to where the variable column order is set
    df_profile = pd.DataFrame(
        columns=['Profile number', 'Lon', 'Lat', 'Time', 'Depth',
                 'Temperature [C]', 'Salinity [PSU]', 'Oxygen [mL/L]',
                 'Source file name'])

    if temperature_data is not None:
        # self.data.add_temperature_data(
        #     self.produce_data(
        #         time,
        #         depth_data,
        #         temperature_data,
        #         temperature_quality,
        #         longitude,
        #         latitude,
        #         data.filename,
        #         placeholder=temperature_pad,
        #     )
        # )
        # print('T', temperature_data)
        df_profile.loc[:, 'Temperature [C]'] = temperature_data

    if salinity_data is not None:
        # self.data.add_salinity_data(
        #     self.produce_data(
        #         time,
        #         depth_data,
        #         salinity_data,
        #         salinity_quality,
        #         longitude,
        #         latitude,
        #         data.filename,
        #         placeholder=salinity_pad,
        #         computed=salinity_computed,
        #     )
        # )
        # print('S', salinity_data)
        df_profile.loc[:, 'Salinity [PSU]'] = salinity_data

    if oxygen_data is not None:
        # self.data.add_oxygen_data(
        #     self.produce_data(
        #         time,
        #         depth_data,
        #         oxygen_data,
        #         oxygen_quality,
        #         longitude,
        #         latitude,
        #         data.filename,
        #         placeholder=oxygen_pad,
        #         computed=oxygen_computed,
        #         assumed_density=oxygen_assumed_density,
        #     )
        # )
        # print('O', oxygen_data)
        df_profile.loc[:, 'Oxygen [mL/L]'] = oxygen_data

    if temperature_data is not None or salinity_data is not None or oxygen_data is not None:
        # Add lat, lon, time, depth data
        # print(longitude, latitude, time, depth_data, sep='\n')
        df_profile.loc[:, 'Profile number'] = np.repeat(profile_number,
                                                        len(df_profile))
        df_profile.loc[:, 'Lon'] = np.repeat(longitude, len(df_profile))
        df_profile.loc[:, 'Lat'] = np.repeat(latitude, len(df_profile))
        df_profile.loc[:, 'Time'] = time  # np.repeat(time, len(df_profile))
        df_profile.loc[:, 'Depth'] = depth_data
        df_profile.loc[:, 'Source file name'] = np.repeat(
            source_file_name, len(df_profile))
        return df_profile
    else:
        return None


fdir = 'C:\\Users\\HourstonH\\Documents\\charles\\' \
       'ubc_historical_cruises\\ios_wp_ubc_1951_1954\\'
flist = glob.glob(fdir + '*.UBC')
flist.sort()

out_path = fdir

# f = flist[0]
#
# print(os.path.basename(f))
#
# shell = ios.ShellFile.fromfile(f, process_data=True)
#
# dfout = add_data_from_shell(shell, 0, os.path.basename(f))
#
# shell_data = np.array(shell.data)
# depth_data = shell_data[:, 0]
# T_data = shell_data[:, 1]
# S_data = shell_data[:, 3]  # Pre-1978', PPT
# O_data = shell_data[:, 5]

df_total = pd.DataFrame()
for i in range(len(flist)):
    shell = ios.ShellFile.fromfile(flist[i], process_data=True)
    df_total = pd.concat(
        (df_total, add_data_from_shell(shell, i, os.path.basename(flist[i]))))

# Export the df to csv
df_file_name = os.path.join(fdir, 'wp_ubc_1951_1954_data.csv')

df_total.to_csv(df_file_name, index=False)

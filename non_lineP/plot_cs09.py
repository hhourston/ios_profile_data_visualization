import matplotlib.pyplot as plt
import xarray as xr
import os
import pandas as pd
import glob
from helpers import get_profile_st_en_idx
import numpy as np

WDIR = 'C:\\Users\\hourstonh\\Documents\\charles\\more_oxygen_projects\\'


def index_rows(depths, x, max_distance):
    """
    Index observations (rows) in a dataframe that are within `max_distance` of the depth x
    :param depths:
    :param x:
    :param max_distance:
    :return: mask
    """
    return (depths >= x - max_distance) & (depths <= x + max_distance)


def plot_cs09_scott2():
    # plot CTD oxygen time series at 150m, 175m, and 200m

    # Max +/- distance away from each depth to include data from
    # So we look for data between 140m and 160m for the 150m bin
    max_distance = 10

    # Confirm oxygen units - mL/L

    skip03 = ''  # '_skip03'  # ''

    sampling_station = 'CS09'
    input_dir = (f'C:\\Users\\hourstonh\\Documents\\charles\\more_oxygen_projects\\'
                 f'{sampling_station}_04_inexact_duplicate_checks{skip03}\\')
    input_file_path = os.path.join(
        input_dir,
        '{}_CTD_BOT_CHE_data.csv'.format(sampling_station)
    )

    # Add CS09 data first, after it has been qced
    df = pd.read_csv(input_file_path)

    # Check what the deepest measurements are for CS09 - many casts only go down to 185m
    profile_start_idx, profile_end_idx = get_profile_st_en_idx(df.loc[:, 'Profile number'])
    max_depths = [0] * len(profile_start_idx)

    for i in range(len(profile_start_idx)):
        start_idx_i = profile_start_idx[i]
        end_idx_i = profile_end_idx[i]
        max_depths[i] = df.loc[start_idx_i:end_idx_i, 'Depth [m]'].max()

    # Check how many casts come close to 200m depth
    print(max_depths)
    print(len(max_depths))
    print(sum([x >= 190 for x in max_depths]))
    print(sum([185 < x < 190 for x in max_depths]))  # not captured by either 175 or 200m bin

    idx_150 = index_rows(df.loc[:, 'Depth [m]'], 150, max_distance)
    idx_175 = index_rows(df.loc[:, 'Depth [m]'], 175, max_distance)
    idx_200 = index_rows(df.loc[:, 'Depth [m]'], 200, max_distance)

    # Convert time to usable format
    df['Time_dt'] = [np.datetime64(x) for x in df.Time_dt]

    # todo average measurements from same cast that are in the same bin?

    # Add SCOTT2 oxygen at 280m to the same plot

    # Open the scott2 files
    scott2_files = glob.glob(WDIR + 'SCOTT2\\*.nc')
    scott2_files.sort()

    fig, ax = plt.subplots()

    markersize = 6
    markerstyle = 'x'
    ax.scatter(
        df.loc[idx_150, 'Time_dt'], df.loc[idx_150, 'Oxygen [mL/L]'],
        c='magenta', label='CS09 150m', marker=markerstyle, s=markersize, zorder=5
    )
    ax.scatter(
        df.loc[idx_175, 'Time_dt'], df.loc[idx_175, 'Oxygen [mL/L]'],
        c='orange', label='CS09 175m', marker=markerstyle, s=markersize, zorder=5.1
    )
    ax.scatter(
        df.loc[idx_200, 'Time_dt'], df.loc[idx_200, 'Oxygen [mL/L]'],
        c='green', label='CS09 200m', marker=markerstyle, s=markersize, zorder=5.2
    )

    for i in range(len(scott2_files)):
        ds = xr.open_dataset(scott2_files[i])
        if i == 0:
            label = 'SCOTT2 280m'
        else:
            label = None

        ax.plot(ds.time.data, ds.DOXYZZ01.data, c='tab:blue', label=label, zorder=4.9)

    # ax.axhline(y=1.4, c='r')  # Add horizontal line at hypoxia boundary
    # ax.set_xlim()
    ax.set_ylabel('Oxygen [mL/L]')
    plt.legend()

    # plt.title('test plot')
    plt.tight_layout()

    plot_name = os.path.join(WDIR, f'scott2-280m_cs09-150m-175m-200m{skip03}.png')
    plt.savefig(plot_name)
    plt.close()

    return


def plot_cs09_annual_cycle():
    """
    Plot on axis that goes from 1-365 days with different symbols for different 5 year periods
    Thus the plot shows the annual cycle and the years are represented by the symbols.
    Then we can see the annual cycle and think about how to remove it.
    Exclude scott2 here.
    :return:
    """
    # Max +/- distance away from each depth to include data from
    # So we look for data between 140m and 160m for the 150m bin
    max_distance = 10

    # Confirm oxygen units - mL/L

    skip03 = ''  # '_skip03'  # ''

    sampling_station = 'CS09'
    input_dir = (f'C:\\Users\\hourstonh\\Documents\\charles\\more_oxygen_projects\\'
                 f'{sampling_station}_04_inexact_duplicate_checks{skip03}\\')
    input_file_path = os.path.join(
        input_dir,
        '{}_CTD_BOT_CHE_data.csv'.format(sampling_station)
    )

    # Add CS09 data first, after it has been qced
    df = pd.read_csv(input_file_path)

    bin_dict = {
        150: index_rows(df.loc[:, 'Depth [m]'], 150, max_distance),
        175: index_rows(df.loc[:, 'Depth [m]'], 175, max_distance),
        200: index_rows(df.loc[:, 'Depth [m]'], 200, max_distance)
    }

    # Get the day of year
    df['Day_of_year'] = [pd.to_datetime(x).day_of_year for x in df.Time_dt]

    # Get year
    df['Year'] = [pd.to_datetime(x).year for x in df.Time_dt]

    fig, ax = plt.subplots(nrows=3, ncols=1, sharex=True, figsize=(6, 8))

    for i, depth in zip([0, 1, 2], [150, 175, 200]):
        for period_start in range(2000, 2021, 5):
            idx_year = (df.Year >= period_start) & (df.Year < period_start + 5)

            ax[i].scatter(
                df.loc[idx_year & bin_dict[depth], 'Day_of_year'],
                df.loc[idx_year & bin_dict[depth], 'Oxygen [mL/L]'],
                label=f'{period_start}-{period_start + 5}',
                marker='x'
            )
        ax[i].set_title(f'{depth}m')
        ax[i].set_ylabel('Oxygen [mL/L]')
        ax[i].set_ylim(1, 6)
        plt.legend()

    plt.xlim(1, 365)

    plt.xlabel('Day of year')

    plt.tight_layout()

    plot_name = os.path.join(WDIR, f'cs09-150m-175m-200m{skip03}_annual_cycle.png')
    plt.savefig(plot_name)
    plt.close(fig)

    return

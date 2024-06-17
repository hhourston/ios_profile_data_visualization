import matplotlib.pyplot as plt
import xarray as xr
import os
import pandas as pd
import glob
from helpers import get_profile_st_en_idx
import numpy as np

WDIR = 'C:\\Users\\hourstonh\\Documents\\charles\\more_oxygen_projects\\'


def mask_rows(depths, x, max_distance):
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

    idx_150 = mask_rows(df.loc[:, 'Depth [m]'], 150, max_distance)
    idx_175 = mask_rows(df.loc[:, 'Depth [m]'], 175, max_distance)
    idx_200 = mask_rows(df.loc[:, 'Depth [m]'], 200, max_distance)

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
        150: mask_rows(df.loc[:, 'Depth [m]'], 150, max_distance),
        175: mask_rows(df.loc[:, 'Depth [m]'], 175, max_distance),
        200: mask_rows(df.loc[:, 'Depth [m]'], 200, max_distance)
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


def plot_season_anomalies():
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

    # Ignore 200m depth for this...
    bin_dict = {
        150: {'mask': mask_rows(df.loc[:, 'Depth [m]'], 150, max_distance)},
        175: {'mask': mask_rows(df.loc[:, 'Depth [m]'], 175, max_distance)},
        # 200: index_rows(df.loc[:, 'Depth [m]'], 200, max_distance)
    }

    # Get the day of year
    df['Day_of_year'] = [pd.to_datetime(x).day_of_year for x in df.Time_dt]

    # Get year
    df['Year'] = [pd.to_datetime(x).year for x in df.Time_dt]

    # Convert time to usable format
    df['Time_dt'] = [np.datetime64(x) for x in df.Time_dt]

    # Get cluster centers
    # Units in day of year; ignore the day 50 group and day 110ish group
    # Day 150 group: take from _ to _
    #     200        take from _ to _
    #     270        take from 240 to 300 ish
    # Dict key is the center of the cluster and the Dict value is the range of the cluster
    season_clusters = {
        150: {'range': (125, 175)},
        200: {'range': (175, 225)},
        270: {'range': (240, 300)}
    }

    # Compute the mean for each of the 3 clusters for each of 150m and 175m
    for szn in season_clusters.keys():
        print('Day', szn)
        season_clusters[szn]['mask'] = (
            (df.Day_of_year >= season_clusters[szn]['range'][0]) &
            (df.Day_of_year <= season_clusters[szn]['range'][1])
        )

        for depth in bin_dict.keys():
            bin_dict[depth][szn] = df.loc[
                bin_dict[depth]['mask'] & season_clusters[szn]['mask'], 'Oxygen [mL/L]'
            ].mean()
            print(
                f'Depth {depth}m mean: {bin_dict[depth][szn]} and group size:',
                sum(bin_dict[depth]['mask'] & season_clusters[szn]['mask'])
            )

    # Plot from 1990 to present

    # 150m: day 150, day 200, day 270
    # 175m: day 150, day 200, day 270
    # print(bin_dict)
    # print(season_clusters)

    nrows, ncols = [2, 3]
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(9, 4.5), sharex=True)

    for i, depth in zip(range(2), bin_dict.keys()):
        for k, szn in zip(range(3), season_clusters.keys()):
            mask_final = bin_dict[depth]['mask'] & season_clusters[szn]['mask']

            # Plot anomalies by subtracting mean from observations
            # axis_idx = i * (nrows + 1) + k

            ax[i, k].scatter(
                df.loc[mask_final, 'Time_dt'],
                df.loc[mask_final, 'Oxygen [mL/L]'] - bin_dict[depth][szn],
                marker='x',
                c='tab:blue'
            )
            ax[i, k].text(
                x=0.1, y=0.1, s=r'$\overline{x}=$' + str(round(bin_dict[depth][szn], 2)),
                transform=ax[i, k].transAxes
            )
            ax[i, k].set_title(
                f'{depth}m day {season_clusters[szn]["range"][0]}-{season_clusters[szn]["range"][1]}'
            )
            ax[i, k].set_xlim(left=np.datetime64('1990-01-01'), right=np.datetime64('2025-01-01'))
            ax[i, k].tick_params(direction='in', bottom=True, top=True, left=True, right=True)
            ax[i, k].set_xticks(
                ticks=[np.datetime64(str(x)) for x in range(1990, 2026, 5)],
                labels=np.arange(1990, 2026, 5)
            )
            ax[i, k].set_ylim(-1.5, 1.5)
            if k == 0:
                ax[i, k].set_ylabel('Oxygen anomaly [mL/L]')
            if i == 1:
                ax[i, k].tick_params(axis='x', labelrotation=30)

    # plt.tight_layout()
    plot_name = os.path.join(WDIR, f'cs09-150m-175m{skip03}_seasonal_anomalies.png')
    plt.savefig(plot_name)
    plt.close(fig)

    return


def plot_season_anomalies_together():
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

    # Ignore 200m depth for this...
    bin_dict = {
        150: {'mask': mask_rows(df.loc[:, 'Depth [m]'], 150, max_distance)},
        175: {'mask': mask_rows(df.loc[:, 'Depth [m]'], 175, max_distance)},
        # 200: index_rows(df.loc[:, 'Depth [m]'], 200, max_distance)
    }

    # Get the day of year
    df['Day_of_year'] = [pd.to_datetime(x).day_of_year for x in df.Time_dt]

    # Get year
    df['Year'] = [pd.to_datetime(x).year for x in df.Time_dt]

    # Convert time to usable format
    df['Time_dt'] = [np.datetime64(x) for x in df.Time_dt]

    # Get cluster centers
    # Units in day of year; ignore the day 50 group and day 110ish group
    # Day 150 group: take from _ to _
    #     200        take from _ to _
    #     270        take from 240 to 300 ish
    # Dict key is the center of the cluster and the Dict value is the range of the cluster
    season_clusters = {
        150: {'range': (125, 175)},
        200: {'range': (175, 225)},
        270: {'range': (240, 300)}
    }

    # Compute the mean for each of the 3 clusters for each of 150m and 175m
    for szn in season_clusters.keys():
        print('Day', szn)
        season_clusters[szn]['mask'] = (
                (df.Day_of_year >= season_clusters[szn]['range'][0]) &
                (df.Day_of_year <= season_clusters[szn]['range'][1])
        )

        for depth in bin_dict.keys():
            bin_dict[depth][szn] = df.loc[
                bin_dict[depth]['mask'] & season_clusters[szn]['mask'], 'Oxygen [mL/L]'
            ].mean()
            print(
                f'Depth {depth}m mean: {bin_dict[depth][szn]} and group size:',
                sum(bin_dict[depth]['mask'] & season_clusters[szn]['mask'])
            )

    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(4.5, 5.5), sharex=True)

    for i, depth in zip(range(2), bin_dict.keys()):
        for szn in season_clusters.keys():
            mask_final = bin_dict[depth]['mask'] & season_clusters[szn]['mask']
            ax[i].scatter(
                df.loc[mask_final, 'Time_dt'],
                df.loc[mask_final, 'Oxygen [mL/L]'] - bin_dict[depth][szn],
                marker='x',
                # c='tab:blue'
                label=f'Day {season_clusters[szn]["range"][0]}-{season_clusters[szn]["range"][1]}'
            )
        ax[i].set_title(f'{depth}m')
        ax[i].set_xlim(left=np.datetime64('1990-01-01'), right=np.datetime64('2025-01-01'))
        ax[i].tick_params(direction='in', bottom=True, top=True, left=True, right=True)
        ax[i].set_xticks(
            ticks=[np.datetime64(str(x)) for x in range(1990, 2026, 5)],
            labels=np.arange(1990, 2026, 5)
        )
        ax[i].set_ylim(-1.5, 1.5)
        ax[i].set_ylabel('Oxygen anomaly [mL/L]')
        if i == 1:
            ax[i].tick_params(axis='x', labelrotation=30)

    plt.legend()

    plt.tight_layout()

    plot_name = os.path.join(WDIR, f'cs09-150m-175m{skip03}_seasonal_anomalies_together.png')
    plt.savefig(plot_name)
    plt.close(fig)

    return

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


def index_closest_obs(bin_depths: list, df: pd.DataFrame, max_distance):

    # Initialize dictionary
    idx_dict = {}
    for depth in bin_depths:
        idx_dict[depth] = []

    # Start and end indices of profiles
    profile_start_idx, profile_end_idx = get_profile_st_en_idx(df.loc[:, 'Profile number'])

    for i in range(len(profile_start_idx)):
        start_idx_i = profile_start_idx[i]
        end_idx_i = profile_end_idx[i]
        distances = {}
        # Find the closest observation to each depth in 150, 175, 200 and keep if it's within +/- 10 m of the depth
        for depth in bin_depths:
            distances[depth] = df.loc[start_idx_i:end_idx_i, 'Depth [m]'] - depth
            if distances[depth].abs().min() < max_distance:
                # Index of measurement of smallest distance to the depth bin
                idx_min_dist = start_idx_i + np.where(distances[depth].abs() == distances[depth].abs().min())[0][0]
                idx_dict[depth].append(idx_min_dist)

    for depth in idx_dict.keys():
        idx_dict[depth] = np.array(idx_dict[depth])

    return idx_dict


def plot_cs09_scott2():
    # plot CTD oxygen time series at 150m, 175m, and 200m

    # Max +/- distance away from each depth to include data from
    # So we look for data between 140m and 160m for the 150m bin
    max_distance = 10

    bin_depths = [150, 175, 200]

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

    # # Check what the deepest measurements are for CS09 - many casts only go down to 185m
    # profile_start_idx, profile_end_idx = get_profile_st_en_idx(df.loc[:, 'Profile number'])
    #
    # max_depths = [0] * len(profile_start_idx)
    # for i in range(len(profile_start_idx)):
    #     start_idx_i = profile_start_idx[i]
    #     end_idx_i = profile_end_idx[i]
    #     max_depths[i] = df.loc[start_idx_i:end_idx_i, 'Depth [m]'].max()
    #
    # # Check how many casts come close to 200m depth
    # print(max_depths)
    # print(len(max_depths))
    # print(sum([x >= 190 for x in max_depths]))
    # print(sum([185 < x < 190 for x in max_depths]))  # not captured by either 175 or 200m bin

    # idx_150 = []
    # idx_175 = []
    # idx_200 = []

    idx_dict = index_closest_obs(bin_depths, df, max_distance)

    # idx_150 = mask_rows(df.loc[:, 'Depth [m]'], 150, max_distance)
    # idx_175 = mask_rows(df.loc[:, 'Depth [m]'], 175, max_distance)
    # idx_200 = mask_rows(df.loc[:, 'Depth [m]'], 200, max_distance)

    # Convert time to usable format
    df['Time_dt'] = [np.datetime64(x) for x in df.Time_dt]

    # Add SCOTT2 oxygen at 280m to the same plot

    # Open the scott2 files
    scott2_files = glob.glob(WDIR + 'SCOTT2\\*.nc')
    scott2_files.sort()

    fig, ax = plt.subplots()

    markersize = 6
    markerstyle = 'x'
    zorder_start = 5

    for depth, colour in zip(idx_dict.keys(), ['magenta', 'orange', 'chartreuse']):
        ax.scatter(
            df.loc[idx_dict[depth], 'Time_dt'], df.loc[idx_dict[depth], 'Oxygen [mL/L]'],
            c=colour, label=f'CS09 {depth}m', marker=markerstyle, s=markersize, zorder=zorder_start
        )
        zorder_start += 0.1

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

    plot_name = os.path.join(WDIR, f'scott2-280m_cs09-150m-175m-200m{skip03}_v2.png')
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
    bin_depths = [150, 175, 200]

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

    # Index one measurement per bin depth per cast
    # bin_dict = {
    #     150: mask_rows(df.loc[:, 'Depth [m]'], 150, max_distance),
    #     175: mask_rows(df.loc[:, 'Depth [m]'], 175, max_distance),
    #     200: mask_rows(df.loc[:, 'Depth [m]'], 200, max_distance)
    # }
    idx_dict = index_closest_obs(bin_depths, df, max_distance)

    # Get the day of year
    df['Day_of_year'] = [pd.to_datetime(x).day_of_year for x in df.Time_dt]

    # Get year
    df['Year'] = [pd.to_datetime(x).year for x in df.Time_dt]

    fig, ax = plt.subplots(nrows=3, ncols=1, sharex=True, figsize=(6, 8))

    for i, depth in zip([0, 1, 2], [150, 175, 200]):
        for period_start in range(2000, 2021, 5):
            idx_year = np.where((df.Year >= period_start) & (df.Year < period_start + 5))[0]

            ax[i].scatter(
                df.loc[np.intersect1d(idx_year, idx_dict[depth]), 'Day_of_year'],
                df.loc[np.intersect1d(idx_year, idx_dict[depth]), 'Oxygen [mL/L]'],
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

    plot_name = os.path.join(WDIR, f'cs09-150m-175m-200m{skip03}_annual_cycle_v2.png')
    plt.savefig(plot_name)
    plt.close(fig)

    return


def plot_season_anomalies():
    # Max +/- distance away from each depth to include data from
    # So we look for data between 140m and 160m for the 150m bin
    max_distance = 10

    bin_depths = [150, 175]

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
    idx_dict = index_closest_obs(bin_depths, df, max_distance)
    for key in idx_dict.keys():
        idx_dict[key] = {'index': idx_dict[key]}
    # bin_dict = {
    #     150: {'mask': mask_rows(df.loc[:, 'Depth [m]'], 150, max_distance)},
    #     175: {'mask': mask_rows(df.loc[:, 'Depth [m]'], 175, max_distance)},
    #     # 200: index_rows(df.loc[:, 'Depth [m]'], 200, max_distance)
    # }

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
        season_clusters[szn]['index'] = np.where(
            (df.Day_of_year >= season_clusters[szn]['range'][0]) &
            (df.Day_of_year <= season_clusters[szn]['range'][1])
        )[0]

        for depth in idx_dict.keys():
            idx_dict[depth][szn] = df.loc[
                np.intersect1d(idx_dict[depth]['index'], season_clusters[szn]['index']),
                'Oxygen [mL/L]'
            ].mean()
            print(
                f'Depth {depth}m mean: {idx_dict[depth][szn]} and group size:',
                len(np.intersect1d(idx_dict[depth]['index'], season_clusters[szn]['index']))
            )

    # Plot from 1990 to present

    # 150m: day 150, day 200, day 270
    # 175m: day 150, day 200, day 270
    # print(bin_dict)
    # print(season_clusters)

    nrows, ncols = [2, 3]
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(9, 4.5), sharex=True)

    for i, depth in zip(range(2), idx_dict.keys()):
        for k, szn in zip(range(3), season_clusters.keys()):
            idx_final = np.intersect1d(idx_dict[depth]['index'], season_clusters[szn]['index'])

            # Plot anomalies by subtracting mean from observations
            # axis_idx = i * (nrows + 1) + k

            ax[i, k].scatter(
                df.loc[idx_final, 'Time_dt'],
                df.loc[idx_final, 'Oxygen [mL/L]'] - idx_dict[depth][szn],
                marker='x',
                c='tab:blue'
            )
            ax[i, k].text(
                x=0.1, y=0.1, s=r'$\overline{x}=$' + str(round(idx_dict[depth][szn], 2)),
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
    plot_name = os.path.join(WDIR, f'cs09-150m-175m{skip03}_seasonal_anomalies_v2.png')
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

    # # Ignore 200m depth for this...
    # bin_dict = {
    #     150: {'mask': mask_rows(df.loc[:, 'Depth [m]'], 150, max_distance)},
    #     175: {'mask': mask_rows(df.loc[:, 'Depth [m]'], 175, max_distance)},
    #     # 200: index_rows(df.loc[:, 'Depth [m]'], 200, max_distance)
    # }
    bin_depths = [150, 175]
    idx_dict = index_closest_obs(bin_depths, df, max_distance)
    for key in idx_dict.keys():
        idx_dict[key] = {'index': idx_dict[key]}

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
        season_clusters[szn]['index'] = np.where(
                (df.Day_of_year >= season_clusters[szn]['range'][0]) &
                (df.Day_of_year <= season_clusters[szn]['range'][1])
        )[0]

        for depth in idx_dict.keys():
            idx_dict[depth][szn] = df.loc[
                np.intersect1d(idx_dict[depth]['index'], season_clusters[szn]['index']),
                'Oxygen [mL/L]'
            ].mean()
            print(
                f'Depth {depth}m mean: {idx_dict[depth][szn]} and group size:',
                len(np.intersect1d(idx_dict[depth]['index'], season_clusters[szn]['index']))
            )

    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(4.5, 5.5), sharex=True)

    for i, depth in zip(range(2), idx_dict.keys()):
        for szn in season_clusters.keys():
            idx_final = np.intersect1d(idx_dict[depth]['index'], season_clusters[szn]['index'])
            ax[i].scatter(
                df.loc[idx_final, 'Time_dt'],
                df.loc[idx_final, 'Oxygen [mL/L]'] - idx_dict[depth][szn],
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

    plot_name = os.path.join(WDIR, f'cs09-150m-175m{skip03}_seasonal_anomalies_together_v2.png')
    plt.savefig(plot_name)
    plt.close(fig)

    return


def compute_fit(x, y, fit_degrees):
    """
    COPIED FROM 12_plot_o2_on_density_surfaces.py

    Compute least-squares fit on y
    :param x:
    :param y:
    :param fit_degrees: number of degrees to use
    :return: x_linspace and y_hat_linspace, the fitted x and y values
    """
    x_values_sorted = np.array(sorted(x))
    y_values_sorted = np.array([i for _, i in
                                sorted(zip(x, y))])
    # Remove any nans otherwise polynomial crashes
    x_values_sorted = x_values_sorted[~np.isnan(y_values_sorted)]
    y_values_sorted = y_values_sorted[~np.isnan(y_values_sorted)]
    # Update polynomial module access from legacy access
    poly = np.polynomial.Polynomial.fit(
        x_values_sorted, y_values_sorted, deg=fit_degrees)
    # coeffs = poly.coef
    # fit_eqn = np.polynomial.Polynomial(coeffs[::-1]) # must reverse order coeffs
    # y_hat_sorted = fit_eqn(x_values_sorted)
    # ax.plot(x_values_sorted, y_hat_sorted, c=c)
    x_linspace, y_hat_linspace = poly.linspace(n=100)

    # numpy.linalg.LinAlgError: SVD did not converge in Linear Least Squares for OSP
    return x_linspace, y_hat_linspace


def plot_cs09_anomaly_trends():
    """
    Use trend from 26.5 density surface at station P4 and plot over top of cs09 anomalies
    Plot scott2 anomalies as well
    :return:
    """
    days_per_year = 365.25

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

    # # Ignore 200m depth for this...
    # bin_dict = {
    #     150: {'mask': mask_rows(df.loc[:, 'Depth [m]'], 150, max_distance)},
    #     175: {'mask': mask_rows(df.loc[:, 'Depth [m]'], 175, max_distance)},
    #     # 200: index_rows(df.loc[:, 'Depth [m]'], 200, max_distance)
    # }

    bin_depths = [150, 175]
    idx_dict = index_closest_obs(bin_depths, df, max_distance)
    for key in idx_dict.keys():
        idx_dict[key] = {'index': idx_dict[key]}

    # Get the day of year
    df['Day_of_year'] = [pd.to_datetime(x).day_of_year for x in df.Time_dt]

    # Get year
    df['Year'] = [pd.to_datetime(x).year for x in df.Time_dt]

    # Convert time to usable format
    df['Time_dt'] = [np.datetime64(x) for x in df.Time_dt]

    # Calculate time in decimal years
    df['Decimal_year'] = df['Year'] + df['Day_of_year'] / days_per_year

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
        season_clusters[szn]['index'] = np.where(
                (df.Day_of_year >= season_clusters[szn]['range'][0]) &
                (df.Day_of_year <= season_clusters[szn]['range'][1])
        )[0]

        for depth in idx_dict.keys():
            idx_dict[depth][szn] = df.loc[
                np.intersect1d(idx_dict[depth]['index'], season_clusters[szn]['index']),
                'Oxygen [mL/L]'
            ].mean()
            print(
                f'Depth {depth}m mean: {idx_dict[depth][szn]} and group size:',
                len(np.intersect1d(idx_dict[depth]['index'], season_clusters[szn]['index']))
            )

    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(6, 8), sharex=True)

    for i, depth in zip(range(2), idx_dict.keys()):
        for szn in season_clusters.keys():
            idx_final = np.intersect1d(idx_dict[depth]['index'], season_clusters[szn]['index'])
            ax[i].scatter(
                df.loc[idx_final, 'Decimal_year'],
                df.loc[idx_final, 'Oxygen [mL/L]'] - idx_dict[depth][szn],
                marker='x',
                # c='tab:blue'
                label=f'Day {season_clusters[szn]["range"][0]}-{season_clusters[szn]["range"][1]}'
            )
        ax[i].set_title(f'{depth}m')
        ax[i].set_xlim(left=1990, right=2025)
        ax[i].tick_params(direction='in', bottom=True, top=True, left=True, right=True)
        ax[i].set_xticks(ticks=np.arange(1990, 2026, 5))
        ax[i].set_ylim(-4, 4)
        ax[i].set_ylabel('Oxygen anomaly [mL/L]')
        if i == 1:
            ax[i].tick_params(axis='x', labelrotation=30)

    # todo add trend from 26.5 density surface for station P4
    bill_dir = 'D:\\charles\\line_P_data_products\\bill_crawford\\'
    bill_p4_file = os.path.join(
        bill_dir,
        'CrawfordPena Line P 1950-2019 4849-5_May2023.csv'
    )
    df_p4 = pd.read_csv(bill_p4_file)
    o2_avg_colname = df_p4.columns[15]
    print('avg oxy column name:', o2_avg_colname)

    mask_annual = pd.notna(df_p4[o2_avg_colname])
    o2_avg_data = df_p4.loc[mask_annual, o2_avg_colname]
    year_data = df_p4.loc[mask_annual, 'Date'].astype(int)
    x_linspace, y_hat_linspace = compute_fit(
        year_data, o2_avg_data - o2_avg_data.mean(), fit_degrees=2
    )
    ax[0].plot(x_linspace, y_hat_linspace - np.mean(y_hat_linspace), c='y', label='P4 26.5 fit')
    ax[1].plot(x_linspace, y_hat_linspace - np.mean(y_hat_linspace), c='y', label='P4 26.5 fit')

    # todo add scott2 anomalies and compute their trend
    scott2_files = glob.glob(WDIR + 'SCOTT2\\*.nc')
    scott2_files.sort()
    scott2_time = np.array([])
    scott2_oxy = np.array([])

    for fi in scott2_files:
        ds_scott2 = xr.open_dataset(fi)
        time_dt = pd.DatetimeIndex(ds_scott2.time.data)
        decimal_years = time_dt.year + time_dt.day_of_year / days_per_year
        scott2_time = np.concatenate((scott2_time, decimal_years))
        scott2_oxy = np.concatenate((scott2_oxy, ds_scott2.DOXYZZ01.data))

    # Compute anomalies
    scott2_oxy_anom = scott2_oxy - np.nanmean(scott2_oxy)

    ax[0].plot(scott2_time, scott2_oxy_anom, c='magenta', zorder=0.01, label='SCOTT2 280m')
    ax[1].plot(scott2_time, scott2_oxy_anom, c='magenta', zorder=0.01, label='SCOTT2 280m')

    # Add best fit line to scott2 data
    # Need to convert time to decimal years?
    x_fit_scott2, y_hat_scott2 = compute_fit(
        scott2_time, scott2_oxy_anom, fit_degrees=2
    )

    ax[0].plot(x_fit_scott2, y_hat_scott2, c='k', label='SCOTT2 280m fit')
    ax[1].plot(x_fit_scott2, y_hat_scott2, c='k', label='SCOTT2 280m fit')

    plt.legend(ncol=2)

    # plt.tight_layout()

    plot_name = os.path.join(WDIR, f'cs09-scott2-p4_150m-175m{skip03}_seasonal_anomaly_trends_v2.png')
    plt.savefig(plot_name)
    plt.close(fig)

    return

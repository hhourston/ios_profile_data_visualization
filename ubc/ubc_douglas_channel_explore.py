import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
# !python -m pip install basemap
from mpl_toolkits.basemap import Basemap
import os
from tqdm import trange


def subset_single_inlet(
        lon_min, lon_max, lat_min, lat_max, lon_data, lat_data, time_data
):
    subsetter = np.where((lon_data >= lon_min) &
                         (lon_data <= lon_max) &
                         (lat_data >= lat_min) &
                         (lat_data <= lat_max))[0]
    return lon_data[subsetter], lat_data[subsetter], time_data[subsetter]


def subset_coords_by_season(lon, lat, time, season_months):
    # Convert time to stable pandas DatetimeIndex
    dt_months = pd.to_datetime(time).month
    subsetter_season = np.where((dt_months >= season_months[0]) &
                                (dt_months <= season_months[-1]))[0]

    return lon[subsetter_season], lat[subsetter_season]


def subset_coords_by_year(lon, lat, time, select_year):
    try:
        dt_years = pd.to_datetime(time).year
    except AttributeError:
        dt_years = pd.to_datetime(time).dt.year
    subsetter_year = np.where((dt_years == select_year))[0]
    return lon[subsetter_year], lat[subsetter_year]


# ----------------------------------------------------------------------

# Investigate WODselect holdings of 1950's UBC data


def plot_spatial_availability(lon, lat, time, output_dir, plot_title):
    # Initialize figure
    # fig = plt.figure(num=None, figsize=(8, 8), dpi=100)
    fig = plt.figure(figsize=(7.2, 5.4))

    # Iterate through the seasons
    for j in range(4):
        # print(szn_abbrevs[j])
        # Add subplot per season
        ax = fig.add_subplot(2, 2, j + 1)

        # months = np.arange(3 * j + 1, 3 * j + 4)

        # Subset lat and lon by season
        # lon_subset, lat_subset = subset_coords_by_season(
        #     ncdata.lon.data, ncdata.lat.data, ncdata.time.data,
        #     season_months=months)
        # lon_subset, lat_subset = subset_nc_coords(
        #     lon_inlet, lat_inlet,
        #     time_inlet,
        #     season_months=months)
        yr = 1951+j
        lon_subset, lat_subset = subset_coords_by_year(
            lon, lat, time, select_year=yr
        )

        left_lon = -136.
        bot_lat = 47.
        right_lon = -121.
        top_lat = 56.
        # left_lon, bot_lat, right_lon, top_lat = [lonmin, latmin, lonmax, latmax]

        m = setup_basemap(left_lon, bot_lat, right_lon, top_lat)

        # Plot the locations of the samples
        x, y = m(lon_subset, lat_subset)
        # Plot on the subplot ax
        m.scatter(x, y, marker='o', color='r', s=0.5)

        # Make subplot titles
        # ax.set_title(szn_abbrevs[j])
        ax.set_title(yr)

    # Set figure title
    # if inlet_name is not None:
    #     fig_title = 'WODselect UBC 1951-1959 {} data spatial availability'.format(inlet_name)
    # else:
    #     fig_title = 'WODselect UBC 1951-1959 {} data spatial availability'
    plt.suptitle(plot_title)
    plot_name = os.path.join(
        output_dir, plot_title.replace(' ', '-') + '.png')
    plt.savefig(plot_name, dpi=400)
    plt.close(fig)
    return


def setup_basemap(left_lon, bot_lat, right_lon, top_lat):
    # Set up Lambert conformal map
    m = Basemap(llcrnrlon=left_lon, llcrnrlat=bot_lat,
                urcrnrlon=right_lon, urcrnrlat=top_lat,
                projection='lcc',
                resolution='h', lat_0=0.5 * (bot_lat + top_lat),
                lon_0=0.5 * (left_lon + right_lon))

    m.drawcoastlines(linewidth=0.2)
    m.drawparallels(np.arange(bot_lat, top_lat, 3),
                    labels=[1, 0, 0, 0])
    m.drawmeridians(np.arange(left_lon, right_lon, 5),
                    labels=[0, 0, 0, 1])
    # m.drawparallels(np.arange(bot_lat, top_lat, 0.3), labels=[1, 0, 0, 0])
    # m.drawmeridians(np.arange(left_lon, right_lon, 1), labels=[0, 0, 0, 1])
    # m.drawmapboundary(fill_color='white')
    m.fillcontinents(color='0.8')
    return m


def plot_wp_nodc_single_year(lon_wp, lat_wp, time_wp,
                             lon_nodc, lat_nodc, time_nodc,
                             year, output_dir, plot_title):
    # Subset each dataset by year
    lon_wp_subset, lat_wp_subset = subset_coords_by_year(
        lon_wp, lat_wp, time_wp, year)
    lon_nodc_subset, lat_nodc_subset = subset_coords_by_year(
        lon_nodc, lat_nodc, time_nodc, year)

    fig = plt.figure()  # figsize=(7.2, 5.4)

    left_lon = -136.
    bot_lat = 47.
    right_lon = -121.
    top_lat = 56.

    # Plot WP first
    ax1 = fig.add_subplot(1, 2, 1)
    m1 = setup_basemap(left_lon, bot_lat, right_lon, top_lat)
    # Plot the locations of the samples
    x, y = m1(lon_wp_subset, lat_wp_subset)
    # Plot on the subplot ax
    m1.scatter(x, y, marker='o', color='r', s=0.5)
    # Make subplot
    ax1.set_title('Water Properties')

    # Plot NODC
    ax2 = fig.add_subplot(1, 2, 2)
    m2 = setup_basemap(left_lon, bot_lat, right_lon, top_lat)
    # Plot the locations of the samples
    x, y = m2(lon_nodc_subset, lat_nodc_subset)
    # Plot on the subplot ax
    m2.scatter(x, y, marker='o', color='r', s=0.5)
    # Make subplot
    ax2.set_title('NODC')

    plt.suptitle(plot_title)
    plt.tight_layout()
    plot_name = os.path.join(
        output_dir, plot_title.replace(' ', '-') + '.png')
    plt.savefig(plot_name, dpi=400)
    plt.close(fig)
    return


# Plot WOD data

ncfile = 'C:\\Users\\HourstonH\\Documents\\charles\\' \
         'ubc_historical_cruises\\ocldb1660928666.22434_OSD.nc'

output_folder = 'C:\\Users\\HourstonH\\Documents\\charles\\' \
                'ubc_historical_cruises\\plots\\'

ncdata = xr.open_dataset(ncfile)

# print(ncdata.data_vars)
#
# print(ncdata.time.data)

time_dt = pd.to_datetime(ncdata.time.data)
print(sum(time_dt.year <= 1954))
print(sum(time_dt.year < 1951))

# Plot sampling locations in the data

szn_abbrevs = ['JFM', 'AMJ', 'JAS', 'OND']

# douglas_area = [-129.3, -128.5, 53.3, 54.1]
# dixon_area = [-133.5, -130, 53.5, 55]  # dixon entrance north of Haida Gwaii
# portland_canal_area = [-130-44/60-45/3600, -129-45/60-25/3600,
#                        54+40/60+54/3600, 56+3/60+52/3600]

inlet_name = None
# inlet_name = 'Douglas Channel'
# inlet_name = 'Dixon Entrance'
# inlet_name = 'Portland Canal'

# lonmin, lonmax, latmin, latmax = dixon_area
# lonmin, lonmax, latmin, latmax = [np.round(coord, 1) for coord in portland_canal_area]  # douglas_area

# lon_inlet, lat_inlet, time_inlet = subset_single_inlet(
#         lonmin, lonmax, latmin, latmax,
#         ncdata.lon.data, ncdata.lat.data, ncdata.time.data)

nodc_title = 'WODselect UBC 1951-1954 data spatial availability'
plot_spatial_availability(
    ncdata.lon.data, ncdata.lat.data, ncdata.time.data,
    output_folder, nodc_title)

# Plot WP data
csvfile = 'C:\\Users\\HourstonH\\Documents\\charles\\' \
          'ubc_historical_cruises\\ios_wp_ubc_1951_1954\\' \
          'wp_ubc_1951_1954_data.csv'
df_wp = pd.read_csv(csvfile)
profile_start_idx = np.unique(df_wp.loc[:, 'Profile number'],
                              return_index=True)[1]
wp_title = 'WP UBC 1951-1954 data spatial availability'
plot_spatial_availability(
    df_wp.loc[profile_start_idx, 'Lon'].to_numpy(float),
    df_wp.loc[profile_start_idx, 'Lat'].to_numpy(float),
    df_wp.loc[profile_start_idx, 'Time'],
    output_folder, wp_title)

# Plot WP and NODC data on the same plot
wp_nodc_title = 'WP and NODC UBC 1951-1954 data spatial availability'
plot_spatial_availability(
    np.concatenate(
        (df_wp.loc[profile_start_idx, 'Lon'].to_numpy(float),
         ncdata.lon.data)
    ),
    np.concatenate(
        (df_wp.loc[profile_start_idx, 'Lat'].to_numpy(float),
         ncdata.lat.data)
    ),
    np.concatenate(
        (df_wp.loc[profile_start_idx, 'Time'].to_numpy(),
         ncdata.time.data)
    ),
    output_folder,
    wp_nodc_title
)

# Plot single year
for y in range(1951, 1954+1):
    fig_title = 'WP and NODC UBC {} data spatial availability'.format(y)
    plot_wp_nodc_single_year(
        df_wp.loc[profile_start_idx, 'Lon'].to_numpy(float),
        df_wp.loc[profile_start_idx, 'Lat'].to_numpy(float),
        df_wp.loc[profile_start_idx, 'Time'].to_numpy(),
        ncdata.lon.data, ncdata.lat.data, ncdata.time.data,
        y, output_folder, fig_title)

# -------------------------------------------------------------------------

# Plot temporal availability


def count_profiles_per_year(time_data, year_range, season_months):
    # year_range = np.arange(1950, 1959+1)
    time_pd = pd.to_datetime(time_data)
    profile_counts = np.zeros(len(year_range))
    for j in range(len(year_range)):
        subsetter = np.where((time_pd.year == year_range[j]) &
                             (time_pd.month >= season_months[0]) &
                             (time_pd.month <= season_months[-1]))[0]
        profile_counts[j] += len(subsetter)

    return profile_counts


szn_abbrevs = ['JFM', 'AMJ', 'JAS', 'OND']
years_ubc = np.arange(1950, 1959+1)

# Initialize figure
fig = plt.figure(figsize=(7.2, 5.4))

# Iterate through the seasons
for j in range(4):
    # Add subplot per season
    ax = fig.add_subplot(2, 2, j + 1)

    months = np.arange(3 * j + 1, 3 * j + 4)

    counts = count_profiles_per_year(
        ncdata.time.data, years_ubc, months)
    # counts = count_profiles_per_year(
    #     time_inlet, years_ubc, months)

    ax.scatter(years_ubc, counts, s=2)

    ax.set_title(szn_abbrevs[j])

    # Add text box about total number of profiles
    ax.text(0.95, 0.95, 'Total profiles: {}'.format(int(sum(counts))),
            verticalalignment='top', horizontalalignment='right',
            transform=ax.transAxes, fontsize=8)

# Adjust hspace
fig.subplots_adjust(hspace=0.3)

if inlet_name is not None:
    fig_title = 'WODselect UBC 1950-1959 {} data profile counts'.format(inlet_name)
else:
    fig_title = 'WODselect UBC 1950-1959 data profile counts'
plt.suptitle(fig_title)

png_name = os.path.join(
    output_folder, fig_title.replace(' ', '-') + '.png')
plt.savefig(png_name, dpi=400)

plt.close(fig)

# --------------------------------------------------------------------------

# Open IOS WP .ubc data holdings

wp_ubc_file = 'C:\\Users\\HourstonH\\Documents\\charles\\' \
              'ubc_historical_cruises\\' \
              'csv_file_download_list_1950_1954.csv'

wp_ubc_df = pd.read_csv(wp_ubc_file)

wp_time = pd.to_datetime(wp_ubc_df.loc[:, 'START TIME(UTC)']).to_numpy()
wp_lat = wp_ubc_df.loc[:, 'LAT'].to_numpy()
wp_lon = wp_ubc_df.loc[:, 'LON'].to_numpy()

# Iterate through all the NODC profiles and check if each
# one is in IOS WP
nodc_is_in_wp = np.repeat(-1, len(ncdata.time.data))

for i in trange(len(nodc_is_in_wp)):  # len(nodc_is_in_wp)
    indexer = np.where(
        (abs(wp_time - ncdata.time.data[i]) < pd.to_timedelta('1 hour')) &
        (abs(wp_lon - ncdata.lon.data[i]) < 0.2) &
        (abs(wp_lat - ncdata.lat.data[i]) < 0.2)
    )[0]

    if len(indexer) > 0:
        nodc_is_in_wp[i] = indexer[0]
    # # elif len(indexer) > 1:
    # #     nodc_is_in_wp[i] = -2
    # if len(indexer) > 0:
    #     nodc_is_in_wp[i] += 2

print(max(nodc_is_in_wp))
print(sum(nodc_is_in_wp > -1))

for i in range(len(nodc_is_in_wp)):
    if nodc_is_in_wp[i] > -1:
        wp_idx = int(nodc_is_in_wp[i])
        print('nodc: {} time, {} lon, {} lat'.format(
            ncdata.time.data[i], ncdata.lon.data[i], ncdata.lat.data[i]))
        print('wp: {} time, {} lon, {} lat'.format(
            wp_time[wp_idx], wp_lon[wp_idx], wp_lat[wp_idx]))
        print()

# Why did it go from 450 matches to 375 matches?
# Because some have more than one match to the ios wp dataset

nodc_to_check = np.where(nodc_is_in_wp > -1)[0]

nodc_to_check_df = pd.DataFrame(
    data=np.array([nodc_to_check, nodc_is_in_wp[nodc_to_check]]).T,
    columns=['Indices from NODC', 'Indices from WP'])

nodc_to_check_df['WP url'] = wp_ubc_df.loc[nodc_is_in_wp[nodc_to_check], 'FILE_URL'].to_numpy()

wp_to_check_file = os.path.join(output_folder, 'nodc_wp_comparison.csv')
nodc_to_check_df.to_csv(wp_to_check_file, index=False)

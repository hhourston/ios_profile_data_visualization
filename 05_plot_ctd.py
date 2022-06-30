import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import numpy as np
import datetime


def plot_annual_samp_freq(df, stn, png_name):
    # Get data in right format

    # Get indices of all the row (profile) starts and ends
    profile_starts = np.unique(df.loc[:, 'Profile number'],
                               return_index=True)[1]

    # Reduce time from flattened 2D array to 1D array
    time_reduced = pd.to_datetime(
        (df.loc[profile_starts, 'Time'])).array

    num_profs = len(profile_starts)
    num_bins = max(time_reduced.year) - min(time_reduced.year) + 1

    # Manually assign y axis ticks to have only whole number ticks
    num_yticks = max(np.unique(time_reduced.year,
                               return_counts=True)[1])
    yticks = np.arange(num_yticks + 1)

    plt.clf()  # Clear any active plots
    fig, ax = plt.subplots()  # Create a new figure and axis instance

    ax.hist(time_reduced.year, bins=num_bins, align='left',
            label='Number of files: {}'.format(num_profs))
    ax.set_yticks(yticks)
    ax.set_ylabel('Number of Profiles')
    plt.legend()
    plt.title('Station {} Sampling History'.format(stn))
    plt.tight_layout()
    plt.savefig(png_name)
    plt.close(fig)
    return


def plot_monthly_samp_freq(df, stn, png_name):
    # Followed James Hannah's ios-inlets plot code
    # And
    # https://matplotlib.org/stable/gallery/images_contours_and_fields/image_annotated_heatmap.html#sphx-glr-gallery-images-contours-and-fields-image-annotated-heatmap-py
    months = [
        "January",
        "February",
        "March",
        "April",
        "May",
        "June",
        "July",
        "August",
        "September",
        "October",
        "November",
        "December",
    ]

    # Reduce time from flattened 2D array to 1D array
    time_reduced = pd.to_datetime(np.unique(df.loc[:, 'Time'])).array

    # ----------author: James Hannah-----------
    # Get the oceanographic variable values

    # Get min and max year in the df
    min_year = np.min(time_reduced.year)  # END.year
    max_year = np.max(time_reduced.year)  # 0
    year_range = max_year - min_year + 1

    # Initialize array to hold heatmap data
    mthly_counts = np.zeros(
        shape=(year_range, len(months)), dtype='int')

    for i in range(year_range):
        for j in range(len(months)):
            mthly_counts[i, j] = sum(
                (time_reduced.year == min_year + i) &
                (time_reduced.month == j + 1))

    biggest = np.max(mthly_counts)

    plt.clf()  # Close any open active plots
    matplotlib.rc("axes", titlesize=25)
    matplotlib.rc("xtick", labelsize=20)
    matplotlib.rc("ytick", labelsize=20)
    plt.figure(figsize=(40, 10), constrained_layout=True)
    # Display data as an image, i.e., on a 2D regular raster.
    plt.imshow(mthly_counts.T, vmin=0, vmax=biggest, cmap="Blues")
    plt.yticks(ticks=range(12), labels=months)
    plt.xticks(
        ticks=range(0, year_range, 2),
        labels=range(min_year, max_year + 1, 2),
        rotation=45,
        ha="right",
        rotation_mode="anchor",
    )
    for i in range(year_range):
        for j in range(len(months)):
            plt.text(
                i,              # position to place the text
                j,              # "                        "
                mthly_counts[i, j],   # the text (number of profiles)
                ha="center",
                va="center",
                color="k",
                fontsize="large",
            )

    plt.title('Station {} CTD Sampling Frequency by Month'.format(stn))
    plt.axis('tight')
    plt.colorbar()
    plt.savefig(png_name)
    plt.close()

    # Reset values
    matplotlib.rcdefaults()
    plt.axis("auto")

    return


def pad_ragged_array(df, var_name, var_unit):
    # https://stackoverflow.com/questions/16346506/representing-a-ragged-array-in-numpy-by-padding

    # Reduce time from flattened 2D array to 1D array
    time_reduced = pd.to_datetime(np.unique(df.loc[:, 'Time'])).array

    # Add +1 because numpy range not inclusive of end
    min_depth_bin = np.min(df.loc[:, 'Depth bin [m]'])
    max_depth_bin = np.max(df.loc[:, 'Depth bin [m]'])
    depth_reduced = np.arange(min_depth_bin, max_depth_bin + 1)

    unique_depth_mask = df.loc[:, 'Unique binned depth mask']

    # Apply the mask to the dataframe
    df_updated = df.loc[unique_depth_mask]

    # Name of the column in the df containing the variable data
    var_column = '{} [{}]'.format(var_name, var_unit)

    padding_value = np.nan

    # Initialize array for containing variable values
    # Has shape (time, depth)
    var_arr = np.repeat(
        padding_value,
        len(time_reduced) * len(depth_reduced)).reshape(
        (len(time_reduced), len(depth_reduced)))

    # Get indices of all the row (profile) starts and ends
    row_starts = np.unique(
        df_updated.loc[:, 'Profile number'],
        return_index=True)[1]

    df_updated_len = len(df_updated)

    row_ends = np.concatenate((row_starts[1:], [df_updated_len]))

    for i in range(len(row_starts)):
        # Pandas indexing is inclusive of end
        profile_depths = df_updated.loc[
                         row_starts[i]:row_ends[i], 'Depth bin [m]']
        # Use the profile binned depths as the indexer
        # which may only work if the starting depth is zero
        # unless the min depth bin is subtracted
        var_arr[
            i,
            profile_depths - min_depth_bin] = df_updated.loc[
                                              row_starts[i]:row_ends[i],
                                              var_column]

    return time_reduced, depth_reduced, var_arr


def plot_contourf(df, var_name, var_unit, stn, cmap, png_name):
    # Start by padding the ragged profiles
    time_reduced, depth_reduced, var_arr = pad_ragged_array(
        df, var_name, var_unit)

    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 6))
    # f1 = ax.pcolormesh(time_reduced, depth_reduced, var_arr.T,
    #                    cmap='hsv', shading='auto')

    f1 = ax.contourf(time_reduced, depth_reduced, var_arr.T,
                     cmap=cmap)

    # Add the color bar
    cbar = fig.colorbar(f1)
    cbar.set_label('{} [{}]'.format(var_name, var_unit))

    # Invert the y-axis so that depth increases downwards
    plt.gca().invert_yaxis()

    ax.set_xlabel('Time')
    ax.set_ylabel('Depth [m]')
    plt.title('Station {} CTD {}'.format(stn, var_name))

    plt.tight_layout()

    plt.savefig(png_name)
    plt.close()
    return


def data_mask(depth_binned, unique_depth_mask, select_depth):
    mask = (depth_binned == select_depth) & unique_depth_mask
    return mask


def compute_anomalies(data):
    return data - np.mean(data)


def select_binned_data(df, var_name, var_unit):
    # Create mask for each select level
    subsetter_5m = data_mask(
        df.loc[:, 'Depth bin [m]'],
        df.loc[:, 'Unique binned depth mask'], 5)
    subsetter_25m = data_mask(
        df.loc[:, 'Depth bin [m]'],
        df.loc[:, 'Unique binned depth mask'], 25)
    subsetter_50m = data_mask(
        df.loc[:, 'Depth bin [m]'],
        df.loc[:, 'Unique binned depth mask'], 50)
    subsetter_100m = data_mask(
        df.loc[:, 'Depth bin [m]'],
        df.loc[:, 'Unique binned depth mask'], 100)
    subsetter_bottom = data_mask(
        df.loc[:, 'Depth bin [m]'],
        df.loc[:, 'Unique binned depth mask'],
        np.max(df.loc[:, 'Depth bin [m]']))

    # Subset the variable data
    var_col_name = '{} [{}]'.format(var_name, var_unit)

    var_5m = df.loc[subsetter_5m, var_col_name]
    var_25m = df.loc[subsetter_25m, var_col_name]
    var_50m = df.loc[subsetter_50m, var_col_name]
    var_100m = df.loc[subsetter_100m, var_col_name]
    var_bottom = df.loc[subsetter_bottom, var_col_name]

    # Compute the variable anomalies
    anom_5m = compute_anomalies(var_5m)
    anom_25m = compute_anomalies(var_25m)
    anom_50m = compute_anomalies(var_50m)
    anom_100m = compute_anomalies(var_100m)
    anom_bottom = compute_anomalies(var_bottom)

    # Convert time string data to numpy datetime64
    time_5m = pd.to_datetime(df.loc[subsetter_5m, 'Time'])
    time_25m = pd.to_datetime(df.loc[subsetter_25m, 'Time'])
    time_50m = pd.to_datetime(df.loc[subsetter_50m, 'Time'])
    time_100m = pd.to_datetime(df.loc[subsetter_100m, 'Time'])
    time_bottom = pd.to_datetime(df.loc[subsetter_bottom, 'Time'])

    time_list = [time_5m, time_25m, time_50m, time_100m, time_bottom]
    anom_list = [anom_5m, anom_25m, anom_50m, anom_100m, anom_bottom]

    return time_list, anom_list


def plot_anomalies(df, var_name, var_unit, stn, png_name):
    # Make line plot of anomalies at select depths
    # Select depths: 5m, 25m, 50m, 200m, bottom
    select_depths = [5, 25, 50, 100, 'bottom']

    time_list, anom_list = select_binned_data(df, var_name, var_unit)

    time_5m, time_25m, time_50m, time_100m, time_bottom = time_list
    anom_5m, anom_25m, anom_50m, anom_100m, anom_bottom = anom_list

    # Make the plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(time_5m, anom_5m, label='5m', marker='o')
    ax.plot(time_25m, anom_25m, label='25m', marker='^')
    ax.plot(time_50m, anom_50m, label='50m', marker='s')
    ax.plot(time_100m, anom_100m, label='100m', marker='x')
    ax.plot(time_bottom, anom_bottom, label='bottom', marker='v')
    ax.set_xlabel('Time')
    ax.set_ylabel('{} anomaly [{}]'.format(var_name, var_unit))

    # Place legend outside of plot box
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)

    plt.title('Station {} CTD {} anomalies'.format(stn, var_name))

    plt.tight_layout()
    # fig.subplots_adjust(top=0.01)  # Add extra headspace
    # plt.subplots_adjust(left=-0.2, right=-0.1)

    # Save the figure
    plt.savefig(png_name)

    plt.close(fig)
    return


f = 'C:\\Users\\HourstonH\\Documents\\ctd_visualization\\csv\\' \
    'P1_ctd_data_binned_depth_dupl.csv'

# variable, units, var_abbrev, colourmap = [
#     'Temperature', 'C', 'T', 'plasma']
variable, units, var_abbrev, colourmap = [
  'Salinity', 'PSS-78', 'S', 'cividis']
station = 'P1'

# ----------------------Plot counts per year-----------------
hist_fig_name = 'C:\\Users\\HourstonH\\Documents\\ctd_visualization\\' \
                'png\\{}_ctd_annual_freq.png'.format(station)

df_in = pd.read_csv(f)

plot_annual_samp_freq(df_in, station, hist_fig_name)

# ----------------------Plot counts per month per year ------
mth_freq_fig_name = 'C:\\Users\\HourstonH\\Documents\\' \
                    'ctd_visualization\\png\\' \
                    '{}_ctd_monthly_freq.png'.format(station)

df_in = pd.read_csv(f)

plot_monthly_samp_freq(df_in, station, mth_freq_fig_name)

# ----------------------Plot pcolor data---------------------
pcolor_fig_name = 'C:\\Users\\HourstonH\\Documents\\ctd_visualization\\' \
                  'png\\{}_ctd_contourf_{}.png'.format(station, var_abbrev)

df_in = pd.read_csv(f)

plot_contourf(df_in, variable, units, station, colourmap, pcolor_fig_name)

# ----------------------Plot anomalies-----------------------
anom_fig_name = 'C:\\Users\\HourstonH\\Documents\\ctd_visualization\\' \
                'png\\{}_ctd_anomalies_{}.png'.format(station, var_abbrev)

df_in = pd.read_csv(f)

plot_anomalies(df_in, variable, units, station, anom_fig_name)

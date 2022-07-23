import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import numpy as np
# import datetime


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
    plt.title('Station {} CTD Sampling History'.format(stn))
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
    min_depth_bin = int(np.min(df.loc[:, 'Depth bin [m]']))
    max_depth_bin = int(np.max(df.loc[:, 'Depth bin [m]']))
    depth_reduced = np.arange(min_depth_bin, max_depth_bin + 1)

    unique_depth_mask = df.loc[:, 'Unique binned depth mask']

    # Apply the mask to the dataframe
    df_updated = df.loc[unique_depth_mask, :]

    df_updated.reset_index(inplace=True)

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

    row_ends = np.concatenate((row_starts[1:], [df_updated_len])) - 1

    for i in range(len(row_starts)):
        # Pandas indexing is inclusive of end
        # use .to_numpy to convert to numpy array
        # .array converts to pandas array, which can't be used
        # as indices
        profile_depths = df_updated.loc[
                         row_starts[i]:row_ends[i],
                         'Depth bin [m]'].to_numpy(dtype='int')
        # Use the profile binned depths as the indexer
        # which may only work if the starting depth is zero
        # unless the min depth bin is subtracted
        # print(row_starts[i], row_ends[i])
        # print(profile_depths)
        var_arr[
            i,
            profile_depths - min_depth_bin] = df_updated.loc[
                                              row_starts[i]:row_ends[i],
                                              var_column].to_numpy()

    return time_reduced, depth_reduced, var_arr


def scatter_padded_data(df, png_name, var_name, var_unit, depth_lim=None):
    depth_to_plot, binned = ['Depth bin [m]', True]  # 'Depth [m]'
    # time_dt = pd.to_datetime(df.loc[:, 'Time']).to_numpy()

    time_reduced, depth_reduced, var_arr = pad_ragged_array(
        df, var_name, var_unit)

    time_reduced_2d, depth_reduced_2d = np.meshgrid(
        time_reduced, depth_reduced)

    plt.scatter(time_reduced_2d[~np.isnan(var_arr.T)],
                depth_reduced_2d[~np.isnan(var_arr.T)], s=1, alpha=0.5)
    # plt.scatter(time_dt, df.loc[:, depth_to_plot], s=1,
    #             alpha=0.5)

    # Adjust the depth scale if specified
    if depth_lim is not None:
        plt.ylim(top=depth_lim)

    # Invert the y-axis so that depth increases downwards
    plt.gca().invert_yaxis()

    plt.ylabel(depth_to_plot)

    if binned:
        plt.title('Station {} QC depth binned'.format(station))
    else:
        plt.title('Station {} QC depth unbinned'.format(station))

    plt.tight_layout()

    plt.savefig(png_name)
    plt.close()
    return


def plot_contourf(ax, time_reduced, depth_reduced, var_arr, cmap):
    return ax.contourf(time_reduced, depth_reduced, var_arr, cmap=cmap)


def plot_pcolormesh(ax, time_reduced, depth_reduced, var_arr, cmap):
    return ax.pcolormesh(time_reduced, depth_reduced, var_arr, cmap=cmap,
                         shading='auto')


def plot_2d(df, var_name, var_unit, stn, cmap, png_name,
            plot_fn, depth_lim=None):
    # plot_fn: either plot_contourf() or plot_pcolormesh()

    # Start by padding the ragged profiles
    time_reduced, depth_reduced, var_arr = pad_ragged_array(
        df, var_name, var_unit)

    plt.clf()  # Close any open active plots

    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 6))
    # f1 = ax.pcolormesh(time_reduced, depth_reduced, var_arr.T,
    #                    cmap='hsv', shading='auto')

    f1 = plot_fn(ax, time_reduced, depth_reduced, var_arr.T, cmap=cmap)

    # Add the color bar
    cbar = fig.colorbar(f1)
    cbar.set_label('{} [{}]'.format(var_name, var_unit))

    # Adjust the depth scale if specified
    if depth_lim is not None:
        ax.set_ylim(top=depth_lim)

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


def compute_anomalies(data, time):
    # Compute anomalies for a single depth
    # For each depth, calculate the average of each year
    # then take the average over all the years

    years = np.arange(np.min(time.dt.year),
                      np.max(time.dt.year) + 1)
    yearly_means = np.zeros(len(years))

    for i in range(len(years)):
        year_mask = time.dt.year == years[i]
        yearly_means[i] = np.nanmean(data[year_mask])

    all_time_mean = np.nanmean(yearly_means)
    return all_time_mean, data - all_time_mean


def get_common_max_depth(df):
    # Find the common deepest depth in all profiles from
    # one station
    prof_numbers, prof_start_ind = np.unique(
        df.loc[:, 'Profile number'], return_index=True)
    prof_end_ind = np.concatenate([prof_start_ind[1:], [len(df)]])
    bin_dict = {}  # Depth bin dict
    for i in range(len(prof_numbers)):
        bin_dict[prof_numbers[i]] = df.loc[
                                    prof_start_ind[i]:prof_end_ind[i],
                                    'Depth bin [m]'].to_list()

    min_depth = int(np.min(df.loc[:, 'Depth bin [m]']))
    max_depth = int(np.max(df.loc[:, 'Depth bin [m]']))
    cmax_depth = max_depth  # common max depth

    for i in range(max_depth, min_depth - 1, -1):
        # Check if common_max_depth in every profile
        depth_in_prof = [
            True if cmax_depth in v else False for v in bin_dict.values()
        ]
        # Check if __% of the profiles have the depth in it
        common_cond = sum(depth_in_prof)/sum(
            df.loc[prof_start_ind, 'Unique binned depth mask']) > 0.50
        if common_cond:
            return cmax_depth
        else:
            cmax_depth -= 1

    return None


def select_binned_data(df, var_name, var_unit, select_depths):
    # Returns a data dictionary

    # Subset the variable data
    var_col_name = '{} [{}]'.format(var_name, var_unit)

    max_common_depth = get_common_max_depth(df)

    select_depths = np.concatenate((
        select_depths[select_depths < max_common_depth],
        [max_common_depth]
    ))

    data_dict = {}
    for d in select_depths:
        data_dict[d] = {}
        # Create subsetter by binned depth
        data_dict[d]['subsetter'] = data_mask(
            df.loc[:, 'Depth bin [m]'],
            df.loc[:, 'Unique binned depth mask'], d)
        # Variable data
        data_dict[d]['var'] = df.loc[data_dict[d]['subsetter'],
                                     var_col_name]
        # Datetime
        data_dict[d]['time'] = pd.to_datetime(
            df.loc[data_dict[d]['subsetter'], 'Time'])
        # Compute anomaly
        data_dict[d]['time mean'], data_dict[d]['anom'] = compute_anomalies(
            data_dict[d]['var'], data_dict[d]['time'])

    return data_dict


def plot_anomalies(df, var_name, var_unit, stn, png_name):
    # Make line plot of anomalies at select depths

    # Need as array and not list to take advantage of
    # boolean indexing
    select_depths = np.array([5, 25, 50, 100, 200])

    data_dict = select_binned_data(df, var_name, var_unit,
                                   select_depths)

    # Make the plot
    markers = ['o', '^', 's', 'x', 'v', '*', '+']
    fig, ax = plt.subplots(figsize=(10, 6))

    # data_dict keys are
    for i, dkey in enumerate(list(data_dict.keys())):
        # Sort data by time
        time_sorted, anom_sorted = zip(
            *sorted(zip(data_dict[dkey]['time'],
                        data_dict[dkey]['anom']))
        )
        ax.plot(time_sorted, anom_sorted,
                label='{}m'.format(dkey), marker=markers[i])

    # Add text about bottom depth
    # TODO figure out where to put the text on each plot
    # By default, this is in data coordinates.
    text_xloc, text_yloc = [0.95, 0.01]
    # Transform the coordinates from data to plot coordinates
    # max_depth >= common maximum depth
    max_depth = int(np.nanmax(df.loc[:, 'Depth bin [m]']))
    ax.text(text_xloc, text_yloc,
            '{} bottom depth = {}m'.format(stn, max_depth),
            verticalalignment='bottom', horizontalalignment='right',
            transform=ax.transAxes, fontsize='large')

    # # Plot bottom separately
    # bottom_key = list(data_dict.keys())[-1]
    # ax.plot(data_dict[bottom_key]['time'],
    #         data_dict[bottom_key]['anom'], label='bottom',
    #         marker=markers[i + 1])

    # Reset y-axis limits
    ybot, ytop = ax.get_ylim()
    if abs(ybot) > abs(ytop):
        ax.set_ylim(bottom=ybot, top=abs(ybot))
    elif abs(ybot) < abs(ytop):
        ax.set_ylim(bottom=-abs(ytop), top=ytop)

    # ax.set_xlabel('Time')
    ax.set_ylabel('{} anomaly [{}]'.format(var_name, var_unit))

    # Place legend outside of plot box
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left',
              borderaxespad=0)

    plt.title('Station {} CTD {} anomalies'.format(stn, var_name))

    plt.tight_layout()
    # fig.subplots_adjust(top=0.01)  # Add extra headspace
    # plt.subplots_adjust(left=-0.2, right=-0.1)

    # Save the figure
    plt.savefig(png_name)

    plt.close(fig)
    return


# -----------------------------------------------------------
# Make dict to make iteration easier
variable_dict = {'Temperature':
                 {'units': 'C', 'abbrev': 'T', 'cmap': 'YlOrBr'},  # Reds
                 'Salinity':
                 {'units': 'PSS-78', 'abbrev': 'S', 'cmap': 'Blues'},
                 'Oxygen':
                 {'units': 'mL/L', 'abbrev': 'O', 'cmap': 'jet'}}

# 'SI01'  # '59'  # '42'  # 'GEO1'  # 'LBP3'  # 'LB08'  # 'P1'
station = 'LBP3'

f = 'C:\\Users\\HourstonH\\Documents\\ctd_visualization\\csv\\' \
    '{}_ctd_data_binned_depth_dupl.csv'.format(station)

# f = 'C:\\Users\\HourstonH\\Documents\\ctd_visualization\\csv\\' \
#     '{}_ctd_data_no_qc_binned_depth_dupl.csv'.format(station)

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

# -------------------Choose variable to plot-----------------
# variable, units, var_abbrev, colourmap = [
#     'Temperature', 'C', 'T', 'plasma']
# variable, units, var_abbrev, colourmap = [
#   'Salinity', 'PSS-78', 'S', 'cividis']
# variable, units, var_abbrev, colourmap = [
#     'Oxygen', 'mL/L', 'O', 'cividis']

# ----------------------Plot contour data---------------------

# If station=LBP3, use the y limit
y_lim = 200

df_in = pd.read_csv(f)

for key in variable_dict.keys():
    print(key)
    variable = key
    units = variable_dict[key]['units']
    colourmap = variable_dict[key]['cmap']
    var_abbrev = variable_dict[key]['abbrev']

    contour_fig_name = 'C:\\Users\\HourstonH\\Documents\\' \
                       'ctd_visualization\\png\\' \
                       '{}_ctd_contourf_{}.png'.format(station, var_abbrev)

    plot_2d(df_in, variable, units, station, colourmap,
            contour_fig_name, plot_contourf, y_lim)

# ----------------------Plot pcolormesh data----------------
# To compare against data gaps in contourf data

df_in = pd.read_csv(f)

for key in variable_dict.keys():
    print(key)
    variable = key
    units = variable_dict[key]['units']
    colourmap = variable_dict[key]['cmap']
    var_abbrev = variable_dict[key]['abbrev']

    contour_fig_name = 'C:\\Users\\HourstonH\\Documents\\' \
                       'ctd_visualization\\png\\' \
                       '{}_ctd_pcolormesh_{}.png'.format(station, var_abbrev)

    plot_2d(df_in, variable, units, station, colourmap,
            contour_fig_name, plot_pcolormesh)

# ----------------------Plot anomalies-----------------------

df_in = pd.read_csv(f)

# ddict = select_binned_data(df_in, variable, units,
#                            select_depths=np.array([5, 25, 50, 100, 200]))

for key in variable_dict.keys():
    variable = key
    units = variable_dict[key]['units']
    colourmap = variable_dict[key]['cmap']
    var_abbrev = variable_dict[key]['abbrev']

    anom_fig_name = 'C:\\Users\\HourstonH\\Documents\\ctd_visualization\\' \
                    'png\\{}_ctd_anomalies_{}_v3.png'.format(
                        station, var_abbrev)

    plot_anomalies(df_in, variable, units, station, anom_fig_name)

# ---------------------------scatter padded data--------------------------

df_in = pd.read_csv(f)

figname = 'C:\\Users\\HourstonH\\Documents\\ctd_visualization\\' \
              'png_noQC\\{}_ctd_qc_binned_padded_scatter.png'.format(station)

variable = 'Temperature'
units = variable_dict[variable]['units']
scatter_padded_data(df_in, figname, variable, units)

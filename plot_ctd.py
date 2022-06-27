import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def ctd_pcolor():
    fig, ax = plt.subplots()
    ax.pcolormesh()
    return


def select_by_depth(depth_data, profile_number, select_depth):
    # Find indices
    subsetter = np.where(
        select_depth - 0.5 < depth_data < select_depth + 0.5)[0]

    # Ensure that there is only one observation selected from
    # each profile
    num_profiles = len(np.unique(profile_number))

    if len(np.unique(profile_number[subsetter])) == num_profiles:
        return subsetter
    elif len(np.unique(profile_number[subsetter])) > num_profiles:
        # Find the observation depth closest to the select depth
        # Remove the other observation depth(s) that were farther
        pass
    else:
        return None


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

# variable, units, var_abbrev = ['Temperature', 'C', 'T']
variable, units, var_abbrev = ['Salinity', 'PSS-78', 'S']
station = 'P1'

fig_name = 'C:\\Users\\HourstonH\\Documents\\ctd_visualization\\png\\' \
           '{}_ctd_anomalies_{}.png'.format(station, var_abbrev)

df_in = pd.read_csv(f)

plot_anomalies(df_in, variable, units, station, fig_name)

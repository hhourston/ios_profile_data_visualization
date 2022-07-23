import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Scatter plot the QCed ctd data to check for diagonal gaps


def scatter_ctd(f, station, depth_lim=None):
    depth_to_plot, binned = ['Depth bin [m]', True]  # 'Depth [m]'
    # depth_to_plot, binned = ['Depth [m]', False]  # 'Depth [m]'

    df = pd.read_csv(f)



    # # Perform lat lon checks only
    # median_lat = np.median(df.loc[:, 'Latitude [deg N]'])
    # median_lon = np.median(df.loc[:, 'Longitude [deg E]'])
    #
    # print('Median {} lon and lat: {}, {}'.format(station, median_lon,
    #                                              median_lat))
    #
    # print('Min and max {} lat: {}, {}'.format(
    #     station, np.nanmin(df.loc[:, 'Latitude [deg N]']),
    #     np.nanmax(df.loc[:, 'Latitude [deg N]'])))
    #
    # print('Min and max {} lon: {}, {}'.format(
    #     station, np.nanmin(df.loc[:, 'Longitude [deg E]']),
    #     np.nanmax(df.loc[:, 'Longitude [deg E]'])))

    # latlon_mask = (df.loc[:, 'Latitude [deg N]'] > median_lat - 0.1) & \
    #               (df.loc[:, 'Latitude [deg N]'] < median_lat + 0.1) & \
    #               (df.loc[:, 'Longitude [deg E]'] > median_lon - 0.1) & \
    #               (df.loc[:, 'Longitude [deg E]'] < median_lon + 0.1)

    time_dt = pd.to_datetime(df.loc[:, 'Time']).to_numpy()

    plt.scatter(time_dt, df.loc[:, depth_to_plot], s=1,
                alpha=0.5)

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

    figname = 'C:\\Users\\HourstonH\\Documents\\ctd_visualization\\' \
              'png_noQC\\{}_ctd_qc_binned_scatter.png'.format(station)
    plt.savefig(figname)
    plt.close()


# '59'  # '42'  # 'GEO1'  # 'LBP3'  # 'LB08'  # 'P1'
stn = '59'  # 42

# f = 'C:\\Users\\HourstonH\\Documents\\ctd_visualization\\csv\\' \
#     '{}_ctd_data_qc.csv'.format(station)

fname = 'C:\\Users\\HourstonH\\Documents\\ctd_visualization\\' \
        'csv\\{}_ctd_data_binned_depth_dupl.csv'.format(stn)

for s in ['42']:  # ['GEO1', 'LB08', 'P1']:
    fname = 'C:\\Users\\HourstonH\\Documents\\ctd_visualization\\' \
            'csv\\{}_ctd_data_binned_depth_dupl.csv'.format(s)
    scatter_ctd(fname, s)   # , depth_lim=200)


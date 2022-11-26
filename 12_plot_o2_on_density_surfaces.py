import glob
import gsw
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt


OXY_UMOL_PER_ML = 44.661
M3_PER_L = 0.001


def do_scatter_plot(ax, sigma_theta, station: str, oxy_data, year_data,
                    marker_color):
    """
    plot the input data
    :param ax: matplotlib.axes.Axes class object
    :param sigma_theta: numeric; for this case 26.5, 26.7 or 26.9
    :param station: station name
    :param oxy_data: the dependent variable
    :param year_data: the independent variable
    :param marker_color: for the plot scatter points
    :return:
    """
    return ax.scatter(
            year_data,
            oxy_data,
            label='{} {}'.format(station, np.round(sigma_theta, 1)),
            marker='o',
            s=20,
            edgecolor='k',
            c=marker_color
        )


def format_scatter_plot(ax, station: str):
    """
    Format the scatter plot of oxygen on density surfaces vs time
    :param ax: matplotlib.axes.Axes class object
    :param station: station name
    :return: nothing
    """
    ybot, ytop = plt.ylim()
    ax.set_ylim(top=ytop + 30)  # Give enough space for legend
    # Major and minor ticks sticking inside and outside the axes
    # Set one minor tick between each major tick
    ax.minorticks_on()
    major_xticks = ax.get_xticks(minor=False)  # in data coords
    minor_xticks = major_xticks[:-1] + (major_xticks[1] - major_xticks[0]) / 2
    ax.set_xticks(ticks=minor_xticks, minor=True)
    major_yticks = ax.get_yticks(minor=False)  # in data coords
    minor_yticks = major_yticks[:-1] + (major_yticks[1] - major_yticks[0]) / 2
    ax.set_yticks(ticks=minor_yticks, minor=True)
    plt.tick_params(which='major', direction='inout')
    plt.tick_params(which='minor', direction='in')
    # Send grid lines to back
    ax.set_axisbelow(True)
    # Set labels
    ax.set_ylabel(r'Oxygen [$\mu$mol/kg]')
    ax.set_title('Annual average oxygen concentration at {}'.format(station))
    # Add legend with specific properties
    plt.legend(loc='upper center', ncol=3, edgecolor='k', framealpha=1)
    plt.tight_layout()
    return


def compute_fit(x, y, fit_degrees):
    """
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


def plot_avg_oxy_on_density_surfaces(df_file, png_name, station,
                                     include_fit=False, fit_deg=None,
                                     start_year=None):
    """
    Plot annually averaged oxygen on constant potential density anomaly
    surfaces.
    :param start_year: int or None
    :param df_file: file name of csv containing the data
    :param png_name: file name to be assigned to png of plot
    :param station: name of station
    :param include_fit: include best-fit line in plot, default False
    :param fit_deg: best-fit degrees of freedom, default 1 (straight line)
    :return:
    """
    df = pd.read_csv(df_file)

    fig, ax = plt.subplots()
    # Add grid lines
    plt.grid(color='lightgrey')

    for sigma_theta, c in zip(
            np.unique(df.loc[:, 'Potential density anomaly bin [kg/m]'])[::-1],
            ['r', 'y', 'b']):
        if start_year is not None:
            msk = np.where(
                (df['Potential density anomaly bin [kg/m]'] == sigma_theta) &
                (df['Year'] >= start_year)
            )[0]
        else:
            msk = np.where(
                df['Potential density anomaly bin [kg/m]'] == sigma_theta
            )[0]
        # Scatter the points
        year_data = df.loc[msk, 'Year'].to_numpy(dtype='int32')
        oxy_data = df.loc[msk, 'Average oxygen [umol/kg]'
                          ].to_numpy(dtype='float')

        do_scatter_plot(ax, sigma_theta, station, oxy_data, year_data, c)

        # Add best-fit line
        if include_fit:
            if len(msk) < 3:
                print('Warning: not enough points to plot best fit line',
                      'for {} density surface'.format(sigma_theta))
                continue
            x_linspace, y_hat_linspace = compute_fit(year_data, oxy_data,
                                                     fit_deg)
            ax.plot(x_linspace, y_hat_linspace, c=c)

    format_scatter_plot(ax, station)

    plt.savefig(png_name)
    plt.close(fig)
    return


def update_bill_oxy_plot(df_file_265: str, df_file_267: str,
                         df_file_269: str, station: str,
                         fit_deg: int, png_name: str):
    """

    :param df_file_265: csv file containing oxygen data at 26.5 density level
    :param df_file_267: csv file containing oxygen data at 26.7 density level
    :param df_file_269: csv file containing oxygen data at 26.9 density level
    :param station: name of station
    :param fit_deg: number of degrees to use for fitting a curve to the time series
    :param png_name: file name to use for output plot
    :return: nothing
    """
    # Version of the above function but for different df structure
    fig, ax = plt.subplots()
    # Add grid lines
    plt.grid(color='lightgrey')

    for sigma_theta, df_file, c in zip(
        [26.9, 26.7, 26.5],
        [df_file_269, df_file_267, df_file_265],
        ['r', 'y', 'b']
    ):
        df = pd.read_csv(df_file)

        # if 'O2 ann avg (umol/kg)' in df.columns:
        #     o2_avg_colname = 'O2 ann avg (umol/kg)'
        # elif 'O2 Ann Avg (umol/kg) ' in df.columns:
        #     o2_avg_colname = 'O2 Ann Avg (umol/kg) '
        # else:
        #     print('Oxygen data in umol/kg not found')
        o2_avg_colname = df.columns[15]
        print('avg oxy column name:', o2_avg_colname)

        mask_annual = pd.notna(df[o2_avg_colname])
        o2_avg_data = df.loc[mask_annual, o2_avg_colname]
        year_data = df.loc[mask_annual, 'Date'].astype(int)

        do_scatter_plot(ax, sigma_theta, station, o2_avg_data, year_data, c)

        # Add best-fit line
        if len(mask_annual) < 3:
            print('Warning: not enough points to plot best fit line',
                      'for {} density surface'.format(sigma_theta))
            continue
        x_linspace, y_hat_linspace = compute_fit(year_data, o2_avg_data,
                                                 fit_deg)
        ax.plot(x_linspace, y_hat_linspace, c=c)

    format_scatter_plot(ax, station)
    plt.savefig(png_name)
    plt.close(fig)
    return


# ------------------------------------------------------------------------

# Plot the oxygen on constant density surfaces

# parent_dir = 'C:\\Users\\HourstonH\\Documents\\charles\\' \
#              'line_P_data_products\\csv\\has_osd_ctd_flags\\'
# parent_dir = 'C:\\Users\\HourstonH\\Documents\\charles\\' \
#              'bottom_oxygen\\'
parent_dir = 'D:\\lineP\\csv_data\\'

# for each station: P4 and P26, LB08
stn = 'P4'
station_name = 'OSP' if stn == 'P26' else stn
# print(stn, station_name)
# data_types = 'CTD_BOT_CHE_OSD'
best_fit_degrees = 2 if stn == 'P4' else 1  # One for P26 and 2 for P4

first_year_to_plot = 1950 if stn == 'P4' else None

average_file = os.path.join(
    parent_dir,
    '11_annual_avg_on_dens_surfaces\\{}_data.csv'.format(stn))
# average_file = os.path.join(
#     parent_dir,
#     '11N_annual_avg_on_dens_surfaces\\{}_ctd_data_qc.csv'.format(stn))

plot_name = average_file.replace(
    '.csv',
    '_oxy_vs_pot_dens_anom_{}degfit_1950-2022.png'.format(best_fit_degrees))

# Why is best fit truncated early in Crawford and Pena (2020)? How to do?
plot_avg_oxy_on_density_surfaces(average_file, plot_name,
                                 station_name, True, best_fit_degrees,
                                 first_year_to_plot)

# # ---------------------------Bill data---------------------------------
# # uncomment to plot bill's data
# """
# bill_dir = 'C:\\Users\\HourstonH\\Documents\\charles\\' \
#            'line_P_data_products\\bill_crawford\\masked\\'
#
# p26_file = bill_dir + 'CrawfordPena Line P 1950-2015 26 oxy annual avg.csv'
#
# p4_file = bill_dir + 'CrawfordPena Line P 1950-2015 4849 oxy annual avg.csv'
#
# bill_p26_plot_name = bill_dir + 'CrawfordPena P26 1950-2015 oxy vs sigma-theta.png'
# bill_p4_plot_name = bill_dir + 'CrawfordPena P4 1950-2015 oxy vs sigma-theta.png'
#
# # P26 P4
# station_name = 'P26'
# plot_avg_oxy_on_density_surfaces(p26_file, bill_p26_plot_name,
#                                  station_name, include_fit=True, fit_deg=1)
#
# station_name = 'P4'
# plot_avg_oxy_on_density_surfaces(p4_file, bill_p4_plot_name,
#                                  station_name, include_fit=True, fit_deg=2)
# """
#
# bill_dir = 'C:\\Users\\HourstonH\\Documents\\charles\\' \
#            'line_P_data_products\\bill_crawford\\'
#
# bill_p4_files = glob.glob(bill_dir + '*4849*_Nov2022.csv')
# bill_p4_files.sort()
# bill_p26_files = glob.glob(bill_dir + '*26_Nov2022.csv')
# bill_p26_files.sort()
#
# indf = pd.read_csv(bill_p4_files[0])
#
# stn = 'P4'
# update_bill_oxy_plot(*bill_p4_files, stn, 2,
#                      bill_dir + f'CrawfordPena_{stn}_1950-2022_o2.png')
#
# stn = 'P26'
# update_bill_oxy_plot(*bill_p26_files, stn, 1,
#                      bill_dir + f'CrawfordPena_{stn}_1955-2022_o2.png')
#
# # # Check if any rows are missing oxygen data
# # print(indf.loc[pd.isna(indf['Ox (umol/kg) '])])
# # # Convert oxy units of these data to umol/kg
# # # Compute density from absolute salinity and conservative temperature
# # mask = pd.isna(indf['Ox (umol/kg) '])
# # pressure = gsw.p_from_z(-indf.loc[mask, 'Depth'].to_numpy(float),
# #                         indf.loc[mask, 'Latitude'].to_numpy(float))
# # density = gsw.rho(indf.loc[mask, 'Absolute Salinity'].to_numpy(float),
# #                   indf.loc[mask, 'Conservative Temperature'].to_numpy(float),
# #                   pressure)
# # indf.loc[mask, 'Ox (umol/kg) '] = [
# #     o * OXY_UMOL_PER_ML / (d * M3_PER_L)
# #     for o, d in zip(indf.loc[mask, 'Ox (ml/L)'].to_numpy(float),
# #                     density)
# # ]
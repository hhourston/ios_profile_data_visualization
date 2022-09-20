import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt


def plot_avg_oxy_on_density_surfaces(df_file, png_name, station,
                                     include_fit=False, fit_deg=None):
    """
    Plot annually averaged oxygen on constant potential density anomaly
    surfaces.
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
        msk = np.where(
            df.loc[:, 'Potential density anomaly bin [kg/m]'] == sigma_theta
        )[0]
        # Scatter the points
        year_data = df.loc[msk, 'Year'].to_numpy(dtype='int32')
        oxy_data = df.loc[msk, 'Average oxygen [umol/kg]'
                          ].to_numpy(dtype='float')
        ax.scatter(
            year_data,
            oxy_data,
            label='{} {}'.format(station, np.round(sigma_theta, 1)),
            marker='o',
            s=20,
            edgecolor='k',
            c=c
        )
        # Add best-fit line
        if include_fit:
            x_values_sorted = np.array(sorted(year_data))
            y_values_sorted = np.array([i for _, i in
                                        sorted(zip(year_data, oxy_data))])
            # Remove any nans otherwise polynomial crashes
            x_values_sorted = x_values_sorted[~np.isnan(y_values_sorted)]
            y_values_sorted = y_values_sorted[~np.isnan(y_values_sorted)]
            # Update polynomial module access from legacy access
            poly = np.polynomial.Polynomial.fit(
                x_values_sorted, y_values_sorted, deg=fit_deg)
            # coeffs = poly.coef
            # fit_eqn = np.polynomial.Polynomial(coeffs[::-1]) # must reverse order coeffs
            # y_hat_sorted = fit_eqn(x_values_sorted)
            # ax.plot(x_values_sorted, y_hat_sorted, c=c)
            x_linspace, y_hat_linspace = poly.linspace(n=100)
            ax.plot(x_linspace, y_hat_linspace, c=c)
            # numpy.linalg.LinAlgError: SVD did not converge in Linear Least Squares for OSP

    ybot, ytop = plt.ylim()
    ax.set_ylim(top=ytop + 30)  # Give enough space for legend
    # Major and minor ticks sticking inside and outside the axes
    # Set one minor tick between each major tick
    ax.minorticks_on()
    major_xticks = ax.get_xticks(minor=False)  # in data coords
    minor_xticks = major_xticks[:-1] + (major_xticks[1] - major_xticks[0])/2
    ax.set_xticks(ticks=minor_xticks, minor=True)
    major_yticks = ax.get_yticks(minor=False)  # in data coords
    minor_yticks = major_yticks[:-1] + (major_yticks[1] - major_yticks[0]) / 2
    ax.set_yticks(ticks=minor_yticks, minor=True)
    plt.tick_params(which='major', direction='inout')
    plt.tick_params(which='minor', direction='in')
    # Send grid lines to back
    ax.set_axisbelow(True)
    # Set labels
    ax.set_ylabel('Oxygen [umol/kg]')
    ax.set_title('Annual average oxygen concentration at {}'.format(station))
    # Add legend with specific properties
    plt.legend(loc='upper center', ncol=3, edgecolor='k', framealpha=1)

    plt.tight_layout()

    plt.savefig(png_name)
    plt.close(fig)
    return


# ------------------------------------------------------------------------

# Plot the oxygen on constant density surfaces

parent_dir = 'C:\\Users\\HourstonH\\Documents\\charles\\' \
             'line_P_data_products\\csv\\has_osd_ctd_flags\\'
# parent_dir = 'C:\\Users\\HourstonH\\Documents\\charles\\' \
#              'bottom_oxygen\\'

# for each station: P4 and P26, LB08
stn = 'P26'
# station_name = stn
station_name = 'OSP'
# print(stn, station_name)
# data_types = 'CTD_BOT_CHE_OSD'
best_fit_degrees = 1  # One for P26 and 2 for P4

average_file = os.path.join(
    parent_dir,
    '11N_annual_avg_on_dens_surfaces\\{}_data.csv'.format(stn))
plot_name = average_file.replace(
    '.csv',
    '_oxy_vs_pot_dens_anom_{}degfit.png'.format(best_fit_degrees))

# Why is best fit truncated early in Crawford and Pena (2020)? How to do?
plot_avg_oxy_on_density_surfaces(average_file, plot_name,
                                 station_name, True, best_fit_degrees)


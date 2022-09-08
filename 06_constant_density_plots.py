import pandas as pd
import numpy as np
import gsw
import os
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

# Charles Line P data product request


def calculate_density(
    temperature_C, salinity_SP, pressure_dbar, longitude, latitude
):
    """author: James Hannah (with some modifications by Hana)
    """
    assumed = False
    if all(x is not None for x in [temperature_C, salinity_SP, pressure_dbar]):
        # Calculate Absolute Salinity from Practical Salinity
        salinity_SA = gsw.SA_from_SP(salinity_SP, pressure_dbar, longitude, latitude)
        # Calculate Conservative Temperature of seawater from in-situ temperature
        temperature_conserv = gsw.CT_from_t(salinity_SA, temperature_C, pressure_dbar)
        # Compute density in kg/m
        density = gsw.rho(
            salinity_SA,
            temperature_conserv,
            pressure_dbar,
        )
        # print(len(density))
    else:
        density = []
    # if len(density) == 0:
    #     print(
    #         "Not enough data to accurately compute density. Calculating density as though all values are 0"
    #     )
        assumed = True
        # density = np.repeat(np.nan, )
    #     # density = np.full(length, gsw.rho([0], [0], 0)[0])
    return density, assumed


def calculate_pot_dens_anom(
        temperature_C, salinity_SP, pressure_dbar, longitude, latitude
):
    # Calculate potential density anomalies

    # Calculate Absolute Salinity from Practical Salinity
    salinity_SA = gsw.SA_from_SP(salinity_SP, pressure_dbar,
                                 longitude, latitude)
    # Calculate Conservative Temperature of seawater from in-situ temperature
    temperature_conserv = gsw.CT_from_t(salinity_SA, temperature_C,
                                        pressure_dbar)

    # sigma1: Calculates potential density anomaly with reference
    # pressure of 1000 dbar, this being this particular potential
    # density minus 1000 kg/m^3
    pot_dens_anom = gsw.sigma0(salinity_SA, temperature_conserv)
    return pot_dens_anom


def interp_oxy_to_density_surfaces(df_file_name, out_df_name,
                                   select_densities):
    # Interpolate oxygen to constant density surfaces
    # Select density surfaces defined in densities parameter
    # Returns oxygen interpolated to select density surfaces

    # NOTE: Calling interp1d with NaNs present in input values
    # results in undefined behaviour. Data must be checked prior
    # to using this function.
    # Do interpolation for each profile
    in_df = pd.read_csv(df_file_name)
    profile_start_idx = np.unique(in_df.loc[:, 'Profile number'],
                                  return_index=True)[1]

    # Minus 1 to account for pandas inclusive indexing
    profile_end_idx = np.concatenate((profile_start_idx[1:] - 1,
                                      np.array([len(in_df)])))

    # Initialize dataframe to hold interpolated data
    # Need time, density, oxygen value
    out_df_columns = ['Profile number', 'Time',
                      'Potential density anomaly level [kg/m]',
                      'Oxygen interpolated [umol/kg]']
    out_df = pd.DataFrame(columns=out_df_columns)

    for i in range(len(profile_start_idx)):
        st = profile_start_idx[i]
        en = profile_end_idx[i]
        if all(np.isnan(in_df.loc[st:en, 'Oxygen [umol/kg]'].to_numpy(float))):
            # Skip the profile
            continue
        else:
            # If x_new is outside the interpolation range, use fill_value
            # and do not extrapolate or raise ValueError
            interp_fn = interp1d(
                in_df.loc[st:en, 'Potential density anomaly [kg/m]'].to_numpy(float),
                in_df.loc[st:en, 'Oxygen [umol/kg]'].to_numpy(float),
                kind='linear',
                bounds_error=False,
                fill_value=np.nan
            )
            oxy_interpolated = interp_fn(select_densities)

            # Append results to dataframe
            df_append = pd.DataFrame(
                np.array([np.repeat(in_df.loc[st, 'Profile number'],
                                    len(select_densities)),
                          np.repeat(in_df.loc[st, 'Time'],
                                    len(select_densities)),
                          select_densities,
                          oxy_interpolated
                          ]).T,
                columns=out_df_columns
            )

            out_df = pd.concat((out_df, df_append))

            out_df.reset_index(inplace=True, drop=True)

    # Print summary statistics

    # print(len(out_df))
    # print(out_df)
    # to_numpy converts Series to numpy object type array
    # unless dtype is specified
    print(np.nanmin(
        out_df.loc[:, 'Oxygen interpolated [umol/kg]'].to_numpy(float)))
    print(np.nanmax(
        out_df.loc[:, 'Oxygen interpolated [umol/kg]'].to_numpy(float)))

    # out_df.drop(columns='index', inplace=True)

    out_df.to_csv(out_df_name, index=False)
    return


def annual_avg_on_density_surfaces(df_file: str, output_file_name: str):
    df = pd.read_csv(df_file)
    # Convert time to pandas datetime
    df['Datetime'] = pd.to_datetime(df.loc[:, 'Time'])
    years_available = np.sort(np.unique(df.loc[:, 'Datetime'].dt.year))

    # Initialize dataframe to hold annual averages
    density_column = 'Potential density anomaly level [kg/m]'
    df_avg = pd.DataFrame(
        columns=['Year', density_column,
                 'Average interpolated oxygen [umol/kg]'])

    # Take the average for each year and density level
    for i in range(len(years_available)):
        for sigma_theta in df.loc[:2, density_column]:
            indexer = np.where(
                (df.loc[:, 'Datetime'].dt.year == years_available[i]) &
                (df.loc[:, density_column] == sigma_theta)
            )[0]
            df_avg.loc[len(df_avg)] = [
                years_available[i], sigma_theta, np.nanmean(
                    df.loc[indexer,
                           'Oxygen interpolated [umol/kg]'].to_numpy(
                        float))
            ]

    df_avg.to_csv(output_file_name, index=False)
    return


def plot_avg_oxy_on_density_surfaces(df_file, png_name, station):
    df = pd.read_csv(df_file)

    fig, ax = plt.subplots()
    # Add grid lines
    plt.grid()

    for sigma_theta, c in zip(
            np.unique(df.loc[:, 'Potential density anomaly level [kg/m]'])[::-1],
            ['r', 'y', 'b']):
        msk = np.where(
            df.loc[:, 'Potential density anomaly level [kg/m]'] == sigma_theta
        )[0]
        # Scatter the points
        ax.scatter(
            df.loc[msk, 'Year'].to_numpy(dtype='int32'),
            df.loc[msk, 'Average interpolated oxygen [umol/kg]'].to_numpy(dtype='float'),
            label='{} {}'.format(station, np.round(sigma_theta, 1)),
            marker='o',
            s=20,
            edgecolor='k',
            c=c
        )
        # TODO Add best-fit line to scatter points as in for Burke Inlet
        # Convert time to seconds first
        # time_ts = time_pd
        # z = np.polyfit()
    ybot, ytop = plt.ylim()
    ax.set_ylim(top=ytop + 30)  # Give enough space for legend
    ax.set_ylabel('Oxygen [umol/kg]')
    ax.set_title('Annual average oxygen concentration at {}'.format(station))
    # Add legend with specific properties
    plt.legend(loc='upper center', ncol=3, edgecolor='k', framealpha=1)

    plt.tight_layout()

    plt.savefig(png_name)
    plt.close(fig)
    return


# ---------------------------------------------------------------------
# Compute density at each observation level
# Interpolate oxygen data onto constant density surfaces
# Selected density surfaces: 1026.5 to 1026.9 kg/m^3
# Make a plot with all three oxygen vs density surface on it
# for each station: P4 and P26
stn = 'P26'
# station_name = stn
station_name = 'OSP'

in_dir = 'C:\\Users\\HourstonH\\Documents\\charles\\' \
         'line_P_data_products\\csv\\02b_inexact_duplicate_check\\'
in_file = os.path.join(
    in_dir, '{}_CTD_BOT_CHE_data.csv'.format(stn))

density_dir = 'C:\\Users\\HourstonH\\Documents\\charles\\' \
          'line_P_data_products\\csv\\06_potential_density_anomalies\\'
density_file = os.path.join(density_dir, os.path.basename(in_file))

df_in = pd.read_csv(in_file)

# Compute pressure from depth
# Depth must be converted to positive up!
df_in['Pressure [dbar]'] = gsw.p_from_z(
    -df_in.loc[:, 'Depth [m]'].to_numpy(float),
    df_in.loc[:, 'Latitude [deg N]'].to_numpy(float)
)

# # Compute density
# # PSS-78 salinity units: Practical Salinity, unitless
# in_df['Density [kg/m]'] = calculate_density(
#     in_df.loc[:, 'Temperature [C]'].to_numpy(),
#     in_df.loc[:, 'Salinity [PSS-78]'].to_numpy(),
#     in_df.loc[:, 'Pressure [dbar]'].to_numpy(),
#     in_df.loc[:, 'Longitude [deg E]'].to_numpy(),
#     in_df.loc[:, 'Latitude [deg N]'].to_numpy()
# )[0]
#
# # Print summary statistics
# print(np.nanmin(in_df.loc[:, 'Density [kg/m]']))
# print(np.nanmax(in_df.loc[:, 'Density [kg/m]']))

# Compute potential density anomaly
# PSS-78 salinity units: Practical Salinity, unitless
df_in['Potential density anomaly [kg/m]'] = calculate_pot_dens_anom(
    df_in.loc[:, 'Temperature [C]'].to_numpy(float),
    df_in.loc[:, 'Salinity [PSS-78]'].to_numpy(float),
    df_in.loc[:, 'Pressure [dbar]'].to_numpy(float),
    df_in.loc[:, 'Longitude [deg E]'].to_numpy(float),
    df_in.loc[:, 'Latitude [deg N]'].to_numpy(float)
)

# pot_dens = gsw.pot_rho_t_exact(
#     salinity_SA,
#     df_in.loc[:, 'Temperature [C]'].to_numpy(float),
#     df_in.loc[:, 'Pressure [dbar]'].to_numpy(float), p_ref=0)

# sigma1: Calculates potential density anomaly with reference
# pressure of 1000 dbar, this being this particular potential
# density minus 1000 kg/m^3

# Print summary statistics
print(np.nanmin(df_in.loc[:, 'Potential density anomaly [kg/m]']))
print(np.nanmax(df_in.loc[:, 'Potential density anomaly [kg/m]']))

df_in.to_csv(density_file, index=False)

# ------------------------------------------------------------------------

# densities = np.array([1026.5, 1026.7, 1026.9])

potential_densities = np.array([26.5, 26.7, 26.9])

# # -----Test-----
# idx = np.max(in_df.loc[:, 'Profile number'])
#
# prof_idx = np.where(in_df.loc[:, 'Profile number'] == idx)[0]
#
# interp_fn = interp1d(
#     in_df.loc[prof_idx, 'Density [kg/m]'].to_numpy(),
#     in_df.loc[prof_idx, 'Oxygen [mL/L]'].to_numpy(),
#     kind='linear'
# )
# oxy_interpolated = interp_fn(densities)
# # --------------

# Perform the interpolation on each profile
interp_dir = 'C:\\Users\\HourstonH\\Documents\\charles\\' \
             'line_P_data_products\\csv\\' \
             '07_interpolate_to_pot_dens_anom_surfaces\\'
interp_file = os.path.join(interp_dir, os.path.basename(in_file))

interp_oxy_to_density_surfaces(density_file, interp_file,
                               potential_densities)

# ------------------------------------------------------------------
# Annual averaging on each density surface
average_dir = 'C:\\Users\\HourstonH\\Documents\\charles\\' \
              'line_P_data_products\\csv\\' \
              '08_annual_avg_on_dens_surfaces\\'
average_file = os.path.join(average_dir, os.path.basename(in_file))
annual_avg_on_density_surfaces(interp_file, average_file)

# ------------------------------------------------------------------
# Plot the oxygen on constant density surfaces

stn = 'P26'
# station_name = stn
station_name = 'OSP'

average_file = 'C:\\Users\\HourstonH\\Documents\\charles\\' \
               'line_P_data_products\\csv\\' \
               '08_annual_avg_on_dens_surfaces\\P26_CTD_BOT_CHE_data.csv'
plot_name = average_file.replace('.csv', '_oxy_vs_pot_dens_anom.png')
plot_avg_oxy_on_density_surfaces(average_file, plot_name, station_name)


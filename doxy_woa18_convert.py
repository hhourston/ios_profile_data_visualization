import gsw
import numpy as np
import pandas as pd

# Convert the ranges DOXY Coastal Pacific table from umol/kg to mL/L
f_in = 'C:\\Users\\HourstonH\\Documents\\NEP_climatology\\' \
       'literature\\WOA docs\\wod18_users_manual_tables\\' \
       'wod18_ranges_DOXY_Coast_N_Pac_ml_l.csv'

df_in = pd.read_csv(f_in)

# Calculate the pressure from depth using gsw
# Choose latitude and longitude value
lat = 48
lon = -125

# z = -depth
df_in['sea_pressure_min [dbar]'] = gsw.p_from_z(
    -df_in.loc[:, 'Depth_min'].to_numpy(), lat)
df_in['sea_pressure_max [dbar]'] = gsw.p_from_z(
    -df_in.loc[:, 'Depth_max'].to_numpy(), lat)
df_in['sea_pressure_mean [dbar]'] = np.mean(
    [df_in['sea_pressure_min [dbar]'].to_numpy(),
     df_in['sea_pressure_max [dbar]'].to_numpy()], axis=0)

# Calculate Absolute Salinity from Practical Salinity
# What pressure values to use? min or max or mean or other?
df_in['salinity_SA_min'] = gsw.SA_from_SP(
    df_in['S_Coast_N_Pac_min'].to_numpy(),
    df_in['sea_pressure_mean [dbar]'].to_numpy(), lon, lat)
df_in['salinity_SA_max'] = gsw.SA_from_SP(
    df_in['S_Coast_N_Pac_max'].to_numpy(),
    df_in['sea_pressure_mean [dbar]'].to_numpy(), lon, lat)
# df_in['salinity_SA_mean'] = np.mean(
#     [df_in['salinity_SA_min'].to_numpy(),
#      df_in['salinity_SA_max'].to_numpy()], axis=0)

# Calculate density using absolute salinity
df_in['density_min'] = gsw.rho(df_in['salinity_SA_min'].to_numpy(),
                               df_in['T_Coast_N_Pac_max'].to_numpy(),
                               df_in['sea_pressure_min [dbar]'].to_numpy())
df_in['density_max'] = gsw.rho(df_in['salinity_SA_max'].to_numpy(),
                               df_in['T_Coast_N_Pac_min'].to_numpy(),
                               df_in['sea_pressure_max [dbar]'].to_numpy())

# Check that max > min density
diffs = df_in['density_max'] - df_in['density_min']
print(np.all(diffs > 0))

oxygen_umol_per_ml = 44.661
metre_cube_per_litre = 0.001

# Finally convert oxygen
df_in['Coast_N_Pac_min'] = [o * d / oxygen_umol_per_ml * metre_cube_per_litre
                            for o, d in
                            zip(df_in['O_Coast_N_Pac_min_umol_kg'].to_numpy(),
                                df_in['density_min'].to_numpy())]

df_in['Coast_N_Pac_max'] = [o * d / oxygen_umol_per_ml * metre_cube_per_litre
                            for o, d in
                            zip(df_in['O_Coast_N_Pac_max_umol_kg'].to_numpy(),
                                df_in['density_max'].to_numpy())]

f_out = 'C:\\Users\\HourstonH\\Documents\\NEP_climatology\\' \
       'literature\\WOA docs\\wod18_users_manual_tables\\' \
       'wod18_ranges_DOXY_Coast_N_Pac_ml_l_filled.csv'

df_in.to_csv(f_out, index=False)

import pandas as pd
import os
import numpy as np
from haversine import haversine

P4_COORDINATES = (48+39/60, -(126+40/60))
P4_SEARCH_RADIUS = max(
    [24.65056612250387, 37.00682860346698]
)/2  # 1/2 the distance from p4 to p5; distance (p3 to p4) < distance (p4 to p5)

parent_dir = 'D:\\lineP\\P4_raw_data\\meds\\'

infile = parent_dir + 'a_MEDS_profile_prof.csv'

indf = pd.read_csv(infile, na_values='NaN')

indf.drop(columns=['FLU1', 'Q_FLU1', 'NTRA', 'Q_NTRA', 'PHOS', 'Q_PHOS',
                   'SLCA', 'Q_SLCA'],
          inplace=True)
mask_out_ios = indf['SOURCE_ID'] != 'IOS '
print(len(indf), sum(mask_out_ios))

indf = indf.loc[mask_out_ios, :]
indf.reset_index(drop=True, inplace=True)

# Get df into the right format
# Will need to merge rows with same lat/lon/time/depth
# since doxy is usually in a different row than corresponding TS
# for each oxygen measurement, check for matching TS
mask_has_o2 = ~pd.isna(indf['DOXY'])
print(sum(mask_has_o2))

outdf = pd.DataFrame(columns=indf.columns)

# # Iterate through non-nan oxygen observations
# for i in indf.index[mask_has_o2]:
#     if pd.notna(indf.loc[i, 'TEMP']) and (pd.notna(indf.loc[i, 'SSAL'])
#                                           or pd.notna(indf.loc[i, 'PSAL'])):
#         outdf.loc[len(outdf)] = indf.loc[i, :]
#     else:
#         Y, M, D, t, lon, lat, dp = indf.loc[
#             i,
#             ['OBS_YEAR', 'OBS_MONTH', 'OBS_DAY', 'OBS_TIME',
#              'LONGITUDE (+E)', 'LATITUDE (+N)', 'DEPTH_PRESS']
#         ]
#         matching_rows_mask = ((indf['OBS_YEAR'] == Y) &
#                               (indf['OBS_MONTH'] == M) &
#                               (indf['OBS_DAY'] == D) &
#                               (indf['OBS_TIME'] == t) &
#                               (indf['LONGITUDE (+E)'] == lon) &
#                               (indf['LATITUDE (+N)'] == lat) &
#                               (indf['DEPTH_PRESS'] == dp) &
#                               (~pd.isna(indf['TEMP'])) &
#                               (~pd.isna(indf['PSAL']) | ~pd.isna(indf['SSAL']))
#                               )
#         # Check if there are any matches other than the i-th row we're on
#         # id_vars = ['OBS_YEAR', 'OBS_MONTH', 'OBS_DAY', 'OBS_TIME',
#         #            'LONGITUDE (+E)', 'LATITUDE (+N)']
#         id_vars = indf.columns[:-10]
#         melted = indf.melt(id_vars=id_vars).dropna()
#         pivoted = melted.pivot_table(index=id_vars, columns="variable", values="value")
#         if sum(matching_rows_mask) > 1:
#             # Merge all the non-nan entries into one row
#             pass

# Do lat/lon check before trying to solve problem of mulitiple rows for 1 cast
distances = np.array(
    [haversine((lat_i, lon_i), P4_COORDINATES)
     for lat_i, lon_i in zip(indf.loc[:, 'LATITUDE (+N)'],
                             indf.loc[:, 'LONGITUDE (+E)'])])
latlon_mask = distances <= P4_SEARCH_RADIUS

print(sum(latlon_mask & mask_has_o2))  # 0, so no meds o2 data for P4 :(

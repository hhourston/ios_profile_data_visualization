import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from haversine import haversine
import os


def plot_stn_coords(lon, lat, figname: str, station: str,
                    lon_rng: tuple, lat_rng: tuple, year_rng: tuple,
                    lineP_df=None, true_coords=None):
    fig, ax = plt.subplots(figsize=(6, 4.5))
    plt.grid(color='lightgrey')
    ax.scatter(lon, lat, s=3)
    if lineP_df is not None:
        ax.scatter(lineP_df.loc[:, 'Lon ddegrees E'],
                   lineP_df.loc[:, 'Lat ddegrees N'], marker='*', c='k',
                   label='Nominal Line P coordinates')
        for stn in ['P4', 'P12', 'P20', 'P26']:
            # ax.annotate(stn, (lineP_df.loc[stn, 'Lon ddegrees E'],
            #                   lineP_df.loc[stn, 'Lat ddegrees N'])
            #             )
            ax.text(lineP_df.loc[stn, 'Lon ddegrees E'],
                    lineP_df.loc[stn, 'Lat ddegrees N'], stn,
                    rotation=45)
    if true_coords is not None:
        ax.scatter([true_coords[0]], [true_coords[1]], marker='*', c='r',
                   label=f'Nominal {station} position')
        # ax.scatter('',
        #            marker='*', c='chartreuse', label='Nominal Line P coordinates')
        plt.legend(loc='upper right')
    ax.set(xlim=lon_rng, ylim=lat_rng)
    ax.set_aspect('equal')  # 'box')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title(
        f'Crawford & Pena {station} observations, {year_rng[0]}-{year_rng[1]}')
    plt.tight_layout()
    plt.savefig(figname)
    plt.close(fig)
    return


parent_dir = 'C:\\Users\\HourstonH\\Documents\\charles\\' \
             'line_P_data_products\\bill_crawford\\'

# p4_file = parent_dir + 'CrawfordPena Line P 1950-2019 4849-5.csv'
# p4_file = parent_dir + 'CrawfordPena Line P 1950-2019 4849-7.csv'
p4_file = parent_dir + 'CrawfordPena Line P 1950-2019 4849-9.csv'

# p4_df = pd.read_csv(p4_265_file, nrows=1615, skip_blank_lines=True)
# p4_df = pd.read_csv(p4_file, nrows=861, skip_blank_lines=True)
p4_df = pd.read_csv(p4_file, nrows=662, skip_blank_lines=True)

"""
print(p4_265_df.columns)
print(np.min(p4_265_df.loc[:, 'Longitude']))
print(np.max(p4_265_df.loc[:, 'Longitude']))
print(np.max(p4_265_df.loc[:, 'Longitude']) - np.min(p4_265_df.loc[:, 'Longitude']))

print(np.min(p4_265_df.loc[:, 'Latitude']))
print(np.max(p4_265_df.loc[:, 'Latitude']))
print(np.max(p4_265_df.loc[:, 'Latitude']) - np.min(p4_265_df.loc[:, 'Latitude']))

p4_nodc_mask = np.array(['NODC' in fname for fname in
                         p4_265_df.loc[:, 'File'].astype(str)])

print(np.min(p4_265_df.loc[p4_nodc_mask, 'Longitude']))
print(np.max(p4_265_df.loc[p4_nodc_mask, 'Longitude']))
print(np.max(p4_265_df.loc[p4_nodc_mask, 'Longitude']) - np.min(p4_265_df.loc[p4_nodc_mask, 'Longitude']))

print(np.min(p4_265_df.loc[p4_nodc_mask, 'Latitude']))
print(np.max(p4_265_df.loc[p4_nodc_mask, 'Latitude']))
print(np.max(p4_265_df.loc[p4_nodc_mask, 'Latitude']) - np.min(p4_265_df.loc[p4_nodc_mask, 'Latitude']))

print(np.min(p4_265_df.loc[~p4_nodc_mask, 'Longitude']))
print(np.max(p4_265_df.loc[~p4_nodc_mask, 'Longitude']))
print(np.max(p4_265_df.loc[~p4_nodc_mask, 'Longitude']) - np.min(p4_265_df.loc[~p4_nodc_mask, 'Longitude']))

print(np.min(p4_265_df.loc[~p4_nodc_mask, 'Latitude']))
print(np.max(p4_265_df.loc[~p4_nodc_mask, 'Latitude']))
print(np.max(p4_265_df.loc[~p4_nodc_mask, 'Latitude']) - np.min(p4_265_df.loc[~p4_nodc_mask, 'Latitude']))
"""

lineP_coord_file = parent_dir + 'lineP_coordinates_ddegrees.csv'
lineP_coord_df = pd.read_csv(lineP_coord_file, index_col=[0])

nominal_p4_coords = (-(126 + 40.0 / 60), 48 + 39.0 / 60)
p4_coords_png = parent_dir + 'p4_coords_1950-2015.png'
plot_stn_coords(p4_df.loc[:, 'Longitude'],
                p4_df.loc[:, 'Latitude'], p4_coords_png, 'P4',
                (-146, -123), (44, 54), (1950, 2015),
                lineP_coord_df,
                true_coords=nominal_p4_coords)

# ----------------------------------------------------------------
# p26_file = parent_dir + 'CrawfordPena Line P 1950-2019 P526.csv'
# p26_file = parent_dir + 'CrawfordPena Line P 1950-2019 P726.csv'
p26_file = parent_dir + 'CrawfordPena Line P 1950-2019 P926.csv'

# p26_df = pd.read_csv(p26_file, nrows=1833)
# p26_df = pd.read_csv(p26_file, nrows=1592)
p26_df = pd.read_csv(p26_file, nrows=1435)

"""
print(np.min(p26_265_df.loc[:, 'Longitude']))
print(np.max(p26_265_df.loc[:, 'Longitude']))
print(np.max(p26_265_df.loc[:, 'Longitude']) - np.min(p26_265_df.loc[:, 'Longitude']))

print(np.min(p26_265_df.loc[:, 'Latitude']))
print(np.max(p26_265_df.loc[:, 'Latitude']))
print(np.max(p26_265_df.loc[:, 'Latitude']) - np.min(p26_265_df.loc[:, 'Latitude']))
"""

nominal_p26_coords = (-145, 50)
p26_coords_png = parent_dir + 'P26_coords_1955-2015.png'
plot_stn_coords(p26_df.loc[:, 'Longitude'],
                p26_df.loc[:, 'Latitude'], p26_coords_png, 'P26',
                (-150, -123), (44, 56), (1955, 2015), lineP_coord_df,
                nominal_p26_coords)

"""
lineP_coord_file = parent_dir + 'lineP_coordinates.txt'
lineP_coord_df = pd.read_table(lineP_coord_file, skiprows=1)
lineP_coord_df['Lat ddegrees N'] = [
    float(x.split('°')[0]) + float(x.split('°')[1])/60
    for x in lineP_coord_df.loc[:, 'Latitude N']]
lineP_coord_df['Lon ddegrees E'] = [
    -float(x.split('°')[0]) - float(x.split('°')[1])/60
    for x in lineP_coord_df.loc[:, 'Longitude W']]

lineP_coord_df.to_csv(parent_dir + 'lineP_coordinates_ddegrees.csv',
                      index=False)
"""

# Do subsetting
# Compute distance between p3 and p4 and between p4 and p5
# Returned units are km
km_to_decimal_degrees = 1 / 111

p3_to_p4 = haversine(
    (lineP_coord_df.loc['P3', 'Lat ddegrees N'],
     lineP_coord_df.loc['P3', 'Lon ddegrees E']),
    (lineP_coord_df.loc['P4', 'Lat ddegrees N'],
     lineP_coord_df.loc['P4', 'Lon ddegrees E'])
)
p4_to_p5 = haversine(
    (lineP_coord_df.loc['P4', 'Lat ddegrees N'],
     lineP_coord_df.loc['P4', 'Lon ddegrees E']),
    (lineP_coord_df.loc['P5', 'Lat ddegrees N'],
     lineP_coord_df.loc['P5', 'Lon ddegrees E'])
)
p35_to_p26 = haversine(
    (lineP_coord_df.loc['P35', 'Lat ddegrees N'],
     lineP_coord_df.loc['P35', 'Lon ddegrees E']),
    (lineP_coord_df.loc['P26', 'Lat ddegrees N'],
     lineP_coord_df.loc['P26', 'Lon ddegrees E'])
)
print(p3_to_p4, p4_to_p5, p35_to_p26)
# 24.65056612250387 37.00682860346698 49.793944479268674

"""
# Search area to use around OSP: OSP lat, lon +/- p35_to_p26/2
lat0, lat1 = [
    lineP_coord_df.loc['P26', 'Lat ddegrees N'] - p35_to_p26/2 * km_to_decimal_degrees,
    lineP_coord_df.loc['P26', 'Lat ddegrees N'] + p35_to_p26/2 * km_to_decimal_degrees]
lon0, lon1 = [
    lineP_coord_df.loc['P26', 'Lon ddegrees E'] - p35_to_p26/2 * km_to_decimal_degrees,
    lineP_coord_df.loc['P26', 'Lon ddegrees E'] + p35_to_p26/2 * km_to_decimal_degrees]
print(lon0, lon1, lat0, lat1)
# -145.22429704720392 -144.77570295279608 49.77570295279609 50.22429704720391

# So search within [-145.5, -144.5] and [49.5, 50.5] for TSO
"""

p4_radius_strict = max([p3_to_p4, p4_to_p5]) / 2
p26_radius_strict = p35_to_p26 / 2
print(p4_radius_strict, p26_radius_strict)

# 24 km limit used in Cummins & Ross (2020)
p4_distances = np.array(
    [haversine((lat_i, lon_i),
               (lineP_coord_df.loc['P4', 'Lat ddegrees N'],
                lineP_coord_df.loc['P4', 'Lon ddegrees E']))
     for lat_i, lon_i in zip(p4_df.loc[:, 'Latitude'],
                             p4_df.loc[:, 'Longitude'])])
p4_latlon_mask = p4_distances <= p4_radius_strict

p26_distances = np.array(
    [haversine((lat_i, lon_i),
               (lineP_coord_df.loc['P26', 'Lat ddegrees N'],
                lineP_coord_df.loc['P26', 'Lon ddegrees E']))
     for lat_i, lon_i in zip(p26_df.loc[:, 'Latitude'],
                             p26_df.loc[:, 'Longitude'])])
p26_latlon_mask = p26_distances <= p26_radius_strict

# Save the masks to a csv file so as to not have to remake them
columns_to_ignore = ['Avg Ox%', 'Avg D', 'Avg T', 'Avg S', 'Avg O2 (ml/L) ',
                     'Avg O2 umol/kg', 'MS Ox%', 'MS D', 'MS T', 'MS S',
                     'MS O2 (ml/L) ', 'MS O2 umol/kg']

p4_output_df = p4_df.drop(columns_to_ignore, axis=1)
p4_output_df['is_close_to_station'] = p4_latlon_mask
p4_mask_filename = p4_file.replace('.csv', '_masked.csv')
p4_mask_filename = os.path.join(
    os.path.dirname(p4_mask_filename), 'masked',
    os.path.basename(p4_mask_filename.replace('2019', '2015')))
p4_output_df.to_csv(p4_mask_filename, index=False)

p26_output_df = p26_df.copy(deep=True)
p26_output_df['is_close_to_station'] = p26_latlon_mask
p26_mask_filename = p26_file.replace('.csv', '_masked.csv')
p26_mask_filename = os.path.join(
    os.path.dirname(p26_mask_filename), 'masked',
    os.path.basename(p26_mask_filename.replace('2019', '2015')))
p26_output_df.to_csv(p26_mask_filename, index=False)

# ------------------------------------------------------------------------------

# Plot the masked data

input_dir = 'C:\\Users\\HourstonH\\Documents\\charles\\' \
            'line_P_data_products\\bill_crawford\\'
p4_265_file = input_dir + 'CrawfordPena Line P 1950-2019 4849-5_masked.csv'
p26_265_file = input_dir + 'CrawfordPena Line P 1950-2019 P526_masked.csv'

p4_df = pd.read_csv(p4_265_file)
p26_df = pd.read_csv(p26_265_file)

lineP_coord_file = input_dir + 'lineP_coordinates_ddegrees.csv'
lineP_coord_df = pd.read_csv(lineP_coord_file, index_col=[0])

nominal_p4_coords = (-(126 + 40.0 / 60), 48 + 39.0 / 60)

p4_png = input_dir + 'crawford_P4_26-5_inside_stn_radius.png'
p4_msk = p4_df.loc[:, 'is_close_to_station']
plot_stn_coords(p4_df.loc[p4_msk, 'Longitude'], p4_df.loc[p4_msk, 'Latitude'],
                p4_png, 'P4', (-146, -123), (44, 54), (1950, 2015),
                lineP_coord_df,
                true_coords=nominal_p4_coords)

# Now p26
nominal_p26_coords = (-145, 50)

p26_png = input_dir + 'crawford_P26_26-5_inside_stn_radius.png'
p26_msk = p26_df.loc[:, 'is_close_to_station']
plot_stn_coords(p26_df.loc[p26_msk, 'Longitude'], p26_df.loc[p26_msk, 'Latitude'],
                p26_png, 'P26', (-150, -123), (44, 56), (1955, 2015),
                lineP_coord_df,
                true_coords=nominal_p26_coords)

# Plot outside station radius
p4_png = input_dir + 'crawford_P4_26-5_outside_stn_radius.png'
plot_stn_coords(p4_df.loc[~p4_msk, 'Longitude'], p4_df.loc[~p4_msk, 'Latitude'],
                p4_png, 'P4', (-146, -123), (44, 54), (1950, 2015),
                lineP_coord_df,
                true_coords=nominal_p4_coords)

p26_png = input_dir + 'crawford_P26_26-5_outside_stn_radius.png'
plot_stn_coords(p26_df.loc[~p26_msk, 'Longitude'], p26_df.loc[~p26_msk, 'Latitude'],
                p26_png, 'P26', (-150, -123), (44, 56), (1955, 2015),
                lineP_coord_df,
                true_coords=nominal_p26_coords)

# ------------------------Check origin of data in data gaps----------------------

p4_265_file = parent_dir + 'CrawfordPena Line P 1950-2019 4849-5.csv'
p4_265_df = pd.read_csv(p4_265_file, nrows=1615, skip_blank_lines=True)
print(len(p4_265_df))
p4_265_df.dropna(axis='index', inplace=True, subset=['Date'])
print(len(p4_265_df))

mask_1970s = (p4_265_df.Date - 1970 >= 0) & (p4_265_df.Date - 1970 < 10)
print(sum(mask_1970s))
mask_1940s = (p4_265_df.Date - 1940 >= 0) & (p4_265_df.Date - 1940 < 10)
print(sum(mask_1940s))

mask_nodc = np.array(['NODC' in f for f in p4_265_df.File])
print(sum(mask_nodc))
mask_ios = ~mask_nodc  # np.array(['IOS' in f for f in p4_265_df.File])
print(sum(mask_ios))

print(len(p4_265_df))
print(p4_265_df.loc[~(mask_nodc | mask_ios), 'File'])

p4_265_df.loc[(mask_1970s & mask_ios), ['Date', 'Latitude', 'Longitude', 'File']].to_csv(
    input_dir + 'p4_265_1970s_ios_file_list.csv', index=False)

print(p4_265_df.loc[mask_1970s & mask_ios, ['Date', 'Latitude', 'Longitude', 'File']])
print(sum(mask_1970s & mask_nodc))
print(sum(mask_1970s & mask_ios))

p4_distances = np.array(
    [haversine((lat_i, lon_i),
               (lineP_coord_df.loc['P4', 'Lat ddegrees N'],
                lineP_coord_df.loc['P4', 'Lon ddegrees E']))
     for lat_i, lon_i in zip(p4_265_df.loc[mask_1970s, 'Latitude'],
                             p4_265_df.loc[mask_1970s, 'Longitude'])])
p4_latlon_mask = p4_distances <= p4_radius_strict

print(p4_265_df.loc[mask_1970s, ['Latitude', 'Longitude']])
print(p4_265_df.loc[mask_1940s, ['Latitude', 'Longitude']])

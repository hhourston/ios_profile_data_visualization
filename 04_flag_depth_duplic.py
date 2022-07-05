import pandas as pd
import numpy as np

# Flag or remove bin depth duplicates in each profile
# if they occur

station = 'GEO1'  # 'LBP3'  # 'LB08'  # 'P1'
f = 'C:\\Users\\HourstonH\\Documents\\ctd_visualization\\csv\\' \
    '{}_ctd_data_binned.csv'.format(station)
df_in = pd.read_csv(f)

# Profile start indices
prof_start_ind = np.unique(df_in.loc[:, 'Profile number'],
                           return_index=True)[1]

# Initialize column for holding duplicate binned depth mask
df_in['Unique binned depth mask'] = np.repeat(True, len(df_in))

# Iterate through all of the profiles
for i in range(len(prof_start_ind)):
    # Set profile end index
    if i == len(prof_start_ind) - 1:
        prof_end_ind = len(df_in)
    else:
        # Pandas indexing is inclusive so need the -1
        prof_end_ind = prof_start_ind[i + 1]

    # Get profile data; np.arange not inclusive of end which we want here
    indices = np.arange(prof_start_ind[i], prof_end_ind)

    # Check for binned depth duplicates
    # Keep the row with the closer observed depth to the binned depth
    # Flag or remove the other

    depth_binned = df_in.loc[indices, 'Depth bin [m]']
    # depth_observed = df_in.loc[indices, 'Depth [m]']

    # Keep first occurrence here
    df_in.loc[
        indices[1:], 'Unique binned depth mask'] = np.diff(depth_binned) > 0

    # # Didn't work properly...
    # df_in.loc[
    #     indices, 'Unique binned depth mask'] = depth_binned.duplicated(
    #     keep='first')


# Print summary statistics
print('Number of observations in:', len(df_in))
print('Number of observations out:',
      sum(df_in.loc[:, 'Unique binned depth mask']))

# Apply mask
# mask_duplicates = df_in.loc[:, 'Duplicate binned depth mask']
# df_out = df_in.loc[mask_duplicates]
df_out = df_in

# Export dataframe to csv
df_out_name = f.replace('binned', 'binned_depth_dupl')
df_out.to_csv(df_out_name, index=False)

import pandas as pd
import numpy as np

in_file = 'C:\\Users\\HourstonH\\Documents\\charles\\' \
          'line_P_data_products\\csv\\01_convert\\' \
          'P26_CTD_BOT_CHE_data.csv'

# How many oxygen profiles per year?

df = pd.read_csv(in_file)

profile_start_idx = np.unique(df.loc[:, 'Profile number'],
                              return_index=True)[1]

profile_end_idx = np.concatenate((profile_start_idx[1:] - 1,
                                  np.array([len(df)])))

years = np.arange(1956, 2022+1)

df['Year'] = pd.to_datetime(df.loc[:, 'Time']).dt.year

for y in years:
    print(y, len(np.where(df.loc[:, 'Year'] == y)[0]))

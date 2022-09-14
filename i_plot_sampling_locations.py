import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Explore CTD sampling locations

station = 'SI01'  # 'SI01'  # '59'  # '42'  # 'GEO1'  # 'LBP3'  # 'LB08'  # 'P1'
ctd_infile = 'C:\\Users\\HourstonH\\Documents\\ctd_visualization\\' \
             'csv\\{}_ctd_data.csv'.format(station)

ctd_df = pd.read_csv(ctd_infile)

prof_start_ind = np.unique(ctd_df.loc[:, 'Profile number'],
                           return_index=True)[1]

fig, ax = plt.subplots()

ax.scatter(ctd_df.loc[prof_start_ind, 'Longitude [deg E]'].to_numpy(),
           ctd_df.loc[prof_start_ind, 'Latitude [deg N]'].to_numpy(), s=1)

ax.scatter([-123.538], [48.520], c='r', marker='*')

plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('Station {}'.format(station))

plt.savefig('C:\\Users\\HourstonH\\Documents\\ctd_visualization\\'
            'png\\{}_location.png'.format(station))

plt.close(fig)
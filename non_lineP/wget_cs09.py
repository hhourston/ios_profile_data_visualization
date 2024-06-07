import wget
import numpy as np
from tqdm import trange
# import tqdm
import os
# import glob
# from Samantha Huntington
# import pandas as pd

wdir = 'C:\\Users\\hourstonh\\Documents\\charles\\more_oxygen_projects\\'
download_list = 'wget_netcdf_file_download_list_CS09_5km.csv'
output_dir = os.path.join(wdir, 'CS09')

address = np.genfromtxt(wdir + download_list, dtype=str)

for i in trange(address.size):
    f = os.path.join(wdir, os.path.basename(address[i]))
    if not os.path.exists(f):
        wget.download('https://' + address[i], out=output_dir)

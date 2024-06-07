import os
import pandas as pd
from helpers import get_profile_st_en_idx
from tqdm import trange
import numpy as np


def run_check(inFilePath, outFilePath, station):
    dfin = pd.read_csv(inFilePath)
    oxygen_column = dfin.columns[
        ['Oxygen' in colname for colname in dfin.columns]][0]
    # oxygen_unit = oxygen_column.split('[')[1][:-1]
    profile_start_idx, profile_end_idx = get_profile_st_en_idx(dfin.loc[:, 'Profile number'])

    mask_na = pd.Series([0] * len(dfin))
    counter_na = 0

    for i in trange(len(profile_start_idx)):
        start_idx_i = profile_start_idx[i]
        end_idx_i = profile_end_idx[i]

        if all(dfin.loc[start_idx_i:end_idx_i, oxygen_column].isna()):
            mask_na.loc[start_idx_i:end_idx_i] = 1
            counter_na += 1

    dfout = dfin.loc[mask_na == 0, :]

    # Print summary statistics to text file
    summary_statistics_file = os.path.join(
        os.path.dirname(outFilePath),
        '{}_drop_missing_o2_summary_statistics.txt'.format(station)
    )

    with open(summary_statistics_file, 'a') as txtfile:
        txtfile.write('Source file: {}\n'.format(inFilePath))
        txtfile.write('Output file: {}\n'.format(outFilePath))
        txtfile.write(
            'Number of profiles in: {}\n'.format(len(profile_start_idx))
        )
        txtfile.write('Number of profiles out: {}\n\n'.format(counter_na))

    dfout.to_csv(outFilePath, index=False)
    return


parent_dir = 'C:\\Users\\hourstonh\\Documents\\charles\\more_oxygen_projects\\'
sampling_station = 'CS09'
input_file_path = os.path.join(
    parent_dir,
    f'{sampling_station}_01_convert',
    f'{sampling_station}_CTD_BOT_CHE_data.csv'
)
output_file_path = os.path.join(
    parent_dir,
    f'{sampling_station}_02b_remove_casts_missing_o2',
    os.path.basename(input_file_path)
)

run_check(input_file_path, output_file_path, sampling_station)
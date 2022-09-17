import numpy as np


def get_profile_st_en_idx(profile_numbers):
    profile_start_idx = np.unique(profile_numbers,
                                  return_index=True)[1]
    # Minus 1 to account for pandas inclusive indexing
    profile_end_idx = np.concatenate(
        (profile_start_idx[1:] - 1, np.array([len(profile_numbers)]))
    )
    return profile_start_idx, profile_end_idx
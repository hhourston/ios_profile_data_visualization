# Copied from NEP_climatology project (mostly)


import pandas as pd
import numpy as np
from tqdm import trange


def vvd_gradient_check(var_df, grad_df, grad_var, verbose=False):
    # Value vs depth gradient check
    # Check for gradients, inversions and zero sensitivity
    # df: value vs depth dataframe
    # grad_df: dataframe from WOA18 containing maximum gradient, inversion,
    #          and zero sensitivity index values to check vvd data against

    if grad_var == 'Temperature':
        var_with_unit = 'Temperature [C]'
    elif grad_var == 'Salinity':
        var_with_unit = 'Salinity [PSS-78]'
    elif grad_var == 'Oxygen':
        var_with_unit = 'Oxygen [mL/L]'

    nobs = len(var_df)

    grad_mask = np.repeat(True, nobs)

    prof_start_ind = np.unique(var_df.loc[:, 'Profile number'],
                               return_index=True)[1]

    # Iterate through all of the profiles
    for i in trange(len(prof_start_ind)):  # len(prof_start_ind) 20
        # Set profile end index
        if i == len(prof_start_ind) - 1:
            prof_end_ind = len(var_df)
        else:
            prof_end_ind = prof_start_ind[i + 1]

        # Get profile data; np.arange not inclusive of end which we want here
        # df.loc is inclusive of the end
        indices = np.arange(prof_start_ind[i], prof_end_ind)
        depths = var_df.loc[indices, 'Depth [m]']
        values = var_df.loc[indices, var_with_unit]

        if verbose:
            print('Got values')

        # Try to speed up computations by skipping profiles with only 1 measurement
        if len(depths) <= 1:
            continue
        else:
            # Use numpy built-in gradient method (uses 2nd order central differences)
            # Need fix for divide by zero
            gradients = np.gradient(values, depths)

            # Find the rate of change of gradient
            d_gradients = np.diff(gradients)

            # Create masks accordingly
            # If depth <= 400m and gradient < -max, apply one set of criteria
            # If depth > 400m and gradient < -max, apply other set of criteria...
            # MGV: maximum gradient value
            # Gradient is a decrease in value over depth (hence the (-) sign)
            # Inversion is an increase in value over depth
            mask_MGV_lt_400 = (depths <= 400) & \
                (gradients > -grad_df.loc[grad_var, 'MGV_Z_lt_400m'])
            mask_MGV_gt_400 = (depths > 400) & \
                (gradients > -grad_df.loc[grad_var, 'MGV_Z_gt_400m'])
            mask_MIV_lt_400 = (depths <= 400) & \
                (gradients < grad_df.loc[grad_var, 'MIV_Z_lt_400m'])
            mask_MIV_gt_400 = (depths > 400) & \
                (gradients < grad_df.loc[grad_var, 'MIV_Z_gt_400m'])

            # for m in [mask_MGV_lt_400, mask_MGV_gt_400, mask_MIV_lt_400,
            #           mask_MIV_gt_400]:
            #     print(sum(m))

            if verbose:
                print('Created MGV/MIV masks')

            # Zero sensitivity check
            # Only flag observations with Value = 0
            # If there are zero-as-missing-values at the very surface, then
            # the ZSI check wouldn't find them because it needs the gradient
            # Need to initialize the masks first
            mask_ZSI_lt_400 = np.repeat(True, len(indices))
            mask_ZSI_gt_400 = np.repeat(True, len(indices))

            mask_ZSI_lt_400[1:] = ~((depths[1:] <= 400) &
                                    (d_gradients < -grad_df.loc[grad_var, 'MGV_Z_lt_400m'] *
                                     grad_df.loc[grad_var, 'ZSI']) &
                                    (values[1:] == 0.))
            mask_ZSI_gt_400[1:] = ~((depths[1:] > 400) &
                                    (d_gradients < -grad_df.loc[grad_var, 'MGV_Z_gt_400m'] *
                                     grad_df.loc[grad_var, 'ZSI']) &
                                    (values[1:] == 0.))

            # print(sum(mask_ZSI_lt_400), sum(mask_ZSI_gt_400), sep='\n')

            if verbose:
                print('Created ZSI subsetters')

            # Merge the masks
            prof_merged_masks = (mask_MGV_lt_400 | mask_MGV_gt_400) & \
                                (mask_MIV_lt_400 | mask_MIV_gt_400) & \
                                (mask_ZSI_lt_400 | mask_ZSI_gt_400)

            grad_mask[indices] = prof_merged_masks

    return grad_mask


__author__ = 'James Hannah'

import numpy as np

"""File copied from 
https://github.com/cyborgsphinx/ios-inlets/blob/main/convert.py
"""

import logging
import math
import gsw
import numpy


def warn_wrong_units(expected, actual, filename):
    logging.warning(
        f"Cowardly refusing to perform the conversion from {actual} to {expected} in {filename}"
    )


def calculate_density(
    length, temperature_C, salinity_SP, pressure_dbar, longitude, latitude, filename
):
    assumed = False
    if all(x is not None for x in [temperature_C, salinity_SP, pressure_dbar]):
        salinity_SA = gsw.SA_from_SP(salinity_SP, pressure_dbar, longitude, latitude)
        density = gsw.rho(
            salinity_SA,
            gsw.CT_from_t(salinity_SA, temperature_C, pressure_dbar),
            pressure_dbar,
        )
    else:
        density = []
    if len(density) == 0:
        logging.warning(
            f"Not enough data in {filename} to accurately compute density. Calculating density as though all values are 0"
        )
        assumed = True
        density = numpy.full(length, gsw.rho([0], [0], 0)[0])
    return density, assumed


def convert_umol_kg_to_mL_L(
    oxygen_umol_kg,
    longitude,
    latitude,
    temperature_C=None,
    salinity_SP=None,
    pressure_dbar=None,
    filename="unknown file",
):
    oxygen_umol_per_ml = 44.661
    metre_cube_per_litre = 0.001
    density, assumed_density = calculate_density(
        len(oxygen_umol_kg),
        temperature_C,
        salinity_SP,
        pressure_dbar,
        longitude,
        latitude,
        filename,
    )
    return (
        numpy.fromiter(
            (
                o * d / oxygen_umol_per_ml * metre_cube_per_litre
                for o, d in zip(oxygen_umol_kg, density)
            ),
            float,
            count=len(oxygen_umol_kg),
        ),
        assumed_density,
    )


def convert_percent_to_mL_L(
    oxygen_percent,
    temperature_C,
    salinity_SP,
    filename="no filename given",
):
    # fix conversion
    if temperature_C is not None and salinity_SP is not None:
        T90_to_T68 = 1.00024
        kelvin_offset = 273.15
        temperature_C_T68 = temperature_C * T90_to_T68
        # Scale as per GG
        x = np.log((298.15 - temperature_C_T68)/(kelvin_offset + temperature_C_T68))
        # constants for Eqn (8) of Garcia and Gordon 1992 for fit to Benson
        # and Krause data (cm3/dm2 = ml/l)
        a0 = 2.00907
        a1 = 3.22014
        a2 = 4.05010
        a3 = 4.94457
        a4 = -2.56847e-1
        a5 = 3.88767
        b0 = -6.24523e-3
        b1 = -7.37614e-3
        b2 = -1.03410e-2
        b3 = -8.17083e-3
        c0 = -4.88682e-7
        # Eqn (8) of Garcia and Gordon 1992
        lnC = np.fromiter(
            (
                a0 + a1*xi + a2*(xi**2) + a3*(xi**3) + a4*(xi**4)
                + a5*(xi**5)
                + S*(b0 + b1*xi + b2*(xi**2) + b3*(xi**3)) + c0*(S**2)
                for S, xi in zip(salinity_SP, x)
            ),
            float,
            count=len(oxygen_percent)
        )
        oxsol = np.exp(lnC)
        oxygen_ml_l = oxygen_percent * oxsol / 100
        return oxygen_ml_l
        # -----------------------------------------------------------
        # # function from en.wikipedia.org/wiki/Oxygenation_(environmental)
        # kelvin_offset = 273.15
        # temperature_K = [t + kelvin_offset for t in temperature_C]
        # A1 = -173.4292
        # A2 = 249.6339
        # A3 = 143.3483
        # A4 = -21.8492
        # B1 = -0.033096
        # B2 = 0.014259
        # B3 = -0.001700
        # return numpy.fromiter(
        #     (
        #         (o / 100)
        #         * math.exp(
        #             A1
        #             + (A2 * 100 / t)
        #             + (A3 * math.log(t / 100))
        #             + (A4 * t / 100)
        #             + (s * (B1 + (B2 * t / 100) + (B3 * ((t / 100) ** 2))))
        #         )
        #         for o, t, s in zip(oxygen_percent, temperature_K, salinity_SP)
        #     ),
        #     float,
        #     count=len(oxygen_percent),
        # )
    else:
        logging.warning(
            f"Not enough data from {filename} to convert oxygen from % to mL/L. Ignoring file"
        )
        return None


def ml_l_to_umol_kg(oxy_ml_l, longitude, latitude, temperature_C,
                    salinity_SP, pressure_dbar, filename):
    # Convert oxygen mL/L units to umol/kg units
    oxygen_umol_per_ml = 44.661
    metre_cube_per_litre = 0.001
    density, assumed_density = calculate_density(
        len(oxy_ml_l),
        temperature_C,
        salinity_SP,
        pressure_dbar,
        longitude,
        latitude,
        filename,
    )
    return (
        np.fromiter(
            (
                o * oxygen_umol_per_ml / (rho * metre_cube_per_litre)
                for o, rho in zip(oxy_ml_l, density)
            ),
            dtype=float,
            count=len(oxy_ml_l),
        ),
        assumed_density
    )


def convert_salinity(salinity, units, filename):
    """Converts salinity into PSU

    Arguments
    salinity - raw salinity data
    units - unit of measure for the salinity data
    filename - the name of the file the data came from (only used for logs if something goes wrong)

    Returns (oxygen in PSU, whether a computation was needed)
    """
    if salinity is None:
        return None, False
    elif units.lower() in ["psu", "pss-78"]:
        return salinity, False
    elif units.lower() in ["ppt"] or units.lower().startswith("'ppt"):
        return gsw.SP_from_SK(salinity), True
    elif units.lower() in ["umol/kg"]:
        g_per_umol = 58.44 / 1000 / 1000
        return gsw.SP_from_SR(salinity * g_per_umol), True
    else:
        warn_wrong_units("PSU", units, filename)
        return None, False


def convert_oxygen(
    oxygen,
    units,
    longitude,
    latitude,
    temperature_C,
    salinity_SP,
    pressure_dbar,
    filename,
):
    """Converts oxygen concentration into mL/L

    Arguments
    oxygen - raw oxygen data
    units - unit of measure for the oxygen data
    longitude - the longitude where the oxygen data was measured
    latitude - the latitude where the oxygen data was measured
    temperature_C - the temperature (in Celcius) associated with the oxygen data
    salinity_SP - the salinity (in PSU) associated with the oxygen data
    pressure_dbar - the pressure (in dbar) associated with the oxygen data
    filename - the name of the file the data came from (only used for logs if something goes wrong)

    Returns (oxygen in mL/L, whether a computation was needed, whether density was assumed)
    """
    if oxygen is None:
        return None, False, False
    elif units.lower() in ["ml/l"]:
        return oxygen, False, False
    #                                  V this is V this when parsed with ObsFile.py
    elif units.lower() in ["umol/kg", "mmol/m", "mmol/m**3"]:
        data, assumed_density = convert_umol_kg_to_mL_L(
            oxygen,
            longitude,
            latitude,
            temperature_C,
            salinity_SP,
            pressure_dbar,
            filename=filename,
        )
        return data, True, assumed_density
    elif units.lower() in ["mg/l"]:
        oxygen_mg_per_mL = 1.429
        data = oxygen / oxygen_mg_per_mL
        return data, True, False
    elif units in ["%"]:
        data = convert_percent_to_mL_L(
            oxygen, temperature_C, salinity_SP, filename=filename
        )
        return data, False, False
    else:
        warn_wrong_units("mL/L", units, filename)
        return None, False, False

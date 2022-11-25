# ios_profile_data_visualization
This repository contains code for three similar projects. It was originally made for producing standard plots of temperature, salinity and oxygen CTD (conductivity, temperature, depth) data from select Institute of Ocean Sciences (IOS) stations in the Salish Sea, BC. The code has since been expanded on. Processing steps have been added to allow the user to plot oxygen on the potential density anomaly surfaces 26.5, 26.7, and 26.9, which correspond to the potential densities 1026.5 kg/$m^3$, 1026.7 kg/$m^3$, and 1026.9 kg/$m^3$, respectively. Such plots were made for stations P4 and P26 on Line P using CTD and bottle data from the IOS and NOAA (National Oceanographic and Atmospheric Administration) archives. The repository also contains code for exploring data coverage of the Douglas Channel over 1951-1954 -- see the folder "UBC_data" for more. 

Author: Hana Hourston (@hhourston)

## Requirements
* Python >= 3.7

## Processing steps
1. Assembly of netCDF CTD data into one CSV table for each station (and originator if applicable)
   1. Perform unit conversions: salinity to PSS-78, oxygen to mL/L or to umol/kg
   2. Include temperature, salinity and oxygen originator (NODC) quality flags
2. Apply the originator flags
   1. For NODC, a flag value of "0" indicates data that passed all quality control checks, so data having a non-zero flag are discarded. NODC flags are listed below. 
3. Latitude, longitude, depth, range, and gradient checks
   1. Use the depth, range and gradient limits from Garcia et al. (2018) and follow the steps in Garcia et al. (2019)
4. If other data types are included (e.g., CHE and BOT), check for more than one profile from the same time and location. Keep only the profile with the higher depth resolution (i.e., the CTD profile) 
5. Data binning to the nearest whole-number depth in meters
6. Flagging of binned depth duplicates in profiles
7. Plotting: Annual sampling frequency, monthly sampling frequency, filled contours, anomalies at select depths

## Steps to plot oxygen data on potential density anomaly surfaces 26.9, 26.7 and 26.5
8. Compute the potential density anomaly at each observed level in the data using TEOS-10. Use a reference pressure of 0 dbar
9. Linearly interpolate oxygen observations sampled with hydro bottles at discrete depths to 1m vertical resolution. Do not interpolate if any of oxygen, potential temperature or absolute salinity are spaced more than 0.2 potential density anomaly units apart.
10. Make the selection of oxygen from the 1m resolution profiles only if the computed potential density anomaly is within 0.005 units of the specified density anomaly (i.e., bin oxygen data to the select density anomalies)
11. Compute annual averages at each potential density anomaly surface (one single average for each year)
12. Make scatter plots of the annually-averaged oxygen on the select potential density surfaces (as in Crawford and Pena, 2021)
      1. Include a best-fit line through the set of annually-averaged oxygen points for each potential density anomaly surface

All times are in UTC. 
The first step involving the assembly of netCDF CTD data into a 
CSV table also includes unit conversions as needed. If not 
already the case, salinity is converted into PSS-78 and oxygen 
is converted into $\mu mol/kg$. Temperature units are kept as 
degrees Celsius.  
\
The latitude and longitude of each profile are checked to ensure 
that all are within +/- 0.075 decimal degrees of the median 
station coordinates. The median is used instead of the mean here 
because the former is robust to outliers.  
\
The depth, range, and gradient checks are taken from the NCEI WOA18.  
\
The monthly sampling frequency plot code was based off of https://github.com/cyborgsphinx/ios-inlets.  
\
The steps to plot oxygen data on select density surfaces are 
taken from Crawford and Peña (2016).

# References
Crawford, B. and Peña, A. (2021). Oxygen in subsurface waters on 
the B.C. Shelf. In *State of the Physical, Biological and 
Selected Fishery Resources of Pacific Canadian Marine Ecosystems 
in 2020*, edited by J. L. Boldt, A. Javorski and P. C. Chandler. 
https://waves-vagues.dfo-mpo.gc.ca/library-bibliotheque/4098297x.pdf

Crawford, W. R. and Peña, M. A. (2016). Decadal Trends in Oxygen 
Concentration in Subsurface Waters of the Northeast Pacific 
Ocean, Atmosphere-Ocean, 54:2, 171-192, 
DOI: 10.1080/07055900.2016.1158145

Garcia H. E., K.W. Weathers, C.R. Paver, I. Smolyar, T.P. Boyer,
R.A. Locarnini, M.M. Zweng, A.V. Mishonov, O.K. Baranova, D. 
Seidov, and J.R. Reagan (2019). World Ocean Atlas 2018, Volume 
3: Dissolved Oxygen, Apparent Oxygen Utilization, and Dissolved 
Oxygen Saturation. A. Mishonov Technical Editor. 
*NOAA Atlas NESDIS 83*, 38pp. (Available at https://www.nodc.noaa.gov/OC5/woa18/pubwoa18.html).

Garcia, H. E., T. P. Boyer, R. A. Locarnini, O. K. Baranova, M. M. Zweng (2018). World Ocean Database 2018: User’s Manual (prerelease). A.V. Mishonov, Technical Ed., NOAA, Silver Spring, MD (Available at https://www.NCEI.noaa.gov/OC5/WOD/pr_wod.html). 

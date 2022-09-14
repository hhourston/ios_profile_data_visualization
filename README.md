# ios_ctd_visualization
Visualization of netCDF CTD data collected by the Institute of Ocean Sciences. Support for netCDF BOT (bottle) and CHE (chemistry) data added 2022-09-02.

Author: Hana Hourston (@hhourston)

# Requirements
* Python >= 3.7

# Processing steps
1. Assembly of netCDF CTD data into one CSV table for each station (and originator if applicable)
   1. Perform unit conversions: salinity to PSS-78, oxygen to mL/L or to umol/kg
   2. Include temperature, salinity and oxygen originator (NODC) quality flags
2. Apply the originator flags
   1. For NODC, a flag value of "0" indicates data that passed all quality control checks, so data having a non-zero flag are discarded. NODC flags are listed below. 
3. Latitude, longitude, depth, range, and gradient checks \
   2b. If CHE and BOT data are included, check for CHE and BOT profiles from the same time and location as CTD profiles. Keep only the profile with the higher depth resolution (i.e., the CTD profile) 
4. Data binning to the nearest whole-number depth in meters
5. Flagging of binned depth duplicates in profiles
6. Plotting: Annual sampling frequency, monthly sampling frequency, filled contours, anomalies at select depths

# Plot oxygen data on density surfaces
7. Compute the potential density anomaly at each observed level in the data using TEOS-10. Use a reference pressure of 0 dbar
8. Linearly interpolate data from step 2b onto the potential density anomaly surfaces 26.9, 26.7 and 26.5 
9. Compute annual averages at each potential density anomaly surface (one single average for each year)
10. Make scatter plots of oxygen on the density surfaces defined in step 7 (as in Crawford and Pena, 2021)
      1. Include a best-fit line through the set of points for each potential density anomaly surface

The first step involving the assembly of netCDF CTD data into a CSV table also includes unit conversions as needed. If not already the case, salinity is converted into PSS-78 and oxygen is converted into $\mu mol/kg$. Temperature and fluorescence units are kept as degrees Celsius and $mg/m^3$, respectively.
\
The latitude and longitude of each profile are checked to ensure that all are within +/- 0.075 decimal degrees of the median station coordinates. The median is used instead of the mean here because the former is robust to outliers.  
\
The depth, range, and gradient checks are taken from the NCEI WOA18.  
\

# References
Crawford, B. and Pena, A. (2021). Oxygen in subsurface waters on the B.C. Shelf. In *State of the Physical, Biological and Selected Fishery Resources of Pacific Canadian Marine Ecosystems in 2020*, edited by J. L. Boldt, A. Javorski and P. C. Chandler. https://waves-vagues.dfo-mpo.gc.ca/library-bibliotheque/4098297x.pdf

Garcia H. E., K.W. Weathers, C.R. Paver, I. Smolyar, T.P. Boyer, R.A. Locarnini, M.M. Zweng, A.V. Mishonov, O.K. Baranova, D. Seidov, and J.R. Reagan (2019). World Ocean Atlas 2018, Volume 3: Dissolved Oxygen, Apparent Oxygen Utilization, and Dissolved Oxygen Saturation. A. Mishonov Technical Editor. *NOAA Atlas NESDIS 83*, 38pp.

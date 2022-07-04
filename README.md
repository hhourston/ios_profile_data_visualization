### Note: In progress

# ios_ctd_visualization
Visualization of CTD data collected by the Institute of Ocean Sciences.

# Requirements
* Python >= 3.7

# Processing steps
1. Assembly of netCDF CTD data into one CSV table for each station
2. Latitude, longitude, depth, range, and gradient checks
3. Data binning to the nearest whole-number depth in meters
4. Flagging of binned depth duplicates in profiles
5. Plotting: Annual sampling frequency, monthly sampling frequency, filled contours, anomalies


The first step involving the assembly of netCDF CTD data into a CSV table also includes unit conversions as needed. If not already the case, salinity is converted into PSU and oxygen is converted into $\mu mol/kg$. Temperature and fluorescence units are kept as degrees Celsius and $mg/m^3$, respectively.
\
The latitude and longitude of each profile are checked to ensure that all are within +/- 0.1 decimal degrees of the median station coordinates. The median is used instead of the mean here because the former is robust to outliers.  
\
The depth, range, and gradient checks are taken from the NCEI WOA18.  
\
The anomalies are computed for each binned depth that the user wants to plot. For a specific depth, the computation consists of taking the mean for each year, then taking the mean over all the yearly means (hence, a "mean of means"). 

# References
Garcia H. E., K.W. Weathers, C.R. Paver, I. Smolyar, T.P. Boyer, R.A. Locarnini, M.M. Zweng, A.V. Mishonov, O.K. Baranova, D. Seidov, and J.R. Reagan (2019). World Ocean Atlas 2018, Volume 3: Dissolved Oxygen, Apparent Oxygen Utilization, and Dissolved Oxygen Saturation. A. Mishonov Technical Editor. *NOAA Atlas NESDIS 83*, 38pp.

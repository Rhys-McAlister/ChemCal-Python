# ChemCal Python

** Work in progress ** 
ChemCal Python is a reimplementation of the R package [chemCal](https://pkgdown.jrwb.de/chemCal/) in python using the Streamlit library and service to host the app. 

There is still some work to be done recreating the remaining functions in the original package and updating the uncertainty calculation to use the formula defined in Massart et al. (1997).

## Using the package
  
The first section of the streamlit app allows for either the uploading of a csv or the manual entry of data. Care should be taken to enter the column names exactly as they appear in the csv to ensure that the columns are selected correctly.   
  
Following the data entry, the next field will ask for the number of replicates performed for any unknown sample. This value is important in the uncertainty calculation, which so far is calculated by:  
  
$U = t * {s_{\hat{x}}}_0$



Where   
```math
{s_{\hat{x}}}_0 = S_\frac{y}{x} \sqrt{\frac{1}{m} + \frac{1}{n}}
``` 
  
  
The data is fitted using the ordinary least squares method from the `statsmodels` package.  
  
The latter part of the app relays a calibration curve where the regression line is plotted over the data with the uncertainty range above and below the line. The plot also contains the regression formula and $R^2$ data for completeness.

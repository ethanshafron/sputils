# sputils, aka SPatial UTILitieS

### A perhaps moderately useful library for common tasks in remote sensing and predictive mapping

-------------------------------------------------------------------------------

### GeoMap, the base class

Some basic functionality:
```
>>>from sputils.map_utils import GeoMap
>>>f = "/home/es182091e/libtest.tif"
>>>m = GeoMap(f)
```

Get the WKT projection info
>>>m.proj
'PROJCS["NAD83 / UTM zone 13N",GEOGCS["NAD83",DATUM["North_American_Datum_1983",SPHEROID["GRS 1980",6378137,298.257222101,AUTHORITY["EPSG","7019"]],AUTHORITY["EPSG","6269"]],PRIMEM["Greenwich",0,AUTHORITY["EPSG","8901"]],UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]],AUTHORITY["EPSG","4269"]],PROJECTION["Transverse_Mercator"],PARAMETER["latitude_of_origin",0],PARAMETER["central_meridian",-105],PARAMETER["scale_factor",0.9996],PARAMETER["false_easting",500000],PARAMETER["false_northing",0],UNIT["metre",1,AUTHORITY["EPSG","9001"]],AXIS["Easting",EAST],AXIS["Northing",NORTH],AUTHORITY["EPSG","26913"]]'
```

Get the file name of the underlying dataset
```
>>>m.name
'/home/es182091e/libtest.tif'
```

Convert raster from bandsXcolsXrows to rows*colsXbands for pixel-wise model building.
```
>>>m.to_table(bands=[1,2]) 
array([[-9999., -9999.],
       [-9999., -9999.],
       [-9999., -9999.],
       ...,
       [-9999., -9999.],
       [-9999., -9999.],
       [-9999., -9999.]], dtype=float32)
```

Get the nodata value
```
>>>m.ndv
-9999.0
```

Band names are pulled from band descriptions if they exist, else they are just ints mapped to ints in a dict. Set band descriptions though, it's worth it.
```
>>>m.band_names ### mapping of variable names to band numbers in dict
{'elevation': 1, 'slope': 2}
```

You can fit statsmodels or sklearn models with this method. If you use a formula, it will use statsmodels, if you use the X/Y _bands arguments, it will use sklearn.
Gradient boosting, random forests, and multilayer perceptron are all available right now, for either classification or regression. For statsmodels, you can use GLMs, OLS, or mixed linear models. The fit_model() method returns statsmodels or sklearn model objects.

```
>>>m.fit_model("OLS", formula = "elevation~slope", sample_size = 40000) 
<statsmodels.regression.linear_model.RegressionResultsWrapper object at 0x7f73547443d0>
>>>m.fit_model("OLS", formula = "elevation~slope", sample_size = 40000).summary()
                            OLS Regression Results
==============================================================================
Dep. Variable:              elevation   R-squared:                       0.142
Model:                            OLS   Adj. R-squared:                  0.142
Method:                 Least Squares   F-statistic:                     6614.
Date:                Wed, 17 Nov 2021   Prob (F-statistic):               0.00
Time:                        11:45:08   Log-Likelihood:            -2.8105e+05
No. Observations:               40000   AIC:                         5.621e+05
Df Residuals:                   39998   BIC:                         5.621e+05
Df Model:                           1
Covariance Type:            nonrobust
==============================================================================
                 coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------
Intercept   1139.7229      2.040    558.794      0.000    1135.725    1143.721
slope         14.0425      0.173     81.327      0.000      13.704      14.381
==============================================================================
Omnibus:                    15990.289   Durbin-Watson:                   2.002
Prob(Omnibus):                  0.000   Jarque-Bera (JB):            74305.159
Skew:                           1.929   Prob(JB):                         0.00
Kurtosis:                       8.450   Cond. No.                         17.8
==============================================================================

Notes:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.


```

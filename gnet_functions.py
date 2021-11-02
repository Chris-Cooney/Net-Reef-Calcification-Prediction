def Gnet_data(reef_lat, reef_lon):

    '''
    This funtion collects and cleans data from the eReefs netCDF
    file available on the AIMS OpenDAP server.
    
    https://dapds00.nci.org.au/thredds/dodsC/fx3/model_data/gbr4_bgc_GBR4_H2p0_B2p0_Chyd_Dnrt.ncml
    
    args:
    
    - reef_lat : latitudinal coordinate of reef location
    - reef_lon : longitudinal coordinate of reef location
    '''
    
    # Required packages for function.
    import numpy as np
    import numpy.ma as ma
    import pandas as pd
    import xarray as xr

    from matplotlib import pyplot as plt
    %config InlineBackend.figure_format = 'retina'
    plt.ion()  # To trigger the interactive inline mode

    import seaborn as sns
    import cmocean

    %matplotlib inline

    # First we load a dataset. 

    # Since the data is provided via an [OPeNDAP](https://en.wikipedia.org/wiki/OPeNDAP) server, we can load it directly without downloading anything:
    df = xr.open_dataset("https://dapds00.nci.org.au/thredds/dodsC/fx3/model_data/gbr4_bgc_GBR4_H2p0_B2p0_Chyd_Dnrt.ncml")
    df


    # Variable selection

    # Here we select the necesary variables from the larger dataset for our model

    ds = df[[ 'temp', 'Gnet', 'PH', 'botz']]
    ds

    print(' model spatial extent:\n')
    print(' - Longitudinal extent:',np.nanmin(ds['longitude']),np.nanmax(ds['longitude']))
    print(' - Latitudinal extent:',np.nanmin(ds['latitude']),np.nanmax(ds['latitude']))


    # Removing non-finite values
    ds.latitude.values = np.nan_to_num(ds.latitude.values)
    ds.longitude.values = np.nan_to_num(ds.longitude.values)

    # Mask nans
    lat = ma.masked_invalid(ds.latitude.values)
    lon = ma.masked_invalid(ds.longitude.values)


    # Reef Selection
    # Next we select the reef of interest within the bounds of the GBR

    reeflat, reeflon = reef_lat, reef_lon

    #We need to make sure the coordinates we provide are trnaslated to the data wee have and so we code to recieve the closest corresponding point

    # Find the closest point base on the coordinates (lon, lat) we can do that a bit better with a kd-tree if you want
    # to analyse multiple reefs
    j_idx, i_idx = np.where((np.abs(lat-reeflat)<0.02)&(np.abs(lon-reeflon)<0.02))
    j_reef = j_idx[0]
    i_reef = i_idx[0]



    # Checking the returned i,j index:
    print(' Closest data point found to input coordinates:\n')
    print(' - Reef lon position ',reeflon,' Found closest lon position in the data ',lon[j_reef,i_reef])
    print(' - Reef lat position ',reeflat,' Found closest lat position in the data ',lat[j_reef,i_reef])


    # Now that we have the position (j,i) in the dataset we can extract the variables at this specific location:

    reef_Gnet = ds.sel(k=(43))
    reef_Gnet = reef_Gnet.sel(j=j_reef, i=i_reef).drop_vars({'longitude','latitude', 'botz', 'zc'})
    reef_Gnet


    # There needs to be some final data cleaning due to duplicates in the time series

    # step 1
    reef_Gnet["time"] = reef_Gnet["time"].dt.floor("D")

    # step 2
    val,idx = np.unique(reef_Gnet.time, return_index=True)
    reef_Gnet = reef_Gnet.isel(time=idx)

    # By querying the min/max net calcification we can confirm that point selected has calcification Data

    print(' model Gnet limits:\n')
    print(' - Gnet min/max:',np.nanmin(reef_Gnet['Gnet']),np.nanmax(reef_Gnet['Gnet']))


    # Summary plots to assess the validity of data obtained

    reef_Gnet_pd = reef_Gnet.to_dataframe()
    fig, axes = plt.subplots(2,2,figsize=(15,15))

    corr = reef_Gnet_pd.corr()

    sns.heatmap(corr, 
        vmin=-1, vmax=1, center=0,
        cmap=sns.diverging_palette(20, 220, n=200),
        square=True, annot=True, ax= axes[0,0])

    ax = sns.lineplot(data=reef_Gnet_pd, x='time', y='Gnet',ax= axes[0,1]) 
    axes[0,1].tick_params(axis='x', labelrotation=30)


    sns.scatterplot(data = reef_Gnet_pd, x='PH',y='Gnet',ax= axes[1,1])
    sns.scatterplot(data = reef_Gnet_pd, x='temp',y='Gnet',ax= axes[1,0])

    plt.suptitle("Summary", fontsize=25)
    axes[0,0].set_title('Correlation')
    axes[0,1].set_title('Gnet over time')
    axes[1,0].set_title('Gnet Versus temp')
    axes[1,1].set_title('Gnet Versus pH')
    
    return reef_Gnet

 import os
import xarray as xr
import datetime as dt
from dateutil.relativedelta import relativedelta


def temp_PH(reef_lat, reef_lon, fname):

    """
    This funtion collects and cleans data from the eReefs netCDF
    file available on the AIMS OpenDAP server.

    Hydrodynamic model: "https://thredds.ereefs.aims.gov.au/thredds/dodsC/s3://aims-ereefs-public-prod/derived/ncaggregate/ereefs/gbr4_v2/monthly-monthly/EREEFS_AIMS-CSIRO_gbr4_v2_hydro_monthly-monthly-"
    Biogechemical model: "https://thredds.ereefs.aims.gov.au/thredds/dodsC/s3://aims-ereefs-public-prod/derived/ncaggregate/ereefs/GBR4_H2p0_B3p1_Cq3b_Dhnd/monthly-monthly/EREEFS_AIMS-CSIRO_GBR4_H2p0_B3p1_Cq3b_Dhnd_bgc_monthly-monthly-"

    args:

    - reef_lat : latitudinal coordinate of reef location
    - reef_lon : longitudinal coordinate of reef location
    - fname : file name
    """

    # Define the datasets to draw from
    base_url = "https://thredds.ereefs.aims.gov.au/thredds/dodsC/s3://aims-ereefs-public-prod/derived/ncaggregate/ereefs/gbr4_v2/monthly-monthly/EREEFS_AIMS-CSIRO_gbr4_v2_hydro_monthly-monthly-"
    base_url2 = "https://thredds.ereefs.aims.gov.au/thredds/dodsC/s3://aims-ereefs-public-prod/derived/ncaggregate/ereefs/GBR4_H2p0_B3p1_Cq3b_Dhnd/monthly-monthly/EREEFS_AIMS-CSIRO_GBR4_H2p0_B3p1_Cq3b_Dhnd_bgc_monthly-monthly-"

    # Biogeochemical
    # Define starting and ending date of the netcdf file we want to load
    start_date = dt.date(2010, 12, 1)
    end_date = dt.date(2019, 4, 30)
    delta = relativedelta(months=+1)

    # Now perform a while loop to open the netcdf file and extract the relevant dataset for the site of interest
    step = True
    biofiles = []
    while start_date <= end_date:

        # Read individual file from the OpeNDAP server
        date = str(start_date.year) + "-" + format(start_date.month, "02")
        start_date += delta
        biofiles.append(f"{base_url2}{date}.nc")

    bio = xr.open_mfdataset(biofiles)
    print("Open multiple netcdf bio files from server")

    # Hydrodynamic
    # Define starting and ending date of the netcdf file we want to load
    start_date = dt.date(2010, 12, 1)
    end_date = dt.date(2019, 4, 30)
    delta = relativedelta(months=+1)

    # Now perform a while loop to open the netcdf file and extract the relevant dataset for the site of interest
    hydrofiles = []
    while start_date <= end_date:

        # Read individual file from the OpeNDAP server
        date = str(start_date.year) + "-" + format(start_date.month, "02")
        start_date += delta
        hydrofiles.append(f"{base_url}{date}.nc")

    hydro = xr.open_mfdataset(hydrofiles)
    print("Open multiple netcdf hydro files from server")

    # Reef Selection
    # Next we select the reef of interest within the bounds of the GBR

    reeflat, reeflon = reef_lat, reef_lon

    # select depth 1.5m
    ds_biodepth = bio.sel(k=15)
    ds_hydrodepth = hydro.sel(k=15)

    # pH
    PH = ds_biodepth.PH
    PH = PH.sel(longitude=reeflon, latitude=reeflat, method="nearest").drop_vars(
        {"longitude", "latitude", "zc"}
    )
    print("Selecting PH for chosen coordinates")

    # Temperature
    temp = ds_hydrodepth.temp
    temp = temp.sel(longitude=reeflon, latitude=reeflat, method="nearest").drop_vars(
        {"longitude", "latitude", "zc"}
    )
    print("Selecting Temperature for chosen coordinates")

    # Combine the two variables
    data = xr.merge([temp, PH],)
    data["time"] = data["time"].dt.floor("D")
    print("Merging variables into a combined Xarray dataset")

    out_path = "historical_data"
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    ncout = os.path.join(out_path, fname + ".nc")
    data.to_netcdf(ncout)
    print("Saved data in file: ", ncout)

    return data

def param_predict(dataset, fname):
    
    '''
    This function utlises historical data to construct and ARIMA predictive 
    model of PH and Temperature from 2019 to 2061.
    
    args:
    
    - dataset: netCDF data file with historical pH and temperature data.
    - fname: name of file to be saved 
    '''
    #packages for model
    import warnings
    import itertools
    import pandas as pd
    import numpy as np
    import statsmodels.api as sm
    import netCDF4
    import xarray as xr
    
    
    from matplotlib import pyplot as plt
    plt.rcParams['figure.figsize'] = (12,7)
    
    ds = xr.open_dataset(dataset)
    ds = ds.to_dataframe()
   
    # PH
    
    PHdata = ds[['PH']].resample('MS').mean()
    
    # Temp
    
    tempdata = ds[['temp']].resample('MS').mean()
    
    print('Summary of PH data')
    print('This graphic shows the rolling avg, Trend, Seasonality and Residuals of the dataset')
    decomposition = sm.tsa.seasonal_decompose(PHdata)
    plt.rcParams["figure.figsize"] = [16,9]
    decomposition.plot()

    plt.show()
    
    from statsmodels.tsa.stattools import adfuller
    def check_stationarity(timeseries):
        result = adfuller(timeseries,autolag='AIC')
        dfoutput = pd.Series(result[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
        print('The test statistic: %f' % result[0])
        print('p-value: %f' % result[1])
        return result[1]

    pvalue = check_stationarity(PHdata)
    
    import itertools
    #set parameter range
    p = range(0,3)
    q = range(0,3)
    d = range(0,2)
    s = range(12,13)
    # list of all parameter combos
    pdq = list(itertools.product(p, d, q))
    seasonal_pdq = list(itertools.product(p, d, q, s))
    # SARIMA model pipeline

    warnings.filterwarnings("ignore") # specify to ignore warning messages
    res = []
    params = []
    param_seasonals = []

    for param in pdq:
        for param_seasonal in seasonal_pdq:
            try:
                mod = sm.tsa.statespace.SARIMAX(PHdata,
                                        order=param,
                                        seasonal_order=param_seasonal)
                results = mod.fit(max_iter = 25)
                res.append(results.aic)
                params.append(param)
                param_seasonals.append(param_seasonal)
            except:
                continue
    
    res = np.array(res)
    k = res.argmin()
    print('Minimum AIC',res.argmin())
    print('ARIMA{}x{}12 - AIC:{}'.format(params[k], param_seasonals[k], res[k]))
    
    PH_mod = sm.tsa.statespace.SARIMAX(PHdata,
                                    order= params[k],
                                    seasonal_order= param_seasonals[k],
                                    enforce_stationarity=False,
                                    enforce_invertibility=False)
    results = PH_mod.fit()

    print(results.summary().tables[1])

    print('PH Graphical diagnostic')
    results.plot_diagnostics(figsize=(15, 12))
    plt.show()

    pred_dynamic_PH = results.get_prediction(start=pd.to_datetime('2016-01-01'), dynamic=True, full_results=True)
    pred_dynamic_ci = pred_dynamic_PH.conf_int()

    # Extract the predicted and true values of our time series
    PH_forecasted = pred_dynamic_PH.predicted_mean
    PH_truth = PHdata.PH

    # Compute the mean square error
    mse = ((PH_forecasted - PH_truth) ** 2).mean()
    print('The Mean Squared Error of our forecasts is {}'.format(round(mse, 6)))

    # Get forecast 502 steps ahead in future
    pred_PH = results.get_forecast(steps=502)

    # Get confidence intervals of forecasts
    pred_ci = pred_PH.conf_int()
    
    # Plot the forecast
    ax = PHdata.plot(label='observed', figsize=(15, 6))
    pred_PH.predicted_mean.plot(ax=ax, label='Forecast')
    ax.set_title('Forecast of pH 2019-2061')
    ax.set_xlabel('Date')
    ax.set_ylabel('PH Levels')

    plt.legend()
    plt.show()
    
    print('Repeating the process for temperature')
    tempdata = ds
    tempdata= tempdata[['temp']].resample('MS').mean()
    
    print('Summary of PH data')
    print('This graphic shows the rolling avg, Trend, Seasonality and Residuals of the dataset')
    decomposition = sm.tsa.seasonal_decompose(PHdata)
    plt.rcParams["figure.figsize"] = [16,9]
    decomposition.plot()

    plt.show()
    
    pvalue = check_stationarity(tempdata)
    
    import itertools
    #set parameter range
    p = range(0,3)
    q = range(0,3)
    d = range(0,2)
    s = range(12,13)
    # list of all parameter combos
    pdq = list(itertools.product(p, d, q))
    seasonal_pdq = list(itertools.product(p, d, q, s))
    # SARIMA model pipeline

    warnings.filterwarnings("ignore") # specify to ignore warning messages
    res = []
    params = []
    param_seasonals = []

    for param in pdq:
        for param_seasonal in seasonal_pdq:
            try:
                mod = sm.tsa.statespace.SARIMAX(tempdata,
                                        order=param,
                                        seasonal_order=param_seasonal)
                results = mod.fit(max_iter = 25)
                res.append(results.aic)
                params.append(param)
                param_seasonals.append(param_seasonal)
            except:
                continue
    
    res = np.array(res)
    k = res.argmin()
    print('Minimum AIC',res.argmin())
    print('ARIMA{}x{}12 - AIC:{}'.format(params[k], param_seasonals[k], res[k]))
    
    temp_mod = sm.tsa.statespace.SARIMAX(tempdata,
                                    order= params[k],
                                    seasonal_order= param_seasonals[k],
                                    enforce_stationarity=False,
                                    enforce_invertibility=False)


    
    results = temp_mod.fit()

    print(results.summary().tables[1])
    
    results.plot_diagnostics(figsize=(15, 12))
    plt.suptitle("Temperature Graphical Summary", fontsize=25)
    plt.show()
    
    pred_dynamic_temp = results.get_prediction(start=pd.to_datetime('2016-01-01'), dynamic=True, full_results=True)
    pred_dynamic_ci = pred_dynamic_temp.conf_int()

    # Extract the predicted and true values of our time series
    temp_forecasted = pred_dynamic_temp.predicted_mean
    temp_truth = tempdata.temp

    # Compute the mean square error
    mse = ((temp_forecasted - temp_truth) ** 2).mean()
    print('The Mean Squared Error of our forecasts is {}'.format(round(mse, 6)))

    # Get forecast 502 steps ahead in future
    pred_temp = results.get_forecast(steps=502)

    # Get confidence intervals of forecasts
    pred_ci = pred_temp.conf_int()

    ax = tempdata.plot(label='observed', figsize=(15, 6))
    pred_temp.predicted_mean.plot(ax=ax, label='Forecast')
    ax.set_title('Forecasted temperature 2019-2061')
    ax.set_xlabel('Date')
    ax.set_ylabel('temp Levels')

    plt.legend()
    plt.show()

    # Combine both the datasets for export
    data = {'time': pred_temp.predicted_mean.index,
        'temp': pred_temp.predicted_mean.values,
        'ph': pred_PH.predicted_mean.values}
    
    # Now we combine all the data
    df = pd.DataFrame(data, columns=['time','temp', 'ph'])
    df = df.set_index('time', drop=True)
    

    # Save Data
    xr.Dataset(df.to_xarray()).to_netcdf(f"future_data/{fname}_forecast.nc")
    
    return df
    
def Gnet_pred(forecast,Gnet,fname):
    
    '''
    This function takes the data produced by the param_predict() and Gnet_data() functions to produce a forecast of net calcification to 2061
    
    args:
    
    - forecast: fname.nc data produced by the param_predict() function. Should contain forecasted values of temperature and pH
    - Gnet: fname.nc data produced by the Gnet_data() function. Should contain values of temperature, pH and Gnet obtained from AIMS eReefs data
    - fname: Name of saved projected gnet data
    '''
    
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import xarray as xr
    import seaborn as sns
    
    ds = xr.open_dataset(f'future_data/{forecast}.nc')
    X_forecast = ds.to_dataframe()
    X_forecast.rename(columns={'ph': 'PH'}, inplace=True)
    
    date_rng = pd.date_range(start='2019-04-01', end='2061-01-01', freq='MS')
    
    data = xr.open_dataset(f'Reef-Data/{Gnet}.nc')
    data = data.to_dataframe()
    
    X_var = data[['temp', 'PH']]
    y_var = data['Gnet'] # dependent variable
    
    
    import statsmodels.api as sm
    from termcolor import colored as cl

    sm_X_var = sm.add_constant(X_var)
    
    mlr_model = sm.GLM(y_var, sm_X_var)
    mlr_reg = mlr_model.fit()
    print(cl(mlr_reg.summary(), attrs = ['bold']))
    
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(X_var, y_var, test_size = 0.2, random_state = 0)

    lr = LinearRegression()
    lr.fit(X_train, y_train)

    y_pred = lr.predict(X_test)
    
    df=pd.DataFrame({'Actual':y_test, 'Predicted':y_pred})
    
    from sklearn import metrics
    print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
    print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
    print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
    print('R-Squared :', lr.score(X_test, y_test))
    
    y_forecast = lr.predict(X_forecast)
    gnet_forecast = pd.DataFrame({'time':date_rng,'Predicted_Gnet':y_forecast})
    gnet_forecast.index = pd.to_datetime(gnet_forecast['time'])
    gnet_forecast.drop(['time'], axis=1, inplace=True)
    
    change = xr.Dataset(gnet_forecast.to_xarray())

    change = change.sel(time='2061-01-01')-change.sel(time='2019-04-01')

    print(cl(f'Change in net calcification (2019-2061): {round((((change.Predicted_Gnet.values)*86400)/1000),4)}(grams/m-2/day-1)', attrs = ['bold']))
    fig, axes = plt.subplots(3,1,figsize=(15,15))

    sns.lineplot(data= df,x = 'time', y= 'Actual' , color="r", label="Actual Values", ax= axes[0])
    sns.lineplot(data= df, x = 'time', y= 'Predicted', color="b", label="Predicted Values", ax= axes[0])

    sns.kdeplot(y_pred, color = 'r', label = 'Predicted Values', ax= axes[1])
    sns.kdeplot(y_test, color = 'b', label = 'Actual Values', ax= axes[1])

    sns.lineplot(data= gnet_forecast, x='time', y='Predicted_Gnet', color="b", ax= axes[2])

    plt.suptitle("Graphical Summary", fontsize= 25)
    axes[0].set_title('Actual vs Predicted Values')
    axes[1].set_title('Actual vs Predicted Values')
    axes[2].set_title('Forecasted net calcification')

    plt.show()
    
    # Save
    xr.Dataset(data.to_xarray()).to_netcdf(f"future_Gnet/{fname}_pred.nc")
    
    return gnet_forecast


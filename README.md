# Net Reef Calcification in Response to Climate Change

## Background

Severe environmental degradation and global climate change as a result of anthropogenic activities are a defining characteristic of the Anthropocene. At the current rate of climate change, ecosystems once thought to be inexhaustible are beginning to display increasingly damaging stress that is both persistent and acute. Loss of biodiversity and habitat are becoming more regular and some ecosystems are beginning to locally collapse under the consistent pressures of human demand and elevated atmospheric carbon dioxide (CO2) levels. The marine environment is no exception and is becoming increasingly impacted from the combined effects of increased sea surface temperature (SST), ocean acidification and ocean deoxygenation.

Coral reefs are large biological structures situated in shallow, tropical coastal waters. As flagship ecosystems, they support 25% of all marine species and provide 10% of global fisheries despite covering <0.1% of the ocean floor. As such, coral reefs represent an important natural resource that directly provides ecosystem services and economic opportunities to 0.5 billion people worldwide.In Australia, the Great Barrier Reef (GBR) alone is composed of 2,900 individual reefs, supports exceptional levels of biodiversity (600 species of coral, 1625 species of fish), and is valued in excess of ~$65 billion (Harriott, 2001; Economics, 2017).  The GBR is recognised as a UNESCO World Heritage Site and directly provides employment for 65,000 people through a combination of ecotourism and fisheries. But current estimates indicate that ca. 65% of corals have been lost worldwide as a result, with the GBR losing ca. 30% of hard corals following sequential mass-bleaching events in 2016 and 2017 which were linked to marine heatwaves. 

Calcification is significant proxy for coral health. The process of precipitation of calcium carbonate by the coral organism is both represetnative of metabolic function, growth and environmental impacts on corals. As such coral calcification represents a measure of health of individual coral, reefs and the whole GBR.

<img src="Gnet-anom.gif" width="900" height="800"/>
Figure 1: Monthly average Net reef calcification anomolies (mg/m-2/day-1) ranging from 2016-2019. Data sourced from <a href="https://portal.ereefs.info/map" target="_top">AIMS eReefs</a>

## Hypothesis

The purpose of the following code is to adress the question:

### “What is the effect of changing temperature and pH levels on coral calcification in the Great Barrier Reef?”

In order to do so this repositry provides a series of notebooks and functions to extract, forecast and predict net calcification data for reefs across the GBR to 2061

## 1. <a href="1. Reef Data.ipynb" target="_top"> Reef Data</a>

This section outlines the process for collection and cleaning of data from the eReefs dataset.

```python
def Gnet_data(reef_lat, reef_lon):

    '''
    This funtion collects and cleans data from the eReefs netCDF
    file available on the AIMS OpenDAP server.
    
    https://dapds00.nci.org.au/thredds/dodsC/fx3/model_data/gbr4_bgc_GBR4_H2p0_B2p0_Chyd_Dnrt.ncml
    
    args:
    
    - reef_lat : latitudinal coordinate of reef location
    - reef_lon : longitudinal coordinate of reef location
    '''

```

## 2 <a href="2. Historical data and 2061 forecast.ipynb" target="_top"> Historical data and 2061 forecast </a>

### 2.1.Temperature and pH data collection
This section outlines the process for collection and cleaning of data from the eReefs dataset

```python
def temp_PH(reef_lat, reef_lon, fname):

    """
    This funtion collects and cleans data from the eReefs netCDF
    file available on the AIMS OpenDAP server.

    Hydrodynamic model: "https://thredds.ereefs.aims.gov.au/thredds/dodsC/s3://aims-ereefs-public-prod/derived/ncaggregate/ereefs/gbr4_v2/monthly-monthly/EREEFS_AIMS-CSIRO_gbr4_v2_hydro_monthly-monthly-2010-12"
    Biogechemical model: "https://thredds.ereefs.aims.gov.au/thredds/dodsC/s3://aims-ereefs-public-prod/derived/ncaggregate/ereefs/GBR4_H2p0_B3p1_Cq3b_Dhnd/monthly-monthly/EREEFS_AIMS-CSIRO_GBR4_H2p0_B3p1_Cq3b_Dhnd_bgc_monthly-monthly-2010-12"

    args:

    - reef_lat : latitudinal coordinate of reef location
    - reef_lon : longitudinal coordinate of reef location
    - fname : file name
    """

```

### 2.2. Forecasting of temperature and pH parameters

This section utilises the data collected in section 1 to generate forecasted changes of temperature and pH from 2019 to 2061. This will provide the data needed to predicte calcification responses to future environmnetal variables.


```python
def param_predict(dataset, fname):
    
    '''
    This function utlises historical data to construct and ARIMA predictive 
    model of PH and Temperature from 2019 to 2061.
    
    args:
    
    - dataset: netCDF data file with historical pH and temperature data.
    - fname: name of file to be saved 
    '''
```


## 3 <a href="3. Multilinear predictions.ipynb" target="_top">Multilinear predictions </a>

This section outlines the methodology used to generate predictions of net reef calcification in response to forecasted temperature and pH parameters

```python
def Gnet_pred(forecast,Gnet,fname):
    
    '''
    This function takes the data produced by the param_predict() and Gnet_data() functions to produce a forecast of net calcification to 2061
    
    args:
    
    - forecast: fname.nc data produced by the param_predict() function. Should contain forecasted values of temperature and pH
    - Gnet: fname.nc data produced by the Gnet_data() function. Should contain values of temperature, pH and Gnet obtained from AIMS eReefs data
    - fname: Name of saved projected gnet data
    '''
    
```
    
## 4 <a href="4. Interactive map.html" target="_top"> Interactive map </a>

This final section displays the change in net calcification of 9 individual reefs across the the 3 major sections of the GBR:

- Northern
- Central 
- Southern

<img src="net calcification change.png" width="900" height="800"/>
Figure 2. Change in net calcification (grams/m^2/day) 2019-2061.

# Remarks

Full process layouts are described in notebooks linked to titles

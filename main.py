# Information about the data
# Perrin Freres monthly champagne sales millions ?64-?72

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from pandas.plotting import autocorrelation_plot          # for auto regressive model
from pandas.tseries.offsets import DateOffset             # create date offset to make predictions into the future
from statsmodels.graphics.api import qqplot
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf
from statsmodels.tsa.arima.model import ARIMA             # For non-seasonal data
from statsmodels.tsa.stattools import adfuller            # Testing For Stationarity                      



# ARIMA and Seasonal ARIMA
# Autoregressive Integrated Moving Averages
# The general process for ARIMA models is the following:

# Step 1: Visualize the Time Series Data
# Step 2: Make the time series data stationary
# Step 3: Plot the Correlation and AutoCorrelation Charts
# Step 4: Construct the ARIMA Model or Seasonal ARIMA based on the data
#         - lags is the amount of time the prediction is based upon
# Step 5: Use the model to make predictions

# read the data
df = pd.read_csv("Data/perrin-freres-monthly-champaigne.csv")


# print(df.shape)
# (105, 2)

# print(df.head())
#      Month  Perrin Freres monthly champagne sales millions ?64-?72
# 0  1964-01                                               2815
# 1  1964-02                                               2672
# 2  1964-03                                               2755
# 3  1964-04                                               2721
# 4  1964-05                                               2946

# print(df.tail())
#        Month  Perrin Freres monthly champagne sales millions ?64-?72
# 100  1972-05                                               4618
# 101  1972-06                                               5312
# 102  1972-07                                               4298
# 103  1972-08                                               1413
# 104  1972-09                                               5877


### --------- Cleaning up the data --------- ###
# change names of columns
df.columns=["Month","Sales"]
# print(df.head())
#      Month  Sales
# 0  1964-01   2815
# 1  1964-02   2672
# 2  1964-03   2755
# 3  1964-04   2721
# 4  1964-05   2946

# Remove NaN values
# Drop last 2 rows
# I did this through the csv so unneccessary in code
# df.drop(106,axis=0,inplace=True)
# df.drop(105,axis=0,inplace=True)
# print(df.tail())

# Convert Month into Datetime objects
# when pandas reads in the date it does so as a string, we will need to convert
# this to a workable format for our needs
df['Month']=pd.to_datetime(df['Month'])
# print(df.head())
#        Month  Sales
# 0 1964-01-01   2815
# 1 1964-02-01   2672
# 2 1964-03-01   2755
# 3 1964-04-01   2721
# 4 1964-05-01   2946

# set the month column as the index
df.set_index('Month',inplace=True)
# print(df.head())
#             Sales
# Month
# 1964-01-01   2815
# 1964-02-01   2672
# 1964-03-01   2755
# 1964-04-01   2721
# 1964-05-01   2946


# print(df.describe())
#               Sales
# count    105.000000
# mean    4761.152381
# std     2553.502601
# min     1413.000000
# 25%     3113.000000
# 50%     4217.000000
# 75%     5221.000000
# max    13916.000000


### --------- Visualize the Data --------- ###
# chart1
# df.plot()
# plt.title("Chart 1")
# plt.show()


### --------- Testing For Stationarity --------- ###
# our goal will be to make the data stationary. The graph shows a high degree
# of influence from the seasons with the summers appearing as positive peaks 
# and the winters appearing as negative peaks, which makes sense given the 
# growing season

# perform Dickey-Fuller test
# most important vlue is the p-value
# The Dickey-Fuller test is a type of hypothesis testing where:
# - Ho: It is non stationary
# - H1: It is stationary
# Basically if the p-value is less than 0.05 than the data is stationary, if
# it is greater than 0.05 then it is non-stationary and will require further
# work
test_result = adfuller(df['Sales'])
# print(test_result)
# (-1.8335930563276184, 0.36391577166024713, 11, 93, 
# {'1%': -3.502704609582561, '5%': -2.8931578098779522, 
# '10%': -2.583636712914788}, 1478.4633060594724) 



# function that outputs the results of the Dickey-Fuller in an easily readable 
# format
def adfuller_test(sales):
    result=adfuller(sales)
    labels = ['ADF Test Statistic','p-value','#Lags Used','Number of Observations Used']
    for value,label in zip(result,labels):
        print(label+' : '+str(value) )
    if result[1] <= 0.05:
        print("strong evidence against the null hypothesis(Ho), reject the null hypothesis. Data has no unit root and is stationary")
    else:
        print("weak evidence against null hypothesis, time series has a unit root, indicating it is non-stationary ")
    

# print(adfuller_test(df['Sales']))
# ADF Test Statistic : -1.8335930563276184
# p-value : 0.36391577166024713
# #Lags Used : 11
# Number of Observations Used : 93
# weak evidence against null hypothesis, time series 
# has a unit root, indicating it is non-stationary   
# None

# If the data is non-stationary what do we do to make it stationary? First
# thing that we can do is differencing

# Differencing
# print(df['Sales'].shift(1))
# Month
# 1964-01-01       NaN
# 1964-02-01    2815.0
# 1964-03-01    2672.0
# 1964-04-01    2755.0
# 1964-05-01    2721.0
#                ...
# 1972-05-01    4788.0
# 1972-06-01    4618.0
# 1972-07-01    5312.0
# 1972-08-01    4298.0
# 1972-09-01    1413.0
# Name: Sales, Length: 105, dtype: float64

# subtract the sales column shifted by one row from the sales column to 
# get the change between the two dates
df['Sales First Difference'] = df['Sales'] - df['Sales'].shift(1)
# print(df.head())
# Month
# 1964-01-01   2815                     NaN
# 1964-02-01   2672                  -143.0
# 1964-03-01   2755                    83.0
# 1964-04-01   2721                   -34.0
# 1964-05-01   2946                   225.0

# shift sales column by 12 (months so 1 year total, to get the difference between
# annual seasons) and substract from the original sales column
df['Seasonal First Difference'] = df['Sales']-df['Sales'].shift(12)
# print(df.head(14))
#             Sales  Sales First Difference  Seasonal First Difference
# Month
# 1964-01-01   2815                     NaN                        NaN
# 1964-02-01   2672                  -143.0                        NaN
# 1964-03-01   2755                    83.0                        NaN
# 1964-04-01   2721                   -34.0                        NaN
# 1964-05-01   2946                   225.0                        NaN
# 1964-06-01   3036                    90.0                        NaN
# 1964-07-01   2282                  -754.0                        NaN
# 1964-08-01   2212                   -70.0                        NaN
# 1964-09-01   2922                   710.0                        NaN
# 1964-10-01   4301                  1379.0                        NaN
# 1964-11-01   5764                  1463.0                        NaN
# 1964-12-01   7312                  1548.0                        NaN
# 1965-01-01   2541                 -4771.0                     -274.0
# 1965-02-01   2475                   -66.0                     -197.0


# Again test dickey fuller test
# Make sure to use .dropna() to drop the NaN values otherwise you will have 
# a lot of errors 
# print(adfuller_test(df['Seasonal First Difference'].dropna()))
# ADF Test Statistic : -7.626619157213166
# p-value : 2.0605796968136632e-11
# #Lags Used : 0
# Number of Observations Used : 92
# strong evidence against the null hypothesis(Ho), reject the null hypothesis. 
# Data has no unit root and is stationary
# None

# chart2
# df['Seasonal First Difference'].plot()
# plt.title("Chart 2 Seasonal Difference")
# plt.show()

# Auto Regressive Model
# chart3
# autocorrelation_plot(df['Sales'])
# plt.title("Chart 3 Auto Regressive Model")
# plt.show()


# Final Thoughts on Autocorrelation and Partial Autocorrelation
# Identification of an AR model is often best done with the PACF.

# For an AR model, the theoretical PACF “shuts off” past the order of the 
# model. The phrase “shuts off” means that in theory the partial 
# autocorrelations are equal to 0 beyond that point. Put another way, the 
# number of non-zero partial autocorrelations gives the order of the AR model. 
# By the “order of the model” we mean the most extreme lag of x that is used 
# as a predictor.
# Identification of an MA model is often best done with the ACF rather than 
# the PACF.

# For an MA model, the theoretical PACF does not shut off, but instead tapers 
# toward 0 in some manner. A clearer pattern for an MA model is in the ACF. 
# The ACF will have non-zero autocorrelations only at lags involved in the 
# model.
# p,d,q 
# - p AR model lags 
# - d differencing 
# - q MA lags

# chart4
# .iloc set to 13 because we are using the Seasonal First Difference wherein the
# first twelve rows contain NaN values
# fig = plt.figure(figsize=(12,8))
# ax1 = fig.add_subplot(211)
# fig = sm.graphics.tsa.plot_acf(df['Seasonal First Difference'].iloc[13:],lags=40,ax=ax1)
# ax2 = fig.add_subplot(212)
# fig = sm.graphics.tsa.plot_pacf(df['Seasonal First Difference'].iloc[13:],lags=40,ax=ax2)
# plt.title("Chart 4 Partial Autocorrelation Seasonal First Difference")
# plt.show()


# For non-seasonal data
# looking at the above graph we see the shut off (the drop from the first and highest
# value to the first point that starts the continuous series) occurs around 1, 
# so p=1. We are looking at only a single seasonal shift so with our differencing
# d=1. q=0 or 1. so order will = (p,d,q)
model=ARIMA(df['Sales'],order=(1,1,1))
model_fit=model.fit()
# print(model_fit.summary())
#                                SARIMAX Results
# ==============================================================================   
# Dep. Variable:                  Sales   No. Observations:                  105   
# Model:                 ARIMA(1, 1, 1)   Log Likelihood                -952.814   
# Date:                Wed, 10 May 2023   AIC                           1911.627   
# Time:                        13:18:33   BIC                           1919.560   
# Sample:                    01-01-1964   HQIC                          1914.841   
#                          - 09-01-1972
# Covariance Type:                  opg
# ==============================================================================   
#                  coef    std err          z      P>|z|      [0.025      0.975]   
# ------------------------------------------------------------------------------   
# ar.L1          0.4545      0.114      3.999      0.000       0.232       0.677   
# ma.L1         -0.9666      0.056    -17.316      0.000      -1.076      -0.857   
# sigma2      5.226e+06   6.17e+05      8.473      0.000    4.02e+06    6.43e+06   
# ===================================================================================
# Ljung-Box (L1) (Q):                   0.91   Jarque-Bera (JB):                 2.59
# Prob(Q):                              0.34   Prob(JB):                         0.27
# Heteroskedasticity (H):               3.40   Skew:                             0.05
# Prob(H) (two-sided):                  0.00   Kurtosis:                         3.77
# ===================================================================================

# Warnings:
# [1] Covariance matrix calculated using the outer product of gradients (complex-step).


# chart5
# df['forecast']=model_fit.predict(start=90,end=103,dynamic=True)
# df[['Sales','forecast']].plot(figsize=(12,8))
# plt.title("Chart 5 Sales vs Forecast")
# plt.show()


### --------- For Seasonal Data, SARIMAX --------- ###

# the seasonal_order = (p,d,q,[the number of values shifted])
model = sm.tsa.statespace.SARIMAX(df['Sales'],order=(1, 1, 1),seasonal_order=(1,1,1,12))
results = model.fit()
# RUNNING THE L-BFGS-B CODE

#            * * *

# Machine precision = 2.220D-16
#  N =            5     M =           10
#  This problem is unconstrained.

# At X0         0 variables are exactly at the bounds

# At iterate    0    f=  7.07295D+00    |proj g|=  4.80911D-02

# At iterate    5    f=  7.04942D+00    |proj g|=  1.53381D-02

# At iterate   10    f=  7.04713D+00    |proj g|=  2.99455D-04

# At iterate   15    f=  7.04708D+00    |proj g|=  5.05920D-03

# At iterate   20    f=  7.04413D+00    |proj g|=  1.18430D-02

# At iterate   25    f=  7.03252D+00    |proj g|=  1.06110D-03

# At iterate   30    f=  7.03240D+00    |proj g|=  5.83554D-05

#            * * *

# Tit   = total number of iterations
# Tnf   = total number of function evaluations
# Tnint = total number of segments explored during Cauchy searches
# Skip  = number of BFGS updates skipped
# Nact  = number of active bounds at final generalized Cauchy point
# Projg = norm of the final projected gradient
# F     = final function value

#            * * *

#    N    Tit     Tnf  Tnint  Skip  Nact     Projg        F
#     5     31     34      1     0     0   1.499D-05   7.032D+00
#   F =   7.0324006467397933

# CONVERGENCE: REL_REDUCTION_OF_F_<=_FACTR*EPSMCH


# check to see accuracy of the model by using known values, predict start 
# from row 90 of the data through row 103 of the data
# chart6
# df['forecast']=results.predict(start=90,end=103,dynamic=True)
# df[['Sales','forecast']].plot(figsize=(12,8))
# plt.title("Chart 6 Sales vs Forecast")
# plt.show()


# Predict for dates into the future 
# print(df.index[-1])
# 1972-09-01 00:00:00

# create a variable future_dates that looks from the last date from our data to 
# 24 months into the future
future_dates = [df.index[-1]+ DateOffset(months=x)for x in range(0,24)]
# print(f"future_dates: \n{future_dates}")
# future_dates:
# [Timestamp('1972-09-01 00:00:00'), Timestamp('1972-10-01 00:00:00'), Timestamp('
# 1972-11-01 00:00:00'), Timestamp('1972-12-01 00:00:00'), Timestamp('1973-01-01 0
# 0:00:00'), Timestamp('1973-02-01 00:00:00'), Timestamp('1973-03-01 00:00:00'), T
# imestamp('1973-04-01 00:00:00'), Timestamp('1973-05-01 00:00:00'), Timestamp('19
# 73-06-01 00:00:00'), Timestamp('1973-07-01 00:00:00'), Timestamp('1973-08-01 00:
# 00:00'), Timestamp('1973-09-01 00:00:00'), Timestamp('1973-10-01 00:00:00'), Tim
# estamp('1973-11-01 00:00:00'), Timestamp('1973-12-01 00:00:00'), Timestamp('1974
# -01-01 00:00:00'), Timestamp('1974-02-01 00:00:00'), Timestamp('1974-03-01 00:00
# :00'), Timestamp('1974-04-01 00:00:00'), Timestamp('1974-05-01 00:00:00'), Times
# tamp('1974-06-01 00:00:00'), Timestamp('1974-07-01 00:00:00'), Timestamp('1974-0
# 8-01 00:00:00')]

# create a dataframe of the newly built future_dates
future_datest_df = pd.DataFrame(index=future_dates[1:],columns=df.columns)
# print(future_datest_df.tail())
#            Sales Sales First Difference Seasonal First Difference
# 1974-04-01   NaN                    NaN                       NaN
# 1974-05-01   NaN                    NaN                       NaN
# 1974-06-01   NaN                    NaN                       NaN
# 1974-07-01   NaN                    NaN                       NaN
# 1974-08-01   NaN                    NaN                       NaN

# add the new dataframe to the original dataframe as rows in a new dataframe
future_df = pd.concat([df,future_datest_df])
# use the model to create predictions for the future dates
future_df['forecast'] = results.predict(start = 104, end = 129, dynamic= True)  

# future_df.to_csv("Data/future_df.csv")



# chart7
# future_df[['Sales', 'forecast']].plot(figsize=(12, 8)) 
# plt.title("Chart 7 Sales vs Forecast")
# plt.show()



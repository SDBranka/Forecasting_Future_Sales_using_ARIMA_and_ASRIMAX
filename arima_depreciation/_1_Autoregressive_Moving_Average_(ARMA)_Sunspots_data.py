import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.api import qqplot
from statsmodels.tsa.arima_process import ArmaProcess

# print(sm.datasets.sunspots.NOTE)
#     Number of Observations - 309 (Annual 1700 - 2008)
#     Number of Variables - 1
#     Variable name definitions::

#         SUNACTIVITY - Number of sunspots for each year

#     The data file contains a 'YEAR' variable that is not returned by load.  


dta = sm.datasets.sunspots.load_pandas().data
# print(dta.head())
#      YEAR  SUNACTIVITY
# 0  1700.0          5.0
# 1  1701.0         11.0
# 2  1702.0         16.0
# 3  1703.0         23.0
# 4  1704.0         36.0

dta.index = pd.Index(sm.tsa.datetools.dates_from_range("1700", "2008"))
dta.index.freq = dta.index.inferred_freq
del dta["YEAR"]
# print(dta.head())
#             SUNACTIVITY
# 1700-12-31          5.0
# 1701-12-31         11.0
# 1702-12-31         16.0
# 1703-12-31         23.0
# 1704-12-31         36.0


# chart1
# dta.plot(figsize=(12, 8))
# plt.title("Chart 1")
# plt.show()

# chart2
# fig = plt.figure(figsize=(12, 8))
# ax1 = fig.add_subplot(211)
# fig = sm.graphics.tsa.plot_acf(dta.values.squeeze(), lags=40, ax=ax1)
# ax2 = fig.add_subplot(212)
# fig = sm.graphics.tsa.plot_pacf(dta, lags=40, ax=ax2)
# plt.title("Chart 2 Partial Autocorrelation")
# plt.show()

arma_mod20 = ARIMA(dta, order=(2, 0, 0)).fit()
# print(arma_mod20.params)
# const      49.746198
# ar.L1       1.390633
# ar.L2      -0.688573
# sigma2    274.727181
# dtype: float64


arma_mod30 = ARIMA(dta, order=(3, 0, 0)).fit()
# print(arma_mod20.aic, arma_mod20.bic, arma_mod20.hqic)
# 2622.6370933008184 2637.570458408409 2628.6074811460644

# print(arma_mod30.params)
# const      49.751911
# ar.L1       1.300818
# ar.L2      -0.508102
# ar.L3      -0.129644
# sigma2    270.101140
# dtype: float64

# Does our model obey the theory?
# print(sm.stats.durbin_watson(arma_mod30.resid.values))
# 1.9564953607422593


# chart3
# fig = plt.figure(figsize=(12, 8))
# ax = fig.add_subplot(111)
# ax = arma_mod30.resid.plot(ax=ax)
# plt.title("Chart 3 Arma_mod30")
# plt.show()


resid = arma_mod30.resid
# print(stats.normaltest(resid))
# NormaltestResult(statistic=49.84393225684826, pvalue=1.5015079370537675e-11)


# Chart4
# fig = plt.figure(figsize=(12, 8))
# ax = fig.add_subplot(111)
# fig = qqplot(resid, line="q", ax=ax, fit=True)
# plt.title("Chart 4 Resid")
# plt.show()


# Chart 5
# fig = plt.figure(figsize=(12, 8))
# ax1 = fig.add_subplot(211)
# fig = sm.graphics.tsa.plot_acf(resid.values.squeeze(), lags=40, ax=ax1)
# ax2 = fig.add_subplot(212)
# fig = sm.graphics.tsa.plot_pacf(resid, lags=40, ax=ax2)
# plt.title("Chart 5 Partial Auotcorrelation Resid")
# plt.show()



r, q, p = sm.tsa.acf(resid.values.squeeze(), fft=True, qstat=True)
data = np.c_[np.arange(1, 25), r[1:], q, p]

table = pd.DataFrame(data, columns=["lag", "AC", "Q", "Prob(>Q)"])
# print(table.set_index("lag"))
#             AC          Q      Prob(>Q)
# lag
# 1.0   0.009170   0.026239  8.713184e-01
# 2.0   0.041793   0.572982  7.508939e-01
# 3.0  -0.001338   0.573544  9.024612e-01
# 4.0   0.136086   6.408642  1.706385e-01
# 5.0   0.092465   9.111351  1.047043e-01
# 6.0   0.091947  11.792661  6.675737e-02
# 7.0   0.068747  13.296552  6.520425e-02
# 8.0  -0.015022  13.368601  9.978086e-02
# 9.0   0.187590  24.641072  3.394963e-03
# 10.0  0.213715  39.320758  2.230588e-05
# 11.0  0.201079  52.359565  2.346490e-07
# 12.0  0.117180  56.802479  8.580351e-08
# 13.0 -0.014057  56.866630  1.895209e-07
# 14.0  0.015398  56.943864  4.000370e-07
# 15.0 -0.024969  57.147642  7.746545e-07
# 16.0  0.080916  59.295052  6.876728e-07
# 17.0  0.041138  59.852008  1.111674e-06
# 18.0 -0.052022  60.745723  1.549418e-06
# 19.0  0.062496  62.040011  1.832778e-06
# 20.0 -0.010303  62.075305  3.383285e-06
# 21.0  0.074453  63.924941  3.195540e-06
# 22.0  0.124954  69.152955  8.984238e-07
# 23.0  0.093162  72.069214  5.803579e-07
# 24.0 -0.082152  74.344911  4.716005e-07

# This indicates a lack of fit.


# In-sample dynamic prediction. How good does our model do?
predict_sunspots = arma_mod30.predict("1990", "2012", dynamic=True)
# print(predict_sunspots)
# 1990-12-31    167.048337
# 1991-12-31    140.995022
# 1992-12-31     94.862115
# 1993-12-31     46.864439
# 1994-12-31     11.246106
# 1995-12-31     -4.718265
# 1996-12-31     -1.164628
# 1997-12-31     16.187246
# 1998-12-31     39.022948
# 1999-12-31     59.450799
# 2000-12-31     72.171269
# 2001-12-31     75.378329
# 2002-12-31     70.438480
# 2003-12-31     60.733987
# 2004-12-31     50.204383
# 2005-12-31     42.078584
# 2006-12-31     38.116648
# 2007-12-31     38.456730
# 2008-12-31     41.965644
# 2009-12-31     46.870948
# 2010-12-31     51.424877
# 2011-12-31     54.401403
# 2012-12-31     55.323515
# Freq: A-DEC, Name: predicted_mean, dtype: float64


def mean_forecast_err(y, yhat):
    return y.sub(yhat).mean()

# print(mean_forecast_err(dta.SUNACTIVITY, predict_sunspots))
# 5.634832982023026

# Exercise: Can you obtain a better fit for the Sunspots model? (Hint: sm.tsa.AR 
# has a method select_order)

# Simulated ARMA(4,1): Model Identification is Difficult
np.random.seed(1234)
# include zero-th lag
arparams = np.array([1, 0.75, -0.65, -0.55, 0.9])
maparams = np.array([1, 0.65])


# Let’s make sure this model is estimable.

arma_t = ArmaProcess(arparams, maparams)

# print(arma_t.isinvertible)
# True

# print(arma_t.isstationary)
# False
# What does this mean?

# chart6
# fig = plt.figure(figsize=(12, 8))
# ax = fig.add_subplot(111)
# ax.plot(arma_t.generate_sample(nsample=50))
# plt.title("Chart 6")
# plt.show()


arparams = np.array([1, 0.35, -0.15, 0.55, 0.1])
maparams = np.array([1, 0.65])
arma_t = ArmaProcess(arparams, maparams)
# print(arma_t.isstationary)
# True

arma_rvs = arma_t.generate_sample(nsample=500, burnin=250, scale=2.5)

# chart7
# fig = plt.figure(figsize=(12, 8))
# ax1 = fig.add_subplot(211)
# fig = sm.graphics.tsa.plot_acf(arma_rvs, lags=40, ax=ax1)
# ax2 = fig.add_subplot(212)
# fig = sm.graphics.tsa.plot_pacf(arma_rvs, lags=40, ax=ax2)
# plt.title("Chart 7 Partial AutoCorrelation")
# plt.show()


# For mixed ARMA processes the Autocorrelation function is a mixture of 
# exponentials and damped sine waves after (q-p) lags.

# The partial autocorrelation function is a mixture of exponentials and dampened 
# sine waves after (p-q) lags.


lags = int(10 * np.log10(arma_rvs.shape[0]))
arma11 = ARIMA(arma_rvs, order=(1, 0, 1)).fit()
resid = arma11.resid
r, q, p = sm.tsa.acf(resid, nlags=lags, fft=True, qstat=True)
data = np.c_[range(1, lags + 1), r[1:], q, p]
table = pd.DataFrame(data, columns=["lag", "AC", "Q", "Prob(>Q)"])
# print(table.set_index("lag"))
#   warn('Non-invertible starting MA parameters found.'
#             AC           Q      Prob(>Q)
# lag
# 1.0   0.270060   36.685475  1.388098e-09
# 2.0  -0.176298   52.350878  4.286979e-12
# 3.0  -0.421845  142.222416  1.253837e-30
# 4.0  -0.051476  143.563346  4.871219e-30
# 5.0   0.082234  146.992377  5.829146e-30
# 6.0   0.233983  174.809780  4.290587e-35
# 7.0   0.008907  174.850176  2.380860e-34
# 8.0  -0.016885  174.995617  1.156108e-33
# 9.0  -0.084050  178.606983  9.897911e-34
# 10.0  0.030537  179.084658  3.628929e-33
# 11.0 -0.011907  179.157428  1.529834e-32
# 12.0  0.014855  179.270923  6.020929e-32
# 13.0 -0.064748  181.431636  8.659643e-32
# 14.0 -0.017771  181.594739  3.073604e-31
# 15.0 -0.000056  181.594741  1.133492e-30
# 16.0  0.072659  184.332597  1.138034e-30
# 17.0 -0.006596  184.355206  3.905586e-30
# 18.0 -0.046422  185.477426  7.858948e-30
# 19.0 -0.060970  187.417275  1.064560e-29
# 20.0  0.037853  188.166535  2.427430e-29
# 21.0  0.022328  188.427785  6.741894e-29
# 22.0 -0.027919  188.837101  1.711515e-28
# 23.0 -0.089912  193.091014  7.662734e-29
# 24.0 -0.017374  193.250189  2.103499e-28
# 25.0  0.101231  198.665345  5.539993e-29
# 26.0  0.074205  201.581157  4.388529e-29


arma41 = ARIMA(arma_rvs, order=(4, 0, 1)).fit()
resid = arma41.resid
r, q, p = sm.tsa.acf(resid, nlags=lags, fft=True, qstat=True)
data = np.c_[range(1, lags + 1), r[1:], q, p]
table = pd.DataFrame(data, columns=["lag", "AC", "Q", "Prob(>Q)"])
# print(table.set_index("lag"))
#   warn('Non-invertible starting MA parameters found.'
#             AC          Q  Prob(>Q)
# lag
# 1.0  -0.000872   0.000382  0.984401
# 2.0  -0.004506   0.010618  0.994705
# 3.0   0.024208   0.306582  0.958786
# 4.0   0.007680   0.336427  0.987343
# 5.0  -0.008645   0.374327  0.996007
# 6.0   0.075956   3.305717  0.769613
# 7.0   0.006781   3.329128  0.852979
# 8.0   0.010347   3.383750  0.908022
# 9.0   0.006191   3.403342  0.946139
# 10.0  0.005413   3.418351  0.969798
# 11.0 -0.030234   3.887545  0.973092
# 12.0 -0.015469   4.010623  0.983244
# 13.0 -0.023504   4.295353  0.987625
# 14.0 -0.024390   4.602570  0.990612
# 15.0 -0.016077   4.736327  0.994133
# 16.0  0.037052   5.448290  0.993036
# 17.0 -0.003529   5.454760  0.996164
# 18.0 -0.030166   5.928642  0.996478
# 19.0 -0.057528   7.655593  0.989813
# 20.0  0.023367   7.941106  0.992250
# 21.0 -0.023638   8.233890  0.994071
# 22.0  0.002937   8.238419  0.996470
# 23.0 -0.056039   9.890916  0.991931
# 24.0 -0.025371  10.230346  0.993530
# 25.0  0.082610  13.836503  0.964508
# 26.0  0.038937  14.639312  0.963579


# Exercise: How good of in-sample prediction can you do for another series, say, 
# CPI¶

macrodta = sm.datasets.macrodata.load_pandas().data
macrodta.index = pd.Index(sm.tsa.datetools.dates_from_range("1959Q1", "2009Q3"))
cpi = macrodta["cpi"]
# Hint:
# chart8
# fig = plt.figure(figsize=(12, 8))
# ax = fig.add_subplot(111)
# ax = cpi.plot(ax=ax)
# ax.legend()
# plt.title("Chart 8")
# plt.show()


# P-value of the unit-root test, resoundingly rejects the null of a unit-root.
print(sm.tsa.adfuller(cpi)[1])
# 0.9904328188337421



























































































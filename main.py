# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

import pandas as pd
import time

from Merton76 import Merton76
from NormalInverseGaussian import NormalInverseGaussian
from VarianceGamma import VarianceGamma


def filterData(data):
    newData = data[(data['Moneyness'] > 0.8) & (data['Moneyness'] < 1.2)]
    nExpirations = newData['Expiration Date of the Option'].unique()[:10]
    firstDate = newData.iloc[0]['The Date of this Price']
    newData = newData.loc[newData['Expiration Date of the Option'].isin(nExpirations)]
    dataForFirstDate = newData[newData['The Date of this Price']==firstDate]
    return dataForFirstDate

def loadData(filename: str):
    df = pd.read_excel(filename)
    df["Moneyness"] = 0
    df["TTM"] = 0
    df["The Date of this Price"] = pd.to_datetime(df["The Date of this Price"], format='%Y%m%d')
    df["Expiration Date of the Option"] = pd.to_datetime(df["Expiration Date of the Option"], format='%Y%m%d')
    df["Moneyness"] = df["Moneyness"].astype(float)
    df["TTM"] = df["TTM"].astype(float)
    return df

'''Calibrates M76 for SPX for date of 01-03-2017 with Expiration Date 06-30-2017'''
def runMerton76ForSPXSingleExpirationDate(filename: str):
    data = loadData(filename)
    S0 = data.iloc[0]['Current Price Of Underlying']
    K = data.iloc[0]['Strike Price']
    T = (data.iloc[0]['Expiration Date of the Option'] - data.iloc[0]['The Date of this Price']).days / 365
    r = .03
    impVol = data.iloc[0]['Implied Vol']
    dividendRate = .02

    for index, row in data.iterrows():
        moneyness = row['Current Price Of Underlying'] / row["Strike Price"]
        data.at[index, "Moneyness"] = moneyness
        data.at[index, "TTM"] = (row["Expiration Date of the Option"] - row["The Date of this Price"]).days / 365.

    m76 = Merton76(S0, K, T, r, impVol, dividendRate, 'Call', data)
    start = time.perf_counter()
    calibratedParams = m76.calibrate()
    stop = time.perf_counter()
    print('To Calibrate Single Exp M76 took '+str(stop-start) + 'seconds')
    m76.generate_plot(calibratedParams, data, True)

'''Calibrates VG for SPX for date of 01-03-2017 with Expiration Date 06-30-2017'''
def runVGForSPXSingleExpirationDate(filename: str):
    data = loadData(filename)
    S0 = data.iloc[0]['Current Price Of Underlying']
    K = data.iloc[0]['Strike Price']
    T = (data.iloc[0]['Expiration Date of the Option'] - data.iloc[0]['The Date of this Price']).days / 365
    r = .03
    impVol = data.iloc[0]['Implied Vol']
    dividendRate = .02

    for index, row in data.iterrows():
        moneyness = row['Current Price Of Underlying'] / row["Strike Price"]
        data.at[index, "Moneyness"] = moneyness
        data.at[index, "TTM"] = (row["Expiration Date of the Option"] - row["The Date of this Price"]).days / 365.

    vg = VarianceGamma(S0, K, T, r, impVol, dividendRate, 'Call', data)
    start = time.perf_counter()
    calibratedParams = vg.calibrate()
    stop = time.perf_counter()
    print('To Calibrate Single Exp VG took ' + str(stop - start) + 'seconds')
    vg.generate_plot(calibratedParams, data, True)

'''Calibrates VG for SPX for date of 01-03-2017 with Expiration Date 06-30-2017'''
def runNIGForSPXSingleExpirationDate(filename: str):
    data = loadData(filename)
    S0 = data.iloc[0]['Current Price Of Underlying']
    K = data.iloc[0]['Strike Price']
    T = (data.iloc[0]['Expiration Date of the Option'] - data.iloc[0]['The Date of this Price']).days / 365
    r = .03
    impVol = data.iloc[0]['Implied Vol']
    dividendRate = .02

    for index, row in data.iterrows():
        moneyness = row['Current Price Of Underlying'] / row["Strike Price"]
        data.at[index, "Moneyness"] = moneyness
        data.at[index, "TTM"] = (row["Expiration Date of the Option"] - row["The Date of this Price"]).days / 365.

    nig = NormalInverseGaussian(S0, K, T, r, impVol, dividendRate, 'Call', data)
    start = time.perf_counter()
    calibratedParams = nig.calibrate()
    stop = time.perf_counter()
    print('To Calibrate Single Exp NIG took ' + str(stop - start) + 'seconds')
    nig.generate_plot(calibratedParams, data, True)


''' Below are Calibrations for 3 Expiries'''
'''Calibrates M76 for SPX for date of 01-03-2017 with Expiration Date 06-30-2017'''
def runMerton76ForSPXMultipleExpirationDates(filename: str):
    data = loadData(filename)
    S0 = data.iloc[0]['Current Price Of Underlying']
    K = data.iloc[0]['Strike Price']
    T = (data.iloc[0]['Expiration Date of the Option'] - data.iloc[0]['The Date of this Price']).days / 365
    r = .03
    impVol = data.iloc[0]['Implied Vol']
    dividendRate = .02

    for index, row in data.iterrows():
        moneyness = row['Current Price Of Underlying'] / row["Strike Price"]
        data.at[index, "Moneyness"] = moneyness
        data.at[index, "TTM"] = (row["Expiration Date of the Option"] - row["The Date of this Price"]).days / 365.

    data = filterData(data)
    m76 = Merton76(S0, K, T, r, impVol, dividendRate, 'Call', data)
    start = time.perf_counter()
    calibratedParams = m76.calibrate()
    stop = time.perf_counter()
    print('To Calibrate Multiple Exp M76 took ' + str(stop - start) + 'seconds')
    m76.generate_plot(calibratedParams, data)

'''Calibrates VG for SPX for date of 01-03-2017 with Expiration Date 06-30-2017'''
def runVGForSPXMultipleExpirationDates(filename: str):
    data = loadData(filename)
    S0 = data.iloc[0]['Current Price Of Underlying']
    K = data.iloc[0]['Strike Price']
    T = (data.iloc[0]['Expiration Date of the Option'] - data.iloc[0]['The Date of this Price']).days / 365
    r = .03
    impVol = data.iloc[0]['Implied Vol']
    dividendRate = .02

    for index, row in data.iterrows():
        moneyness = row['Current Price Of Underlying'] / row["Strike Price"]
        data.at[index, "Moneyness"] = moneyness
        data.at[index, "TTM"] = (row["Expiration Date of the Option"] - row["The Date of this Price"]).days / 365.

    data = filterData(data)
    vg = VarianceGamma(S0, K, T, r, impVol, dividendRate, 'Call', data)
    start = time.perf_counter()
    calibratedParams = vg.calibrate()
    stop = time.perf_counter()
    print('To Calibrate Multi Exp VG took ' + str(stop - start) + 'seconds')
    vg.generate_plot(calibratedParams, data)

'''Calibrates VG for SPX for date of 01-03-2017 with Expiration Date 06-30-2017'''
def runNIGForSPXMultipleExpirationDates(filename: str):
    data = loadData(filename)
    S0 = data.iloc[0]['Current Price Of Underlying']
    K = data.iloc[0]['Strike Price']
    T = (data.iloc[0]['Expiration Date of the Option'] - data.iloc[0]['The Date of this Price']).days / 365
    r = .03
    impVol = data.iloc[0]['Implied Vol']
    dividendRate = .02

    for index, row in data.iterrows():
        moneyness = row['Current Price Of Underlying'] / row["Strike Price"]
        data.at[index, "Moneyness"] = moneyness
        data.at[index, "TTM"] = (row["Expiration Date of the Option"] - row["The Date of this Price"]).days / 365.

    data = filterData(data)
    nig = NormalInverseGaussian(S0, K, T, r, impVol, dividendRate, 'Call', data)
    start = time.perf_counter()
    calibratedParams = nig.calibrate()
    stop = time.perf_counter()
    print('To Calibrate Multi Exp NIG took ' + str(stop - start) + 'seconds')
    nig.generate_plot(calibratedParams, data)

#print('\n\nStarting Single Exp Calibrations for M76 \n\n')
#runMerton76ForSPXSingleExpirationDate('spx012017SingleExp.xlsx')
#print('\n\nStarting Single Exp Calibrations for VG \n\n')
#runVGForSPXSingleExpirationDate('spx012017SingleExp.xlsx')
#print('\n\nStarting Single Exp Calibrations for NIG \n\n')
#runNIGForSPXSingleExpirationDate('spx012017SingleExp.xlsx')

'''Will start calibrations for 3 expirations'''
#print('\n\nStarting Multi Exp Calibrations for M76 \n\n')
#runMerton76ForSPXMultipleExpirationDates('spx012017.xlsx')
#print('\n\nStarting Multi Exp Calibrations for VG \n\n')
#runVGForSPXMultipleExpirationDates('spx012017.xlsx')
print('\n\nStarting Multi Exp Calibrations for NIG \n\n')
runNIGForSPXMultipleExpirationDates('spx012017.xlsx')

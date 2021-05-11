# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

import pandas as pd
import time

from EuropeanOption import EuropeanOption
from Merton76 import Merton76
from NormalInverseGaussian import NormalInverseGaussian
from VarianceGamma import VarianceGamma

interestRate: float = .0054
dividendRateSPX: float = .02
dividendRateNDX: float = .0173
spxData: any
npxData: any

##TODO: Does Calling FilterData after getting S0, K, etc. cause those values to be incorrect when using filterData()

def filterData(data):
    newData = data[(data['Moneyness'] > 0.8) & (data['Moneyness'] < 1.2)]
    nExpirations = newData['Expiration Date of the Option'].unique()[:7]
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

def loadDataNDX(filename: str):
    df = pd.read_excel(filename)
    df["Moneyness"] = 0
    df["TTM"] = 0
    df["Implied Vol"] = 0
    df["The Date of this Price"] = pd.to_datetime(df["The Date of this Price"], format='%Y%m%d')
    df["Expiration Date of the Option"] = pd.to_datetime(df["Expiration Date of the Option"], format='%Y%m%d')
    df["Moneyness"] = df["Moneyness"].astype(float)
    df["TTM"] = df["TTM"].astype(float)
    df["Implied Vol"] = df["Implied Vol"].astype(float)
    return df

def runMerton76ForNDXSingleExpirationDate(filename: str):
    data = loadDataNDX(filename)
    S0 = data.iloc[0]['Current Price Of Underlying']
    K = data.iloc[0]['Strike Price']
    T = (data.iloc[0]['Expiration Date of the Option'] - data.iloc[0]['The Date of this Price']).days / 365

    for index, row in data.iterrows():
        premium = row['Premium']
        moneyness = row['Current Price Of Underlying'] / row["Strike Price"]
        data.at[index, "Moneyness"] = moneyness
        data.at[index, "TTM"] = (row["Expiration Date of the Option"] - row["The Date of this Price"]).days / 365.
        option = EuropeanOption(S0, row['Strike Price'], 0, data.at[index, 'TTM'], interestRate, .15, dividendRateNDX, 'Call', premium)
        data.at[index, "Implied Vol"] = option.imp_vol()

    impVol = data.iloc[0]['Implied Vol']
    m76 = Merton76(S0, K, T, interestRate, impVol, dividendRateNDX, 'Call', data)
    start = time.perf_counter()
    calibratedParams = m76.calibrate()
    stop = time.perf_counter()
    print('To Calibrate Single Exp M76 for NDX Data took ' + str(stop - start) + 'seconds')
    m76.generate_plot(calibratedParams, data, True, True)

def runVGForNDXSingleExpirationDate(filename: str):
    data = loadDataNDX(filename)
    S0 = data.iloc[0]['Current Price Of Underlying']
    K = data.iloc[0]['Strike Price']
    T = (data.iloc[0]['Expiration Date of the Option'] - data.iloc[0]['The Date of this Price']).days / 365

    for index, row in data.iterrows():
        premium = row['Premium']
        moneyness = row['Current Price Of Underlying'] / row["Strike Price"]
        data.at[index, "Moneyness"] = moneyness
        data.at[index, "TTM"] = (row["Expiration Date of the Option"] - row["The Date of this Price"]).days / 365.
        option = EuropeanOption(S0, row['Strike Price'], 0, data.at[index, 'TTM'], interestRate, .15, dividendRateNDX, 'Call', premium)
        data.at[index, "Implied Vol"] = option.imp_vol()

    impVol = data.iloc[0]['Implied Vol']
    vg = VarianceGamma(S0, K, T, interestRate, impVol, dividendRateNDX, 'Call', data)
    start = time.perf_counter()
    calibratedParams = vg.calibrate()
    stop = time.perf_counter()
    print('To Calibrate Single Exp VG For NDX took ' + str(stop - start) + 'seconds')
    vg.generate_plot(calibratedParams, data, True, True)

def runNIGForNDXSingleExpirationDate(filename: str):
    data = loadDataNDX(filename)
    S0 = data.iloc[0]['Current Price Of Underlying']
    K = data.iloc[0]['Strike Price']
    T = (data.iloc[0]['Expiration Date of the Option'] - data.iloc[0]['The Date of this Price']).days / 365

    for index, row in data.iterrows():
        premium = row['Premium']
        moneyness = row['Current Price Of Underlying'] / row["Strike Price"]
        data.at[index, "Moneyness"] = moneyness
        data.at[index, "TTM"] = (row["Expiration Date of the Option"] - row["The Date of this Price"]).days / 365.
        option = EuropeanOption(S0, row['Strike Price'], 0, data.at[index, 'TTM'], interestRate, .15, dividendRateNDX, 'Call', premium)
        data.at[index, "Implied Vol"] = option.imp_vol()

    impVol = data.iloc[0]['Implied Vol']
    normalInverseGaussian = NormalInverseGaussian(S0, K, T, interestRate, impVol, dividendRateNDX, 'Call', data)
    start = time.perf_counter()
    calibratedParams = normalInverseGaussian.calibrate()
    stop = time.perf_counter()
    print('To Calibrate Single Exp NIG For NDX took ' + str(stop - start) + 'seconds')
    normalInverseGaussian.generate_plot(calibratedParams, data, True, True)

def runMerton76ForNDXMultipleExpirationDates(filename: str):
    data = loadDataNDX(filename)
    S0 = data.iloc[0]['Current Price Of Underlying']

    for index, row in data.iterrows():
        premium = row['Premium']
        moneyness = row['Current Price Of Underlying'] / row["Strike Price"]
        data.at[index, "Moneyness"] = moneyness
        data.at[index, "TTM"] = (row["Expiration Date of the Option"] - row["The Date of this Price"]).days / 365.
        option = EuropeanOption(S0, row['Strike Price'], 0, data.at[index, 'TTM'], interestRate, .15, dividendRateNDX, 'Call', premium)
        data.at[index, "Implied Vol"] = option.imp_vol()

    data=filterData(data)
    m76 = Merton76(
        data.iloc[0]['Current Price Of Underlying'],
        data.iloc[0]['Strike Price'],
        (data.iloc[0]['Expiration Date of the Option'] - data.iloc[0]['The Date of this Price']).days / 365,
        interestRate,
        data.iloc[0]['Implied Vol'],
        dividendRateSPX,
        'Call',
        data)

    start = time.perf_counter()
    calibratedParams = m76.calibrate()
    stop = time.perf_counter()
    print('To Calibrate Multi Exp M76 for NDX took ' + str(stop - start) + 'seconds')
    m76.generate_plot(calibratedParams, data, False, True)

def runVGForNDXMultipleExpirationDates(filename: str):
    data = loadDataNDX(filename)
    S0 = data.iloc[0]['Current Price Of Underlying']
    K = data.iloc[0]['Strike Price']
    T = (data.iloc[0]['Expiration Date of the Option'] - data.iloc[0]['The Date of this Price']).days / 365

    for index, row in data.iterrows():
        premium = row['Premium']
        moneyness = row['Current Price Of Underlying'] / row["Strike Price"]
        data.at[index, "Moneyness"] = moneyness
        data.at[index, "TTM"] = (row["Expiration Date of the Option"] - row["The Date of this Price"]).days / 365.
        option = EuropeanOption(S0, row['Strike Price'], 0, data.at[index, 'TTM'], interestRate, .15, dividendRateNDX, 'Call', premium)
        data.at[index, "Implied Vol"] = option.imp_vol()

    data=filterData(data)
    impVol = data.iloc[0]['Implied Vol']
    vg = VarianceGamma(
        data.iloc[0]['Current Price Of Underlying'],
        data.iloc[0]['Strike Price'],
        (data.iloc[0]['Expiration Date of the Option'] - data.iloc[0]['The Date of this Price']).days / 365,
        interestRate,
        data.iloc[0]['Implied Vol'],
        dividendRateSPX,
        'Call',
        data)

    start = time.perf_counter()
    calibratedParams = vg.calibrate()
    stop = time.perf_counter()
    print('To Calibrate Multi Exp VG for NDX took ' + str(stop - start) + 'seconds')
    vg.generate_plot(calibratedParams, data, False, True)

def runNIGForNDXMultipleExpirationDates(filename: str):
    data = loadDataNDX(filename)
    S0 = data.iloc[0]['Current Price Of Underlying']

    for index, row in data.iterrows():
        premium = row['Premium']
        moneyness = row['Current Price Of Underlying'] / row["Strike Price"]
        data.at[index, "Moneyness"] = moneyness
        data.at[index, "TTM"] = (row["Expiration Date of the Option"] - row["The Date of this Price"]).days / 365.
        option = EuropeanOption(S0, row['Strike Price'], 0, data.at[index, 'TTM'], interestRate, .15, dividendRateNDX, 'Call', premium)
        data.at[index, "Implied Vol"] = option.imp_vol()

    data=filterData(data)
    normalInverseGaussian = NormalInverseGaussian(
        data.iloc[0]['Current Price Of Underlying'],
        data.iloc[0]['Strike Price'],
        (data.iloc[0]['Expiration Date of the Option'] - data.iloc[0]['The Date of this Price']).days / 365,
        interestRate,
        data.iloc[0]['Implied Vol'],
        dividendRateSPX,
        'Call',
        data)

    start = time.perf_counter()
    calibratedParams = normalInverseGaussian.calibrate()
    stop = time.perf_counter()
    print('To Calibrate Multi Exp NIG For NDX took ' + str(stop - start) + 'seconds')
    normalInverseGaussian.generate_plot(calibratedParams, data, False, True)

''' Does All Calibration for For SPX Data '''
#############################################


'''Calibrates M76 for SPX for date of 01-03-2017 with Expiration Date 06-30-2017'''
def runMerton76ForSPXSingleExpirationDate(filename: str):
    data = loadData(filename)
    S0 = data.iloc[0]['Current Price Of Underlying']
    K = data.iloc[0]['Strike Price']
    T = (data.iloc[0]['Expiration Date of the Option'] - data.iloc[0]['The Date of this Price']).days / 365
    impVol = data.iloc[0]['Implied Vol']

    for index, row in data.iterrows():
        moneyness = row['Current Price Of Underlying'] / row["Strike Price"]
        data.at[index, "Moneyness"] = moneyness
        data.at[index, "TTM"] = (row["Expiration Date of the Option"] - row["The Date of this Price"]).days / 365.

    m76 = Merton76(S0, K, T, interestRate, impVol, dividendRateSPX, 'Call', data)
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
    impVol = data.iloc[0]['Implied Vol']

    for index, row in data.iterrows():
        moneyness = row['Current Price Of Underlying'] / row["Strike Price"]
        data.at[index, "Moneyness"] = moneyness
        data.at[index, "TTM"] = (row["Expiration Date of the Option"] - row["The Date of this Price"]).days / 365.

    vg = VarianceGamma(S0, K, T, interestRate, impVol, dividendRateSPX, 'Call', data)
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
    impVol = data.iloc[0]['Implied Vol']

    for index, row in data.iterrows():
        moneyness = row['Current Price Of Underlying'] / row["Strike Price"]
        data.at[index, "Moneyness"] = moneyness
        data.at[index, "TTM"] = (row["Expiration Date of the Option"] - row["The Date of this Price"]).days / 365.

    nig = NormalInverseGaussian(S0, K, T, interestRate, impVol, dividendRateSPX, 'Call', data)
    start = time.perf_counter()
    calibratedParams = nig.calibrate()
    stop = time.perf_counter()
    print('To Calibrate Single Exp NIG took ' + str(stop - start) + 'seconds')
    nig.generate_plot(calibratedParams, data, True)


''' Below are Calibrations for 10 Expiries'''
'''Calibrates M76 for SPX for date of 01-03-2017 with Expiration Date 06-30-2017'''
def runMerton76ForSPXMultipleExpirationDates(filename: str):
    data = loadData(filename)

    for index, row in data.iterrows():
        moneyness = row['Current Price Of Underlying'] / row["Strike Price"]
        data.at[index, "Moneyness"] = moneyness
        data.at[index, "TTM"] = (row["Expiration Date of the Option"] - row["The Date of this Price"]).days / 365.

    data = filterData(data)
    m76 = Merton76(
        data.iloc[0]['Current Price Of Underlying'],
        data.iloc[0]['Strike Price'],
        (data.iloc[0]['Expiration Date of the Option'] - data.iloc[0]['The Date of this Price']).days / 365,
        interestRate,
        data.iloc[0]['Implied Vol'],
        dividendRateSPX,
        'Call',
        data)

    start = time.perf_counter()
    calibratedParams = m76.calibrate()
    stop = time.perf_counter()
    print('To Calibrate Multiple Exp M76 took ' + str(stop - start) + 'seconds')
    m76.generate_plot(calibratedParams, data)

'''Calibrates VG for SPX for date of 01-03-2017 with Expiration Date 06-30-2017'''
def runVGForSPXMultipleExpirationDates(filename: str):
    data = loadData(filename)

    for index, row in data.iterrows():
        moneyness = row['Current Price Of Underlying'] / row["Strike Price"]
        data.at[index, "Moneyness"] = moneyness
        data.at[index, "TTM"] = (row["Expiration Date of the Option"] - row["The Date of this Price"]).days / 365.

    data = filterData(data)

    vg = VarianceGamma(data.iloc[0]['Current Price Of Underlying'],
        data.iloc[0]['Strike Price'],
        (data.iloc[0]['Expiration Date of the Option'] - data.iloc[0]['The Date of this Price']).days / 365,
        interestRate,
        data.iloc[0]['Implied Vol'],
        dividendRateSPX,
        'Call',
        data)

    start = time.perf_counter()
    calibratedParams = vg.calibrate()
    stop = time.perf_counter()
    print('To Calibrate Multi Exp VG took ' + str(stop - start) + 'seconds')
    vg.generate_plot(calibratedParams, data)

'''Calibrates VG for SPX for date of 01-03-2017 with Expiration Date 06-30-2017'''
def runNIGForSPXMultipleExpirationDates(filename: str):
    data = loadData(filename)

    for index, row in data.iterrows():
        moneyness = row['Current Price Of Underlying'] / row["Strike Price"]
        data.at[index, "Moneyness"] = moneyness
        data.at[index, "TTM"] = (row["Expiration Date of the Option"] - row["The Date of this Price"]).days / 365.

    data = filterData(data)
    nig = NormalInverseGaussian(
        data.iloc[0]['Current Price Of Underlying'],
        data.iloc[0]['Strike Price'],
        (data.iloc[0]['Expiration Date of the Option'] - data.iloc[0]['The Date of this Price']).days / 365,
        interestRate,
        data.iloc[0]['Implied Vol'],
        dividendRateSPX,
        'Call',
        data)

    start = time.perf_counter()
    calibratedParams = nig.calibrate()
    stop = time.perf_counter()
    print('To Calibrate Multi Exp NIG took ' + str(stop - start) + 'seconds')
    nig.generate_plot(calibratedParams, data)

# '''NDX Calibrations'''
print('\n\nStarting NDX Single Exp Calibrations for M76 \n\n')
runMerton76ForNDXSingleExpirationDate('012017NDXSingleExp.xlsx')
print('\n\nStarting NDX Single Exp Calibrations for M76 \n\n')
runVGForNDXSingleExpirationDate('012017NDXSingleExp.xlsx')
print('\n\nStarting NDX Single Exp Calibrations for M76 \n\n')
runNIGForNDXSingleExpirationDate('012017NDXSingleExp.xlsx')

'''Will start calibrations for 7 expirations'''
print('\n\nStarting NDX Multi Exp Calibrations for M76 \n\n')
runMerton76ForNDXMultipleExpirationDates('012017NDXData.xlsx')
print('\n\nStarting NDX Multi Exp Calibrations for VG \n\n')
runVGForNDXMultipleExpirationDates('012017NDXData.xlsx')
print('\n\nStarting NDX Multi Exp Calibrations for NIG \n\n')
runNIGForNDXMultipleExpirationDates('012017NDXData.xlsx')

print('\n\nStarting SPX Single Exp Calibrations for M76 \n\n')
runMerton76ForSPXSingleExpirationDate('spx012017SingleExp.xlsx')
print('\n\nStarting SPX Single Exp Calibrations for VG \n\n')
runVGForSPXSingleExpirationDate('spx012017SingleExp.xlsx')
print('\n\nStarting SPX Single Exp Calibrations for NIG \n\n')
runNIGForSPXSingleExpirationDate('spx012017SingleExp.xlsx')
#
'''Will start calibrations for 7 expirations'''
print('\n\nStarting SPX Multi Exp Calibrations for M76 \n\n')
runMerton76ForSPXMultipleExpirationDates('spx012017.xlsx')
print('\n\nStarting SPX Multi Exp Calibrations for VG \n\n')
runVGForSPXMultipleExpirationDates('spx012017.xlsx')
print('\n\nStarting SPX Multi Exp Calibrations for NIG \n\n')
runNIGForSPXMultipleExpirationDates('spx012017.xlsx')

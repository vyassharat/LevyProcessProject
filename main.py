# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

import pandas as pd

from Merton76 import Merton76
from NormalInverseGaussian import NormalInverseGaussian
from VarianceGamma import VarianceGamma


def filterData(data):
    newData = data[(data['Moneyness'] > 0.8) & (data['Moneyness'] < 1.2)]
    filteredData = newData[newData['Expiration Date of the Option'].isin(['12/18/2020'])]
    return filteredData

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
    calibratedParams = m76.calibrate()
    m76.generate_plot(calibratedParams, data)

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
    calibratedParams = vg.calibrate()
    vg.generate_plot(calibratedParams, data)

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
    calibratedParams = nig.calibrate()
    nig.generate_plot(calibratedParams, data)

#runMerton76ForSPXSingleExpirationDate('spx012017SingleExp.xlsx')
#runVGForSPXSingleExpirationDate('spx012017SingleExp.xlsx')
runNIGForSPXSingleExpirationDate('spx012017SingleExp.xlsx')
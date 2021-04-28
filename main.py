# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

import pandas as pd

from Merton76 import Merton76

S0 = 3629.65

def filterData(data):
    newData = data[(data['Moneyness'] > 0.8) & (data['Moneyness'] < 1.2)]
    filteredData = newData[newData['Expiration Date of the Option'].isin(['12/18/2020'])]
    return filteredData

def loadData(filename: str):
    df = pd.read_excel(filename)
    df["Moneyness"] = 0
    df["Call"] = 0
    df["TTM"] = 0
    df["The Date of this Price"] = pd.DatetimeIndex(df["The Date of this Price"])
    df["Expiration Date of the Option"] = pd.DatetimeIndex(df["Expiration Date of the Option"])
    df["Moneyness"] = df["Moneyness"].astype(float)
    df["Call"] = df["Call"].astype(float)
    df["TTM"] = df["TTM"].astype(float)
    return df

data = loadData('spxcalls20201125.xlsx')
for index, row in data.iterrows():
    startDate = str(row["The Date of this Price"])
    endDate = str(row["Expiration Date of the Option"])
    data.at[index, "Call"] = (row['Highest Closing Bid Across All Exchanges'] + row[
        'Lowest  Closing Ask Across All Exchanges']) / 2
    moneyness = S0 / row["Strike"]
    data.at[index, "Moneyness"] = moneyness
    data.at[index, "TTM"] = (row["Expiration Date of the Option"] - row["The Date of this Price"]).days / 365.

filteredData = filterData(data)
options = filteredData

m76 = Merton76(S0, 3700, 1, .015, .2, .0162, 'Call', options)
m76.calibrate()
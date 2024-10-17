import pandas as pd


file_path = 'ENB2012_data.xlsx'
excel_data = pd.ExcelFile(file_path)


data = excel_data.parse('Φύλλο1')

X = data.iloc[:, :-2]
y1 = data.iloc[:, -2]
y2 = data.iloc[:, -1]

data1 = pd.concat([X, y1], axis=1)

data2 = pd.concat([X, y2], axis=1)

data1.to_csv('data1.csv', index=False)
data2.to_csv('data2.csv', index=False)

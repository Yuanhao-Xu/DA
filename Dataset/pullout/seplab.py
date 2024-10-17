import pandas as pd


file_path = 'data.csv'
data = pd.read_csv(file_path)


cleaned_data = data.dropna()


features = cleaned_data.iloc[:, :-2]
label_1 = cleaned_data.iloc[:, -1]
label_2 = cleaned_data.iloc[:, -2]


dataset_with_ifss = pd.concat([features, label_1], axis=1)
dataset_with_fmax = pd.concat([features, label_2], axis=1)


dataset_with_ifss.to_csv('dataset_ifss.csv', index=False)
dataset_with_fmax.to_csv('dataset_fmax.csv', index=False)

print("Dataset processing is complete, and it has been saved as a CSV file.")

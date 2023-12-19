import pathlib
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

files = [f for f in pathlib.Path().glob("./output/*.csv")]
for index, file in enumerate(files):
    print(f"{index} - {file}")

headers = ["AccX [mg]", "AccY [mg]", "AccZ [mg]", "State"]
dataset = pd.DataFrame()

while True:
    entry = input("Enter the id of the file you want to add to the dataset or 'exit' to exit:")

    if entry.lower() == 'exit' or entry.lower() == '':
        break

    file_selected = files[int(entry)]
    print(f"File selected: {file_selected}")

    data = pd.read_csv(file_selected, usecols=headers)

    # Padding ou troncature des données
    if data.shape[0] < 100:
        # Créer un DataFrame de padding avec les mêmes en-têtes
        padding = pd.DataFrame(np.zeros((100 - data.shape[0], len(headers))), columns=headers)
        data = pd.concat([data, padding], ignore_index=True)
    elif data.shape[0] > 100:
        data = data.iloc[:100]

    dataset = pd.concat([dataset, data], ignore_index=True)

plt.figure(figsize=(10, 6))
plt.subplot(2, 1, 1)
plt.plot(dataset['AccX [mg]'], label='AccX as a function of T')
plt.plot(dataset['AccY [mg]'], label='AccY as a function of T')
plt.plot(dataset['AccZ [mg]'], label='AccZ as a function of T')
plt.xlabel('Point number')
plt.ylabel('Acceleration')
plt.title('Dataset')

plt.subplot(2, 1, 2)
plt.plot(dataset['State'])
plt.xlabel('Point number')
plt.ylabel('State')
plt.title('Dataset state')

plt.grid(True)
plt.show()

dataset_name = input("Enter the name of the dataset to export: ")
dataset.to_csv(f'datasets/{dataset_name}.csv', index=False, header=True)

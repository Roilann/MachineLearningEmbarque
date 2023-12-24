import json
import pathlib

import numpy as np
import pandas as pd
import tensorflow as tf
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler


def print_dataset(dataset, title: str = 'Dataset state'):
    plt.figure(figsize=(10, 6))
    plt.subplot(2, 1, 1)
    plt.plot(dataset['AccX [mg]'], label='AccX as a function of T')
    plt.plot(dataset['AccY [mg]'], label='AccY as a function of T')
    plt.plot(dataset['AccZ [mg]'], label='AccZ as a function of T')
    plt.xlabel('Point number')
    plt.ylabel('Acceleration')
    plt.title(title)

    plt.subplot(2, 1, 2)
    plt.plot(dataset['State'])
    plt.xlabel('Point number')
    plt.ylabel('State')

    plt.grid(True)
    plt.show()


h5_files_path = [f for f in pathlib.Path().glob("./models/*/*.h5")]
json_files_path = [f for f in pathlib.Path().glob("./models/*/*.json")]

models_to_load = []

for index, file_path in enumerate(h5_files_path):
    if file_path.name.split('.')[0] == json_files_path[index].name.split('.')[0]:
        models_to_load.append({
            'h5': file_path,
            'json': json_files_path[index]
        })

        print(f"{len(models_to_load) - 1} - {file_path.name.split('.')[0]}")

models = []
results = []

while True:
    entry = input("Enter the id of the model you want to add to the dataset or 'exit' to exit:")

    if entry.lower() == 'exit' or entry.lower() == '':
        break

    model_structure_file = open(models_to_load[int(entry)]['json'], 'r')
    model_structure = model_structure_file.read()
    model_structure_file.close()

    model = tf.keras.models.model_from_json(model_structure)
    model.load_weights(models_to_load[int(entry)]['h5'])

    models.append(model)
    results.append({
        'id': int(entry),
        'result': 0,
    })

headers = ["AccX [mg]", "AccY [mg]", "AccZ [mg]", "State"]

files_path = [f for f in pathlib.Path().glob("./datasets/*.csv")]
files_path = files_path + [f for f in pathlib.Path().glob("./datasets/*/concat_data_+_dataset.csv")]
files_path = files_path + [f for f in pathlib.Path().glob("./datasets/*/compile_data_+_dataset.csv")]

for index, file_path in enumerate(files_path):
    print(f"{index} - {file_path}")

while True:
    entry = input("Enter the id of the file you want to test or 'exit' to exit:")

    if entry.lower() == 'exit' or entry.lower() == '':
        break

    file_selected = files_path[int(entry)]

    data = pd.read_csv(file_selected, usecols=headers)
    if data.shape[0] % 100:
        print(f"File {file_selected} has {data.shape[0]} rows, which is not a multiple of 100, so it will be ignored")
        continue
    else:
        print(f"File selected: {file_selected.name}")

    print_dataset(data)

    data = data.values
    dataset_count = len(data) // 100

    E_raw_dataset = data[:, :-1]
    Y_raw_dataset = data[:, -1]

    scaler = MinMaxScaler(feature_range=(0, 1))
    E_raw_dataset = scaler.fit_transform(E_raw_dataset)

    E_datasets = np.array_split(E_raw_dataset, dataset_count)
    Y_datasets = np.array_split(Y_raw_dataset, dataset_count)
    Y_datasets = [lot[0] for lot in Y_datasets]

    print("Processing ...")

    for E_index, E_dataset in enumerate(E_datasets):
        for model_index, model in enumerate(models):
            E_dataset_expand = np.expand_dims(E_dataset, axis=0)
            result = model.predict(E_dataset_expand, verbose=0)

            if (result[0][0] >= 0.5) == Y_datasets[E_index]:
                results[model_index]['result'] = results[model_index]['result'] + 1

    scores_to_show = []
    models_to_show = []

    for model_index, model in enumerate(models):
        scores_to_show.append((results[model_index]['result'] / dataset_count) * 100)
        models_to_show.append(str(results[model_index]['id']))

        plt.bar(models_to_show, scores_to_show)
        for i in range(len(models_to_show)):
            plt.text(i, scores_to_show[i], str(round(scores_to_show[i])) + ' %', ha='center')

        plt.title('Performances of differents models')
        plt.xlabel('Models id')
        plt.ylabel('Scores (%)')
        plt.show()

        results[model_index]['result'] = 0

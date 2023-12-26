import pathlib
import time

import numpy as np
import pandas as pd
import tensorflow as tf
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler

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

result = {
    'result_per_dataset': [],
    'dataset_difficulties': {
        'AccX [mg]': [],
        'AccY [mg]': [],
        'AccZ [mg]': [],
    },
    'dataset_expected_difficulties': [],
}

entry = input("Enter the id of the model you want to add or 'exit' to exit:")

if entry.lower() == 'exit' or entry.lower() == '':
    quit()

model_structure_file = open(models_to_load[int(entry)]['json'], 'r')
model_structure = model_structure_file.read()
model_structure_file.close()

model = tf.keras.models.model_from_json(model_structure)
model.load_weights(models_to_load[int(entry)]['h5'])

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

    csv_data = pd.read_csv(file_selected, usecols=headers)
    if csv_data.shape[0] % 100:
        print(f"File {file_selected} has {csv_data.shape[0]} rows, which is not a multiple of 100, so it will be ignored")
        continue
    else:
        print(f"File selected: {file_selected.name}")

    low_threshold = input("Type 'exit' to quit or enter low results threshold: ")
    if low_threshold.lower() == 'exit':
        break
    else:
        low_threshold = float(low_threshold)

    high_threshold = float(input("Enter high result threshold: "))

    data = csv_data.values
    dataset_count = len(data) // 100

    E_raw_dataset = data[:, :-1]
    Y_raw_dataset = data[:, -1]

    scaler = MinMaxScaler(feature_range=(0, 1))
    E_raw_dataset = scaler.fit_transform(E_raw_dataset)

    E_datasets = np.array_split(E_raw_dataset, dataset_count)
    Y_datasets = np.array_split(Y_raw_dataset, dataset_count)

    start_time = time.time()
    time.sleep(0.2)
    for E_index, E_dataset in enumerate(E_datasets):
        if E_index % 5 == 0:

            time_elapsed = time.time() - start_time
            time_for_5_datasets = time_elapsed / (E_index + 1)
            datasets_remaining = dataset_count - (E_index + 1)

            time_remaining = datasets_remaining * time_for_5_datasets

            minutes = int(time_remaining // 60)
            seconds = int(time_remaining % 60)

            print(f"\rProcessing dataset {E_index + 1}/{dataset_count}, time remaining: {minutes}:{seconds:02d}", end="")

        E_dataset_expand = np.expand_dims(E_dataset, axis=0)
        prediction = model.predict(E_dataset_expand, verbose=0, use_multiprocessing=True)

        if low_threshold <= prediction[0][0] <= high_threshold:
            result_per_dataset = np.full(100, prediction[0][0]).tolist()

            result['result_per_dataset'] = np.concatenate((result['result_per_dataset'],
                                                           result_per_dataset), axis=0)

            result['dataset_difficulties']['AccX [mg]'] = np.concatenate((result['dataset_difficulties']['AccX [mg]'],
                                                                          E_dataset.transpose()[0]), axis=0)

            result['dataset_difficulties']['AccY [mg]'] = np.concatenate((result['dataset_difficulties']['AccY [mg]'],
                                                                          E_dataset.transpose()[1]), axis=0)

            result['dataset_difficulties']['AccZ [mg]'] = np.concatenate((result['dataset_difficulties']['AccZ [mg]'],
                                                                          E_dataset.transpose()[2]), axis=0)

            result['dataset_expected_difficulties'] = np.concatenate((result['dataset_expected_difficulties'],
                                                                      Y_datasets[E_index]), axis=0)

    print()

    plt.figure(figsize=(10, 6))

    plt.subplot(3, 1, 1)
    plt.plot(result['dataset_difficulties']['AccX [mg]'], label='AccX as a function of T')
    plt.plot(result['dataset_difficulties']['AccY [mg]'], label='AccY as a function of T')
    plt.plot(result['dataset_difficulties']['AccZ [mg]'], label='AccZ as a function of T')
    plt.ylabel('Acceleration')
    plt.legend()
    plt.title('Models result')

    plt.subplot(3, 1, 2)
    plt.plot(result['result_per_dataset'], label=f"Model")
    result['result_per_dataset'] = []
    plt.ylabel('Result')
    plt.legend()

    plt.subplot(3, 1, 3)
    plt.plot(result['dataset_expected_difficulties'], label="Expected results")
    plt.xlabel('Point number')
    plt.ylabel('Result')
    plt.legend()

    plt.grid(True)
    plt.tight_layout()
    plt.show()

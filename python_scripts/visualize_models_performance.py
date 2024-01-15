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

models = []
results = []

while True:
    entry = input("Enter the id of the model you want to add or 'exit' to exit:")

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
        'global_result': 0,
        'global_non_pendulum': 0,
        'global_pendulum': 0,
        'result_per_dataset': [],
        'raw_result_per_dataset': [],
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

    csv_data = pd.read_csv(file_selected, usecols=headers)
    if csv_data.shape[0] % 100:
        print(f"File {file_selected} has {csv_data.shape[0]} rows, which is not a multiple of 100, so it will be ignored")
        continue
    else:
        print(f"File selected: {file_selected.name}")

    data = csv_data.values
    dataset_count = len(data) // 100

    E_raw_dataset = data[:, :-1]
    Y_raw_dataset = data[:, -1]

    scaler = MinMaxScaler(feature_range=(0, 1))
    E_raw_dataset = scaler.fit_transform(E_raw_dataset)

    E_datasets = np.array_split(E_raw_dataset, dataset_count)
    Y_datasets = np.array_split(Y_raw_dataset, dataset_count)
    Y_datasets = [lot[0] for lot in Y_datasets]

    pendulum_cout = len([lot for lot in Y_datasets if lot])
    non_penulum_count = len([lot for lot in Y_datasets if not lot])

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

        for model_index, model in enumerate(models):
            E_dataset_expand = np.expand_dims(E_dataset, axis=0)
            result = model.predict(E_dataset_expand, verbose=0, use_multiprocessing=True)

            result_per_dataset = (result[0][0] >= 0.5)
            if (result[0][0] >= 0.5) == Y_datasets[E_index]:
                results[model_index]['global_result'] = results[model_index]['global_result'] + 1

            if result[0][0] >= 0.5:
                results[model_index]['global_pendulum'] = results[model_index]['global_pendulum'] + 1
            else:
                results[model_index]['global_non_pendulum'] = results[model_index]['global_non_pendulum'] + 1

            result_per_dataset = np.full(100, result_per_dataset).tolist()
            raw_result_per_dataset = np.full(100, result[0][0]).tolist()

            results[model_index]['result_per_dataset'] = np.concatenate((results[model_index]['result_per_dataset'],
                                                                         result_per_dataset), axis=0)

            results[model_index]['raw_result_per_dataset'] = np.concatenate((results[model_index]['raw_result_per_dataset'],
                                                                             raw_result_per_dataset), axis=0)

    print()

    global_scores_to_show = []
    global_pendulum_scores_to_show = []
    global_non_pendulum_scores_to_show = []
    models_to_show = []

    for model_index, model in enumerate(models):
        global_scores_to_show.append(round((results[model_index]['global_result'] / dataset_count) * 100))
        global_pendulum_scores_to_show.append(round((results[model_index]['global_pendulum'] / pendulum_cout) * 100))
        global_non_pendulum_scores_to_show.append(round((results[model_index][
                                                             'global_non_pendulum'] / non_penulum_count) * 100))

        models_to_show.append(str(results[model_index]['id']))

        results[model_index]['global_result'] = 0
        results[model_index]['global_pendulum'] = 0
        results[model_index]['global_non_pendulum'] = 0

    bar_width = 0.3

    show_global_scores = np.arange(len(global_scores_to_show))
    show_global_pendulum_scores = [x + bar_width for x in show_global_scores]
    show_global_non_pendulum_scores = [x + bar_width for x in show_global_pendulum_scores]

    global_scores_bar = plt.bar(show_global_scores,
                                global_scores_to_show,
                                color='blue',
                                width=bar_width,
                                edgecolor='grey',
                                label='Global score')
    global_pendulum_scores_bar = plt.bar(show_global_pendulum_scores,
                                         global_pendulum_scores_to_show,
                                         color='red',
                                         width=bar_width,
                                         edgecolor='grey',
                                         label='Pendulum score')
    global_non_pendulum_scores_bar = plt.bar(show_global_non_pendulum_scores,
                                             global_non_pendulum_scores_to_show,
                                             color='green',
                                             width=bar_width,
                                             edgecolor='grey',
                                             label='Non pendulum score')


    def add_labels(bars):
        for bar in bars:
            height = bar.get_height()
            plt.annotate('{}{}'.format(height, " %"),
                         xy=(bar.get_x() + bar.get_width() / 2, height),
                         xytext=(0, 3),
                         textcoords="offset points",
                         ha='center', va='bottom')


    add_labels(global_scores_bar)
    add_labels(global_pendulum_scores_bar)
    add_labels(global_non_pendulum_scores_bar)

    plt.xlabel('Models id')
    plt.xticks([r + bar_width / 2 for r in range(len(show_global_scores))], models_to_show)
    plt.title(f"Models performance on {file_selected.name} dataset")
    plt.ylabel('Scores (%)')

    plt.legend()
    plt.show()

    plt.figure(figsize=(10, 6))

    plt.subplot(4, 1, 1)
    plt.plot(csv_data['AccX [mg]'], label='AccX as a function of T')
    plt.plot(csv_data['AccY [mg]'], label='AccY as a function of T')
    plt.plot(csv_data['AccZ [mg]'], label='AccZ as a function of T')
    plt.ylabel('Acceleration')
    plt.legend()
    plt.title('Models result')

    plt.subplot(4, 1, 2)
    for model_index in range(len(models)):
        plt.plot(np.array(results[model_index]['result_per_dataset']), label=f"Model {results[model_index]['id']}")
        results[model_index]['result_per_dataset'] = []
    plt.ylabel('Result')
    plt.legend()

    plt.subplot(4, 1, 3)
    for model_index in range(len(models)):
        plt.plot(np.array(results[model_index]['raw_result_per_dataset']), label=f"Model {results[model_index]['id']}")
        results[model_index]['raw_result_per_dataset'] = []
    plt.ylabel('Raw result')
    plt.legend()

    plt.subplot(4, 1, 4)
    plt.plot(csv_data['State'], label="Expected results")
    plt.xlabel('Point number')
    plt.ylabel('Result')
    plt.legend()

    plt.grid(True)
    plt.tight_layout()
    plt.show()

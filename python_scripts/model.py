import os
import pathlib
import time

import numpy as np
import pandas as pd
import tensorflow as tf
from keras.src.callbacks import ReduceLROnPlateau, EarlyStopping
from keras.src.regularizers import l2
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from utils import DATA_POINTS, print_dataset

# Settings to modify in the AI

# Split parameters
TEST_SPLIT = 0.3
VAL_SPLIT = 0.5

# Optimizers
rms_optimizer = tf.keras.optimizers.RMSprop(momentum=0.1)

OPTIMIZER = rms_optimizer

# Loss

binary_crossentropy_loss = tf.keras.losses.BinaryCrossentropy()

LOSS = binary_crossentropy_loss

# Metrics
metrics_accuracy = ['accuracy']
metrics_binary_accuracy_auc = [tf.keras.metrics.BinaryAccuracy(), tf.keras.metrics.AUC()]
metrics_precision_recall = [tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
metrics_accuracy_precision_recall = ['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]

METRICS = metrics_precision_recall

# Main parameters
EPOCHS = 200
BATCH_SIZE = 20


def visualize_error(error_lot, data):
    # Ensure data is a 2D NumPy array
    if not isinstance(data, np.ndarray) or data.ndim != 2:
        raise ValueError("Data must be a 2D NumPy array.")

    previous_lot = (error_lot - 1) * 100
    next_lot = (error_lot + 2) * 100

    # Check if previous_lot and next_lot + 1 are within the bounds of data
    if previous_lot < 0 or next_lot + 1 > data.shape[0]:
        raise ValueError("Indices out of bounds for the given data.")

    # Confirm that dataset in print_dataset has the required columns
    required_columns = ['AccX [mg]', 'AccY [mg]', 'AccZ [mg]']
    missing_columns = [col for col in required_columns if col not in headers]

    if missing_columns:
        raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")

    selected_lot = data[previous_lot:next_lot + 1, :]
    print_dataset(selected_lot, f"Error for bloc {previous_lot} to {next_lot}")


def scale_sequences(sequences):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit_transform(sequences)
    return sequences


seed = int(time.time())
np.random.seed(seed)
tf.random.set_seed(seed)

parent_directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
dataset_folder_path = parent_directory + "/datasets/"

files_path = [f for f in pathlib.Path(dataset_folder_path).glob("*.csv")]
files_path = files_path + [f for f in pathlib.Path(dataset_folder_path).glob("*/concat_data*.csv")]
files_path = files_path + [f for f in pathlib.Path(dataset_folder_path).glob("*/compile_data*.csv")]
files_path = files_path + [f for f in pathlib.Path(dataset_folder_path).glob("*/concat_data_+_dataset*.csv")]
files_path = files_path + [f for f in pathlib.Path(dataset_folder_path).glob("*/compile_data_+_dataset*.csv")]

for index, file_path in enumerate(files_path):
    print(f"{index} - {file_path}")

entry = input("Enter the id of the file you want to test or 'exit' to exit: ")

file_selected = files_path[int(entry)]

headers = ["AccX [mg]", "AccY [mg]", "AccZ [mg]", "State"]
raw_dataset = pd.read_csv(file_selected, usecols=headers).values

E_raw_dataset = raw_dataset[:, :-1]
Y_raw_dataset = raw_dataset[:, -1]

# Vérification si la longueur de raw_dataset est un multiple de 200
if len(raw_dataset) % DATA_POINTS != 0:
    print(f'Nb lignes : {len(raw_dataset)}')
    error_message = f"Le nombre de lignes dans le fichier CSV n'est pas un multiple de {DATA_POINTS}."
    raise ValueError(error_message)

scaler = MinMaxScaler(feature_range=(0, 1))
E_raw_dataset = scaler.fit_transform(E_raw_dataset)
print(f"Dataset scaled with MinMaxScaler: {scaler.data_min_} - {scaler.data_max_}")

# Division du dataset en sous-tableaux
E_dataset = np.array_split(E_raw_dataset, len(raw_dataset) // DATA_POINTS)
Y_dataset = np.array_split(Y_raw_dataset, len(raw_dataset) // DATA_POINTS)

# Initialize a variable to store the lot number
error_lot_number = None

# Vérifier que toutes les étiquettes sont identiques dans chaque lot
for i, lot in enumerate(Y_dataset):
    if np.any(lot != lot[0]):
        error_lot_number = i
        print(f'Voir lot : {lot}')
        visualize_error(error_lot_number, raw_dataset)
        raise ValueError(f"Toutes les étiquettes ne sont pas identiques dans un lot {error_lot_number}"
                         + f" de {DATA_POINTS}.")

# Réduire chaque lot à une seule étiquette
Y_dataset = [lot[0] for lot in Y_dataset]

# Conversion en numpy arrays
E_dataset = np.asarray(E_dataset)
Y_dataset = np.asarray(Y_dataset)

E_train_dataset, E_test_dataset, Y_train_dataset, Y_test_dataset = train_test_split(E_dataset, Y_dataset,
                                                                                    test_size=TEST_SPLIT,
                                                                                    random_state=seed)
l2_regularizer = l2(0.01)

# L'ajout de EarlyStopping dans l'optique d'arrêter un mauvais apprentissage et donc de gagner du temps
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=40,
    verbose=1,
    mode='min',
    restore_best_weights=True
)

# L'ajout de ReduceLROnPlateau est a essayer pour voir si cela améliore les résultats
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.01,
    patience=20,
    verbose=1,
    mode='min',
    min_lr=0.001
)

model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(DATA_POINTS, activation='relu',
                          input_shape=(E_train_dataset.shape[1], E_train_dataset.shape[2])),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(50, activation='relu', kernel_regularizer=l2_regularizer),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(25, activation='relu', kernel_regularizer=l2_regularizer),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer=OPTIMIZER,
              loss=LOSS,
              metrics=METRICS)

history = model.fit(E_train_dataset,
                    Y_train_dataset,
                    shuffle=True,
                    epochs=EPOCHS,
                    batch_size=BATCH_SIZE,
                    validation_split=VAL_SPLIT,
                    callbacks=[early_stopping, reduce_lr])

scores = model.evaluate(E_test_dataset, Y_test_dataset)
print("\nEvaluation sur le test data %s: %.2f - %s: %.2f%% " % (
    model.metrics_names[0], scores[0], model.metrics_names[1], scores[1] * 100))

plt.figure()
plt.plot(history.history['recall'])
plt.plot(history.history['val_recall'])
plt.title('Model recall')
plt.ylabel('Recall')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper left')
plt.show()

plt.figure()
plt.plot(history.history['precision'])
plt.plot(history.history['val_precision'])
plt.title('Model precision')
plt.ylabel('Precision')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper left')
plt.show()

plt.figure()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper left')
plt.show()

# save the model
model_name = input("Do you want to save the model? (File name/n) ")
if model_name != "n" and model_name != "":
    os.makedirs('../models/' + model_name, exist_ok=True)
    model.save('../models/' + model_name + '/' + model_name + '.h5')

    model_structure = model.to_json()
    with open('../models/' + model_name + '/' + model_name + '.json', "w") as json_file:
        json_file.write(model_structure)

    plt.figure()
    plt.plot(history.history['recall_1'])
    plt.plot(history.history['val_recall'])
    plt.title('Model recall')
    plt.ylabel('Recall')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='upper left')
    plt.savefig('../models/' + model_name + '/' + model_name + '_recall_plot.png')

    plt.figure()
    plt.plot(history.history['precision'])
    plt.plot(history.history['val_precision'])
    plt.title('Model precision')
    plt.ylabel('Precision')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='upper left')
    plt.savefig('../models/' + model_name + '/' + model_name + '_precision_plot.png')

    plt.figure()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='upper left')
    plt.savefig('../models/' + model_name + '/' + model_name + '_loss_plot.png')

    print("Model saved.")

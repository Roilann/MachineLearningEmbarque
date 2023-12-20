import time

import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


DATA_POINTS = 100


def scale_sequences(sequences):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit_transform(sequences)
    return sequences


seed = int(time.time())
np.random.seed(seed)
tf.random.set_seed(seed)

raw_dataset = pd.read_csv('datasets/dataset.csv').values

E_raw_dataset = raw_dataset[:, :-1]
Y_raw_dataset = raw_dataset[:, -1]

# Vérification si la longueur de raw_dataset est un multiple de 200
if len(raw_dataset) % DATA_POINTS != 0:
    raise ValueError("Le nombre de lignes dans le fichier CSV n'est pas un multiple de 200.")

scaler = MinMaxScaler(feature_range=(0, 1))
E_raw_dataset = scaler.fit_transform(E_raw_dataset)

# Division du dataset en sous-tableaux
E_dataset = np.array_split(E_raw_dataset, len(raw_dataset) // DATA_POINTS)
Y_dataset = np.array_split(Y_raw_dataset, len(raw_dataset) // DATA_POINTS)

# Vérifier que toutes les étiquettes sont identiques dans chaque lot
for lot in Y_dataset:
    if np.any(lot != lot[0]):
        raise ValueError("Toutes les étiquettes ne sont pas identiques dans un lot de 100.")

# Réduire chaque lot à une seule étiquette
Y_dataset = [lot[0] for lot in Y_dataset]

# Conversion en numpy arrays
E_dataset = np.asarray(E_dataset)
Y_dataset = np.asarray(Y_dataset)

E_train_dataset, E_test_dataset, Y_train_dataset, Y_test_dataset = train_test_split(E_dataset, Y_dataset, test_size=0.2,
                                                                                    random_state=seed)
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(DATA_POINTS, activation='relu', input_shape=(E_train_dataset.shape[1], E_train_dataset.shape[2])),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(50, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(25, activation='relu'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.5),
              loss='binary_crossentropy', metrics=['accuracy'])

# Résumé du modèle
# model.summary()

model.fit(E_train_dataset, Y_train_dataset, shuffle=False, epochs=100, batch_size=10, validation_split=0, verbose=0)

# evaluate the model
scores = model.evaluate(E_test_dataset, Y_test_dataset)
print("\nEvaluation sur le test data %s: %.2f - %s: %.2f%% " % (
    model.metrics_names[0], scores[0], model.metrics_names[1], scores[1] * 100))

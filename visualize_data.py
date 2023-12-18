import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# Charger le fichier CSV
df = pd.read_csv('input/chaotique_02.csv')

# Assure-toi que les colonnes appropriées sont sélectionnées
temps = df['T [ms]']
acceleration_x = df['AccX [mg]']
acceleration_y = df['AccY [mg]']
acceleration_z = df['AccZ [mg]']
gyro_x = df['GyroX [mdps]']
gyro_y = df['GyroY [mdps]']
gyro_z = df['GyroZ [mdps]']
data = df['AccX [mg]'].values

# Calcul de la DFT
dft_result = np.fft.fft(data)

# Calcul de la FFT
fft_result = np.fft.fft(data)

# Fréquences associées à la transformation
freq = np.fft.fftfreq(len(data))

# Tracé des résultats
plt.figure(figsize=(12, 6))

plt.subplot(2, 1, 1)
plt.plot(freq, np.abs(dft_result))
plt.title('Transformée de Fourier Discrète (DFT)')
plt.xlabel('Fréquence')
plt.ylabel('Amplitude')

plt.subplot(2, 1, 2)
plt.plot(freq, np.abs(fft_result))
plt.title('Transformée de Fourier Rapide (FFT)')
plt.xlabel('Fréquence')
plt.ylabel('Amplitude')

plt.tight_layout()
plt.show()

# Plot des données
plt.figure(figsize=(12, 6))
plt.plot(temps, acceleration_x, label='Accélération X')
plt.plot(temps, acceleration_y, label='Accélération Y')
plt.plot(temps, acceleration_z, label='Accélération Z')

plt.title('Données de l\'accéléromètre au fil du temps')
plt.xlabel('Temps')
plt.ylabel('Accélération')
plt.legend()
plt.show()

# Plot en 3D
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.plot(temps, acceleration_x, acceleration_y, label='Trajectoire 3D de l\'accélération')
ax.set_xlabel('Temps')
ax.set_ylabel('Accélération X')
ax.set_zlabel('Accélération Y')
ax.legend()
plt.show()

# Plot des données
plt.figure(figsize=(12, 6))
plt.plot(temps, gyro_x, label='Gyro X')
plt.plot(temps, gyro_y, label='Gyro Y')
plt.plot(temps, gyro_z, label='Gyro Z')

plt.title('Données du gyroscope au fil du temps')
plt.xlabel('Temps')
plt.ylabel('Gyro')
plt.legend()
plt.show()

# Plot en 3D
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.plot(temps, gyro_x, gyro_y, label='Trajectoire 3D du gyroscope')
ax.set_xlabel('Temps')
ax.set_ylabel('Gyro X')
ax.set_zlabel('Gyro Y')
ax.legend()
plt.show()

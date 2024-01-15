from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

# Minimum amount of points to determine if the movement is swinging or not
DATA_POINTS = 100


def print_dataset(dataset, title: str = 'Dataset state'):
    plt.figure(figsize=(10, 6))
    plt.subplot(2, 1, 1)

    if isinstance(dataset, np.ndarray):
        # Handle NumPy array
        plt.plot(dataset[:, 0], label='AccX as a function of T')
        plt.plot(dataset[:, 1], label='AccY as a function of T')
        plt.plot(dataset[:, 2], label='AccZ as a function of T')
    elif isinstance(dataset, pd.DataFrame):
        # Handle Pandas DataFrame
        plt.plot(dataset['AccX [mg]'], label='AccX as a function of T')
        plt.plot(dataset['AccY [mg]'], label='AccY as a function of T')
        plt.plot(dataset['AccZ [mg]'], label='AccZ as a function of T')
    else:
        raise ValueError("Unsupported dataset type. Must be a NumPy array or Pandas DataFrame.")

    plt.xlabel('Point number')
    plt.ylabel('Acceleration')
    plt.title(title)

    plt.subplot(2, 1, 2)

    if isinstance(dataset, np.ndarray):
        # Handle NumPy array
        plt.plot(dataset[:, 3])
    elif isinstance(dataset, pd.DataFrame):
        # Handle Pandas DataFrame
        plt.plot(dataset['State'])
    else:
        raise ValueError("Unsupported dataset type. Must be a NumPy array or Pandas DataFrame.")

    plt.xlabel('Point number')
    plt.ylabel('State')

    plt.grid(True)
    plt.show()


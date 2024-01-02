from matplotlib import pyplot as plt

# Minimum amount of points to determine if the movement is swinging or not
DATA_POINTS = 100


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

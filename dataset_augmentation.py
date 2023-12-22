

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load your original accelerometer data from CSV
def load_data(csv_path):
    return pd.read_csv(csv_path)

# Random Rotation
def apply_random_rotation(data):
    rotation_angle = np.random.uniform(-10, 10)  # Adjust the range as needed
    rotated_data = data.apply(lambda x: np.roll(x, int(rotation_angle)))
    return rotated_data

# Time Warping
def apply_time_warping(data):
    num_points = len(data.columns)
    warp_factor = np.random.uniform(0.9, 1.1)  # Adjust the range as needed
    warped_data = data.apply(lambda x: np.interp(np.linspace(0, 1, num_points), np.linspace(0, 1, num_points * warp_factor), x))
    return warped_data

# Magnitude Scaling
def apply_magnitude_scaling(data):
    scale_factor = np.random.uniform(0.8, 1.2)  # Adjust the range as needed
    scaled_data = data * scale_factor
    return scaled_data

# Jittering
def apply_jittering(data):
    jitter_factor = 0.01  # Adjust the factor as needed
    jittered_data = data + np.random.normal(0, jitter_factor, size=data.shape)
    return jittered_data

# Temporal Subsampling
def apply_temporal_subsampling(data):
    subsample_factor = np.random.choice([1, 2, 3])  # Adjust choices as needed
    subsampled_data = data.iloc[::subsample_factor, :]
    return subsampled_data

# Bandpass Filtering
def apply_bandpass_filter(data, low_cutoff, high_cutoff):
    # Implement your bandpass filter (e.g., using scipy.signal)
    # Make sure to adjust the cutoff frequencies based on your data
    pass

# Gaussian Noise
def apply_gaussian_noise(data):
    noise_factor = 0.1  # Adjust the factor as needed
    noisy_data = data + np.random.normal(0, noise_factor, size=data.shape)
    return noisy_data

# Visualization function
def visualize_data(original_data, augmented_data):
    plt.figure(figsize=(10, 6))

    # Plot original data on top
    plt.subplot(2, 1, 1)
    plt.plot(original_data)
    plt.title('Original Data')

    # Plot augmented data on bottom
    plt.subplot(2, 1, 2)
    plt.plot(augmented_data)
    plt.title('Augmented Data')

    plt.tight_layout()
    plt.show()

# Example usage:
csv_path = "input/chaotique_01.csv"
original_data = load_data(csv_path)

# Apply augmentation functions and visualize after each
rotated_data = apply_random_rotation(original_data)
visualize_data(original_data, rotated_data)

# warped_data = apply_time_warping(original_data)
# visualize_data(original_data, warped_data)

scaled_data = apply_magnitude_scaling(original_data)
visualize_data(original_data, scaled_data)

jittered_data = apply_jittering(original_data)
visualize_data(original_data, jittered_data)

subsampled_data = apply_temporal_subsampling(original_data)
visualize_data(original_data, subsampled_data)

noisy_data = apply_gaussian_noise(original_data)
visualize_data(original_data, noisy_data)

# For bandpass filtering, you'll need to implement or use a specific library.

# Use the augmented data for training your AI model.
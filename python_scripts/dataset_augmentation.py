import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.ndimage import laplace

from utils import print_dataset, DATA_POINTS


# Load your original accelerometer data from CSV
def load_data(csv_path):
    return pd.read_csv(csv_path)


# Random Rotation
def apply_random_rotation(data):
    max_iterations = 4
    rotate_data = data
    for iteration in range(1, max_iterations):
        # Separate data with the amount of iteration
        separation = 2 ** iteration
        bloc_size = len(rotate_data) // separation
        # Clear output buffer to use concat after
        saved_data = rotate_data
        rotate_data = pd.DataFrame()
        for nb_rotation in range(0, separation):
            # Define data to rotate
            start_bloc_index = nb_rotation * bloc_size
            end_bloc_index = (nb_rotation + 1) * bloc_size
            bloc = saved_data.iloc[start_bloc_index:end_bloc_index]
            # Rotate data
            if not bloc.empty:
                rotated_bloc = bloc.apply(lambda x: np.roll(x, int(len(bloc) // 2)))
                rotate_data = pd.concat([rotate_data, rotated_bloc], ignore_index=True)
        # visualize_data(original_data, rotate_data, f"Rotated data number {iteration}")
    return rotate_data


# Magnitude Scaling
def apply_magnitude_scaling(data, exclude_column='State', min_range=0.8, max_range=1.2):
    scale_factor = np.random.uniform(min_range, max_range)  # Adjust the range as needed
    # Identify non-'State' columns
    non_state_columns = data.columns[data.columns != exclude_column]

    # Apply scaling only to non-'State' columns
    scaled_data = data.copy()
    scaled_data[non_state_columns] *= scale_factor
    return scaled_data


# Jittering
def apply_jittering(data, exclude_column='State', jitter_factor=25):
    # Identify numeric columns (excluding 'time')
    numeric_columns = data.select_dtypes(include=[np.number]).columns

    # Exclude 'time' from numeric columns
    numeric_columns = [col for col in numeric_columns if col != exclude_column]

    # Extract the numeric columns for jittering
    data_to_jitter = data[numeric_columns].values

    # Apply jittering to the numeric columns
    jittered_numeric_data = data_to_jitter + np.random.normal(0, jitter_factor, size=data_to_jitter.shape)

    # Create a copy of the original data to avoid modifying it in place
    data_jittered = data.copy()

    # Replace the original numeric values with the jittered values
    data_jittered[numeric_columns] = jittered_numeric_data

    return data_jittered


# Temporal Subsampling
def apply_temporal_subsampling(data):
    subsample_factor = np.random.choice([1, 2, 3])  # Adjust choices as needed
    subsampled_data = data.iloc[::subsample_factor, :]
    return subsampled_data


# Gaussian Noise
def apply_gaussian_noise(data, exclude_column='', sigma=1.0):
    blurred_data = data.copy()
    # Get the list of column names excluding the specified column
    columns_to_blur = [col for col in data.columns if col != exclude_column]

    # Apply Gaussian blur to selected columns
    for column in columns_to_blur:
        blurred_data[column] = gaussian_filter1d(data[column], sigma)

    return blurred_data


# Laplace enhancement
def apply_laplace(data, exclude_column='State', alpha=1.0):
    # Copy the input DataFrame to avoid modifying the original data
    df_enhanced = data.copy()

    # Get the list of column names excluding the specified column
    columns_to_enhance = [col for col in df_enhanced.columns if col != exclude_column]

    # Apply Laplacian filter to enhance features in selected columns
    for column in columns_to_enhance:
        df_enhanced[column] += alpha * laplace(df_enhanced[column])

    return df_enhanced


# Visualization function
def visualize_data(original_data, augmented_data, data_name):
    plt.figure(figsize=(10, 6))

    # Plot original data on top
    plt.subplot(2, 1, 1)
    plt.plot(original_data)
    plt.title('Original Data')

    # Plot augmented data on bottom
    plt.subplot(2, 1, 2)
    plt.plot(augmented_data)
    plt.title(data_name)

    plt.tight_layout()
    plt.show()


def compare_data(data1, data2):
    print("Original Data Stats:")
    print("Mean:", np.mean(data1))
    print("Std:", np.std(data1))

    print("\nModified Data Stats:")
    print("Mean:", np.mean(data2))
    print("Std:", np.std(data2))


def apply_and_visualize(original_data, augmented_data, title, save_path):
    returned_data = augmented_data(original_data)
    if VISUEL:
        visualize_data(original_data, returned_data, title)
    if DEBUG:
        compare_data(original_data, returned_data)
    if SAVING_DATA_SEPARATE:
        if returned_data.shape[0] % DATA_POINTS:
            print(f"File has {returned_data.shape[0]} rows, which is not a multiple of {DATA_POINTS}, so it will be "
                  f"ignored")
        else:
            returned_data.to_csv(save_path + f'/{title}' + '.csv', index=False, header=True)
    return returned_data


# Example usage:

# Use to be sure there has been a difference (if not visible enough)
DEBUG = 0
# Show plots
VISUEL = 0
"""
To use the program properly set at least 1 of the two macros to '1'
- data_separate allows to generate each type of augmentation from the original 
       resulting in a LOT OF POINTS
- data_compile allows to generate each type of augmentation over the previous one, 
       resulting in the same amount of points as the original
       /!\ if not set properly, could be unusable /!\
"""
DATA_SEPARATE = 1
DATA_COMPILE = 1

"""
To use the program properly activate at least 1 of the macros with the same name as the macro previously activated
eg. data_seperate = 1   ====>   saving_data_seperate = 1

** data_separate **
- saving_data_separate
        save each augmentation data in its own file
- saving_data_separate_concat 
        save each augmentation in a single file (with a LOTS OF POINTS)
- saving_data_separate_concat_dataset 
        save each augmentation in a single file (with a LOTS OF POINTS) + the original dataset

** data_compile **
- saving_data_compile
        save the data augmentation in 1 file
- saving_data_compile_dataset
        save the data augmentation and the dataset in the same file
"""
SAVING_DATA_SEPARATE = 1
SAVING_DATA_SEPARATE_CONCAT = 1
SAVING_DATA_SEPARATE_CONCAT_DATASET = 1

SAVING_DATA_COMPILE = 1
SAVING_DATA_COMPILE_DATASET = 1

csv_path = "../datasets/dataset_v4.csv"

original_data = load_data(csv_path)

directory, file = os.path.split(csv_path)
filename, ext = os.path.splitext(file)
dir_path = os.path.join(directory, filename)

# Saving datasets
if (SAVING_DATA_SEPARATE or SAVING_DATA_SEPARATE_CONCAT or SAVING_DATA_SEPARATE_CONCAT_DATASET
        or SAVING_DATA_COMPILE or SAVING_DATA_COMPILE_DATASET):
    os.makedirs(dir_path, exist_ok=True)

# rotated_data = pd.DataFrame()
scaled_data = pd.DataFrame()
# subsampled_data = pd.DataFrame()
jittered_data = pd.DataFrame()
# noisy_data = pd.DataFrame()
enhanced_data = pd.DataFrame()

# Apply augmentation functions and visualize after each
if DATA_SEPARATE:
    # rotated_data = apply_and_visualize(original_data, apply_random_rotation, "Rotated data", dir_path)
    scaled_data = apply_and_visualize(original_data, apply_magnitude_scaling, "Scaling data", dir_path)
    # subsampled_data = apply_and_visualize(original_data, apply_temporal_subsampling, "Subsampling data", dir_path)
    jittered_data = apply_and_visualize(original_data, apply_jittering, "Jittered data", dir_path)
    # noisy_data = apply_and_visualize(original_data, apply_gaussian_noise, "Gaussian noise data", dir_path)
    enhanced_data = apply_and_visualize(original_data, apply_laplace, "Laplace data", dir_path)

    # Concat them
    if SAVING_DATA_SEPARATE_CONCAT:
        concat_data = pd.concat(
            [scaled_data, jittered_data, enhanced_data])
        if concat_data.shape[0] % DATA_POINTS:
            print(
                f"File has {concat_data.shape[0]} rows, which is not a multiple of {DATA_POINTS}, so it will be "
                f"ignored")
        else:
            concat_data.to_csv(dir_path + '/concat_data' + '.csv', index=False, header=True)

    # Concat them with original data
    if SAVING_DATA_SEPARATE_CONCAT_DATASET:
        concat_data = pd.concat(
            [original_data, scaled_data, jittered_data, enhanced_data])
        if concat_data.shape[0] % DATA_POINTS:
            print(
                f"File has {concat_data.shape[0]} rows, which is not a multiple of {DATA_POINTS}, so it will be"
                f" ignored")
        else:
            concat_data.to_csv(dir_path + '/concat_data_+_dataset' + '.csv', index=False, header=True)

# Add every augmentation layer over the previous one
# /!\ if not set properly could result in unusable data /!\
if DATA_COMPILE:
    # rotated_data = apply_and_visualize(original_data, apply_random_rotation, "Rotated data", dir_path)
    # subsampled_data = apply_and_visualize(scaled_data, apply_temporal_subsampling, "Subsampling data", dir_path)
    scaled_data = apply_and_visualize(original_data, apply_magnitude_scaling, "Scaling data", dir_path)
    jittered_data = apply_and_visualize(scaled_data, apply_jittering, "Jittered data", dir_path)
    # noisy_data = apply_and_visualize(jittered_data, apply_gaussian_noise, "Gaussian noise data", dir_path)
    enhanced_data = apply_and_visualize(jittered_data, apply_laplace, "Laplace data", dir_path)
    visualize_data(original_data, enhanced_data, 'Compiled data')

    if SAVING_DATA_COMPILE:
        if enhanced_data.shape[0] % DATA_POINTS:
            print(
                f"File has {enhanced_data.shape[0]} rows, which is not a multiple of {DATA_POINTS}, so it will be "
                f"ignored")
        else:
            enhanced_data.to_csv(dir_path + '/compile_data' + '.csv', index=False, header=True)

    if SAVING_DATA_COMPILE_DATASET:
        concat_data = pd.concat([original_data, enhanced_data])
        if concat_data.shape[0] % DATA_POINTS:
            print(
                f"File has {concat_data.shape[0]} rows, which is not a multiple of {DATA_POINTS}, so it will "
                f"be ignored")
        else:
            saved_path = dir_path + '/compile_data_+_dataset' + '.csv'
            concat_data.to_csv(saved_path, index=False, header=True)

            # Verify saved file
            saved_csv = pd.read_csv(saved_path)
            if 'State' in saved_csv.columns:
                print_dataset(saved_csv, 'After saving the file')

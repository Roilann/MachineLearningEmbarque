import pathlib
import pandas as pd
import os
from utils import print_dataset, DATA_POINTS


def strike(text):
    result = ''
    for c in text:
        result = result + c + '\u0336'
    return result


# Navigate to the parent directory
parent_directory = os.path.dirname(os.getcwd())
# Construct the path to the output folder in the parent directory
output_folder_path = os.path.join(parent_directory, "output")

# Use pathlib to find CSV files in the output folder
files_path = [f for f in pathlib.Path(output_folder_path).glob("*.csv")]
for index, file_path in enumerate(files_path):
    print(f"{index} - {file_path}")

headers = ["AccX [mg]", "AccY [mg]", "AccZ [mg]", "State"]
dataset = pd.DataFrame()

while True:
    entry = input("Enter the id of the file you want to add to the dataset or 'exit' to exit:")

    if entry.lower() == 'exit' or entry.lower() == '':
        break

    file_selected = files_path[int(entry)]

    data = pd.read_csv(file_selected, usecols=headers)
    if data.shape[0] % DATA_POINTS:
        print(f"File {file_selected} has {data.shape[0]} rows, which is not a multiple of {DATA_POINTS}, so it will "
              f"be ignored")
        continue
    else:
        print(f"File selected: {file_selected.name}")

    action = input("Do you want to add this values in the dataset (Y/n): ")
    if action.lower() == 'y' or action.lower() == 'yes' or action.lower() == '':
        dataset = pd.concat([dataset, data], ignore_index=True)
        print_dataset(dataset, 'Actual dataset')

print_dataset(dataset)

dataset_name = input("Enter the name of the dataset to export: ")
dataset.to_csv(f'../datasets/{dataset_name}.csv', index=False, header=True)

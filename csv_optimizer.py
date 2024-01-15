import os
import csv
from decimal import Decimal, ROUND_HALF_UP

# Basic round up numbers to closest
# Give it a folder and will round up all numbers in all the csv in the selected folder

def round_decimal(value):
    try:
        # Try converting the value to a Decimal
        decimal_value = Decimal(value)

        # Round the decimal value to the closest
        rounded_value = decimal_value.quantize(Decimal('1'), rounding=ROUND_HALF_UP)

        return rounded_value
    except:
        # If conversion to Decimal fails, return the original value
        return value


def process_csv(input_file, output_file):
    with open(input_file, 'r') as csv_input, open(output_file, 'w', newline='') as csv_output:
        reader = csv.reader(csv_input)
        writer = csv.writer(csv_output)

        for row in reader:
            # Process each column in the row
            rounded_row = [round_decimal(value) for value in row]

            # Write the rounded row to the output file
            writer.writerow(rounded_row)


def process_folder(input_folder):
    for filename in os.listdir(input_folder):
        if filename.endswith('.csv'):
            input_file_path = os.path.join(input_folder, filename)
            output_file_path = os.path.join(input_folder, filename.replace('.csv', '_optimized.csv'))
            process_csv(input_file_path, output_file_path)
            # Remove the old CSV file
            os.remove(input_file_path)


# Example usage:
input_folder_path = 'datasets/dataset_v4'
process_folder(input_folder_path)

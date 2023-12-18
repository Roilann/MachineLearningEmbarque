import pandas as pd
import matplotlib.pyplot as plt

# Read the specified file
file_name = input("Enter the file name: ")
data = pd.read_csv('input/' + file_name + '.csv')

plt.figure(figsize=(10, 6))
plt.plot(data['T [ms]'], data['AccX [mg]'], label='AccX as a function of T')
plt.plot(data['T [ms]'], data['AccY [mg]'], label='AccY as a function of T')
plt.plot(data['T [ms]'], data['AccZ [mg]'], label='AccZ as a function of T')
plt.xlabel('Time (T) in ms')
plt.ylabel('Acceleration')
plt.title('Accelerometer Data as a Function of T')
plt.legend()
plt.grid(True)
plt.show()

data_to_export = pd.DataFrame()

# Loop to ask for time ranges
while True:
    response = input("Type 'exit' to quit or enter T min (in ms): ")
    if response.lower() == 'exit':
        break
    else:
        t_min = int(response)

    t_max = int(input("Enter T max (in ms): "))

    # Filter data for the specified time range
    filtered_data = data[(data['T [ms]'] >= t_min) & (data['T [ms]'] <= t_max)]
    data_to_export = pd.concat([data_to_export, filtered_data])
    print(f"Number of selected rows: {len(filtered_data)}")

    plt.figure(figsize=(10, 6))
    plt.plot(filtered_data['T [ms]'], filtered_data['AccX [mg]'], label='AccX as a function of T')
    plt.plot(filtered_data['T [ms]'], filtered_data['AccY [mg]'], label='AccY as a function of T')
    plt.plot(filtered_data['T [ms]'], filtered_data['AccZ [mg]'], label='AccZ as a function of T')
    plt.xlabel('Time (T) in ms')
    plt.ylabel('Acceleration')
    plt.title('Accelerometer Data as a Function of T')
    plt.legend()
    plt.grid(True)
    plt.show()

column_value = int(input("Enter a value (0 or 1) for the new column: "))
data_to_export['State'] = column_value

# Select only the columns you want to export
columns_to_export = ['T [ms]', 'AccX [mg]', 'AccY [mg]', 'AccZ [mg]', 'State']
data_to_export = data_to_export[columns_to_export]

# Export data to a new CSV file
export_file_name = input("Enter the name of the CSV file for export: ")
data_to_export.to_csv('output/' + export_file_name + '.csv', index=False, header=True)

print(f"Data exported to file '{export_file_name}'.")

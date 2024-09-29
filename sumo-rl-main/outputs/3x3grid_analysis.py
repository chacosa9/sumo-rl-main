import xml.etree.ElementTree as ET
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load and parse the tripinfo XML file
tripinfo_path = 'tripinfo.xml'  # Update the path if necessary
tree = ET.parse(tripinfo_path)
root = tree.getroot()

# Extract relevant trip information from the XML file
trip_data = []
for trip in root.findall('tripinfo'):
    trip_info = {
        'id': trip.get('id'),
        'depart': float(trip.get('depart')),
        'arrival': float(trip.get('arrival')),
        'duration': float(trip.get('duration')),
        'routeLength': float(trip.get('routeLength')),
        'waitingTime': float(trip.get('waitingTime')),
        'timeLoss': float(trip.get('timeLoss')),
        'speedFactor': float(trip.get('speedFactor'))
    }
    trip_data.append(trip_info)

# Create a DataFrame for analysis
df = pd.DataFrame(trip_data)

# Save the DataFrame to a CSV file
csv_file_path = 'tripinfo_output.csv'  # Specify the desired file name
df.to_csv(csv_file_path, index=False)
print(f"Data successfully saved to {csv_file_path}")

# 1. Summary statistics of the data
summary_stats = df.describe()
print("Summary statistics of the trips:\n", summary_stats)

# Plot 1: Distribution of Trip Duration
plt.figure(figsize=(10, 6))
plt.hist(df['duration'], bins=30, color='blue', edgecolor='black', alpha=0.7)
plt.title('Distribution of Trip Duration', fontsize=14)
plt.xlabel('Trip Duration (seconds)', fontsize=12)
plt.ylabel('Number of Vehicles', fontsize=12)
plt.grid(True)
plt.show()

# Plot 2: Distribution of Waiting Time
plt.figure(figsize=(10, 6))
plt.hist(df['waitingTime'], bins=30, color='green', edgecolor='black', alpha=0.7)
plt.title('Distribution of Waiting Time at Traffic Lights', fontsize=14)
plt.xlabel('Waiting Time (seconds)', fontsize=12)
plt.ylabel('Number of Vehicles', fontsize=12)
plt.grid(True)
plt.show()

# Plot 3: Route Length vs Trip Duration
plt.figure(figsize=(10, 6))
plt.scatter(df['routeLength'], df['duration'], color='red', alpha=0.7)
plt.title('Route Length vs Trip Duration', fontsize=14)
plt.xlabel('Route Length (meters)', fontsize=12)
plt.ylabel('Trip Duration (seconds)', fontsize=12)
plt.grid(True)
plt.show()

# Plot 4: Speed Factor vs Time Loss
plt.figure(figsize=(10, 6))
plt.scatter(df['speedFactor'], df['timeLoss'], color='purple', alpha=0.7)
plt.title('Speed Factor vs Time Loss', fontsize=14)
plt.xlabel('Speed Factor', fontsize=12)
plt.ylabel('Time Loss (seconds)', fontsize=12)
plt.grid(True)
plt.show()

# 5. Correlation Analysis
correlation_matrix = df.corr()
print("Correlation Matrix:\n", correlation_matrix)

# Plot 5: Heatmap of Correlations
plt.figure(figsize=(10, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Between Variables', fontsize=14)
plt.show()

# Define signal round duration (e.g., 60 seconds per signal round)
round_duration = 60  # Adjust this value as needed to represent the duration of a signal round

# Calculate the signal round for each trip based on departure time
df['signal_round'] = (df['depart'] // round_duration).astype(int)

# Count vehicles departing in each signal round
vehicles_per_round = df.groupby('signal_round').size().reset_index(name='vehicle_count')
print("Vehicles per signal round:\n", vehicles_per_round)

# Aggregate waiting time by each signal round
waiting_time_per_round = df.groupby('signal_round')['waitingTime'].sum().reset_index()

# Plot Total Waiting Time per Signal Round
plt.figure(figsize=(10, 6))
plt.plot(waiting_time_per_round['signal_round'] * round_duration, waiting_time_per_round['waitingTime'], label='Total Waiting Time')
plt.title('Total Waiting Time of Vehicles Per Signal Round', fontsize=14)
plt.xlabel('Time Step (s)', fontsize=12)
plt.ylabel('Total Waiting Time of Vehicles (s)', fontsize=12)
plt.grid(True)
plt.legend()
plt.show()

# Plot the number of vehicles per signal round
plt.figure(figsize=(10, 6))
plt.plot(vehicles_per_round['signal_round'] * round_duration, vehicles_per_round['vehicle_count'], label='Vehicle Count', color='orange')
plt.title('Number of Vehicles Per Signal Round', fontsize=14)
plt.xlabel('Time Step (s)', fontsize=12)
plt.ylabel('Number of Vehicles', fontsize=12)
plt.grid(True)
plt.legend()
plt.show()

# Report key findings
avg_trip_duration = df['duration'].mean()
avg_waiting_time = df['waitingTime'].mean()
max_waiting_time = df['waitingTime'].max()

print(f"Key Findings:")
print(f"1. Average Trip Duration: {avg_trip_duration:.2f} seconds")
print(f"2. Average Waiting Time: {avg_waiting_time:.2f} seconds")
print(f"3. Maximum Waiting Time: {max_waiting_time:.2f} seconds")

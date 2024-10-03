import xml.etree.ElementTree as ET
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def parse_tripinfo_xml(file_path):
    """Parses the XML file and returns a DataFrame with relevant trip data."""
    tree = ET.parse(file_path)
    root = tree.getroot()

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

    return pd.DataFrame(trip_data)

def plot_comparison(df1, df2, file1_label='Original Simulation', file2_label='RL Simulation', round_duration=60):
    """Plots a comparison of waiting times between two DataFrames."""
    # Calculate signal round for each trip based on departure time
    df1['signal_round'] = (df1['depart'] // round_duration).astype(int)
    df2['signal_round'] = (df2['depart'] // round_duration).astype(int)
    
    # Aggregate waiting time by each signal round
    waiting_time_per_round_1 = df1.groupby('signal_round')['waitingTime'].sum().reset_index()
    waiting_time_per_round_2 = df2.groupby('signal_round')['waitingTime'].sum().reset_index()

    # Plot both lines for comparison
    plt.figure(figsize=(10, 6))

    # File 1 line
    sns.lineplot(x=waiting_time_per_round_1['signal_round'] * round_duration, 
                 y=waiting_time_per_round_1['waitingTime'], 
                 label=file1_label, color='orange')

    # File 2 line
    sns.lineplot(x=waiting_time_per_round_2['signal_round'] * round_duration, 
                 y=waiting_time_per_round_2['waitingTime'], 
                 label=file2_label, color='blue')

    # Titles and labels
    plt.title('Comparison of Total Waiting Time Per Signal Round', fontsize=14)
    plt.xlabel('Time Step (s)', fontsize=12)
    plt.ylabel('Total Waiting Time of Vehicles (s)', fontsize=12)
    plt.legend()
    plt.grid(True)
    
    # Show the plot
    plt.show()

# Example Usage
file1 = 'D:\\mproject\\1\\sumo-rl-main\\tripinfo_1.xml'  # Path to the first XML file
file2 = 'D:\\mproject\\1\\sumo-rl-main\\tripinfo_2.xml'  # Path to the second XML file

# Parse the XML files
df1 = parse_tripinfo_xml(file1)
df2 = parse_tripinfo_xml(file2)

# Compare and plot the results
plot_comparison(df1, df2, file1_label='Original Simulation', file2_label='RL Simulation')

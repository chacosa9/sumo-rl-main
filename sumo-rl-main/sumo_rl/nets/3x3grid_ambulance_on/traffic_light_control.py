import traci
import sumolib

# Corrected file path with forward slashes
sumoCmd = ["sumo-gui", "-c", "D:\mproject\1\sumo-rl-main\sumo-rl-main\sumo_rl\nets\3x3grid_ambulance_on\3x3grid.sumocfg"]  # Replace with the correct path to your .sumocfg file

try:
    print("Starting SUMO...")
    traci.start(sumoCmd)  # Start SUMO
    print("SUMO started successfully.")
except Exception as e:
    print(f"Error starting SUMO: {e}")

# Get the traffic lights and vehicle information
def change_traffic_lights(emergency_vehicle_id):
    try:
        while traci.simulation.getMinExpectedNumber() > 0:
            traci.simulationStep()

            # Check if the emergency vehicle is in the simulation
            vehicle_ids = traci.vehicle.getIDList()
            print(f"Vehicles in simulation: {vehicle_ids}")

            if emergency_vehicle_id in vehicle_ids:
                vehicle_position = traci.vehicle.getRoadID(emergency_vehicle_id)
                print(f"Emergency vehicle position: {vehicle_position}")
                traffic_light_id = traci.trafficlight.getIDList()

                # Logic to turn the traffic light green when the emergency vehicle is near
                for tl_id in traffic_light_id:
                    controlled_lanes = traci.trafficlight.getControlledLanes(tl_id)
                    if vehicle_position in controlled_lanes:
                        # Set the traffic light to green
                        traci.trafficlight.setRedYellowGreenState(tl_id, 'GGGrrr')  # Example for 3-phase traffic light
                        print(f"Traffic light {tl_id} turned green for emergency vehicle")

            # Step through the simulation
            traci.simulationStep()

    except Exception as e:
        print(f"Error during simulation: {e}")
    finally:
        print("Closing TraCI...")
        traci.close()

# Main function
if __name__ == "__main__":
    emergency_vehicle_id = "emergency_1"  # The ID of your emergency vehicle
    change_traffic_lights(emergency_vehicle_id)
    print("Simulation complete.")

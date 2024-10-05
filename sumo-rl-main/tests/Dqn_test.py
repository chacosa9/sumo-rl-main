import os
import numpy as np
import pandas as pd
import torch
from sumo_rl.environment.env import SumoEnvironment
from sumo_rl.agents.dqn_agent import DQNAgent

# Check if SUMO_HOME is set up properly
if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    os.environ["PATH"] += os.pathsep + tools
else:
    raise ImportError("Please declare the environment variable 'SUMO_HOME'")

# Load all trained models
MODEL_DIR = '/Users/rakanalrasheed/Desktop/SDAIA Bootcamp/Capstone/sumo-rl-main/sumo-rl-main/experiments/models'
models = {}

# Setup the SUMO environment
env = SumoEnvironment(
    net_file='/Users/rakanalrasheed/Desktop/SDAIA Bootcamp/Capstone/sumo-rl-main/sumo-rl-main/sumo_rl/nets/3x3grid/3x3Grid2lanes.net.xml',
    route_file='/Users/rakanalrasheed/Desktop/SDAIA Bootcamp/Capstone/sumo-rl-main/sumo-rl-main/sumo_rl/nets/3x3grid/routes14000.rou.xml',
    use_gui=True,  # or False if you don't want to visualize the simulation
    num_seconds=20000,
    delta_time=5,
    reward_fn="diff-waiting-time",  # or your custom reward function
    single_agent=False,
)

# Define the correct state and action dimensions by fetching from environment
for ts_id in env.ts_ids:
    state_space = env.observation_spaces(ts_id)
    action_space = env.action_spaces(ts_id)

    # Fetch state space and action space dimensions
    state_space_dim = state_space.shape[0] if hasattr(state_space, 'shape') else state_space
    action_space_dim = action_space.n if hasattr(action_space, 'n') else action_space

    # Load the models into the DQN agents
    models[ts_id] = DQNAgent(state_space=state_space, action_space=action_space, epsilon=0.0)

    # Load saved model weights
    model_path = os.path.join(MODEL_DIR, f'dqn_{ts_id}_episode_1.pt')
    models[ts_id].model.load_state_dict(torch.load(model_path))

# Simulation parameters
num_episodes = 10
max_steps = int(env.sim_max_time / env.delta_time)

# Test the models
for episode in range(num_episodes):
    print(f"Starting episode {episode + 1}/{num_episodes}")
    obs = env.reset()
    done = {"__all__": False}
    step = 0
    total_vehicles_passed = {ts_id: 0 for ts_id in env.ts_ids}
    total_waiting_time = 0
    total_rewards = {ts_id: 0.0 for ts_id in env.ts_ids}
    total_trip_time = 0
    vehicle_ids = set()  # To track unique vehicle IDs

    while not done["__all__"]:
        # Select actions based on the loaded DQN models
        actions = {}
        for ts_id in env.ts_ids:
            state = obs[ts_id].reshape(1, -1)  # Reshape for the model input

            # Predict the action and detach the tensor from the computation graph
            action = np.argmax(models[ts_id].model(torch.FloatTensor(state)).detach().numpy())
            actions[ts_id] = action

        # Step in the environment
        next_obs, rewards, done, _ = env.step(actions)
        obs = next_obs

        # Log number of vehicles passing through each signal
        for ts_id in env.ts_ids:
            total_vehicles_passed[ts_id] += env.traffic_signals[ts_id].get_total_queued()
            total_rewards[ts_id] += rewards[ts_id]

        # Calculate total waiting time
        total_waiting_time += sum(env.sumo.vehicle.getWaitingTime(veh) for veh in env.sumo.vehicle.getIDList())

        # Calculate the number of vehicles in the system and their trip times
        current_vehicle_ids = set(env.sumo.vehicle.getIDList())
        for veh in current_vehicle_ids:
            if veh not in vehicle_ids:
                vehicle_ids.add(veh)
                # Approximate trip time using accumulated waiting time
                trip_time = env.sumo.vehicle.getAccumulatedWaitingTime(veh)
                total_trip_time += trip_time

        # Increment the step
        step += 1

    print(f"Episode {episode + 1} finished after {step} steps")

    # Calculate metrics for this episode
    avg_waiting_time = total_waiting_time / len(vehicle_ids) if len(vehicle_ids) > 0 else 0
    avg_trip_time = total_trip_time / len(vehicle_ids) if len(vehicle_ids) > 0 else 0
    num_vehicles_in_system = len(env.sumo.vehicle.getIDList())

    # Save metrics for this episode
    episode_metrics = {
        "episode": episode + 1,
        "total_vehicles_passed": total_vehicles_passed,
        "avg_waiting_time": avg_waiting_time,
        "cumulative_reward": total_rewards,
        "num_vehicles_in_system": num_vehicles_in_system,
        "avg_trip_time": avg_trip_time,
    }

    # Convert to DataFrame and save for each episode
    df_episode = pd.DataFrame([episode_metrics])
    df_episode.to_csv(f"test_results_modified/test_results_episode_{episode + 1}.csv", index=False)

env.close()

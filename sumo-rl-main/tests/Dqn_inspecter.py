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

    print(f"State space for {ts_id}: {state_space}")
    print(f"Action space for {ts_id}: {action_space}")

    # Fetch state space dimension
    state_space_dim = state_space.shape[0] if hasattr(state_space, 'shape') else state_space
    # Fetch action space dimension
    action_space_dim = action_space.n if hasattr(action_space, 'n') else action_space

    # Load the models into the DQN agents
    models[ts_id] = DQNAgent(state_space=state_space, action_space=action_space, epsilon=0.0)

    # Log the epsilon value (for checking exploration rate)
    print(f"Epsilon value for Agent {ts_id}: {models[ts_id].epsilon}")

    # Load saved model weights
    model_path = os.path.join(MODEL_DIR, f'dqn_{ts_id}_episode_1.pt')
    models[ts_id].model.load_state_dict(torch.load(model_path))

    # Log the model weights for sanity check
    print(f"Model weights for Agent {ts_id}: {models[ts_id].model.state_dict()}")

# Simulation parameters
num_episodes = 10
max_steps = int(env.sim_max_time / env.delta_time)

# Test the models
for episode in range(num_episodes):
    print(f"\n--- Starting episode {episode + 1}/{num_episodes} ---\n")
    obs = env.reset()
    done = {"__all__": False}
    step = 0
    total_rewards = {ts_id: 0 for ts_id in env.ts_ids}  # Initialize total rewards for each agent

    while not done["__all__"]:
        # Select actions based on the loaded DQN models
        actions = {}
        for ts_id in env.ts_ids:
            state = obs[ts_id].reshape(1, -1)  # Reshape for the model input
            # Predict the action and detach the tensor from the computation graph
            action = np.argmax(models[ts_id].model(torch.FloatTensor(state)).detach().numpy())
            actions[ts_id] = action

            # Log the state, action, and other details for debugging
            print(f"Step {step}, Agent {ts_id}: State: {state}, Action: {action}")

        # Step in the environment
        next_obs, rewards, done, _ = env.step(actions)
        obs = next_obs

        # Log rewards for the step and accumulate total rewards
        for ts_id in rewards.keys():
            total_rewards[ts_id] += rewards[ts_id]
            print(f"Step {step}, Agent {ts_id}: Reward: {rewards[ts_id]} (Total so far: {total_rewards[ts_id]})")

        step += 1

    print(f"\nEpisode {episode + 1} finished after {step} steps")
    print(f"Total rewards per agent for episode {episode + 1}: {total_rewards}")

    # Log or save the episode results (e.g., average waiting time, trip time, etc.)
    episode_results = env.metrics
    print(f"Metrics for Episode {episode + 1}: {episode_results}")  # Print the metrics for inspection

    # Convert the metrics to a DataFrame
    df_results = pd.DataFrame(episode_results)
    print(f"Results DataFrame for Episode {episode + 1}:")
    print(df_results.head())  # Preview the DataFrame before saving
    
    # Save to CSV
    df_results.to_csv(f"test_results_episode_{episode + 1}.csv", index=False)

env.close()

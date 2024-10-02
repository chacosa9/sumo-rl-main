import os
import numpy as np
from sumo_rl.environment import SumoEnvironment
from sumo_rl.agents.dqn_agent import DQNAgent
import sys
# sys.path.append("/Users/rakanalrasheed/Desktop/SDAIA Bootcamp/Capstone/sumo-rl-main/sumo-rl-main")

# Add the parent directory (sumo-rl-main) to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


# Directory to save model weights and logs
SAVE_DIR = 'models/'

def train_dqn(env, num_episodes=500, max_steps=2000, save_every=10):
    """Train the DQN agent for traffic signal control using SUMO simulation.

    Args:
        env: SumoEnvironment instance.
        num_episodes: Number of episodes to train.
        max_steps: Maximum steps per episode.
        save_every: Number of episodes after which to save the agent model.
    """
    # Initialize DQN agents for each traffic signal
# Initialize DQN agents for each traffic signal
    agents = {
        ts: DQNAgent(
            state_space=env.observation_spaces(ts),
            action_space=env.action_spaces(ts)
        )
        for ts in env.ts_ids
    }


    for episode in range(num_episodes):
        obs = env.reset()
        done = {"__all__": False}
        step = 0
        total_reward = 0

        while not done["__all__"] and step < max_steps:
            actions = {ts: agents[ts].act(obs[ts]) for ts in env.ts_ids}  # Get actions for each agent
            next_obs, rewards, done, _ = env.step(actions)  # Take a step in the environment

            for ts in env.ts_ids:
                # Store the transition in agent's memory
                agents[ts].remember(
                    state=obs[ts],
                    action=actions[ts],
                    reward=rewards[ts],
                    next_state=next_obs[ts],
                    done=done[ts]
                )
                # Perform a replay step (train on a batch of experiences)
                agents[ts].replay()

                # Update total reward for this episode
                total_reward += rewards[ts]

            # Update observations
            obs = next_obs
            step += 1

        # Update target network after every episode
        for ts in env.ts_ids:
            agents[ts].update_target_network()

        print(f"Episode {episode + 1}/{num_episodes} - Total Reward: {total_reward}")

        # Save the model every 'save_every' episodes
        if (episode + 1) % save_every == 0:
            for ts in env.ts_ids:
                agents[ts].save(os.path.join(SAVE_DIR, f"dqn_{ts}_episode_{episode + 1}.h5"))

    # Save the final model after training
    for ts in env.ts_ids:
        agents[ts].save(os.path.join(SAVE_DIR, f"dqn_{ts}_final.h5"))

if __name__ == "__main__":
    # Load the SUMO environment
    print("Training started...")

    env = SumoEnvironment(
        net_file = '/Users/rakanalrasheed/Desktop/SDAIA Bootcamp/Capstone/sumo-rl-main/sumo-rl-main/sumo_rl/nets/3x3grid/3x3Grid2lanes.net.xml',
        route_file = '/Users/rakanalrasheed/Desktop/SDAIA Bootcamp/Capstone/sumo-rl-main/sumo-rl-main/sumo_rl/nets/3x3grid/routes14000.rou.xml',
        out_csv_name='output/DqnTrainingResults/simulation_results.csv',
        use_gui=True,  # Set to True to visualize simulation
        num_seconds=3000  # Simulate for 10,000 seconds per episode
    )

    # Train the DQN agents
    train_dqn(env, num_episodes=300, max_steps=1000, save_every=1)

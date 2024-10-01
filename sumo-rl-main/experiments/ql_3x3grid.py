import os
import numpy as np
import pandas as pd
from sumo_rl.environment.env import SumoEnvironment
from sumo_rl.agents.ql_agent import QLAgent  # Modified QLAgent with pre-training

if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    os.environ["PATH"] += os.pathsep + tools
else:
    raise ImportError("Please declare the environment variable 'SUMO_HOME'")

# Setting up the environment for the 3x3 grid
env = SumoEnvironment(
    net_file = '/Users/rakanalrasheed/Desktop/SDAIA Bootcamp/Capstone/sumo-rl-main/sumo-rl-main/sumo_rl/nets/3x3grid/3x3Grid2lanes.net.xml',
    route_file = '/Users/rakanalrasheed/Desktop/SDAIA Bootcamp/Capstone/sumo-rl-main/sumo-rl-main/sumo_rl/nets/3x3grid/routes14000.rou.xml',
    out_csv_name="outputs/ql_3x3grid_train_results_preAlgorithm.csv", 
    use_gui=True,
    num_seconds=20000,
    delta_time=5,
    reward_fn="diff-waiting-time",
    single_agent=False,
)

# Create a Q-learning agent for each traffic signal in the environment
agents = {
    ts: QLAgent(
        starting_state=env.reset()[ts],
        state_space=env.observation_spaces(ts),
        action_space=env.action_spaces(ts),
        alpha=0.1,
        gamma=0.99,
        epsilon=1.0,  # Start with high exploration
        epsilon_decay=0.995,  # Slowly decrease exploration
        epsilon_min=0.1  # Minimum exploration threshold
    )
    for ts in env.ts_ids
}

# Q-Learning algorithm parameters
num_episodes = 500
max_steps = int(env.sim_max_time / env.delta_time)

for episode in range(num_episodes):
    obs = env.reset()
    done = {"__all__": False}
    step = 0
    print(f"Starting Episode {episode}")

    while not done["__all__"]:
        actions = {ts: agents[ts].choose_action(obs[ts]) for ts in env.ts_ids}
        next_obs, rewards, done, _ = env.step(actions)

        for ts in env.ts_ids:
            agents[ts].learn(
                state=obs[ts],
                action=actions[ts],
                reward=rewards[ts],
                next_state=next_obs[ts],
                done=done[ts]
            )

        obs = next_obs
        step += 1
        if step % 100 == 0:
            print(f"Step {step}: Action taken {actions}")
    
    # Log episode results every 10 episodes
    if episode % 10 == 0:
        print(f"Episode {episode}/{num_episodes} completed.")
        
    for ts in env.ts_ids:
        np.save(f"outputs/ql_3x3grid_qtable_ep{episode}_{ts}.npy", agents[ts].q_table)
        print("Q-table saved for episode", episode)


# Save Q-tables after training
for ts in env.ts_ids:
    np.save(f"outputs/ql_3x3grid_qtable_{ts}.npy", agents[ts].q_table)

# Close the environment
env.close()
print("Training completed.")

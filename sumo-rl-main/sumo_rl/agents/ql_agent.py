import numpy as np
from sumo_rl.exploration.epsilon_greedy import EpsilonGreedy

class QLAgent:
    """Q-learning Agent with pre-trained heuristic initialization."""

    def __init__(self, starting_state, state_space, action_space, alpha=0.5, gamma=0.95, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.1):
        """Initialize Q-learning agent with pre-trained heuristic."""
        self.state = tuple(starting_state)  # Convert to tuple for hashability
        self.state_space = state_space
        self.action_space = action_space
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon  # For exploration vs exploitation
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.q_table = {self.state: [0 for _ in range(action_space.n)]}

        # Accumulated reward for performance monitoring
        self.acc_reward = 0
        
        # Pre-training heuristic: Fixed time slot allocation for each light
        self.pretrained_policy = self.initialize_pretrained_policy()

    def initialize_pretrained_policy(self):
        """Pre-trained heuristic policy (e.g., prioritize lanes with higher queue lengths)."""
        pretrained_policy = {}
        
        # Example heuristic: pre-define actions for a few sample states (for learning kickstart)
        for _ in range(100):  # Sample 100 initial states
            state_sample = self.state_space.sample()  # Sample from the state space
            state_sample = tuple(state_sample)  # Convert to tuple for hashability
            pretrained_policy[state_sample] = np.random.choice(range(self.action_space.n))  # Random action for pre-training

        return pretrained_policy

    def choose_action(self, state):
        """Choose an action based on epsilon-greedy exploration."""
        state = tuple(state)  # Convert to tuple for hashability
        if np.random.rand() <= self.epsilon:
            # Exploration: random action
            return self.action_space.sample()
        else:
            # Exploitation: follow the pre-trained policy or learned Q-values
            if state in self.q_table:
                return np.argmax(self.q_table[state])
            else:
                # Follow pre-trained policy as fallback if Q-values are not yet learned
                return self.pretrained_policy.get(state, self.action_space.sample())

    def learn(self, state, action, reward, next_state, done):
        """Update Q-table based on experience."""
        state = tuple(state)  # Convert to tuple for hashability
        next_state = tuple(next_state)  # Convert next_state to tuple for hashability

        if next_state not in self.q_table:
            self.q_table[next_state] = [0 for _ in range(self.action_space.n)]
        
        s = state
        s1 = next_state
        a = action
        self.q_table[s][a] = self.q_table[s][a] + self.alpha * (
            reward + self.gamma * max(self.q_table[s1]) - self.q_table[s][a]
        )
        
        # Decay epsilon after each episode
        if done:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        self.acc_reward += reward

import pickle

# Add this save function
    def save_q_table(q_table, file_name="q_table.pkl"):
        """Save the Q-table to a file."""
        with open(file_name, 'wb') as f:
            pickle.dump(q_table, f)

# Wherever your training loop ends for an episode, add this
if done:  # Done means the end of an episode
    save_q_table(agent.q_table, "q_table_current.pkl")

import numpy as np
import pickle

class GreenWaveQLAgent:
    """Q-learning Agent optimized for the Green Wave Algorithm."""

    def __init__(self, starting_state, state_space, action_space, alpha=0.1, gamma=0.99, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.1):
        """Initialize Q-learning agent with Green Wave optimization."""
        self.state = tuple(starting_state)  # Convert to tuple for hashability
        self.state_space = state_space
        self.action_space = action_space
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon  # Exploration vs exploitation
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.q_table = {self.state: [0 for _ in range(action_space.n)]}
        self.acc_reward = 0

    def choose_action(self, state):
        """Choose an action based on epsilon-greedy exploration."""
        state = tuple(state)
        if np.random.rand() <= self.epsilon:
            return self.action_space.sample()  # Exploration
        if state in self.q_table:
            return np.argmax(self.q_table[state])  # Exploitation
        else:
            return self.action_space.sample()

    def learn(self, state, action, reward, next_state, done):
        """Update Q-table based on experience."""
        state = tuple(state)
        next_state = tuple(next_state)

        if next_state not in self.q_table:
            self.q_table[next_state] = [0 for _ in range(self.action_space.n)]
        
        # Q-Learning update
        s, s1, a = state, next_state, action
        self.q_table[s][a] = self.q_table[s][a] + self.alpha * (reward + self.gamma * max(self.q_table[s1]) - self.q_table[s][a])
        
        if done:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        self.acc_reward += reward

    def save_q_table(self, filename='q_table_final.pkl'):
        """Save the Q-table to a file."""
        with open(filename, 'wb') as f:
            pickle.dump(self.q_table, f)
        print("Q-table saved successfully!")

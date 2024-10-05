import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque


# Define the DQN network architecture
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)
    

# Define the DQN Agent
class DQNAgent:
    def __init__(self, state_space, action_space, gamma=0.99, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995, lr=0.001, batch_size=64, memory_size=10000):
        self.state_space = state_space
        self.action_space = action_space
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.lr = lr
        self.batch_size = batch_size
        self.memory = deque(maxlen=memory_size)

        # Initialize the DQN network and target network
        self.model = DQN(self.get_state_size(), self.get_action_size())
        self.target_model = DQN(self.get_state_size(), self.get_action_size())
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.loss_fn = nn.MSELoss()

        # Copy weights from the model to the target network
        self.update_target_network()

    def get_state_size(self):
        """Extracts the state space size (input dimension)."""
        return self.state_space.shape[0] if hasattr(self.state_space, 'shape') else self.state_space.n

    def get_action_size(self):
        """Extracts the action space size (output dimension)."""
        return self.action_space.n if hasattr(self.action_space, 'n') else self.action_space

    def update_target_network(self):
        """Updates the target network with the current model weights."""
        self.target_model.load_state_dict(self.model.state_dict())

    def remember(self, state, action, reward, next_state, done):
        """Stores experience tuples in the replay memory."""
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        """Chooses an action based on epsilon-greedy strategy."""
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.get_action_size())  # Random action
        state = torch.FloatTensor(state).unsqueeze(0)  # Convert state to tensor
        q_values = self.model(state)
        return torch.argmax(q_values).item()  # Greedy action

    def replay(self):
        """Performs experience replay and updates the model."""
        if len(self.memory) < self.batch_size:
            return

        # Sample a mini-batch from the replay memory
        batch = random.sample(self.memory, self.batch_size)
        for state, action, reward, next_state, done in batch:
            target = reward
            if not done:
                next_state = torch.FloatTensor(next_state).unsqueeze(0)
                target = reward + self.gamma * torch.max(self.target_model(next_state)).item()

            state = torch.FloatTensor(state).unsqueeze(0)
            target_f = self.model(state)
            target_f[0][action] = target  # Update the Q-value for the taken action

            # Perform a gradient descent step
            self.optimizer.zero_grad()
            loss = self.loss_fn(self.model(state), target_f)
            loss.backward()
            self.optimizer.step()

        # Decay epsilon for exploration-exploitation balance
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    # def save(self, path):
    #     """Saves the model weights to a file."""
    #     torch.save(self.model.state_dict(), path)
    def save(self, path):
        """Saves the model weights to a file."""
        try:
            torch.save(self.model.state_dict(), path)
            print(f"Model saved successfully at {path}")
        except Exception as e:
            print(f"Error saving model: {e}")


    def load(self, path):
        """Loads the model weights from a file."""
        self.model.load_state_dict(torch.load(path))


import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque

BLACK = 1
WHITE = 2

dim = 3
class QNetwork(nn.Module):
    def __init__(self, input_dim=64, output_dim=64, hidden_dim=256, num_layers=3, learning_rate=0.001):
        super(QNetwork, self).__init__()
        self.learning_rate = learning_rate

        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))

        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())

        layers.append(nn.Linear(hidden_dim, output_dim))
        #layers.append(nn.Softmax(dim=-1))  # Dành cho phân phối xác suất các hành động

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

class QNetworkAgent:
    def __init__(self):
        self.model = QNetwork()
        self.target_model = QNetwork()
        self.target_model.load_state_dict(self.model.state_dict())
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.gamma = 0.95
        self.epsilon = 0.8
        self.epsilon_min = 0
        self.epsilon_decay = 0.9
        self.current_loss = 0.0
        self.losses = []
        self.memory = deque(maxlen=15000)
        self.update_target_steps = 1000
        self.train_step = 0
        self.max_grad_norm = 1.0

    def encode_state(self, obs):
        board = obs['board'].reshape(-1)
        player = obs['turn']
        if player == WHITE:
            board = np.where(board == BLACK, -1, board)
            board = np.where(board == WHITE, 1, board)
        else:
            board = np.where(board == BLACK, 1, board)
            board = np.where(board == WHITE, -1, board)
        return torch.tensor(board, dtype=torch.float32)

    def choose_action(self, state, valid_moves):
        if random.random() < self.epsilon:
            return random.choice(valid_moves)
        self.model.eval()
        with torch.no_grad():
            q_values = self.model(state.unsqueeze(0)).squeeze(0)
            best_idx = max(valid_moves, key=lambda idx: q_values[idx].item())
        self.model.train()
        return best_idx

    def replay_buffer_save(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        minibatch = random.sample(self.memory, batch_size)
        states = torch.stack([s for s, _, _, _, _ in minibatch])
        actions = torch.tensor([a for _, a, _, _, _ in minibatch], dtype=torch.long)
        rewards = torch.tensor([r for _, _, r, _, _ in minibatch], dtype=torch.float32)
        next_states = torch.stack([ns for _, _, _, ns, _ in minibatch])
        dones = torch.tensor([d for _, _, _, _, d in minibatch], dtype=torch.float32)

        # Q(s, a)
        q_values = self.model(states)
        state_action_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        # Q_target(s', a')
        with torch.no_grad():
            next_q_values = self.target_model(next_states)
            max_next_q_values, _ = next_q_values.max(1)
            expected_state_action_values = rewards + self.gamma * max_next_q_values * (1 - dones)

        loss = self.criterion(state_action_values, expected_state_action_values)
        self.current_loss = loss.item()
        self.losses.append(self.current_loss)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        self.optimizer.step()

        # Update target network
        self.train_step += 1
        if self.train_step % self.update_target_steps == 0:
            self.target_model.load_state_dict(self.model.state_dict())

        # Epsilon decay
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
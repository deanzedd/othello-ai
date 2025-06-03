import torch
import torch.nn as nn
import torch.optim as optim
import cupy as cp # Đổi lại thành cp để tránh nhầm lẫn với numpy
import numpy as np # Vẫn cần numpy cho một số thao tác CPU
import random
from collections import deque
from othello import Game 

BLACK = 1
WHITE = 2
EMPTY = 0
PASS_ACTION = -9

# Xác định thiết bị toàn cục
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')

class PolicyNetwork(nn.Module):
    def __init__(self, input_dim=64, output_dim=64, hidden_dim=256, num_layers=3):
        super(PolicyNetwork, self).__init__()
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())

        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())

        layers.append(nn.Linear(hidden_dim, output_dim))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

class PolicyAgent:
    def __init__(self, learning_rate=0.0001, gamma=0.99, epsilon_start=1.0, epsilon_min=0.01, epsilon_decay=0.995):
        self.policy_network = PolicyNetwork().to(device)
        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=learning_rate)
        self.criterion = nn.CrossEntropyLoss()
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.memory = deque(maxlen=10000)
        self.losses = []
        self.current_loss = 0
        self.train_step = 0

    def encode_state(self, obs):
        """
        Mã hóa trạng thái bàn cờ thành vector đầu vào cho mạng nơ-ron.
        - Chuyển đổi bàn cờ 8x8 thành vector 1D 64 phần tử.
        - Chuẩn hóa giá trị: quân của agent là 1, quân đối thủ là -1, ô trống là 0.
        """
        board_raw = obs['board']
        
        # Đảm bảo board_raw là một numpy array và làm phẳng nó
        if isinstance(board_raw, torch.Tensor):
            board_np = board_raw.cpu().numpy().flatten()
        elif isinstance(board_raw, cp.ndarray):
            # Nếu đã là CuPy array, chỉ cần làm phẳng và giữ nguyên CuPy array
            board_np = board_raw.flatten() # board_np sẽ là cupy.ndarray
        elif isinstance(board_raw, np.ndarray):
            board_np = board_raw.flatten()
        else:
            board_np = np.array(board_raw).flatten()

        # CHUYỂN board_np SANG cupy.ndarray TRƯỚC KHI THỰC HIỆN CÁC PHÉP TOÁN CUPY
        # Nếu board_np đã là cp.ndarray từ nhánh elif trên, không cần chuyển lại
        # Nếu board_np là np.ndarray, cần chuyển
        if not isinstance(board_np, cp.ndarray):
            board_cp = cp.asarray(board_np)
        else:
            board_cp = board_np # board_cp đã là cupy array

        player = obs['turn']

        # Chuẩn hóa board state theo góc nhìn của người chơi hiện tại
        # Sử dụng board_cp (cupy.ndarray) cho các phép toán cupy.where
        if player == WHITE:
            board_encoded_cp = cp.where(board_cp == WHITE, 1, cp.where(board_cp == BLACK, -1, 0))
        else:
            board_encoded_cp = cp.where(board_cp == BLACK, 1, cp.where(board_cp == WHITE, -1, 0))
        
        # Chuyển tensor đầu ra lên đúng device
        # Chuyển từ cupy.ndarray sang torch.Tensor và đưa lên GPU
        return torch.tensor(cp.asnumpy(board_encoded_cp), dtype=torch.float32).to(device) 

    def choose_action(self, obs, valid_moves):
        if random.random() < self.epsilon:
            return random.choice(valid_moves)
        
        self.policy_network.eval()
        state_tensor = self.encode_state(obs)
        
        with torch.no_grad():
            action_logits = self.policy_network(state_tensor.unsqueeze(0)).squeeze(0)
            
            mask = torch.full_like(action_logits, float('-inf')).to(device)
            for r, c in valid_moves:
                idx = r * 8 + c
                mask[idx] = 0

            masked_action_logits = action_logits + mask
            action_probs = torch.softmax(masked_action_logits, dim=-1)

            action_probs_np = action_probs.cpu().numpy()
            action_indices_cp = cp.arange(action_probs.size(0))

            chosen_idx_cp = cp.random.choice(action_indices_cp, size=1, p=cp.asarray(action_probs_np))
            chosen_idx = int(chosen_idx_cp.item())

            chosen_row = chosen_idx // 8
            chosen_col = chosen_idx % 8
            
        self.policy_network.train()
        return (chosen_row, chosen_col)

    def store_experience(self, state, action, reward, next_state, done):
        if action == PASS_ACTION:
            return
        if isinstance(action, int):
            action_idx = action 
        else:
            action_idx = action[0] * 8 + action[1]
        self.memory.append((state.cpu(), action_idx, reward, next_state.cpu(), done))

    def update_policy(self, batch_size):
        if len(self.memory) < batch_size:
            return

        minibatch = random.sample(self.memory, batch_size)
        
        states = torch.stack([s for s, _, _, _, _ in minibatch]).to(device)
        actions_indices = torch.tensor([a for _, a, _, _, _ in minibatch], dtype=torch.long).to(device)
        rewards = torch.tensor([r for _, _, r, _, _ in minibatch], dtype=torch.float32).to(device)
        next_states = torch.stack([ns for _, _, _, ns, _ in minibatch]).to(device)
        dones = torch.tensor([d for _, _, _, _, d in minibatch], dtype=torch.float32).to(device)

        action_logits = self.policy_network(states)
        log_probs = torch.log_softmax(action_logits, dim=-1)
        
        chosen_action_log_probs = log_probs.gather(1, actions_indices.unsqueeze(1)).squeeze(1)

        loss = - (chosen_action_log_probs * rewards).mean()

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_network.parameters(), 1.0)
        self.optimizer.step()
        self.current_loss=loss.item()
        self.losses.append(loss.item())
        self.train_step += 1

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
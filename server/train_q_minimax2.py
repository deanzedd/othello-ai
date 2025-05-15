import random
from collections import defaultdict
from test_enviroment import OthelloEnv  # Import môi trường Othello của bạn
from othello import BLACK, WHITE
import pickle
import numpy as np
import copy
import utils
from ai import evaluator # Giả định evaluator.py chứa hàm minimax

class QLearningAgent:
    def __init__(self, alpha=0.1, gamma=0.9, epsilon=0.2, board_size=8):
        self.Q = defaultdict(lambda: defaultdict(float))  # Q[state][action]
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.board_size = board_size
        # Thêm minimax_depth để kiểm soát độ sâu tìm kiếm minimax
        self.minimax_depth = 3 # Độ sâu tìm kiếm minimax, có thể điều chỉnh

    def get_state_id(self, obs):
        # Chuyển trạng thái về dạng hashable (dùng tuple)
        board = tuple(obs['board'].flatten())
        turn = obs['turn']
        return (board, turn)

    def choose_action(self, obs, valid_actions):
        state = self.get_state_id(obs)
        if random.random() < self.epsilon:
            # Khám phá: chọn hành động ngẫu nhiên
            return random.choice(valid_actions)
        else:
            # Khai thác: chọn hành động có Q-value cao nhất
            q_values = self.Q[state]
            # Chọn action có Q-value cao nhất trong các hành động hợp lệ
            best_action = None
            max_q_value = -float('inf')
            for action in valid_actions:
                if q_values[action] > max_q_value:
                    max_q_value = q_values[action]
                    best_action = action
            # Nếu không có hành động hợp lệ nào trong Q-table, chọn ngẫu nhiên
            if best_action is None:
                 return random.choice(valid_actions)
            return best_action


    def learn(self, obs, action, reward, next_obs, done, valid_next_actions):
        state = self.get_state_id(obs)
        next_state = self.get_state_id(next_obs)
        q_sa = self.Q[state][action]

        if done:
            q_target = reward  # Sử dụng phần thưởng cuối cùng khi game kết thúc
        else:
            # Sử dụng giá trị minimax của trạng thái tiếp theo làm mục tiêu
            # Cần sao chép next_obs['board'] để tránh sửa đổi trạng thái thật
            next_board_copy = copy.deepcopy(next_obs['board'])
            # minimax ở đây cần tính cho người chơi hiện tại của next_obs
            minimax_value = evaluator.minimax(next_board_copy, self.minimax_depth, next_obs['turn'] == WHITE, -1000, 1000) # Giả định minimax_depth và isMaximizingPlayer dựa vào lượt chơi

            # Q-target = phần thưởng tức thời + gamma * giá trị của trạng thái tiếp theo (từ minimax)
            q_target = reward + self.gamma * minimax_value # Điều chỉnh công thức Q-target


        self.Q[state][action] += self.alpha * (q_target - q_sa)

    def save_q_table(self, path):
        with open(path, "wb") as f:
            pickle.dump(dict(self.Q), f)

    def load_q_table(self, path):
        try:
            with open(path, 'rb') as f:
                q_dict = pickle.load(f)
                self.Q = defaultdict(lambda: defaultdict(float), q_dict)
            print(f"Loaded Q-table from {path}")
        except FileNotFoundError:
            print(f"No Q-table found at {path}. Starting with an empty table.")


if __name__ == "__main__":
    env = OthelloEnv()
    agent = QLearningAgent()

    num_episodes = 10000 # Số lượng episode để huấn luyện

    # Tải Q-table nếu tồn tại để tiếp tục huấn luyện
    agent.load_q_table("q_table.pkl")


    for episode in range(num_episodes):
        obs, _ = env.reset()
        done = False

        while not done:
            # Lấy danh sách hành động hợp lệ
            valid_moves = env.game.get_valid_moves
            valid_actions = [row * env.board_size + col for row, col in valid_moves]

            if not valid_actions:
                # Nếu không còn hành động nào hợp lệ thì game kết thúc hoặc pass
                break

            action = agent.choose_action(obs, valid_actions)
            next_obs, reward, terminated, truncated, _ = env.step(action)

            done = terminated or truncated

            valid_moves_next = env.game.get_valid_moves
            next_valid_actions = [row * env.board_size + col for row, col in valid_moves_next]


            # Agent học sử dụng cả reward từ môi trường và giá trị minimax
            agent.learn(obs, action, reward, next_obs, done, next_valid_actions)


            obs = next_obs

        if (episode + 1) % 500 == 0:
            print(f"Tập {episode + 1} hoàn tất.")
            # Lưu Q-table định kỳ
            agent.save_q_table(f"q_table_episode_{episode + 1}.pkl")


    env.close()

    # Lưu Q-table cuối cùng ra file
    agent.save_q_table("q_table_final.pkl")
import random
import pickle
from collections import defaultdict
import numpy as np
from ai.ai_player import QLearningPlayer, RandomPlayer, AIPlayer 
from test_enviroment import OthelloEnv  # Import môi trường Othello của bạn

# Giả sử bạn có các class/hàm: Game, AIPlayer (Minimax), QLearningPlayer, utils

# Tham số huấn luyện
alpha = 0.1 # Tốc độ học
gamma = 0.9 # Hệ số chiết khấu
num_episodes = 1000 # Số ván đấu mô phỏng
q_table_save_path = 'q_table_othello_minimax.pkl'

# Khởi tạo Q-table
# Sử dụng defaultdict để tự động gán giá trị mặc định (0.0) cho các cặp trạng thái-hành động chưa thấy
Q = defaultdict(lambda: defaultdict(float))

# Cố gắng tải Q-table nếu đã tồn tại
try:
    with open(q_table_save_path, 'rb') as f:
        loaded_q = pickle.load(f)
        # Kết hợp hoặc thay thế Q hiện tại bằng Q đã tải
        for state, actions in loaded_q.items():
            for action, value in actions.items():
                 Q[state][action] = value
    print(f"Đã tải Q-table từ {q_table_save_path}")
except FileNotFoundError:
    print("Không tìm thấy Q-table cũ, bắt đầu với Q-table rỗng.")
except Exception as e:
    print(f"Lỗi khi tải Q-table: {e}, bắt đầu với Q-table rỗng.")

class QLearningAgent:
    def __init__(self, alpha=0.1, gamma=0.9, epsilon=0.2):
        self.Q = defaultdict(lambda: defaultdict(float))  # Q[state][action]
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

    def get_state_id(self, obs):
        # Chuyển trạng thái về dạng hashable (dùng tuple)
        board = tuple(obs['board'].flatten())
        turn = obs['turn']
        return (board, turn)

    def choose_action(self, obs, valid_actions):
        state = self.get_state_id(obs)
        if random.random() < self.epsilon:
            return random.choice(valid_actions)
        q_values = self.Q[state]
        return max(valid_actions, key=lambda a: q_values[a])

    def learn(self, obs, action, reward, next_obs, done, valid_next_actions):
        state = self.get_state_id(obs)
        next_state = self.get_state_id(next_obs)
        q_sa = self.Q[state][action]
        
        if done or not valid_next_actions:
            q_target = reward
        else:
            q_target = reward + self.gamma * max([self.Q[next_state][a] for a in valid_next_actions])

        self.Q[state][action] += self.alpha * (q_target - q_sa)


# Khởi tạo người chơi Minimax (chuyên gia) và Q-Learner (người học)
minimax_player = AIPlayer(player=1) # Minimax chơi cho người chơi 1
q_learning_player_obj = QLearningPlayer(player=1, q_table_path=None) # Tạo đối tượng nhưng sẽ dùng Q table ở trên
q_learning_player_obj.Q = Q # Gán Q table cho đối tượng Q-Learning


# Hàm lấy state_id (tương tự như trong QLearningPlayer)
def get_state_id(board, turn):
    # Đảm bảo state có thể làm key trong dictionary (immutable)
    return (tuple(np.array(board).flatten()), turn)

# Hàm xác định phần thưởng (ví dụ đơn giản: +1 thắng, -1 thua, 0 hòa, 0 cho các bước trung gian)
def get_reward(game, current_player_value):
    if game.is_game_over():
        score = game.get_scores()
        if score[current_player_value] > score[3 - current_player_value]: # Giả sử người chơi là 1 hoặc 2
            return 1
        elif score[current_player_value] < score[3 - current_player_value]:
            return -1
        else:
            return 0
    return 0 # Phần thưởng 0 cho các bước không kết thúc game

# Hàm chuyển đổi move (row, col) thành action index (0-63 cho bàn 8x8)
def move_to_action(move, board_size=8):
    return move[0] * board_size + move[1]

# Vòng lặp huấn luyện
for episode in range(num_episodes):
    game = Game() # Khởi tạo ván cờ mới
    current_player = 1 # Bắt đầu với người chơi 1 (người mà Q-Learning sẽ học)

    while not game.is_game_over():
        board_state = game.board_state
        state = get_state_id(board_state, current_player)
        valid_moves = game.get_valid_moves # Lấy nước đi hợp lệ cho người chơi hiện tại

        if not valid_moves:
            game.pass_turn() # Bắt buộc phải pass
            current_player = 3 - current_player # Đổi lượt chơi
            continue # Bỏ qua bước cập nhật Q nếu pass

        # Minimax chọn nước đi tốt nhất
        # Đảm bảo MinimaxPlayer chơi cho đúng người chơi hiện tại
        minimax_player.player = current_player
        best_move_by_minimax = minimax_player.findBestMove(board_state)

        if best_move_by_minimax is None: # Trường hợp Minimax không tìm được nước đi (có thể do không có nước đi hợp lệ)
             game.pass_turn()
             current_player = 3 - current_player
             continue

        action = move_to_action(best_move_by_minimax)

        # Lưu trạng thái và hành động trước khi thực hiện nước đi
        prev_state = state
        prev_action = action

        # Thực hiện nước đi
        game.make_move(best_move_by_minimax, current_player)
        next_board_state = game.board_state
        next_state = get_state_id(next_board_state, 3 - current_player) # Trạng thái tiếp theo là lượt của đối phương

        # Nhận phần thưởng (kiểm tra xem ván đấu đã kết thúc chưa)
        reward = get_reward(game, current_player)

        # Q-Learning Update
        # Q(s, a) = Q(s, a) + alpha * [r + gamma * max_a' Q(s', a') - Q(s, a)]

        # Tính max Q ở trạng thái tiếp theo (nếu game chưa kết thúc)
        max_next_q = 0.0
        if not game.is_game_over():
             next_valid_moves = game.get_valid_moves # Lấy nước đi hợp lệ ở trạng thái tiếp theo
             next_valid_actions = [move_to_action(m) for m in next_valid_moves]
             if next_valid_actions: # Đảm bảo có nước đi hợp lệ ở trạng thái tiếp theo
                  next_q_values = q_learning_player_obj.Q[next_state] # Lấy Q values cho next_state
                  max_next_q = max(next_q_values[next_action] for next_action in next_valid_actions)


        # Cập nhật Q value cho trạng thái và hành động hiện tại
        current_q = Q[prev_state][prev_action]
        new_q = current_q + alpha * (reward + gamma * max_next_q - current_q)
        Q[prev_state][prev_action] = new_q

        # Đổi lượt chơi
        current_player = 3 - current_player

    # Sau khi kết thúc mỗi ván (hoặc sau mỗi vài ván), lưu Q-table
    if episode % 100 == 0 or episode == num_episodes - 1: # Lưu định kỳ
        with open(q_table_save_path, 'wb') as f:
            pickle.dump(dict(Q), f) # Lưu lại dưới dạng dict thông thường
        print(f"Đã lưu Q-table sau episode {episode}")

print("Hoàn thành huấn luyện.")

if __name__ == "__main__":
    
    # Cố gắng tải Q-table nếu đã tồn tại
    try:
        with open(q_table_save_path, 'rb') as f:
            loaded_q = pickle.load(f)
            # Kết hợp hoặc thay thế Q hiện tại bằng Q đã tải
            for state, actions in loaded_q.items():
                for action, value in actions.items():
                    Q[state][action] = value
        print(f"Đã tải Q-table từ {q_table_save_path}")
    except FileNotFoundError:
        print("Không tìm thấy Q-table cũ, bắt đầu với Q-table rỗng.")
    except Exception as e:
        print(f"Lỗi khi tải Q-table: {e}, bắt đầu với Q-table rỗng.")
    
    env = OthelloEnv()

    

    # Vòng lặp huấn luyện
    for episode in range(num_episodes):
        game = Game() # Khởi tạo ván cờ mới
        current_player = 1 # Bắt đầu với người chơi 1 (người mà Q-Learning sẽ học)

        
        
        while not game.is_game_over():
            board_state = game.board_state
            state = get_state_id(board_state, current_player)
            valid_moves = game.get_valid_moves # Lấy nước đi hợp lệ cho người chơi hiện tại

            if not valid_moves:
                game.pass_turn() # Bắt buộc phải pass
                current_player = 3 - current_player # Đổi lượt chơi
                continue # Bỏ qua bước cập nhật Q nếu pass

            # Minimax chọn nước đi tốt nhất
            # Đảm bảo MinimaxPlayer chơi cho đúng người chơi hiện tại
            minimax_player.player = current_player
            best_move_by_minimax = minimax_player.findBestMove(board_state)

            if best_move_by_minimax is None: # Trường hợp Minimax không tìm được nước đi (có thể do không có nước đi hợp lệ)
                game.pass_turn()
                current_player = 3 - current_player
                continue

            action = move_to_action(best_move_by_minimax)

            # Lưu trạng thái và hành động trước khi thực hiện nước đi
            prev_state = state
            prev_action = action

            # Thực hiện nước đi
            game.make_move(best_move_by_minimax, current_player)
            next_board_state = game.board_state
            next_state = get_state_id(next_board_state, 3 - current_player) # Trạng thái tiếp theo là lượt của đối phương

            # Nhận phần thưởng (kiểm tra xem ván đấu đã kết thúc chưa)
            reward = get_reward(game, current_player)

            # Q-Learning Update
            # Q(s, a) = Q(s, a) + alpha * [r + gamma * max_a' Q(s', a') - Q(s, a)]

            # Tính max Q ở trạng thái tiếp theo (nếu game chưa kết thúc)
            max_next_q = 0.0
            if not game.is_game_over():
                next_valid_moves = game.get_valid_moves # Lấy nước đi hợp lệ ở trạng thái tiếp theo
                next_valid_actions = [move_to_action(m) for m in next_valid_moves]
                if next_valid_actions: # Đảm bảo có nước đi hợp lệ ở trạng thái tiếp theo
                    next_q_values = q_learning_player_obj.Q[next_state] # Lấy Q values cho next_state
                    max_next_q = max(next_q_values[next_action] for next_action in next_valid_actions)


            # Cập nhật Q value cho trạng thái và hành động hiện tại
            current_q = Q[prev_state][prev_action]
            new_q = current_q + alpha * (reward + gamma * max_next_q - current_q)
            Q[prev_state][prev_action] = new_q

            # Đổi lượt chơi
            current_player = 3 - current_player

        # Sau khi kết thúc mỗi ván (hoặc sau mỗi vài ván), lưu Q-table
        if episode % 100 == 0 or episode == num_episodes - 1: # Lưu định kỳ
            with open(q_table_save_path, 'wb') as f:
                pickle.dump(dict(Q), f) # Lưu lại dưới dạng dict thông thường
            print(f"Đã lưu Q-table sau episode {episode}")

    print("Hoàn thành huấn luyện.")

# Bây giờ, QLearningPlayer có thể sử dụng Q-table đã được huấn luyện bởi Minimax
# Bạn có thể tải Q-table này khi khởi tạo QLearningPlayer
# example_player = QLearningPlayer(player=1, q_table_path='q_table_othello.pkl')
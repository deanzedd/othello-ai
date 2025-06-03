from test_environment import OthelloEnv  # Môi trường Othello
from othello import BLACK, WHITE
from network.policy_network import PolicyAgent # Agent dùng mạng nơ-ron
from network.metrics import TrainingMetrics
from ai.ai_player import RandomPlayer, MinimaxPlayer  # Random player
import os
import torch
from tqdm import tqdm
BATCH_SIZE=32
MODEL_PATH = "models/MCTS_vs_Mini.pt"
PASS_ACTION = -9 # Giá trị này cần khớp với PASS_ACTION_VALUE trong policy_network.py nếu bạn muốn lọc nó


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available else 'cpu')
    print(f'Device: {device}')
    metrics = TrainingMetrics()
    total_black_win = 0
    total_white_win = 0
    total_draw = 0

    num_games = 200

    # Tải model đã huấn luyện
    agent_white = PolicyAgent()
    # Kiểm tra xem mô hình PolicyNetwork có thuộc tính 'model' không
    # Nếu PolicyAgent của bạn trực tiếp chứa PolicyNetwork, bạn sẽ truy cập nó qua policy_network
    if os.path.exists(MODEL_PATH):
        # Giả sử agent_white.policy_network là đối tượng mạng thực sự
        agent_white.policy_network.load_state_dict(torch.load(MODEL_PATH))
        agent_white.policy_network.eval()  # Đặt model về chế độ đánh giá
        print("✅ Model loaded from", MODEL_PATH)
    else:
        print("⚠️ No saved model found, training from scratch.")

    for game_index in tqdm(range(num_games)):
        env = OthelloEnv()
        random_agent_black = MinimaxPlayer(BLACK) # Sử dụng RandomPlayer từ ai_player.py

        observation, info = env.reset()
        done = False
        #current_player = BLACK

        total_reward = 0

        while not done:
            env.render()
            if env.game.turn == WHITE:
                
                # Lấy nước đi hợp lệ dưới dạng tuple (row, col)
                valid_moves_tuples = env.game.get_valid_moves
                
                if valid_moves_tuples:
                    # SỬA LỖI: Truyền đối tượng observation gốc vào choose_action
                    action = agent_white.choose_action(observation, valid_moves_tuples) 
                    # Chuyển đổi tuple (row, col) thành chỉ số 1D cho env.step nếu cần
                    action_for_env_step = action[0] * env.board_size + action[1]
                else:
                    action_for_env_step = PASS_ACTION  # Pass (nếu bạn định nghĩa pass là action -9)
                    action = PASS_ACTION # Lưu trữ PASS_ACTION vào replay buffer

                # Mã hóa trạng thái HIỆN TẠI để lưu vào replay buffer
                state_to_store = agent_white.encode_state(observation)
                
                next_observation, reward, terminated, truncated, info = env.step(action_for_env_step)
                done = terminated or truncated
                
                # Mã hóa trạng thái TIẾP THEO để lưu vào replay buffer
                next_state_to_store = agent_white.encode_state(next_observation)
                
                # Lưu trữ kinh nghiệm vào replay buffer
                agent_white.store_experience(
                    state=state_to_store,
                    action=action, # action ở đây có thể là tuple (row, col) hoặc PASS_ACTION
                    reward=reward,
                    next_state=next_state_to_store,
                    done=done
                )
                
                # Cập nhật chính sách sau mỗi lượt của WHITE nếu đủ kinh nghiệm
                if len(agent_white.memory) > BATCH_SIZE:
                    agent_white.update_policy(BATCH_SIZE)

            else: # Lượt của BLACK (RandomPlayer)
                move = random_agent_black.play(env.game) # play() của RandomPlayer trả về (row, col) hoặc None
                if move is not None:
                    row, col = move
                    action_for_env_step = row * env.board_size + col
                else:
                    action_for_env_step = PASS_ACTION  # Pass
                
                next_observation, reward, terminated, truncated, info = env.step(action_for_env_step)
                done = terminated or truncated

            observation = next_observation
            #current_player = 3 - current_player
            
        total_reward += reward
            #env.render()

        # Ghi nhận kết quả trận đấu
        winner = env.game.get_winner()
        
        if winner == WHITE: # WHITE là 2
            total_white_win += 1
        elif winner == BLACK: # BLACK là 1
            total_black_win += 1
        else: # Hòa
            total_draw += 1
    
        # Cập nhật metrics sau mỗi game
        win_rate = total_white_win / (game_index + 1)
        metrics.update(
            win_rate=win_rate,
            loss=agent_white.current_loss, # SỬA LỖI: Sử dụng current_loss thay vì losses[-1]
            epsilon=agent_white.epsilon,
            reward=total_reward
        )
        
        # Vẽ biểu đồ mỗi 5 game (hoặc số lượng bạn muốn)
        if game_index % 10 == 0 and game_index > 0:
            metrics.plot(save_path=f"training_metrics_{game_index}.png") # Lưu với tên khác nhau
            torch.save(agent_white.policy_network.state_dict(), MODEL_PATH)
            print(f"Đã lưu biểu đồ tại training_metrics_{game_index}.png")
            
    # Vẽ biểu đồ cuối cùng
    metrics.plot("final_training_metrics.png")

    # Lưu model của PolicyNetwork
    # Giả sử agent_white.policy_network là đối tượng mạng thực sự
    torch.save(agent_white.policy_network.state_dict(), MODEL_PATH)
    print(f"💾 Saved model to {MODEL_PATH}")
    
    # Tổng kết
    print("\n=== KẾT QUẢ SAU", num_games, "GAME ===")
    print(f"WHITE (PolicyAgent) thắng: {total_white_win}")
    print(f"BLACK (RandomPlayer) thắng: {total_black_win}")
    print(f"Hòa: {total_draw}")
'''
from test_environment import OthelloEnv  # Môi trường Othello
from othello import BLACK, WHITE
from network.policy_network import PolicyAgent # Agent dùng mạng nơ-ron
from network.metrics import TrainingMetrics
from ai.ai_player import RandomPlayer, MinimaxPlayer  # Random player
import os
import torch
from tqdm import tqdm
BATCH_SIZE=32
MODEL_PATH = "models/MCTS_vs_Mini.pt"
PASS_ACTION = -9 # Giá trị này cần khớp với PASS_ACTION_VALUE trong policy_network.py nếu bạn muốn lọc nó


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available else 'cpu')
    print(f'Device: {device}')
    metrics = TrainingMetrics()
    total_black_win = 0
    total_white_win = 0
    total_draw = 0

    num_games = 200

    # Tải model đã huấn luyện
    agent_white = PolicyAgent()
    # Kiểm tra xem mô hình PolicyNetwork có thuộc tính 'model' không
    # Nếu PolicyAgent của bạn trực tiếp chứa PolicyNetwork, bạn sẽ truy cập nó qua policy_network
    if os.path.exists(MODEL_PATH):
        # Giả sử agent_white.policy_network là đối tượng mạng thực sự
        agent_white.policy_network.load_state_dict(torch.load(MODEL_PATH))
        agent_white.policy_network.eval()  # Đặt model về chế độ đánh giá
        print("✅ Model loaded from", MODEL_PATH)
    else:
        print("⚠️ No saved model found, training from scratch.")

    for game_index in tqdm(range(num_games)):
        env = OthelloEnv()
        random_agent_black = MinimaxPlayer(BLACK) # Sử dụng RandomPlayer từ ai_player.py

        observation, info = env.reset()
        done = False
        current_player = BLACK

        total_reward = 0

        while not done:
            env.render()
            if current_player == WHITE:
                
                # Lấy nước đi hợp lệ dưới dạng tuple (row, col)
                valid_moves_tuples = env.game.get_valid_moves
                
                if valid_moves_tuples:
                    # SỬA LỖI: Truyền đối tượng observation gốc vào choose_action
                    action = agent_white.choose_action(observation, valid_moves_tuples) 
                    # Chuyển đổi tuple (row, col) thành chỉ số 1D cho env.step nếu cần
                    action_for_env_step = action[0] * env.board_size + action[1]
                else:
                    action_for_env_step = PASS_ACTION  # Pass (nếu bạn định nghĩa pass là action -9)
                    action = PASS_ACTION # Lưu trữ PASS_ACTION vào replay buffer

                # Mã hóa trạng thái HIỆN TẠI để lưu vào replay buffer
                state_to_store = agent_white.encode_state(observation)
                
                next_observation, reward, terminated, truncated, info = env.step(action_for_env_step)
                done = terminated or truncated
                
                # Mã hóa trạng thái TIẾP THEO để lưu vào replay buffer
                next_state_to_store = agent_white.encode_state(next_observation)
                
                # Lưu trữ kinh nghiệm vào replay buffer
                agent_white.store_experience(
                    state=state_to_store,
                    action=action, # action ở đây có thể là tuple (row, col) hoặc PASS_ACTION
                    reward=reward,
                    next_state=next_state_to_store,
                    done=done
                )
                
                # Cập nhật chính sách sau mỗi lượt của WHITE nếu đủ kinh nghiệm
                if len(agent_white.memory) > BATCH_SIZE:
                    agent_white.update_policy(BATCH_SIZE)

            else: # Lượt của BLACK (RandomPlayer)
                move = random_agent_black.play(env.game) # play() của RandomPlayer trả về (row, col) hoặc None
                if move is not None:
                    row, col = move
                    action_for_env_step = row * env.board_size + col
                else:
                    action_for_env_step = PASS_ACTION  # Pass
                
                next_observation, reward, terminated, truncated, info = env.step(action_for_env_step)
                done = terminated or truncated

            observation = next_observation
            current_player = 3 - current_player
            
        total_reward += reward
            #env.render()

        # Ghi nhận kết quả trận đấu
        winner = env.game.get_winner()
        
        if winner == WHITE: # WHITE là 2
            total_white_win += 1
        elif winner == BLACK: # BLACK là 1
            total_black_win += 1
        else: # Hòa
            total_draw += 1
    
        # Cập nhật metrics sau mỗi game
        win_rate = total_white_win / (game_index + 1)
        metrics.update(
            win_rate=win_rate,
            loss=agent_white.current_loss, # SỬA LỖI: Sử dụng current_loss thay vì losses[-1]
            epsilon=agent_white.epsilon,
            reward=total_reward
        )
        
        # Vẽ biểu đồ mỗi 5 game (hoặc số lượng bạn muốn)
        if game_index % 10 == 0 and game_index > 0:
            metrics.plot(save_path=f"training_metrics_{game_index}.png") # Lưu với tên khác nhau
            torch.save(agent_white.policy_network.state_dict(), MODEL_PATH)
            print(f"Đã lưu biểu đồ tại training_metrics_{game_index}.png")
            
    # Vẽ biểu đồ cuối cùng
    metrics.plot("final_training_metrics.png")

    # Lưu model của PolicyNetwork
    # Giả sử agent_white.policy_network là đối tượng mạng thực sự
    torch.save(agent_white.policy_network.state_dict(), MODEL_PATH)
    print(f"💾 Saved model to {MODEL_PATH}")
    
    # Tổng kết
    print("\n=== KẾT QUẢ SAU", num_games, "GAME ===")
    print(f"WHITE (PolicyAgent) thắng: {total_white_win}")
    print(f"BLACK (RandomPlayer) thắng: {total_black_win}")
    print(f"Hòa: {total_draw}")
'''

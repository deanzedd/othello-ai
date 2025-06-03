import os
import torch
import numpy as np
from test_environment import OthelloEnv
from ai.ai_player import RandomPlayer
from network.minimax_q_learning import MinimaxQAgent
from network.metrics import TrainingMetrics

PASS_ACTION = -9
WHITE = 2
BLACK = 1
MODEL_PATH = "models/minimax_q_network.pt"

if __name__ == "__main__":
    metrics = TrainingMetrics()
    total_black_win = 0
    total_white_win = 0
    total_draw = 0
    num_games = 1000

    agent_white = MinimaxQAgent()
    if os.path.exists(MODEL_PATH):
        agent_white.model.load_state_dict(torch.load(MODEL_PATH))
        agent_white.model.eval()  # Đặt model về chế độ đánh giá
        print("✅ Model loaded from", MODEL_PATH)
    else:
        print("⚠️ No saved model found, training from scratch.")

    for game_index in range(num_games):
        env = OthelloEnv()
        random_agent_black = RandomPlayer(env.game)

        observation, info = env.reset()
        done = False
        current_player = BLACK

        # Biến tạm lưu trạng thái và action của White
        white_state = None
        white_action = None
        total_reward = 0

        while not done:
            if current_player == WHITE:
                # === PHASE 1: WHITE'S TURN ===
                # Lưu trạng thái hiện tại và action của White
                white_state = agent_white.encode_state(observation)
                valid_moves = env.game.get_valid_moves

                if valid_moves:
                    action = agent_white.choose_action(observation, valid_moves)
                    row, col = action
                else:
                    row, col = -1, -1  # Pass

                # Thực hiện action và lưu tạm
                action_flat = row * env.board_size + col
                next_observation, _, _, _, _ = env.step(action_flat)
                white_action = (row, col)
                
                # Chuyển lượt nhưng CHƯA lưu vào buffer
                current_player = 3 - current_player
                observation = next_observation

            else:
                # === PHASE 2: BLACK'S TURN ===
                # BLACK đi
                move = random_agent_black.play(env.game)
                if move is not None:
                    row, col = move
                else:
                    row, col = -1, -1  # Pass

                # Thực hiện action
                action_flat = row * env.board_size + col
                next_observation, reward, terminated, truncated, info = env.step(action_flat)
                done = terminated or truncated

                # === LƯU VÀO BUFFER SAU KHI CẢ HAI ĐI ===
                if white_state is not None:
                    next_state = agent_white.encode_state(next_observation)
                    
                    # Tính reward tổng hợp cho cả 2 lượt
                    temp_reward = reward if current_player == WHITE else -reward
                    
                    # Lưu vào buffer
                    agent_white.replay_buffer_save(
                        state=white_state,
                        action=white_action,
                        opponent_action=(row, col),
                        reward=temp_reward,
                        next_state=next_state,
                        done=done
                    )
                    white_state = None  # Reset

                # Cập nhật trạng thái
                total_reward += reward
                current_player = 3 - current_player
                observation = next_observation
        # === KẾT THÚC MỘT GAME ===
        
        # Xử lý kết quả
        winner = env.game.get_winner()
        if winner == WHITE:
            total_white_win += 1
        elif winner == BLACK:
            total_black_win += 1
        else:
            total_draw += 1

        # Cập nhật epsilon và metrics

        metrics.update(
            win_rate=total_white_win/(game_index+1),
            loss=agent_white.current_loss,
            epsilon=agent_white.epsilon,
            reward=total_reward
        )

        # Train định kỳ
        if len(agent_white.memory) > 32:
            agent_white.replay(32)

        # Vẽ biểu đồ mỗi 50 game
        if game_index % 50 == 0 and game_index > 0:
            metrics.plot()
            print(f"Đã lưu biểu đồ tại training_metrics_{game_index}.png")

    # Lưu model và hiển thị kết quả
    torch.save(agent_white.model.state_dict(), MODEL_PATH)
    print("\n=== KẾT QUẢ SAU", num_games, "GAMES ===")
    print(f"WHITE thắng: {total_white_win}")
    print(f"BLACK thắng: {total_black_win}")
    print(f"Hòa: {total_draw}")
from test_enviroment import OthelloEnv  # Môi trường Othello
from othello import BLACK, WHITE
from ai.ai_player import QLearningPlayer, RandomPlayer, AIPlayer  # Player của bạn

if __name__ == "__main__":
    total_black_win = 0
    total_white_win = 0
    total_draw = 0

    num_games = 20

    for game_index in range(num_games):
        print(f"\n--- Game {game_index + 1} ---")

        env = OthelloEnv()
        ai_agent_white = QLearningPlayer(WHITE, q_table_path="q_table.pkl")
        random_agent_black = AIPlayer(BLACK)

        observation, info = env.reset()
        done = False
        current_player = BLACK

        while not done:
            if current_player == WHITE:
                action = ai_agent_white.play(env.game)
            else:
                action = random_agent_black.play(env.game)

            # Chuyển đổi action (row, col) thành action index
            if action is not None:
                row, col = action
                action_gym = row * env.board_size + col
            else:
                # Nếu không có nước đi hợp lệ, chọn đại action cuối cùng (xử lý pass)
                action_gym = env.action_space.n - 1

            next_observation, reward, terminated, truncated, info = env.step(action_gym)
            done = terminated or truncated
            observation = next_observation
            current_player = 3 - current_player  # Đổi lượt chơi
            env.render()



        # Ghi nhận kết quả
        print("Game Over!")
        if reward < 0:
            print("BLACK wins!")
            total_black_win += 1
        elif reward > 0:
            print("WHITE wins!")
            total_white_win += 1
        else:
            print("It's a draw!")
            total_draw += 1

        env.close()

    # Tổng kết sau 20 game
    print("\n=== KẾT QUẢ SAU 20 GAME ===")
    print(f"WHITE (AIPlayer) thắng: {total_white_win}")
    print(f"BLACK (MiniMax_Player) thắng: {total_black_win}")
    print(f"Hòa: {total_draw}")


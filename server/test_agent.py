from test_environment import OthelloEnv  # Môi trường Othello
from othello import BLACK, WHITE
from ai.ai_player import QLearningPlayer, RandomPlayer, MinimaxPlayer, MinimaxQLearningPlayer, Policy_Net_Player, MCTS_Player  # Player của bạn

def test_agent(num_games, agent1, agent2):
    """
    Runs a series of Othello games between two agents and returns the results.

    Args:
        num_games (int): The number of games to play.
        agent1 (AIPlayer): The first agent.
        agent2 (AIPlayer): The second agent.

    Returns:
        dict: A dictionary containing the total wins for each agent and the total draws.
              The keys will be the string representation of the agent objects (e.g., "QLearningPlayer(BLACK)")
              and "draws".
    """
    results = {
        str(agent1): 0,
        str(agent2): 0,
        "draws": 0
    }

    # Assign colors to agents for clarity in results
    agent1_color = None
    agent2_color = None

    for game_index in range(num_games):
        print(f"\n--- Game {game_index + 1} ---")

        env = OthelloEnv()

        # Randomly assign black and white for fairness, or based on agent properties if they have a preferred color
        # Here, we'll alternate who goes first or assign based on which agent is passed as argument order
        # For simplicity, let's say agent1 is BLACK and agent2 is WHITE in odd games, and vice-versa in even games.
        """"""
        if game_index % 1 == 0:
            black_player = agent1
            white_player = agent2
            agent1.player = BLACK
            agent2.player = WHITE
            agent1_color = BLACK
            agent2_color = WHITE
            print(f"{str(agent1)} is BLACK, {str(agent2)} is WHITE")
        else:
            black_player = agent2
            white_player = agent1
            agent1.player = WHITE
            agent2.player = BLACK
            agent1_color = WHITE
            agent2_color = BLACK
            print(f"{str(agent2)} is BLACK, {str(agent1)} is WHITE")


        observation, info = env.reset()
        done = False

        while not done:
            current_agent = black_player if env.game.turn == BLACK else white_player
            action = current_agent.play(env.game)

            # Convert action (row, col) to action index
            if action is not None:
                row, col = action
                action_gym = row * env.board_size + col
            else:
                # If no valid moves, choose the last action (handle pass)
                action_gym = env.action_space.n - 1
                print(f"Player {env.game.turn} passes.")

            next_observation, reward, terminated, truncated, info = env.step(action_gym)
            done = terminated or truncated
            observation = next_observation
            #env.render() # Uncomment to visualize each step

        print("Game Over!")
        winner = env.game.winner
        if winner == BLACK:
            print("BLACK wins!")
            env.render()
            if agent1_color == BLACK:
                results[str(agent1)] += 1
            else:
                results[str(agent2)] += 1
        elif winner == WHITE:
            print("WHITE wins!")
            env.render()
            if agent1_color == WHITE:
                results[str(agent1)] += 1
            else:
                results[str(agent2)] += 1
        else:
            print("It's a draw!")
            results["draws"] += 1

        env.close()

    return results


if __name__ == "__main__":
    # Example Usage:
    num_games_to_play = 50

    # Make sure to replace these with actual instances of your player classes
    # For demonstration, let's use RandomPlayer and another RandomPlayer
    # You would replace these with your trained QLearningPlayer, MiniMax_Player, etc.
    agent_a = QLearningPlayer(BLACK) # Example: Assuming AIPlayer can be initialized without a color initially
    agent_b = MCTS_Player(WHITE) # Example: Assuming AIPlayer can be initialized without a color initially

    print(f"\n=== Starting {num_games_to_play} games between {str(agent_a)} and {str(agent_b)} ===")
    game_results = test_agent(num_games_to_play, agent_a, agent_b)

    # Print final results
    print(f"\n=== KẾT QUẢ SAU {num_games_to_play} GAME ===")
    for agent_name, wins in game_results.items():
        if agent_name != "draws":
            print(f"{agent_name} thắng: {wins}")
    print(f"Hòa: {game_results['draws']}")
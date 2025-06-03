import random
import utils
from ai import evaluator
import numpy as np
import torch
import os
from network.q_learning import QNetwork, QNetworkAgent
from network.policy_network1 import PolicyNetwork, PolicyAgent
from network.MCTS1 import MCTSPolicyAgent
import time
from ai import minimax

class MinimaxPlayer:
    def __init__(self, player):
        self.player = player

    def play(self, game):
        """
        Play a move using the Minimax algorithm with alpha-beta pruning.
        Args:
            game: The current game state.
        Returns:
            bestMove: The best move for the player in the form (row, col).
        """
        return self.find_best_move(game.board_state)
    
    def find_best_move(self, board_state):
        # start = time.time()
        max_depth = 4
        time_limit = 7
        best_move = minimax.id_minimax(board_state, self.player, max_depth, time_limit)
        # end = time.time()
        # print(f"Iterative deepening evaluation time: {end - start:.4f} seconds")
        return best_move
    

class RandomPlayer():

    def __init__(self, game):
        self.game = game

    def get_random_move(self, game):
        """
        Get a random move for the AI player. For testing.

        """
        valid_moves = game.get_valid_moves

        #print(evaluator.evaluate(game.board_state))
        return random.choice(valid_moves) if valid_moves else None

    def play(self, game):
        """
        Args:
        board: np.ndarray
            Board of size n x n [6x6 in this case]

        Returns:
            a: int
            Randomly chosen move
        """

        return self.get_random_move(game)
    

q_model_path = "models/q_network_model_sau_hon.pt"
q_model_path1 = "models/q_network.pt"
q_model_path2="models/MCTS_vs_q.pt"
q_model_path3="models/MCTS_vs_mini.pt"

class QLearningPlayer:
    def __init__(self, player):
        self.player = player
        self.board_size = 8

        # Load Q-network
        self.model = QNetwork();
        self.model.load_state_dict(torch.load(q_model_path1))
        print("Model loaded from", q_model_path1)
        self.model.eval()  # Set model to evaluation mode

    def get_state_tensor(self, board, turn):
        if turn == 2:
            board = np.where(board == 1, -1, board)
            board = np.where(board == 2, 1, board)
        else:
            board = np.where(board == 1, 1, board)
            board = np.where(board == 2, -1, board)
        return torch.tensor(board, dtype=torch.float32).reshape(-1)

    def play(self, game):
        valid_moves = game.get_valid_moves
        if not valid_moves:
            return None

        board = np.array(game.board_state)

        state_tensor = self.get_state_tensor(board, self.player)

        with torch.no_grad():
            q_values = self.model(state_tensor).cpu().numpy().flatten()

        # Lọc Q-values theo hành động hợp lệ
        best_action = max(valid_moves, key=lambda a: q_values[a[0] * 8 + a[1]])
        return best_action

class MinimaxQLearningPlayer:
    def __init__(self, player):
        self.player = player
        self.board_size = 8
        from network.minimax_q_learning import MinimaxQAgent
        self.agent = MinimaxQAgent()
        # Load minimax Q-network nếu có
        model_path = "models/minimax_q_network.pt"
        if os.path.exists(model_path):
            self.agent.model.load_state_dict(torch.load(model_path))
            self.agent.model.eval()
            print("MinimaxQ model loaded from", model_path)
        else:
            print("No MinimaxQ model found, using random weights.")

    def get_state_tensor(self, board, turn):
        if turn == 2:
            board = np.where(board == 1, -1, board)
            board = np.where(board == 2, 1, board)
        else:
            board = np.where(board == 1, 1, board)
            board = np.where(board == 2, -1, board)
        return torch.tensor(board, dtype=torch.float32).reshape(-1)

    def play(self, game):
        valid_moves = game.get_valid_moves
        if not valid_moves:
            return None
        board = np.array(game.board_state)
        state_tensor = self.get_state_tensor(board, self.player)
        obs = {'board': board, 'turn': self.player}
        # Sử dụng agent để chọn action minimax Q
        action = self.agent.choose_action(obs, valid_moves)
        return action

class Policy_Net_Player:
    def __init__(self, player):
        self.player = player
        self.board_size = 8
        self.model = PolicyNetwork().to(device='cpu')
        self.model.load_state_dict(torch.load(q_model_path2))
        print("Model loaded from", q_model_path2)
        self.model.eval()  # Set model to evaluation mode
        
    def get_state_tensor(self, board, turn):
        if turn == 2:
            board = np.where(board == 1, -1, board)
            board = np.where(board == 2, 1, board)
        else:
            board = np.where(board == 1, 1, board)
            board = np.where(board == 2, -1, board)
        return torch.tensor(board, dtype=torch.float32).reshape(-1)

    def play(self, game):
        valid_moves = game.get_valid_moves
        if not valid_moves:
            return None

        board = np.array(game.board_state)

        state_tensor = self.get_state_tensor(board, self.player)

        with torch.no_grad():
            q_values = self.model(state_tensor).cpu().numpy().flatten()

        # Lọc Q-values theo hành động hợp lệ
        best_action = max(valid_moves, key=lambda a: q_values[a[0] * 8 + a[1]])
        return best_action
    
class MCTS_Player:
    def __init__(self, player):
        self.player = player
        self.board_size = 8
        self.nigga= PolicyAgent()
        self.model = PolicyNetwork().to(device='cpu')
        self.model.load_state_dict(torch.load(q_model_path3))
        print("Model loaded from", q_model_path3)
        self.model.eval()  # Set model to evaluation mode
        self.agent = MCTSPolicyAgent(policy_agent=self.nigga, n_simulations=1000)
        
    def get_state_tensor(self, board, turn):
        if turn == 2:
            board = np.where(board == 1, -1, board)
            board = np.where(board == 2, 1, board)
        else:
            board = np.where(board == 1, 1, board)
            board = np.where(board == 2, -1, board)
        return torch.tensor(board, dtype=torch.float32).reshape(-1)
    
    def play(self, game):
        valid_moves = game.get_valid_moves
        if not valid_moves:
            return None
        
        board = np.array(game.board_state)
        state_tensor = self.get_state_tensor(board, self.player)

        with torch.no_grad():
            q_values = self.model(state_tensor).cpu().numpy().flatten() 
        
        obs = {'board': board, 'turn': self.player}
        action = self.agent.choose_action(game)
        return action
        
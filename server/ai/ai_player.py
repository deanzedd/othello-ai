import random
import utils
import copy
from ai import evaluator
import numpy as np
from collections import defaultdict
import pickle

class AIPlayer:
    def __init__(self, player):
        self.player = player

    def play(self, game):
        """
        Get the AI bot's move.
        
        TODO: Implement AI algorithms (minimax search)

        """
        return self.findBestMove(game.board_state)

    
    def get_random_move(self, game):
        """
        Get a random move for the AI player. For testing.

        """
        valid_moves = game.get_valid_moves
        print(evaluator.evaluate(game.board_state))
        return random.choice(valid_moves) if valid_moves else None

    
    def findBestMove(self, board_state):
        bestVal = -1001
        bestMove = None
        
        for move in utils.get_valid_moves(board_state, 2):
            # Sao chép board
            temp_board = copy.deepcopy(board_state)
            
            # Áp dụng nước đi lên temp_board
            row, col = move
            utils.make_move(temp_board, row, col, self.player)

            # Đánh giá trạng thái sau khi đi
            value = evaluator.minimax(temp_board, 0, True, -1000, 1000)
            
            if value > bestVal:
                bestVal = value
                bestMove = move

        return bestMove

    
class RandomPlayer():

    def __init__(self, game):
        self.game = game

    def get_random_move(self, game):
        """
        Get a random move for the AI player. For testing.

        """
        valid_moves = game.get_valid_moves

        print(evaluator.evaluate(game.board_state))
        return random.choice(valid_moves) if valid_moves else None
    
    def findBestMove(self, board_state):
        bestVal = -1001
        bestMove = None
        
        for move in utils.get_valid_moves(board_state, 2):
            # Sao chép board
            temp_board = copy.deepcopy(board_state)
            
            # Áp dụng nước đi lên temp_board
            row, col = move
            utils.make_move(temp_board, row, col, self.player)

            # Đánh giá trạng thái sau khi đi
            value = evaluator.minimax(temp_board, 0, True, -1000, 1000)
            
            if value > bestVal:
                bestVal = value
                bestMove = move

        return bestMove
    
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
    
class QLearningPlayer:
    def __init__(self, player, q_table_path, board_size=8):
        self.player = player
        self.board_size = board_size

        # Load Q-table từ file
        with open(q_table_path, 'rb') as f:
            q_dict = pickle.load(f)
            self.Q = defaultdict(lambda: defaultdict(float), q_dict)

    def get_state_id(self, board, turn):
        board = tuple(np.array(board).flatten())
        return (board, turn)

    def play(self, game):
        valid_moves = game.get_valid_moves
        if not valid_moves:
            return None  # Bắt buộc phải pass

        obs = {'board': game.board_state, 'turn': self.player}
        state = self.get_state_id(obs['board'], obs['turn'])
        q_values = self.Q[state]

        # Chuyển valid_moves -> valid_actions
        valid_actions = [r * self.board_size + c for r, c in valid_moves]

        # Chọn action có Q-value cao nhất
        best_action = max(valid_actions, key=lambda a: q_values[a])

        # Chuyển lại thành (row, col)
        return best_action // self.board_size, best_action % self.board_size



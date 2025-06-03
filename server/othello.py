import utils
import copy
EMPTY = 0
BLACK = 1
WHITE = 2


class Game:
    """
    An Othello game. Handles the game state and rules.

    Play moves with :meth:`play`.

    Get the current state of the game with :attr:`board_state`.

    Get the current turn with :attr:`turn`.

    Check for a victory with :attr:`winner`.

    """

    def __init__(self):
        self.board_state = [
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 2, 1, 0, 0, 0],
            [0, 0, 0, 1, 2, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0]
        ]
        self.turn = BLACK
        self.winner = None
        self.is_game_over = False
        self.board_x = 8
        self.board_y = 8
        #board_x = board_y = n = 8

    def copy(self):
        """Tạo bản sao sâu của game"""
        new_game = Game()
        new_game.board_state = copy.deepcopy(self.board_state)
        new_game.turn = self.turn
        new_game.winner = self.winner
        new_game.is_game_over = self.is_game_over
        return new_game
    
    @property
    def get_valid_moves(self):
        return utils.get_valid_moves(self.board_state, self.turn)

    def play(self, player, row, col):
        """
        Play a move in a column.

        Returns the row where the checker lands.

        Raises :exc:`ValueError` if the move is illegal.

        """
        if player != self.turn:
            raise ValueError("It isn't your turn." + str(player) + " Turn:" + str(self.turn) )
        if not utils.is_valid_move(self.board_state, player, row, col):
            raise ValueError("Invalid move.")
        # Game logic here
        self.board_state = utils.make_move(self.board_state, row, col, player)
        opponent = utils.get_opponent(player)
        if utils.get_valid_moves(self.board_state, opponent):
            self.turn = opponent
        else: 
            if not utils.get_valid_moves(self.board_state, player):
                #handle game over
                self.is_game_over = True
                self.winner = utils.get_winner(self.board_state)
                
                
    def getBoardSize(self):
        return (self.board_x, self.board_y)
    
    def getActionSize(self):
        # Return number of actions, n is the board size and +1 is for no-op action
        return self.board_x * self.board_y + 1
    
    def get_winner(self):
        return utils.get_winner(self.board_state)
import time
import utils
from ai import evaluator

INFINITY = 100_000
transposition_table = {}


def id_minimax(board_state, player, max_depth, time_limit=None):
    """
    Iterative deepening wrapper for minimax.
    Args:
        board_state: Current board state.
        player: The player to move.
        max_depth: Maximum search depth.
        time_limit: Optional time limit in seconds.
    Returns:
        best_move: The best move found.
    """
    best_move = None
    start_time = time.time()
    valid_moves = utils.get_valid_moves(board_state, player)
    if not valid_moves:
        return None

    for depth in range(1, max_depth + 1):
        current_best_val = -INFINITY
        current_best_move = None
        for move in valid_moves:
            row, col = move
            next_board = utils.make_move(board_state, row, col, player)
            val = minimax(next_board, depth, player, False, -INFINITY, INFINITY)
            if val > current_best_val:
                current_best_val = val
                current_best_move = move
            if time_limit and (time.time() - start_time) > time_limit:
                return best_move if best_move is not None else current_best_move
        best_move = current_best_move
        if time_limit and (time.time() - start_time) > time_limit:
            break
    return best_move


def minimax(board_state, depth, player, isMax, alpha, beta):
    """
    Minimax algorithm with alpha-beta pruning to evaluate the best move.
    Args:
        board_state: Current state of the board.
        depth: Depth of the search tree.
        isMax: True if it's the maximizing player's turn, False otherwise.
        alpha: Alpha value for pruning.
        beta: Beta value for pruning.
    Returns:
        score: The evaluation score for the current board state.
    """
    # Use transposition table to avoid re-evaluating the same board state
    key = (board_hash(board_state), depth, player, isMax)
    if key in transposition_table:
        return transposition_table[key]
    
    # If the game is over, return the evaluation score based on the winner
    if utils.is_game_over(board_state):
        winner = utils.get_winner(board_state)
        if winner == player:
            return INFINITY
        if winner == utils.get_opponent(player):
            return -INFINITY
        return 0
    
    # When reached the maximum depth, return the evaluation score
    if depth == 0:
        return evaluator.evaluate(board_state, player)
    
    # Xác định player hiện tại
    current_player = player if isMax else utils.get_opponent(player)
    
    # Get valid moves for the current player
    valid_moves = utils.get_valid_moves(board_state, current_player)
    # Sort valid moves by their evaluation score
    valid_moves.sort(key=lambda move: evaluator.evaluate(utils.make_move(board_state, move[0], move[1], current_player), current_player), reverse=isMax)

    if not valid_moves:
        # Mất lượt -> chuyển cho đối thủ nhưng giữ nguyên board
        return minimax(board_state, depth - 1, player, not isMax, alpha, beta)

    best = 0
    if isMax:
        best = -INFINITY
        for move in valid_moves:
            row, col = move
            next_board = utils.make_move(board_state, row, col, current_player)
            val = minimax(next_board, depth - 1, player, not isMax, alpha, beta)
            best = max(best, val)
            alpha = max(alpha, best)
            if beta <= alpha:
                break
    else:
        best = INFINITY
        for move in valid_moves:
            row, col = move
            next_board = utils.make_move(board_state, row, col, current_player)
            val = minimax(next_board, depth - 1, player, not isMax, alpha, beta)
            best = min(best, val)
            beta = min(beta, best)
            if beta <= alpha:
                break
    
    # Save score in the transposition table
    transposition_table[key] = best
    return best


def board_hash(board_state):
    """
    Generate a unique hash for the board state.
    This is used for transposition table to avoid re-evaluating the same board state.
    """
    flat = tuple(tuple(row) for row in board_state)
    return hash(flat)
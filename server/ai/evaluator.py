import utils
import cupy as np
from othello import BLACK, WHITE

INFINITY = 100_000
CORNERS = [(0,0), (0,7), (7,0), (7,7)]
DYNAMIC_WEIGHTS = {
    'early': { 'disc_diff': 0.5, 'corner_diff': 10, 'mobility': 2.0, 'stability': 5.0, 'positional_score': 2.0, 'frontier_discs': -1.5 },
    'mid':   { 'disc_diff': 1.0, 'corner_diff': 10, 'mobility': 1.5, 'stability': 7.0, 'positional_score': 1.0, 'frontier_discs': -1.0 },
    'late':  { 'disc_diff': 5.0, 'corner_diff': 10, 'mobility': 1.0, 'stability': 10.0, 'positional_score': 0.5, 'frontier_discs': -0.5 }
}

transposition_table = {}

def evaluate(board_state, player):
    """
    Evaluate the board state for the given player.
    The evaluation is based on several heuristics:
    - Difference in number of discs
    - Corner ownership
    - Mobility
    - Stability
    - Positional score
    - Frontier discs
    Returns:
        total_score: A float representing the evaluation score for the player.
    """
    # Determine the game phase and set weights accordingly
    weights = DYNAMIC_WEIGHTS[get_game_phase(board_state)]
    # Calculate scores for each heuristic
    disc_diff_score = get_disc_diff(board_state, player)
    corner_diff_score = get_corner_diff(board_state, player)
    mobility_score = get_mobility(board_state, player)
    stability_score = get_stability(board_state, player)
    positional_score = get_positional_score(board_state, player)
    frontier_score = get_frontier_discs(board_state, player)

    # Combine scores with weights (adjust on testing)
    total_score = (
        weights['disc_diff'] * disc_diff_score +
        weights['corner_diff'] * corner_diff_score +
        weights['mobility'] * mobility_score +
        weights['stability'] * stability_score +
        weights['positional_score'] * positional_score +
        weights['frontier_discs'] * frontier_score
    )

    return total_score

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
        return evaluate(board_state, player)
    
    # Xác định player hiện tại
    current_player = player if isMax else utils.get_opponent(player)
    
    # Get valid moves for the current player
    valid_moves = utils.get_valid_moves(board_state, current_player)
    # Sort valid moves by their evaluation score
    valid_moves.sort(key=lambda move: evaluate(utils.make_move(board_state, move[0], move[1], current_player), current_player), reverse=isMax)

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


def get_disc_diff(board_state, player):
    """
    Calculate the difference in the number of discs between the player and the opponent.
    """
    opponent = utils.get_opponent(player)
    player_count = sum(row.count(player) for row in board_state)
    opponent_count = sum(row.count(opponent) for row in board_state)
    if (player_count + opponent_count) == 0:
        return 0
    return (player_count - opponent_count) / (player_count + opponent_count)


def get_corner_diff(board_state, player):
    """
    Calculate the difference in corner ownership between the player and the opponent.
    Corners are considered valuable positions in Othello.
    """
    opponent = utils.get_opponent(player)
    player_corners = sum(1 for corner in CORNERS if board_state[corner[0]][corner[1]] == player)
    opponent_corners = sum(1 for corner in CORNERS if board_state[corner[0]][corner[1]] == opponent)
    if player_corners + opponent_corners == 0:
        return 0
    return (player_corners - opponent_corners) / (player_corners + opponent_corners)


def get_mobility(board_state, player):
    """
    Calculate the mobility of the player.
    Mobility is the number of valid moves available to the player.
    """
    opponent = utils.get_opponent(player)
    player_moves = utils.count_valid_moves(board_state, player)
    opponent_moves = utils.count_valid_moves(board_state, opponent)
    if player_moves + opponent_moves == 0:
        return 0
    return (player_moves - opponent_moves) / (player_moves + opponent_moves)


def get_stability(board, player):
    n = 8
    stable = [[False for _ in range(n)] for _ in range(n)]

    # Mark corners and propagate along edges
    for r, c in CORNERS:
        if board[r][c] != 0:
            stable[r][c] = True
            # Vertical propagation
            if r == 7:
                rr = r
                while rr > 0:
                    rr -= 1
                    if board[rr][c] == board[rr+1][c]:
                        stable[rr][c] = True
                    else:
                        break
            else:
                rr = r
                while rr < 7:
                    rr += 1
                    if board[rr][c] == board[rr-1][c]:
                        stable[rr][c] = True
                    else:
                        break
            # Horizontal propagation
            if c == 7:
                cc = c
                while cc > 0:
                    cc -= 1
                    if board[r][cc] == board[r][cc+1]:
                        stable[r][cc] = True
                    else:
                        break
            else:
                cc = c
                while cc < 7:
                    cc += 1
                    if board[r][cc] == board[r][cc-1]:
                        stable[r][cc] = True
                    else:
                        break

    # Propagate stability inward
    changed = True
    while changed:
        changed = False
        for i in range(1, n-1):
            for j in range(1, n-1):
                if stable[i][j]:
                    continue
                if board[i][j] == 0:
                    continue
                # Check all 4 orthogonal and 4 diagonal directions
                if (
                    (board[i][j] == board[i+1][j] and stable[i+1][j]) and
                    (board[i][j] == board[i-1][j] and stable[i-1][j]) and
                    (board[i][j] == board[i][j+1] and stable[i][j+1]) and
                    (board[i][j] == board[i][j-1] and stable[i][j-1]) and
                    (board[i][j] == board[i+1][j+1] and stable[i+1][j+1]) and
                    (board[i][j] == board[i-1][j-1] and stable[i-1][j-1]) and
                    (board[i][j] == board[i+1][j-1] and stable[i+1][j-1]) and
                    (board[i][j] == board[i-1][j+1] and stable[i-1][j+1])
                ):
                    stable[i][j] = True
                    changed = True

    player_stable = sum(1 for i in range(n) for j in range(n) if board[i][j] == player and stable[i][j])
    opponent_stable = sum(1 for i in range(n) for j in range(n) if board[i][j] == utils.get_opponent(player) and stable[i][j])

    if player_stable + opponent_stable == 0:
        return 0
    return (player_stable - opponent_stable) / (player_stable + opponent_stable)


def get_positional_score(board_state, player):
    """
    Calculate the positional score based on the player's discs.
    Positional score is a heuristic that considers the positions of the discs on the board.
    """
    positional_weights = np.array([
        [100, -20, 10,  5,  5, 10, -20, 100],
        [-20, -50, -2, -2, -2, -2, -50, -20],
        [ 10,  -2,  3,  2,  2,  3,  -2,  10],
        [  5,  -2,  2,  1,  1,  2,  -2,   5],
        [  5,  -2,  2,  1,  1,  2,  -2,   5],
        [ 10,  -2,  3,  2,  2,  3,  -2,  10],
        [-20, -50, -2, -2, -2, -2, -50, -20],
        [100, -20, 10,  5,  5, 10, -20, 100]
    ])
    # If a corner is occupied, the cells adjacent to it are more valuable
    for corner in CORNERS:
        r, c = corner
        if board_state[r][c] != 0:
            # Increase the value of adjacent cells
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    if (dr != 0 or dc != 0) and utils.is_on_board(r + dr, c + dc):
                        positional_weights[r + dr][c + dc] += 50
    # Calculate the positional score
    score = 0
    for r in range(8):
        for c in range(8):
            if board_state[r][c] == player:
                score += positional_weights[r][c]
            elif board_state[r][c] == utils.get_opponent(player):
                score -= positional_weights[r][c]
    return score / 100


def get_frontier_discs(board_state, player):
    """
    Get the number of frontier discs for the player.
    Frontier discs are those that are adjacent to empty squares.
    """
    player_frontier = 0
    opponent_frontier = 0

    for r in range(8):
        for c in range(8):
            if board_state[r][c] == 0:
                continue
            is_frontier = False
            for dr, dc in utils.DIRECTIONS:
                nr, nc = r + dr, c + dc
                if utils.is_on_board(nr, nc) and board_state[nr][nc] == 0:
                    is_frontier = True
                    break
            if is_frontier:
                if board_state[r][c] == player:
                    player_frontier += 1
                else:
                    opponent_frontier += 1

    if player_frontier + opponent_frontier == 0:
        return 0
    return (player_frontier - opponent_frontier) / (player_frontier + opponent_frontier)


def get_game_phase(board_state):
    """
    Determine the game phase based on the number of discs on the board.
    Returns:
        'early', 'mid', or 'late' phase.
    """
    total_discs = utils.get_score(board_state, BLACK) + utils.get_score(board_state, WHITE)
    
    if total_discs < 20:
        return 'early'
    elif total_discs < 50:
        return 'mid'
    else:
        return 'late'
import gymnasium as gym
from gymnasium import spaces
import cupy as np
from othello import Game, BLACK, WHITE  # Import lớp Game của bạn


"""
reset: bắt đầu 1 trò chơi mới
_get_obs: trả về trạng thái
get_reward: phần thưởng
render: hiển thị bàn cờ
"""

class OthelloEnv(gym.Env):
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 4}

    def __init__(self, board_size=8):
        super().__init__()
        self.game = Game()  # Sử dụng lớp Game của bạn
        self.board_size = board_size

        # Không gian hành động: một vị trí trên bàn cờ (hàng * kích thước + cột)
        self.action_space = spaces.Discrete(board_size * board_size)

        # Không gian quan sát: trạng thái bàn cờ và lượt chơi hiện tại
        self.observation_space = spaces.Dict({
            'board': spaces.Box(low=0, high=2, shape=(board_size, board_size), dtype=int),
            'turn': spaces.Discrete(3)  # 1 cho BLACK, 2 cho WHITE
        })

    # bắt đầu 1 trò chơi mới
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.game = Game()
        observation = self._get_obs()
        info = {}
        return observation, info

    #trả về trạng thái trò chơi theo dạng dict [board_state, turn(lượt của ai?)]
    def _get_obs(self):
        return {'board': np.array(self.game.board_state), 'turn': self.game.turn}

    # thực hiện hành động, cập nhật trạng thái trò chơi, trả về thông tin phần thưởng
    def step(self, action):
        # Nếu action là -9: pass turn (bị mất lượt)
        if action == -9:
            self.game.turn = 3 - self.game.turn  # Đổi lượt cho người chơi tiếp theo
            terminated = self.game.is_game_over
            reward = self._get_reward()
            truncated = False
            observation = self._get_obs()
            info = {}
            return observation, reward, terminated, truncated, info

        #action = 8*row+col
        row = action // self.board_size
        col = action % self.board_size
        player = self.game.turn

        try:
            self.game.play(player, row, col)
        except ValueError:
            # Hành động không hợp lệ
            reward = -10  # Phạt nặng hành động không hợp lệ
            terminated = self.game.is_game_over
            truncated = False
            observation = self._get_obs()
            info = {}
            return observation, reward, terminated, truncated, info

        terminated = self.game.is_game_over
        reward = self._get_reward()
        truncated = False
        observation = self._get_obs()
        info = {}

        return observation, reward, terminated, truncated, info

    # tính toán phần thưởng khi trò chơi kết thúc
    def _get_reward(self):
        if self.game.is_game_over:
            if self.game.winner == BLACK:
                return -1
            elif self.game.winner == WHITE:
                return 1
            else:
                return 0  # Hòa
        return 0

    #Hiển thị trạng thái trò chơi (ở đây là in ra bàn cờ).
    def render(self, mode='human'):
        if mode == 'human':
            for row in self.game.board_state:
                print(row)
            
            print(f"Turn: {'BLACK' if self.game.turn == BLACK else 'WHITE'}")
            if self.game.is_game_over:
                print(f"Winner: {self.game.winner}")

    def close(self):
        pass
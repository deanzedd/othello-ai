import torch
import torch.nn as nn
import torch.optim as optim
import cupy as np
import random
import math
import copy
from collections import deque
from othello import Game # Giả sử bạn có file othello.py và utils.py
from network.policy_network import PolicyNetwork, PolicyAgent # Import từ file policy_learning.py đã tạo trước đó

# Định nghĩa các hằng số màu quân cờ
BLACK = 1
WHITE = 2
EMPTY = 0

class MCTSNode:
    """
    Đại diện cho một nút trong cây MCTS.
    Mỗi nút tương ứng với một trạng thái game.
    """
    def __init__(self, game_state: Game, parent=None, move=None, policy_prior=None):
        self.game_state = game_state  # Trạng thái game Othello tại nút này
        self.parent = parent          # Nút cha
        self.move = move              # Nước đi dẫn đến nút này từ nút cha
        self.children = {}            # Dictionary: move -> MCTSNode (nút con)
        self.n_visits = 0             # Số lần nút này được ghé thăm
        self.q_value = 0.0            # Tổng phần thưởng tích lũy từ nút này (tổng các kết quả)
        self.untried_moves = None     # Các nước đi chưa được thử từ trạng thái này
        self.policy_prior = policy_prior # Phân phối xác suất tiền định từ mạng chính sách cho trạng thái của nút này
        
        # Kiểm tra xem trạng thái game có phải là trạng thái kết thúc không
        self.is_terminal = game_state.is_game_over
        self.winner = game_state.winner if self.is_terminal else None

        # Khởi tạo untried_moves khi nút được tạo
        if not self.is_terminal:
            self.untried_moves = list(self.game_state.get_valid_moves)
            # Nếu không có nước đi hợp lệ cho người chơi hiện tại, kiểm tra đối thủ
            if not self.untried_moves:
                temp_game = self.game_state.copy()
                temp_game.turn = 3 - temp_game.turn # Chuyển lượt sang đối thủ
                if not temp_game.get_valid_moves:
                    # Nếu cả hai bên đều không có nước đi, game kết thúc
                    self.is_terminal = True
                    self.winner = self.game_state.get_winner()


class MCTSPolicyAgent:
    """
    Agent sử dụng thuật toán Monte Carlo Tree Search (MCTS) được hướng dẫn bởi một Mạng Chính Sách.
    """
    def __init__(self, policy_agent: PolicyAgent, n_simulations=1000, c_param=2.0):
        """
        Khởi tạo MCTS Policy Agent.
        :param policy_agent: Một instance của PolicyAgent (chứa PolicyNetwork đã huấn luyện).
        :param n_simulations: Số lượng mô phỏng (rollouts) cho mỗi lần tìm kiếm.
        :param c_param: Hằng số khám phá trong công thức UCB1/PUCT.
        """
        self.policy_agent = policy_agent
        self.policy_network = policy_agent.policy_network # Truy cập mạng nơ-ron cơ bản
        self.n_simulations = n_simulations
        self.c_param = c_param
        self.root = None

    def choose_action(self, current_game_state: Game):
        """
        Chạy MCTS để tìm nước đi tốt nhất từ trạng thái game hiện tại,
        được hướng dẫn bởi Mạng Chính Sách.
        :param current_game_state: Đối tượng Game hiện tại.
        :return: Nước đi tốt nhất (row, col).
        """
        # Đảm bảo mạng chính sách ở chế độ đánh giá
        self.policy_network.eval()

        # Lấy xác suất tiền định từ mạng chính sách cho trạng thái gốc
        root_state_obs = {'board': current_game_state.board_state, 'turn': current_game_state.turn}
        root_state_tensor = self.policy_agent.encode_state(root_state_obs)
        with torch.no_grad():
            action_logits = self.policy_network(root_state_tensor.unsqueeze(0)).squeeze(0)
            valid_moves_1d_indices = [r * 8 + c for r, c in current_game_state.get_valid_moves]
            
            # Mask các nước đi không hợp lệ để chỉ lấy xác suất cho các nước đi hợp lệ
            mask = torch.full_like(action_logits, float('-inf'))
            for idx in valid_moves_1d_indices:
                mask[idx] = 0
            
            masked_action_logits = action_logits + mask
            root_policy_prior_probs = torch.softmax(masked_action_logits, dim=-1).cpu().numpy()

        self.root = MCTSNode(current_game_state.copy(), policy_prior=root_policy_prior_probs)

        for _ in range(self.n_simulations):
            leaf_node = self._select(self.root)

            # Nếu nút lá là trạng thái kết thúc, không cần mở rộng/mô phỏng, chỉ lan truyền ngược kết quả
            if leaf_node.is_terminal:
                winner = leaf_node.winner
            else:
                # Mở rộng: Lấy xác suất tiền định từ mạng chính sách cho nút được mở rộng
                expanded_node = self._expand(leaf_node)
                # Mô phỏng: Sử dụng mạng chính sách để thực hiện rollout
                winner = self._simulate(expanded_node.game_state.copy())
                leaf_node = expanded_node # Lan truyền ngược từ nút được mở rộng

            self._backpropagate(leaf_node, winner, current_game_state.turn)

        # Sau tất cả các mô phỏng, chọn nước đi tốt nhất dựa trên số lần ghé thăm
        best_move = None
        max_visits = -1

        # In thông tin các nút con từ gốc để debug/kiểm tra
        # print("\nThông tin các nút con từ gốc sau MCTS:")
        # for move, child_node in self.root.children.items():
        #    print(f"Move: {move}, Visits: {child_node.n_visits}, Q-Value: {child_node.q_value:.2f}")

        for move, child_node in self.root.children.items():
            if child_node.n_visits > max_visits:
                max_visits = child_node.n_visits
                best_move = move
        
        self.policy_network.train() # Đặt mạng về chế độ huấn luyện
        
        if best_move is None:
            # Fallback: nếu MCTS không tìm thấy nước đi, chọn ngẫu nhiên nước đi hợp lệ
            print("MCTS không tìm thấy nước đi tốt nhất. Chọn một nước đi hợp lệ ngẫu nhiên.")
            valid_moves = current_game_state.get_valid_moves
            if valid_moves:
                return random.choice(valid_moves)
            else:
                return None # Không có nước đi hợp lệ nào

        return best_move

    def _select(self, node: MCTSNode):
        """
        Chọn nút con có điểm UCB1/PUCT cao nhất.
        Phiên bản này kết hợp xác suất tiền định của mạng chính sách.
        """
        while not node.is_terminal and not node.untried_moves:
            best_child = None
            best_score = -float('inf')

            for move, child in node.children.items():
                if child.n_visits == 0:
                    score = float('inf') # Ưu tiên khám phá các nút chưa được ghé thăm
                else:
                    # Công thức PUCT (Polynomial Upper Confidence Trees) tương tự AlphaGo Zero
                    # UCB = Q(s,a) + C * P(a|s) * sqrt(N(s)) / (1 + N(s,a))
                    
                    move_idx = move[0] * 8 + move[1]
                    # Lấy xác suất tiền định từ policy_prior của nút cha
                    prior_prob = node.policy_prior[move_idx] if node.policy_prior is not None else 0.01

                    avg_q = child.q_value / child.n_visits
                    
                    score = avg_q + self.c_param * prior_prob * (math.sqrt(node.n_visits) / (1 + child.n_visits))
                
                if score > best_score:
                    best_score = score
                    best_child = child
            
            if best_child is None: # Không nên xảy ra nếu logic game đúng
                break
            node = best_child
        return node

    def _expand(self, node: MCTSNode):
        """
        Mở rộng nút lá bằng cách tạo một nút con mới cho một nước đi chưa thử.
        Xác suất tiền định từ mạng chính sách cho trạng thái của nút con mới cũng được tính toán.
        """
        move = random.choice(node.untried_moves)
        node.untried_moves.remove(move)

        new_game_state = node.game_state.copy()
        try:
            new_game_state.play(new_game_state.turn, move[0], move[1])
        except ValueError:
            print(f"Lỗi: Nước đi {move} không hợp lệ trong trạng thái {node.game_state.board_state}. Điều này không nên xảy ra nếu get_valid_moves đúng.")
            return node # Trả về nút hiện tại nếu có lỗi

        # Lấy xác suất tiền định từ mạng chính sách cho trạng thái của nút con mới
        child_state_obs = {'board': new_game_state.board_state, 'turn': new_game_state.turn}
        child_state_tensor = self.policy_agent.encode_state(child_state_obs)
        
        with torch.no_grad():
            action_logits = self.policy_network(child_state_tensor.unsqueeze(0)).squeeze(0)
            valid_moves_1d_indices = [r * 8 + c for r, c in new_game_state.get_valid_moves]
            
            mask = torch.full_like(action_logits, float('-inf'))
            for idx in valid_moves_1d_indices:
                mask[idx] = 0
            
            masked_action_logits = action_logits + mask
            child_policy_prior_probs = torch.softmax(masked_action_logits, dim=-1).cpu().numpy()

        child_node = MCTSNode(new_game_state, parent=node, move=move, policy_prior=child_policy_prior_probs)
        node.children[move] = child_node
        return child_node

    def _simulate(self, game_state: Game):
        """
        Thực hiện một mô phỏng (rollout) được hướng dẫn bởi chính sách từ trạng thái game hiện tại
        cho đến khi game kết thúc.
        Các nước đi được chọn dựa trên xác suất của Mạng Chính Sách.
        """
        '''
        có thể cái tiến việc chơi mô phỏng này minimax 
        nhưng có thể dẫn đến chơi lâu hơn rất nhiều 
        
        '''
        current_rollout_game = game_state.copy()
        
        # Đảm bảo mạng ở chế độ đánh giá cho mô phỏng
        self.policy_network.eval() 

        while not current_rollout_game.is_game_over:
            valid_moves = current_rollout_game.get_valid_moves
            
            if not valid_moves:
                # Kiểm tra kịch bản cả hai bên đều không có nước đi
                temp_game_check = current_rollout_game.copy()
                temp_game_check.turn = 3 - temp_game_check.turn
                if not temp_game_check.get_valid_moves:
                    current_rollout_game.is_game_over = True
                    current_rollout_game.winner = current_rollout_game.get_winner()
                    break
                else:
                    current_rollout_game.turn = 3 - current_rollout_game.turn
                    continue

            # Sử dụng Mạng Chính Sách để chọn nước đi
            state_obs = {'board': current_rollout_game.board_state, 'turn': current_rollout_game.turn}
            state_tensor = self.policy_agent.encode_state(state_obs)

            with torch.no_grad():
                action_logits = self.policy_network(state_tensor.unsqueeze(0)).squeeze(0)
                
                valid_moves_1d_indices = [r * 8 + c for r, c in valid_moves]
                mask = torch.full_like(action_logits, float('-inf'))
                for idx in valid_moves_1d_indices:
                    mask[idx] = 0
                
                masked_action_logits = action_logits + mask
                action_probs = torch.softmax(masked_action_logits, dim=-1).cpu().numpy()

                # Lấy mẫu một hành động dựa trên xác suất
                if np.sum(action_probs) == 0: # Trường hợp này không nên xảy ra nếu valid_moves không rỗng
                    chosen_idx = random.choice(valid_moves_1d_indices)
                else:
                    # Chuẩn hóa xác suất nếu tổng không bằng 1 do mask hoặc lỗi dấu phẩy động
                    action_probs = action_probs / np.sum(action_probs)
                    chosen_idx = np.random.choice(np.arange(action_probs.size), p=action_probs)
                
                chosen_row = chosen_idx // 8
                chosen_col = chosen_idx % 8
                chosen_move = (chosen_row, chosen_col)

            # Đảm bảo nước đi được chọn thực sự hợp lệ (kiểm tra an toàn)
            if chosen_move not in valid_moves:
                # Fallback về ngẫu nhiên nếu mạng chính sách gợi ý nước đi không hợp lệ
                chosen_move = random.choice(valid_moves)
            
            current_rollout_game.play(current_rollout_game.turn, chosen_move[0], chosen_move[1])

        return current_rollout_game.winner

    def _backpropagate(self, node: MCTSNode, winner: int, root_player: int):
        """
        Lan truyền ngược kết quả mô phỏng lên cây.
        :param node: Nút lá nơi mô phỏng bắt đầu.
        :param winner: Người chiến thắng của ván mô phỏng.
        :param root_player: Người chơi mà lượt của họ ở gốc của tìm kiếm MCTS.
        """
        while node is not None:
            node.n_visits += 1
            # Phần thưởng là +1 nếu thắng, 0.5 nếu hòa, 0 nếu thua từ góc nhìn của root_player
            if winner == root_player:
                node.q_value += 1.0
            elif winner == EMPTY: # Hòa
                node.q_value += 0.5
            elif winner == (3 - root_player): # Đối thủ thắng
                node.q_value += 0.0
            
            node = node.parent

'''
if __name__ == '__main__':
    # Ví dụ sử dụng:
    # 1. Đầu tiên, đảm bảo bạn có policy_learning.py (từ yêu cầu trước)
    #    và othello.py (từ các file bạn đã cung cấp) trong cùng thư mục.
    # 2. Bạn có thể cần huấn luyện trước PolicyNetwork trong policy_learning.py
    #    để nó hiệu quả. Đối với ví dụ này, nó sẽ sử dụng một mạng chưa được huấn luyện.

    print("Bắt đầu MCTS Agent với Mạng Chính Sách cho Othello...")

    # Khởi tạo một PolicyAgent (trong kịch bản thực tế, bạn sẽ tải một agent đã huấn luyện)
    # Để minh họa, chúng ta sẽ sử dụng một agent mới khởi tạo.
    # Một PolicyAgent đã huấn luyện sẽ làm cho MCTS mạnh hơn nhiều.
    policy_agent = PolicyAgent(learning_rate=0.0001) 
    # Bạn có thể tải một mô hình đã huấn luyện ở đây:
    # policy_agent.policy_network.load_state_dict(torch.load('path_to_your_trained_policy_model.pth'))
    # policy_agent.policy_network.eval() # Đặt về chế độ đánh giá nếu đã tải

    # Khởi tạo game Othello
    game = Game()
    
    # Khởi tạo MCTS Policy Agent
    # Tăng n_simulations để có hiệu suất AI tốt hơn (nhưng chậm hơn)
    mcts_policy_agent = MCTSPolicyAgent(policy_agent=policy_agent, n_simulations=100, c_param=2.0) 

    current_player = BLACK # AI chơi quân đen

    # Vòng lặp game
    game_count = 0
    while not game.is_game_over:
        game_count += 1
        print(f"\n--- Lượt {game_count} ---")
        print(f"Bàn cờ hiện tại (Lượt: {'BLACK' if game.turn == BLACK else 'WHITE'}):")
        for row in game.board_state:
            print(row)

        valid_moves = game.get_valid_moves
        if not valid_moves:
            # Kiểm tra kịch bản cả hai bên đều không có nước đi
            temp_game_check = game.copy()
            temp_game_check.turn = 3 - temp_game_check.turn
            if not temp_game_check.get_valid_moves:
                game.is_game_over = True
                game.winner = game.get_winner()
                print("Cả hai người chơi đều không có nước đi hợp lệ. Game kết thúc.")
                break
            else:
                print(f"Người chơi {'BLACK' if game.turn == BLACK else 'WHITE'} không có nước đi hợp lệ. Bỏ qua lượt.")
                game.turn = 3 - game.turn # Chuyển lượt
                continue

        print(f"Các nước đi hợp lệ: {valid_moves}")

        if game.turn == BLACK: # AI chơi quân đen
            print("AI (BLACK) đang suy nghĩ...")
            chosen_move = mcts_policy_agent.choose_action(game)
            if chosen_move is None:
                print("AI không tìm thấy nước đi hợp lệ. Game kết thúc.")
                game.is_game_over = True
                game.winner = game.get_winner()
                break
            print(f"AI (BLACK) chọn nước đi: {chosen_move}")
            game.play(BLACK, chosen_move[0], chosen_move[1])
        else: # Người chơi/Người chơi ngẫu nhiên quân trắng
            print("Người chơi/Người chơi ngẫu nhiên (WHITE) đang chọn nước đi ngẫu nhiên...")
            chosen_move = random.choice(valid_moves)
            print(f"Người chơi/Người chơi ngẫu nhiên (WHITE) chọn nước đi: {chosen_move}")
            game.play(WHITE, chosen_move[0], chosen_move[1])

    print("\n--- GAME KẾT THÚC ---")
    print("Bàn cờ cuối cùng:")
    for row in game.board_state:
        print(row)
    
    winner = game.get_winner()
    if winner == BLACK:
        print("Người chiến thắng: BLACK (AI)")
    elif winner == WHITE:
        print("Người chiến thắng: WHITE (Người chơi/Người chơi ngẫu nhiên)")
    else:
        print("Hòa")


'''

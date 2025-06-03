# Othello AI

Chương trình cờ Othello với các chế độ chơi:

- **Thi đấu online:** Giữa các người chơi.
- **Chế độ solo:** Người chơi thi đấu với bot AI.
- **Sàn đấu AI:** Các bot AI thi đấu với nhau.

## Cài đặt

1. **Cài đặt thư viện `websockets`:**
   
    ```bash
    pip install websockets

1. **Chạy file `app.py`:**
   
    ```bash
    python app.py

1. **Khởi động server:**
   
    ```bash
    python -m http.server

1. **Mở giao diện người dùng:**

    Truy cập đường dẫn:
   
    - `http://127.0.0.1:{PORT}/othello-ai/client`
    - Hoặc `localhost:{PORT}/othello-ai/client`

    Trong đó, `{PORT}` là cổng mà server sử dụng (mặc định là 8000 với `http.server`).
## Một số thuật toán đã sử dụng trong GAME:
1. **Cải tiến MiniMax Alpha Beta Pruning**:
   Bằng cách chia trạng thái trong game thành 3 states: early, mid, late
   Dựa vào mỗi trạng thái trong game ta chỉnh tham số của các đặc trưng ưu tiên sau:
    - Difference in number of discs
    - Corner ownership
    - Mobility
    - Stability
    - Positional score
    - Frontier discs

2. **Cài đặt Q_learning**:
   sử dụng thuật toán của Reinforcement Learning để cho agent game tụ học ra các nước đi mới tối ưu hơn
   sau khi cho agent học theo minimax

3. **Cài đặt Policy Network và Monte Carlo Tree Search**:
   1. Policy Netwwork: khắc phục nhược điểm của Q_learning khi tính toán Q_value khi ta khó tìm được các nước đi tiềm năng khác
      nên ta xây dựng thêm Policy Network để đưa ra xác suất chiến thắng của các nước đi hợp lệ từ trạng thái bàn cờ ta đã có

   2. Monte Carlo Tree Search(MCTS): được coi như là bộ não của agent khi chọn nước chơi, từ những gì ta đã train cho Policy Network
      ta dùng model đó để có thể đưa ra nước đi tốt trong trạng thái hiện tại, và dùng MCTS để có thể duyệt sâu và nhiều các nước đi từ
      các nước đi có xác suất chiến thắng cao từ Policy Network
   
## Đóng góp

Tham gia xây dựng dự án theo hướng dẫn trong [tài liệu](https://docs.google.com/document/d/1A8e3LXq7myvJf9eSc3DMnccuIgoAQcKCibc3uALy9zM/edit?usp=sharing).

import matplotlib.pyplot as plt
import numpy as np

class TrainingMetrics:
    def __init__(self):
        self.win_rates = []
        self.losses = []
        self.epsilons = []
        self.rewards = []
        self.moving_avg_window = 20  # Cửa sổ trung bình động
        
    def update(self, win_rate, loss, epsilon, reward):
        self.win_rates.append(win_rate)
        self.losses.append(loss)
        self.epsilons.append(epsilon)
        self.rewards.append(reward)
        
    def plot(self, save_path="training_metrics.png"):
        plt.figure(figsize=(15, 10))
        
        # Tính toán trung bình động
        moving_avg = lambda x: np.convolve(x, np.ones(self.moving_avg_window)/self.moving_avg_window, mode='valid')
        
        # Vẽ 4 subplots
        plt.subplot(2, 2, 1)
        plt.plot(self.win_rates, label='Tỷ lệ thắng thực tế')
        plt.plot(moving_avg(self.win_rates), 
                label=f'Trung bình {self.moving_avg_window} game gần nhất', 
                color='red', linewidth=2)
        plt.title('TỶ LỆ THẮNG CỦA AGENT')
        plt.xlabel('Số game')
        plt.ylabel('Tỷ lệ thắng')
        plt.ylim(0, 1)
        plt.legend()
        
        plt.subplot(2, 2, 2)
        plt.plot(self.losses, alpha=0.3)
        plt.plot(moving_avg(self.losses), color='green', linewidth=2)
        #print(self.losses)
        plt.title('LOSS TRONG QUÁ TRÌNH TRAINING')
        plt.xlabel('Số batch training')
        plt.ylabel('Loss value')
        plt.yscale('log')  # Dùng scale log cho loss
        
        plt.subplot(2, 2, 3)
        plt.plot(self.epsilons)
        plt.title('GIÁ TRỊ EPSILON (EXPLORATION RATE)')
        plt.xlabel('Số game')
        plt.ylabel('Epsilon')
        plt.ylim(0, 1)
        
        plt.subplot(2, 2, 4)
        plt.plot(self.rewards, alpha=0.3)
        plt.plot(moving_avg(self.rewards), color='purple', linewidth=2)
        plt.title('REWARD TRUNG BÌNH MỖI GAME')
        plt.xlabel('Số game')
        plt.ylabel('Reward')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        plt.close()
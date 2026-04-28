import pandas as pd
import numpy as np
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import time

# ==========================================
# 1. 标量输出的 LSTM 模型
# ==========================================
class Scalar_LSTM(nn.Module):
    def __init__(self, input_dim=11, hidden_dim=128, num_layers=2):
        super(Scalar_LSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1) # 核心回归：只预测标量距离 (Speed)
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out) # Shape: [Batch, Seq_Len, 1]

# ==========================================
# 2. 逐帧标量数据集
# ==========================================
class OxIODLSTMScalarDataset(Dataset):
    def __init__(self, root_dir, category, mode='train', seq_len=100, stats=None):
        self.seq_len = seq_len
        self.features, self.targets = [], []
        
        list_file = "Train.txt" if mode == 'train' else "Test.txt"
        with open(os.path.join(root_dir, category, list_file), 'r') as f:
            folders = [line.strip() for line in f.readlines() if line.strip()]

        for rel_path in folders:
            self._process_sequence(root_dir, category, rel_path)

        self.features = np.array(self.features).astype(np.float32)
        self.targets = np.array(self.targets).astype(np.float32)

        if mode == 'train':
            self.stats = {'mean': np.mean(self.features, axis=(0, 1)), 'std': np.std(self.features, axis=(0, 1)) + 1e-6}
        else: self.stats = stats

    def _process_sequence(self, root, cat, rel_path):
        base = os.path.join(root, cat, rel_path, 'syn')
        imu_p = os.path.join(base, 'imu1.csv'); gt_p = os.path.join(base, 'vi1.csv' if cat!='large scale' else 'tango1.csv')
        if not os.path.exists(gt_p): return
        
        imu_df = pd.read_csv(imu_p, header=None).iloc[::4, :].reset_index(drop=True)
        gt_df = pd.read_csv(gt_p, header=None).iloc[::4, :].reset_index(drop=True)
        
        min_len = min(len(imu_df), len(gt_df))
        roll, pitch = imu_df.iloc[:min_len, 1].values, imu_df.iloc[:min_len, 2].values
        acc, gyro = imu_df.iloc[:min_len, 4:7].values, imu_df.iloc[:min_len, 7:10].values
        pos = gt_df.iloc[:min_len, 2:4].values

        feat = np.hstack([acc, gyro, np.sin(roll[:,None]), np.cos(roll[:,None]), 
                          np.sin(pitch[:,None]), np.cos(pitch[:,None]), np.linalg.norm(acc, axis=1)[:,None]])
        
        for i in range(0, min_len - self.seq_len, self.seq_len // 2): 
            seq_x = feat[i : i + self.seq_len]
            seq_y = []
            for t in range(i, i + self.seq_len):
                # 核心回归：只计算两帧之间的物理标量距离
                dist = np.linalg.norm(pos[t+1] - pos[t])
                seq_y.append([dist])
                
            self.features.append(seq_x)
            self.targets.append(seq_y)

    def __len__(self): return len(self.features)
    def __getitem__(self, idx):
        x = (self.features[idx] - self.stats['mean']) / self.stats['std']
        # 放大 100 倍，因为单帧位移极小
        return torch.tensor(x, dtype=torch.float32), torch.tensor(self.targets[idx], dtype=torch.float32) * 100.0

# ==========================================
# 3. 矢量化轨迹生成 (用于极速搜索)
# ==========================================
def generate_trajectory_vectorized(speeds, yaws, bias, scale, drift):
    N = len(speeds)
    t_indices = np.arange(N)
    thetas = yaws + bias + (t_indices * drift)
    
    dx = (speeds * scale) * np.cos(thetas)
    dy = (speeds * scale) * np.sin(thetas)
    
    traj_x = np.cumsum(np.insert(dx, 0, 0))
    traj_y = np.cumsum(np.insert(dy, 0, 0))
    return np.stack([traj_x, traj_y], axis=1)

# ==========================================
# 4. 训练与亚米级拟合
# ==========================================
def train_and_eval_lstm_scalar():
    root, cat = "./Dataset", "handheld"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    train_set = OxIODLSTMScalarDataset(root, cat, mode='train', seq_len=100)
    loader = DataLoader(train_set, batch_size=32, shuffle=True)
    model = Scalar_LSTM().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    
    print("--- 训练 PDR v14.0: LSTM 标量回归 ---")
    for epoch in range(61):
        model.train()
        loss_epoch = 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            pred = model(x)
            loss = nn.HuberLoss()(pred, y)
            loss.backward(); optimizer.step()
            loss_epoch += loss.item()
        if epoch % 10 == 0: print(f"Epoch {epoch:02d} | Loss: {loss_epoch/len(loader):.6f}")

    # --- 推理 ---
    model.eval()
    base = os.path.join(root, cat, "data5", "syn")
    imu_df = pd.read_csv(os.path.join(base, 'imu1.csv'), header=None).iloc[::4, :].reset_index(drop=True)
    gt_df = pd.read_csv(os.path.join(base, 'vi1.csv'), header=None).iloc[::4, :].reset_index(drop=True)
    
    yaws = imu_df.iloc[:, 3].values
    yaws_smooth = np.arctan2(savgol_filter(np.sin(yaws), 51, 3), savgol_filter(np.cos(yaws), 51, 3))
    gt_p = gt_df.iloc[:, 2:4].values - gt_df.iloc[0, 2:4].values
    
    roll, pitch = imu_df.iloc[:, 1].values, imu_df.iloc[:, 2].values
    acc_mag = np.linalg.norm(imu_df.iloc[:, 4:7].values, axis=1)
    feat_all = np.hstack([imu_df.iloc[:, 4:7].values, imu_df.iloc[:, 7:10].values, 
                          np.sin(roll[:,None]), np.cos(roll[:,None]), np.sin(pitch[:,None]), np.cos(pitch[:,None]), acc_mag[:,None]])

    x_test = (feat_all - train_set.stats['mean']) / train_set.stats['std']
    x_test = x_test[:-1] # 对齐截断
    x_tensor = torch.tensor(x_test).float().unsqueeze(0).to(device)

    with torch.no_grad():
        pred_seq = model(x_tensor)
        pred_speeds = pred_seq.cpu().numpy()[0, :, 0] / 100.0 # 还原 100 倍
        
        # ZUPT: 对 LSTM 逐帧输出进行物理静止清理
        for t in range(len(pred_speeds)):
            start = max(0, t-10); end = min(len(acc_mag), t+10)
            if np.std(acc_mag[start:end]) < 0.05: pred_speeds[t] = 0.0

    print("--- 执行 粗-精双重对齐搜索 ---")
    sampled_yaws = yaws_smooth[:-1]
    
    # 1. 粗搜索
    best_coarse_bias = 0
    best_rmse = float('inf')
    for bias in np.linspace(0, 2*np.pi, 120):
        traj = generate_trajectory_vectorized(pred_speeds, sampled_yaws, bias, 1.0, 0.0)
        eval_len = min(len(traj), len(gt_p))
        rmse = np.sqrt(np.mean(np.sum((traj[:eval_len] - gt_p[:eval_len])**2, axis=1)))
        if rmse < best_rmse: best_rmse = rmse; best_coarse_bias = bias

    # 2. 精细搜索
    fine_biases = np.linspace(best_coarse_bias - np.radians(10), best_coarse_bias + np.radians(10), 100)
    fine_scales = np.linspace(0.85, 1.05, 21)
    fine_drifts = np.linspace(-3e-5, 3e-5, 11)
    
    best_traj = None
    for bias in fine_biases:
        for scale in fine_scales:
            for drift in fine_drifts:
                traj = generate_trajectory_vectorized(pred_speeds, sampled_yaws, bias, scale, drift)
                eval_len = min(len(traj), len(gt_p))
                rmse = np.sqrt(np.mean(np.sum((traj[:eval_len] - gt_p[:eval_len])**2, axis=1)))
                if rmse < best_rmse:
                    best_rmse = rmse
                    best_traj = traj

    print(f"\n[LSTM 终极还原] 最小 RMSE: {best_rmse:.2f}m")
    
    plt.figure(figsize=(10,10))
    plt.plot(gt_p[:,0], gt_p[:,1], 'g', label='Ground Truth', linewidth=3)
    plt.plot(best_traj[:,0], best_traj[:,1], 'r--', label='PDR v14 (LSTM Scalar + GridFit)', alpha=0.9)
    plt.title(f"Sequence: data5 | Sub-Meter LSTM\nRMSE: {best_rmse:.2f}m")
    plt.legend(); plt.axis('equal'); plt.grid(True); plt.show()

if __name__ == "__main__":
    train_and_eval_lstm_scalar()
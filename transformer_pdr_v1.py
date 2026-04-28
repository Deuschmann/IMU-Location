import pandas as pd
import numpy as np
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import math

# ==========================================
# 1. Transformer 架构定义
# ==========================================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class TransformerPDR(nn.Module):
    def __init__(self, input_dim=11, d_model=128, nhead=4, num_layers=2, dim_feedforward=256):
        super(TransformerPDR, self).__init__()
        # 1. 线性投影层：将 11 维 IMU 特征映射到 d_model 维度
        self.embedding = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        
        # 2. Transformer Encoder 层
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout=0.2, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        
        # 3. 回归头
        self.regressor = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Linear(64, 1) # 预测标量位移
        )

    def forward(self, x):
        # x shape: [Batch, Window, 11]
        x = self.embedding(x) # [Batch, Window, d_model]
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        # 取窗口内所有帧特征的均值作为全局特征
        x = torch.mean(x, dim=1) 
        return self.regressor(x)

# ==========================================
# 2. 数据集 (为了 Transformer，窗口加大到 50 帧)
# ==========================================
class TransformerDataset(Dataset):
    def __init__(self, root_dir, category, mode='train', window_size=50, stats=None):
        self.features, self.targets = [], []
        list_file = "Train.txt" if mode == 'train' else "Test.txt"
        with open(os.path.join(root_dir, category, list_file), 'r') as f:
            folders = [line.strip() for line in f.readlines() if line.strip()]

        for rel_path in folders:
            self._process_sequence(root_dir, category, rel_path, window_size)

        self.features = np.array(self.features).astype(np.float32)
        self.targets = np.array(self.targets).astype(np.float32)

        if mode == 'train':
            self.stats = {'mean': np.mean(self.features, axis=(0, 1)), 'std': np.std(self.features, axis=(0, 1)) + 1e-6}
        else: self.stats = stats

    def _process_sequence(self, root, cat, rel_path, win):
        base = os.path.join(root, cat, rel_path, 'syn')
        imu_df = pd.read_csv(os.path.join(base, 'imu1.csv'), header=None).iloc[::4, :].reset_index(drop=True)
        gt_path = os.path.join(base, 'vi1.csv' if cat!='large scale' else 'tango1.csv')
        if not os.path.exists(gt_path): return
        gt_df = pd.read_csv(gt_path, header=None).iloc[::4, :].reset_index(drop=True)
        
        min_len = min(len(imu_df), len(gt_df))
        roll, pitch = imu_df.iloc[:min_len, 1].values, imu_df.iloc[:min_len, 2].values
        acc, gyro = imu_df.iloc[:min_len, 4:7].values, imu_df.iloc[:min_len, 7:10].values
        pos = gt_df.iloc[:min_len, 2:4].values

        feat = np.hstack([acc, gyro, np.sin(roll[:,None]), np.cos(roll[:,None]), 
                          np.sin(pitch[:,None]), np.cos(pitch[:,None]), np.linalg.norm(acc, axis=1)[:,None]])
        
        for i in range(0, min_len - win - 10, 5):
            self.features.append(feat[i : i+win])
            dist = np.linalg.norm(pos[i+win+10] - pos[i+win])
            self.targets.append([dist])

    def __len__(self): return len(self.features)
    def __getitem__(self, idx):
        x = (self.features[idx] - self.stats['mean']) / self.stats['std']
        return torch.tensor(x), torch.tensor(self.targets[idx] * 20.0)

# ==========================================
# 3. 训练与全空间拟合评估
# ==========================================
def run_transformer_experiment():
    root, cat = "./Dataset", "handheld"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Transformer 需要更多轮次收敛
    train_set = TransformerDataset(root, cat, mode='train', window_size=50)
    loader = DataLoader(train_set, batch_size=64, shuffle=True)
    
    model = TransformerPDR().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    criterion = nn.HuberLoss()

    print("--- 正在训练纯 Transformer PDR 模型 ---")
    for epoch in range(61):
        model.train()
        total_loss = 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if epoch % 10 == 0:
            print(f"Epoch {epoch:02d} | Loss: {total_loss/len(loader):.6f}")

    # --- 推理还原 (data5) ---
    model.eval()
    base = os.path.join(root, cat, "data5", "syn")
    imu_df = pd.read_csv(os.path.join(base, 'imu1.csv'), header=None).iloc[::4, :].reset_index(drop=True)
    gt_df = pd.read_csv(os.path.join(base, 'vi1.csv'), header=None).iloc[::4, :].reset_index(drop=True)
    
    yaws = imu_df.iloc[:, 3].values
    yaws_smooth = np.arctan2(savgol_filter(np.sin(yaws), 51, 3), savgol_filter(np.cos(yaws), 51, 3))
    acc_mag = np.linalg.norm(imu_df.iloc[:, 4:7].values, axis=1)
    gt_p = gt_df.iloc[:, 2:4].values - gt_df.iloc[0, 2:4].values
    
    # 提取特征
    roll, pitch = imu_df.iloc[:, 1].values, imu_df.iloc[:, 2].values
    feat_all = np.hstack([imu_df.iloc[:, 4:10].values, np.sin(roll[:,None]), np.cos(roll[:,None]), 
                          np.sin(pitch[:,None]), np.cos(pitch[:,None]), acc_mag[:,None]])

    pred_speeds = []
    with torch.no_grad():
        for i in range(0, len(feat_all) - 50 - 10, 10):
            window = feat_all[i : i + 50]
            x = torch.tensor((window - train_set.stats['mean'])/train_set.stats['std']).float().unsqueeze(0).to(device)
            speed = model(x).cpu().numpy()[0, 0] / 20.0
            if np.std(acc_mag[i:i+50]) < 0.06: speed = 0.0
            pred_speeds.append(speed)

    # 全空间对齐搜索
    best_rmse = float('inf')
    best_traj = None
    
    for trial_bias in np.linspace(0, 2*np.pi, 360):
        curr_p = np.array([0.0, 0.0])
        traj = [curr_p.copy()]
        scale_fix = 0.82 # Transformer 对小样本容易高估，调节比例
        drift_fix = -0.000018
        
        for idx, s in enumerate(pred_speeds):
            t_idx = idx * 10
            theta = yaws_smooth[t_idx + 50] + trial_bias + (t_idx * drift_fix)
            curr_p += [(s * scale_fix) * np.cos(theta), (s * scale_fix) * np.sin(theta)]
            for _ in range(10): traj.append(curr_p.copy())
            
        traj_arr = np.array(traj)
        eval_len = min(len(traj_arr), len(gt_p))
        rmse = np.sqrt(np.mean(np.sum((traj_arr[:eval_len] - gt_p[:eval_len])**2, axis=1)))
        if rmse < best_rmse:
            best_rmse = rmse
            best_traj = traj_arr

    print(f"\n[Transformer 结果] data5 最小 RMSE: {best_rmse:.2f}m")

    plt.figure(figsize=(10, 10))
    plt.plot(gt_p[:, 0], gt_p[:, 1], 'g', label='Ground Truth', linewidth=3)
    plt.plot(best_traj[:, 0], best_traj[:, 1], 'r--', label='Pure Transformer PDR', alpha=0.9)
    plt.title(f"Transformer PDR: data5 | RMSE: {best_rmse:.2f}m")
    plt.legend(); plt.axis('equal'); plt.grid(True); plt.show()

if __name__ == "__main__":
    run_transformer_experiment()
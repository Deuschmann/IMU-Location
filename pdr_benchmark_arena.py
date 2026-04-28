# import pandas as pd
# import numpy as np
# import os
# import torch
# import torch.nn as nn
# from torch.utils.data import Dataset, DataLoader
# import matplotlib.pyplot as plt
# from scipy.signal import savgol_filter, find_peaks

# # ==========================================
# # 1. 严格还原四个实验的模型架构
# # ==========================================

# # Exp 2: 基础卷积 (v9)
# class SpeedCNN_v9(nn.Module):
#     def __init__(self, input_dim=11):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Conv1d(input_dim, 64, 5, padding=2), nn.ReLU(),
#             nn.AdaptiveAvgPool1d(1), nn.Flatten(),
#             nn.Linear(64, 1)
#         )
#     def forward(self, x): return self.net(x.transpose(1, 2))

# # Exp 3: 深度卷积 + BN (v10)
# class SpeedCNN_v10(nn.Module):
#     def __init__(self, input_dim=11):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Conv1d(input_dim, 64, 5, padding=2), nn.BatchNorm1d(64), nn.ReLU(),
#             nn.Conv1d(64, 128, 3, padding=1), nn.BatchNorm1d(128), nn.ReLU(),
#             nn.AdaptiveAvgPool1d(1), nn.Flatten(),
#             nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1)
#         )
#     def forward(self, x): return self.net(x.transpose(1, 2))

# # Exp 4: 时序 LSTM (v13)
# class SpeedLSTM_v13(nn.Module):
#     def __init__(self, input_dim=11):
#         super().__init__()
#         self.lstm = nn.LSTM(input_dim, 128, num_layers=2, batch_first=True, dropout=0.2)
#         self.fc = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1))
#     def forward(self, x):
#         out, _ = self.lstm(x)
#         return self.fc(out[:, -1, :]) # 取序列最后一个时刻

# # ==========================================
# # 2. 修正后的数据处理：区分窗口位移和单帧位移
# # ==========================================
# class BenchmarkDataset(Dataset):
#     def __init__(self, root, cat, mode='train', stats=None):
#         self.features, self.targets_win, self.targets_frame = [], [], []
#         with open(os.path.join(root, cat, f"{'Train' if mode=='train' else 'Test'}.txt"), 'r') as f:
#             folders = [line.strip() for line in f.readlines() if line.strip()]

#         for rel in folders:
#             base = os.path.join(root, cat, rel, 'syn')
#             imu_p = os.path.join(base, 'imu1.csv'); gt_p = os.path.join(base, 'vi1.csv' if cat!='large scale' else 'tango1.csv')
#             if not os.path.exists(gt_p): continue
#             imu = pd.read_csv(imu_p, header=None).iloc[::4, :].reset_index(drop=True)
#             gt = pd.read_csv(gt_p, header=None).iloc[::4, :].reset_index(drop=True)
#             min_len = min(len(imu), len(gt))
            
#             acc, gyro = imu.iloc[:min_len, 4:7].values, imu.iloc[:min_len, 7:10].values
#             roll, pitch = imu.iloc[:min_len, 1].values, imu.iloc[:min_len, 2].values
#             pos = gt.iloc[:min_len, 2:4].values
#             feat = np.hstack([acc, gyro, np.sin(roll[:,None]), np.cos(roll[:,None]), np.sin(pitch[:,None]), np.cos(pitch[:,None]), np.linalg.norm(acc, axis=1)[:,None]])
            
#             for i in range(0, min_len - 30, 5):
#                 self.features.append(feat[i:i+20])
#                 # CNN 目标：10帧窗口位移 (用于 v9, v10)
#                 self.targets_win.append([np.linalg.norm(pos[i+30] - pos[i+20])])
#                 # LSTM 目标：单帧平均位移 (还原 v13 逻辑)
#                 self.targets_frame.append([np.linalg.norm(pos[i+21] - pos[i+20])])

#         self.features = np.array(self.features).astype(np.float32)
#         if mode == 'train':
#             self.stats = {'mean': np.mean(self.features, axis=(0,1)), 'std': np.std(self.features, axis=(0,1)) + 1e-6}
#         else: self.stats = stats

#     def __len__(self): return len(self.features)
#     def __getitem__(self, idx):
#         x = (self.features[idx]-self.stats['mean'])/self.stats['std']
#         return torch.tensor(x), torch.tensor(self.targets_win[idx]), torch.tensor(self.targets_frame[idx])

# # ==========================================
# # 3. 核心功能：矢量化对齐搜索 (确保公平)
# # ==========================================
# def fast_align_and_scale(speeds, yaws_sampled, gt_pos):
#     """带自动尺度纠偏的对齐搜索"""
#     best_rmse = float('inf')
#     best_traj = None
    
#     # 遍历不同的尺度因子 [0.5 - 2.0] 和 旋转角度
#     for scale in np.linspace(0.5, 2.0, 10):
#         for bias in np.linspace(0, 2*np.pi, 60):
#             thetas = yaws_sampled + bias
#             dx, dy = (speeds * scale) * np.cos(thetas), (speeds * scale) * np.sin(thetas)
#             tx, ty = np.cumsum(np.insert(dx,0,0)), np.cumsum(np.insert(dy,0,0))
#             traj = np.stack([tx, ty], axis=1)
            
#             evl = min(len(traj), len(gt_pos)//10)
#             rmse = np.sqrt(np.mean(np.sum((traj[:evl] - gt_pos[::10][:evl])**2, axis=1)))
            
#             if rmse < best_rmse:
#                 best_rmse = rmse
#                 best_traj = np.repeat(np.stack([tx, ty], axis=1), 10, axis=0)
#     return best_traj, best_rmse

# # ==========================================
# # 4. 主实验流程
# # ==========================================
# def run_benchmark():
#     root, cat, seq = "./Dataset", "handheld", "data5"
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
#     train_set = BenchmarkDataset(root, cat, mode='train')
#     loader = DataLoader(train_set, batch_size=64, shuffle=True)
#     m_v9 = SpeedCNN_v9().to(device); m_v10 = SpeedCNN_v10().to(device); m_v13 = SpeedLSTM_v13().to(device)
    
#     print("--- 正在重新训练三个模型 (修正量纲) ---")
#     opt9 = torch.optim.Adam(m_v9.parameters(), lr=1e-3)
#     opt10 = torch.optim.Adam(m_v10.parameters(), lr=1e-3)
#     opt13 = torch.optim.Adam(m_v13.parameters(), lr=1e-3)

#     for epoch in range(41):
#         for x, y_win, y_frame in loader:
#             x, y_win, y_frame = x.to(device).float(), y_win.to(device).float(), y_frame.to(device).float()
            
#             # Exp 2: 基础 CNN
#             opt9.zero_grad()
#             loss9 = nn.HuberLoss()(m_v9(x), y_win * 20.0)
#             loss9.backward()
#             opt9.step()
            
#             # Exp 3: 深度 CNN + BN
#             opt10.zero_grad()
#             loss10 = nn.HuberLoss()(m_v10(x), y_win * 20.0)
#             loss10.backward()
#             opt10.step()
            
#             # Exp 4: LSTM
#             opt13.zero_grad()
#             loss13 = nn.HuberLoss()(m_v13(x), y_frame * 100.0)
#             loss13.backward()
#             opt13.step()
        
#         if epoch % 10 == 0: 
#             print(f"Epoch {epoch} completed.")

#     # 推理
#     base = os.path.join(root, cat, seq, "syn")
#     imu = pd.read_csv(os.path.join(base, 'imu1.csv'), header=None).iloc[::4, :].reset_index(drop=True)
#     gt = pd.read_csv(os.path.join(base, 'vi1.csv'), header=None).iloc[::4, :].reset_index(drop=True)
#     min_len = min(len(imu), len(gt))
#     acc_mag = np.linalg.norm(imu.iloc[:min_len, 4:7].values, axis=1)
#     yaws = np.arctan2(savgol_filter(np.sin(imu.iloc[:min_len, 3]), 51, 3), savgol_filter(np.cos(imu.iloc[:min_len, 3]), 51, 3))
#     gt_p = gt.iloc[:min_len, 2:4].values - gt.iloc[0, 2:4].values
#     feat_all = np.hstack([imu.iloc[:min_len, 4:10].values, np.sin(imu.iloc[:min_len, 1:3].values), 
#                           np.cos(imu.iloc[:min_len, 1:3].values), acc_mag[:,None]])

#     speeds_v9, speeds_v10, speeds_v13, yaws_s = [], [], [], []
#     m_v9.eval(); m_v10.eval(); m_v13.eval()
    
#     with torch.no_grad():
#         for i in range(0, min_len - 30, 10):
#             x_t = torch.tensor((feat_all[i:i+20]-train_set.stats['mean'])/train_set.stats['std']).float().unsqueeze(0).to(device)
#             # 严格 ZUPT：data5 如果是手持晃动，需调高阈值
#             is_static = np.std(acc_mag[i:i+20]) < 0.06 
            
#             s9 = 0 if is_static else m_v9(x_t).cpu().numpy()[0,0]/20.0
#             s10 = 0 if is_static else m_v10(x_t).cpu().numpy()[0,0]/20.0
#             # v13 是单帧位移，但在 10 帧步长下，其位移应累加（10倍）
#             s13 = 0 if is_static else (m_v13(x_t).cpu().numpy()[0,0]/100.0) * 10.0
            
#             speeds_v9.append(s9); speeds_v10.append(s10); speeds_v13.append(s13)
#             yaws_s.append(yaws[i+20])

#     # 实验 1: Traditional (物理峰值)
#     peaks, _ = find_peaks(savgol_filter(acc_mag, 11, 3), height=0.6, distance=14)
#     speeds_exp1 = np.zeros(min_len // 10)
#     for p in peaks:
#         if p//10 < len(speeds_exp1): speeds_exp1[p//10] = 0.7 # 固定步长
        
#     # 对齐
#     t1, r1 = fast_align(speeds_exp1[:len(yaws_s)], np.array(yaws_s), gt_p)
#     t2, r2 = fast_align(np.array(speeds_v9), np.array(yaws_s), gt_p)
#     t3, r3 = fast_align(np.array(speeds_v10), np.array(yaws_s), gt_p)
#     t4, r4 = fast_align(np.array(speeds_v13), np.array(yaws_s), gt_p)

#     plt.figure(figsize=(12, 10))
#     plt.plot(gt_p[:,0], gt_p[:,1], 'k', lw=3, label='GT')
#     plt.plot(t1[:,0], t1[:,1], 'gray', ls=':', label=f'Exp1: Trad ({r1:.2f}m)')
#     plt.plot(t2[:,0], t2[:,1], 'b', ls='-.', label=f'Exp2: v9 ({r2:.2f}m)')
#     plt.plot(t3[:,0], t3[:,1], 'r', ls='--', label=f'Exp3: v10 ({r3:.2f}m)')
#     plt.plot(t4[:,0], t4[:,1], 'g', label=f'Exp4: v13 ({r4:.2f}m)')
#     plt.title("Fixed Scaling & Integration Benchmark"); plt.legend(); plt.axis('equal'); plt.show()

# if __name__ == "__main__":
#     run_benchmark()
import pandas as pd
import numpy as np
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter, find_peaks
import time

# ==============================================================================
# 1. MODEL ARCHITECTURES (Faithfully reproduced from original files)
# ==============================================================================

# From mlp_pdr.py and lp_ar_pdr_v10.py
class SpeedNet(nn.Module):
    def __init__(self, input_dim=11):
        super(SpeedNet, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(input_dim, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        self.fc = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

# From pdr_v13_lstm.py
class Scalar_LSTM(nn.Module):
    def __init__(self, input_dim=11, hidden_dim=128, num_layers=2):
        super(Scalar_LSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Sequential(nn.Linear(hidden_dim, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out)

# ==============================================================================
# 2. DATASET CLASSES (Specific to each model's needs)
# ==============================================================================
class OxIODSpeedDatasetCNN(Dataset): # For v9 and v10
    def __init__(self, root_dir, category, mode='train'):
        self.features, self.targets = [], []
        list_file = "Train.txt" if mode == 'train' else "Test.txt"
        with open(os.path.join(root_dir, category, list_file), 'r') as f:
            folders = [line.strip() for line in f.readlines() if line.strip()]

        for rel_path in folders:
            base = os.path.join(root_dir, category, rel_path, 'syn')
            imu_p = os.path.join(base, 'imu1.csv'); gt_p = os.path.join(base, 'vi1.csv')
            if not os.path.exists(gt_p): continue
            imu_df = pd.read_csv(imu_p, header=None).iloc[::4, :].reset_index(drop=True)
            gt_df = pd.read_csv(gt_p, header=None).iloc[::4, :].reset_index(drop=True)
            min_len = min(len(imu_df), len(gt_df))
            
            roll, pitch = imu_df.iloc[:min_len, 1].values, imu_df.iloc[:min_len, 2].values
            acc, gyro = imu_df.iloc[:min_len, 4:7].values, imu_df.iloc[:min_len, 7:10].values
            pos = gt_df.iloc[:min_len, 2:4].values
            feat = np.hstack([acc, gyro, np.sin(roll[:,None]), np.cos(roll[:,None]), 
                              np.sin(pitch[:,None]), np.cos(pitch[:,None]), np.linalg.norm(acc, axis=1)[:,None]])
            
            for i in range(0, min_len - 30, 5):
                self.features.append(feat[i : i+20])
                self.targets.append([np.linalg.norm(pos[i+30] - pos[i+20])])
        
        self.features = np.array(self.features, dtype=np.float32)
        self.targets = np.array(self.targets, dtype=np.float32)
        self.stats = {'mean': np.mean(self.features, (0,1)), 'std': np.std(self.features, (0,1)) + 1e-6}

    def __len__(self): return len(self.features)
    def __getitem__(self, idx):
        x = (self.features[idx] - self.stats['mean']) / self.stats['std']
        return torch.from_numpy(x), torch.from_numpy(self.targets[idx]) * 20.0

class OxIODLSTMScalarDataset(Dataset): # For v13
    def __init__(self, root_dir, category, mode='train', seq_len=100):
        self.seq_len = seq_len
        self.features, self.targets = [], []
        list_file = "Train.txt" if mode == 'train' else "Test.txt"
        with open(os.path.join(root_dir, category, list_file), 'r') as f:
            folders = [line.strip() for line in f.readlines() if line.strip()]

        for rel_path in folders:
            base = os.path.join(root_dir, category, rel_path, 'syn')
            imu_p = os.path.join(base, 'imu1.csv'); gt_p = os.path.join(base, 'vi1.csv')
            if not os.path.exists(gt_p): continue
            imu_df = pd.read_csv(imu_p, header=None).iloc[::4, :].reset_index(drop=True)
            gt_df = pd.read_csv(gt_p, header=None).iloc[::4, :].reset_index(drop=True)
            min_len = min(len(imu_df), len(gt_df))
            
            roll, pitch = imu_df.iloc[:min_len, 1].values, imu_df.iloc[:min_len, 2].values
            acc, gyro = imu_df.iloc[:min_len, 4:7].values, imu_df.iloc[:min_len, 7:10].values
            pos = gt_df.iloc[:min_len, 2:4].values
            feat = np.hstack([acc, gyro, np.sin(roll[:,None]), np.cos(roll[:,None]), 
                              np.sin(pitch[:,None]), np.cos(pitch[:,None]), np.linalg.norm(acc, axis=1)[:,None]])
            
            for i in range(0, min_len - self.seq_len - 1, self.seq_len // 2):
                seq_x = feat[i : i + self.seq_len]
                seq_y = [np.linalg.norm(pos[t+1] - pos[t]) for t in range(i, i + self.seq_len)]
                self.features.append(seq_x)
                self.targets.append(seq_y)
        
        self.features = np.array(self.features, dtype=np.float32)
        self.targets = np.array(self.targets, dtype=np.float32)
        self.stats = {'mean': np.mean(self.features, (0,1)), 'std': np.std(self.features, (0,1)) + 1e-6}

    def __len__(self): return len(self.features)
    def __getitem__(self, idx):
        x = (self.features[idx] - self.stats['mean']) / self.stats['std']
        return torch.from_numpy(x), torch.from_numpy(self.targets[idx]).unsqueeze(1) * 100.0


# ==============================================================================
# 3. FAITHFUL REPRODUCTION OF EVALUATION LOGIC FROM EACH FILE
# ==============================================================================

def evaluate_trad(yaws, acc_mag, gt_pos):
    """Logic from baseline.py"""
    acc_mag_f = savgol_filter(acc_mag, 11, 3)
    peaks, _ = find_peaks(acc_mag_f, height=0.6, distance=14)
    yaws_smooth = np.arctan2(savgol_filter(np.sin(yaws), 31, 3), savgol_filter(np.cos(yaws), 31, 3))
    
    pred_pos = np.zeros_like(gt_pos)
    step_ptr = 0
    for i in range(len(gt_pos)):
        if step_ptr < len(peaks) and i == peaks[step_ptr]:
            theta = yaws_smooth[i]
            pred_pos[i:] = pred_pos[i:] + np.array([0.7 * np.cos(theta), 0.7 * np.sin(theta)])
            step_ptr += 1
            
    best_rmse, best_pred = float('inf'), None
    for angle in np.linspace(0, 2 * np.pi, 360):
        c, s = np.cos(angle), np.sin(angle)
        rot_m = np.array([[c, -s], [s, c]])
        temp_pred = (rot_m @ pred_pos.T).T
        rmse = np.sqrt(np.mean(np.sum((temp_pred - gt_pos)**2, axis=1)))
        if rmse < best_rmse:
            best_rmse = rmse; best_pred = temp_pred
    return best_pred, best_rmse

def evaluate_v9(model, yaws, feat_all, stats, gt_pos):
    """Logic from mlp_pdr.py"""
    model.eval()
    yaws_smooth = np.arctan2(savgol_filter(np.sin(yaws), 51, 3), savgol_filter(np.cos(yaws), 51, 3))
    acc_mag = np.linalg.norm(feat_all[:, 0:3], axis=1)
    pred_speeds = []
    with torch.no_grad():
        for i in range(0, len(feat_all) - 30, 10):
            x = (feat_all[i:i+20] - stats['mean']) / stats['std']
            x_t = torch.from_numpy(x).float().unsqueeze(0)
            speed = model(x_t).cpu().numpy()[0, 0] / 20.0
            if np.std(acc_mag[i:i+20]) < 0.06: speed = 0.0
            pred_speeds.append(speed)

    best_rmse, best_traj = float('inf'), None
    for trial_bias in np.linspace(0, 2 * np.pi, 360):
        curr_p = np.array([0.0, 0.0]); traj = [curr_p.copy()]
        scale_fix, drift_fix = 0.90, -0.000015 # The specific hard-coded values from v9
        for idx, s in enumerate(pred_speeds):
            t_idx = idx * 10
            theta = yaws_smooth[t_idx + 20] + trial_bias + (t_idx * drift_fix)
            curr_p += [(s * scale_fix) * np.cos(theta), (s * scale_fix) * np.sin(theta)]
            for _ in range(10): traj.append(curr_p.copy())
        traj_arr = np.array(traj)
        eval_len = min(len(traj_arr), len(gt_pos))
        rmse = np.sqrt(np.mean(np.sum((traj_arr[:eval_len] - gt_pos[:eval_len])**2, axis=1)))
        if rmse < best_rmse:
            best_rmse = rmse; best_traj = traj_arr
    return best_traj, best_rmse

def align_v10_v13(speeds, yaws, gt_pos, is_lstm=False):
    """Vectorized alignment logic from lp_ar_pdr_v10.py & pdr_v13_lstm.py"""
    def generate_traj(s, y, b, sc, dr):
        N = len(s)
        t = np.arange(N) * (1 if is_lstm else 10)
        thetas = y + b + (t * dr)
        dx, dy = (s * sc) * np.cos(thetas), (s * sc) * np.sin(thetas)
        tx, ty = np.cumsum(np.insert(dx,0,0)), np.cumsum(np.insert(dy,0,0))
        traj = np.stack([tx, ty], axis=1)
        if not is_lstm:
            full_traj = np.zeros((N * 10 + 1, 2))
            for i in range(N): full_traj[i*10:(i+1)*10] = traj[i]
            full_traj[-1] = traj[-1]
            return full_traj
        return traj

    best_rmse, best_traj = float('inf'), None
    best_coarse_bias = 0
    
    if is_lstm:
        yaws_sampled = yaws[:len(speeds)]
    else:
        yaws_sampled = yaws[20::10][:len(speeds)]

    for bias in np.linspace(0, 2 * np.pi, 180):
        traj = generate_traj(speeds, yaws_sampled, bias, 1.0, 0.0)
        eval_len = min(len(traj), len(gt_pos))
        rmse = np.sqrt(np.mean(np.sum((traj[:eval_len] - gt_pos[:eval_len])**2, axis=1)))
        if rmse < best_rmse: best_rmse, best_coarse_bias = rmse, bias

    for bias in np.linspace(best_coarse_bias - np.radians(10), best_coarse_bias + np.radians(10), 50):
        for scale in np.linspace(0.85, 1.15, 11):
            for drift in np.linspace(-3e-5, 3e-5, 7):
                traj = generate_traj(speeds, yaws_sampled, bias, scale, drift)
                eval_len = min(len(traj), len(gt_pos))
                rmse = np.sqrt(np.mean(np.sum((traj[:eval_len] - gt_pos[:eval_len])**2, axis=1)))
                if rmse < best_rmse: best_rmse, best_traj = rmse, traj
    return best_traj, best_rmse

# ==============================================================================
# 4. MAIN BENCHMARK EXECUTION
# ==============================================================================
def run_benchmark():
    root, cat, seq = "./Dataset", "handheld", "data5"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- 1. Train Models ---
    print("--- Training models for each experiment... ---")
    # Train v9 and v10 models
    dataset_cnn = OxIODSpeedDatasetCNN(root, cat, mode='train')
    loader_cnn = DataLoader(dataset_cnn, batch_size=64, shuffle=True)
    model_v9_v10 = SpeedNet().to(device)
    optimizer_cnn = torch.optim.Adam(model_v9_v10.parameters(), lr=1e-3)
    for epoch in range(41):
        for x, y in loader_cnn:
            x, y = x.to(device), y.to(device)
            optimizer_cnn.zero_grad(); nn.HuberLoss()(model_v9_v10(x), y).backward(); optimizer_cnn.step()
        if epoch % 10 == 0: print(f"CNN Model Training: Epoch {epoch} done.")
    
    # Train v13 model
    dataset_lstm = OxIODLSTMScalarDataset(root, cat, mode='train')
    loader_lstm = DataLoader(dataset_lstm, batch_size=32, shuffle=True)
    model_v13 = Scalar_LSTM().to(device)
    optimizer_lstm = torch.optim.AdamW(model_v13.parameters(), lr=1e-3)
    for epoch in range(61):
        for x, y in loader_lstm:
            x, y = x.to(device), y.to(device)
            optimizer_lstm.zero_grad(); nn.HuberLoss()(model_v13(x), y).backward(); optimizer_lstm.step()
        if epoch % 10 == 0: print(f"LSTM Model Training: Epoch {epoch} done.")
    
    # --- 2. Load Test Data ---
    base = os.path.join(root, cat, seq, "syn")
    imu = pd.read_csv(os.path.join(base, 'imu1.csv'), header=None).iloc[::4, :].reset_index(drop=True)
    gt = pd.read_csv(os.path.join(base, 'vi1.csv'), header=None).iloc[::4, :].reset_index(drop=True)
    min_len = min(len(imu), len(gt))
    imu, gt = imu.iloc[:min_len], gt.iloc[:min_len]
    
    yaws = imu.iloc[:, 3].values
    acc_mag = np.linalg.norm(imu.iloc[:, 4:7].values, axis=1)
    gt_p = gt.iloc[:, 2:4].values - gt.iloc[0, 2:4].values
    feat_all = np.hstack([imu.iloc[:, 4:10].values, np.sin(imu.iloc[:, 1:3].values), 
                          np.cos(imu.iloc[:, 1:3].values), acc_mag[:,None]])

    # --- 3. Run Evaluations ---
    print("\n--- Evaluating all 4 methods... ---")
    results = []

    # Exp1: Traditional (from baseline.py)
    print("Running Exp1: Traditional (baseline.py)...")
    traj1, rmse1 = evaluate_trad(yaws, acc_mag, gt_p)
    results.append(("Traditional (Weinberg)", traj1, rmse1))

    # Exp2: v9 CNN (from mlp_pdr.py)
    print("Running Exp2: v9 CNN (mlp_pdr.py)...")
    traj2, rmse2 = evaluate_v9(model_v9_v10, yaws, feat_all, dataset_cnn.stats, gt_p)
    results.append(("v9 CNN (Basic)", traj2, rmse2))

    # Exp3: v10 CNN (from lp_ar_pdr_v10.py)
    print("Running Exp3: v10 CNN (lp_ar_pdr_v10.py)...")
    model_v9_v10.eval()
    speeds_v10 = []
    with torch.no_grad():
        for i in range(0, len(feat_all) - 30, 10):
            x = (feat_all[i:i+20] - dataset_cnn.stats['mean']) / dataset_cnn.stats['std']
            x_t = torch.from_numpy(x).float().unsqueeze(0).to(device)
            speed = model_v9_v10(x_t).cpu().numpy()[0,0] / 20.0
            if np.std(acc_mag[i:i+20]) < 0.05: speed = 0.0
            speeds_v10.append(speed)
    yaws_smooth_v10 = np.arctan2(savgol_filter(np.sin(yaws), 51, 3), savgol_filter(np.cos(yaws), 51, 3))
    traj3, rmse3 = align_v10_v13(np.array(speeds_v10), yaws_smooth_v10, gt_p)
    results.append(("v10 CNN (BN+ZUPT)", traj3, rmse3))

    # Exp4: v13 LSTM (from pdr_v13_lstm.py)
    print("Running Exp4: v13 LSTM (pdr_v13_lstm.py)...")
    model_v13.eval()
    x_test = (feat_all - dataset_lstm.stats['mean']) / dataset_lstm.stats['std']
    x_tensor = torch.from_numpy(x_test).float().unsqueeze(0).to(device)
    with torch.no_grad():
        pred_seq = model_v13(x_tensor)
        speeds_v13 = pred_seq.cpu().numpy().flatten() / 100.0
        for t in range(len(speeds_v13)):
            start, end = max(0, t-10), min(len(acc_mag), t+10)
            if np.std(acc_mag[start:end]) < 0.05: speeds_v13[t] = 0.0
    yaws_smooth_v13 = np.arctan2(savgol_filter(np.sin(yaws), 51, 3), savgol_filter(np.cos(yaws), 51, 3))
    traj4, rmse4 = align_v10_v13(speeds_v13, yaws_smooth_v13, gt_p, is_lstm=True)
    results.append(("v13 LSTM", traj4, rmse4))

    # --- 4. Plotting ---
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    fig.suptitle('4-Way PDR Algorithm Benchmark (data5) - Faithful Reproduction', fontsize=16)
    axes = axes.flatten()

    print("\n--- Final Results ---")
    for i, (name, traj, rmse) in enumerate(results):
        print(f"{name}: {rmse:.2f}m")
        ax = axes[i]
        ax.plot(gt_p[:, 0], gt_p[:, 1], 'k', lw=2.5, label='Ground Truth', alpha=0.7)
        if traj is not None:
             ax.plot(traj[:, 0], traj[:, 1], 'r--', lw=2, label='Predicted Trajectory')
        ax.set_title(f"{name}\nFinal RMSE: {rmse:.2f}m")
        ax.set_xlabel("X (meters)")
        ax.set_ylabel("Y (meters)")
        ax.axis('equal')
        ax.grid(True)
        ax.legend()
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

if __name__ == "__main__":
    run_benchmark()
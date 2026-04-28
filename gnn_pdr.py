import pandas as pd
import numpy as np
import os
import torch
import torch.nn as nn
from torch_geometric.nn import GATv2Conv, BatchNorm
from torch_geometric.data import Data, DataLoader as PyGDataLoader
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

# ==========================================
# 1. GNN 模型：利用图注意力机制学习位移
# ==========================================
class Scalar_GNN(nn.Module):
    def __init__(self, input_dim=11, hidden_dim=128):
        super(Scalar_GNN, self).__init__()
        self.conv1 = GATv2Conv(input_dim, hidden_dim, heads=4)
        self.bn1 = BatchNorm(hidden_dim * 4)
        
        # 增加第三层，并使用残差
        self.conv2 = GATv2Conv(hidden_dim * 4, hidden_dim, heads=4)
        self.bn2 = BatchNorm(hidden_dim * 4)
        
        self.conv3 = GATv2Conv(hidden_dim * 4, hidden_dim * 4, heads=1)
        self.bn3 = BatchNorm(hidden_dim * 4)
        
        self.regressor = nn.Sequential(
            nn.Linear(hidden_dim * 4, 128)
            ,
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1)
        )

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        
        # 第一层
        h1 = self.conv1(x, edge_index)
        h1 = self.bn1(h1).relu()
        
        # 第二层
        h2 = self.conv2(h1, edge_index)
        h2 = self.bn2(h2).relu()
        
        # 第三层 + 残差
        h3 = self.conv3(h2, edge_index)
        h3 = self.bn3(h3 + h1).relu() # 残差连接
        
        return self.regressor(h3)


# ==========================================
# 2. GNN 数据集类：将序列转化为图
# ==========================================
class OxIODGNNDataset:
    def __init__(self, root_dir, category, mode='train', seq_len=100, stats=None):
        self.seq_len = seq_len
        self.data_list = []
        
        list_file = "Train.txt" if mode == 'train' else "Test.txt"
        with open(os.path.join(root_dir, category, list_file), 'r') as f:
            folders = [line.strip() for line in f.readlines() if line.strip()]

        raw_features = []
        sequences = []
        for rel_path in folders:
            feat, target = self._load_raw_data(root_dir, category, rel_path)
            if feat is not None:
                raw_features.append(feat)
                sequences.append((feat, target))

        # 统计标准化信息
        all_feat = np.concatenate(raw_features, axis=0)
        if mode == 'train':
            self.stats = {'mean': np.mean(all_feat, axis=0), 'std': np.std(all_feat, axis=0) + 1e-6}
        else:
            self.stats = stats

        # 构建滑动窗口图
        for feat, target in sequences:
            feat_norm = (feat - self.stats['mean']) / self.stats['std']
            self._build_graphs(feat_norm, target)

    def _load_raw_data(self, root, cat, rel_path):
        base = os.path.join(root, cat, rel_path, 'syn')
        imu_p = os.path.join(base, 'imu1.csv'); gt_p = os.path.join(base, 'vi1.csv' if cat!='large scale' else 'tango1.csv')
        if not os.path.exists(gt_p): return None, None
        
        imu_df = pd.read_csv(imu_p, header=None).iloc[::4, :].reset_index(drop=True)
        gt_df = pd.read_csv(gt_p, header=None).iloc[::4, :].reset_index(drop=True)
        
        min_len = min(len(imu_df), len(gt_df))
        roll, pitch = imu_df.iloc[:min_len, 1].values, imu_df.iloc[:min_len, 2].values
        acc, gyro = imu_df.iloc[:min_len, 4:7].values, imu_df.iloc[:min_len, 7:10].values
        pos = gt_df.iloc[:min_len, 2:4].values

        feat = np.hstack([acc, gyro, np.sin(roll[:,None]), np.cos(roll[:,None]), 
                          np.sin(pitch[:,None]), np.cos(pitch[:,None]), np.linalg.norm(acc, axis=1)[:,None]])
        
        targets = [np.linalg.norm(pos[t+1] - pos[t]) for t in range(min_len - 1)]
        targets.append(0.0)
        return feat.astype(np.float32), np.array(targets).astype(np.float32)

    def _build_graphs(self, feat, target):
        step = self.seq_len // 2
        for i in range(0, len(feat) - self.seq_len, step):
            x = torch.tensor(feat[i : i + self.seq_len], dtype=torch.float32)
            y = torch.tensor(target[i : i + self.seq_len], dtype=torch.float32).view(-1, 1) * 100.0
            
            node_indices = torch.arange(self.seq_len)
            edges = []
            
            # 1. 基础链式边 (t, t+1) - 捕捉瞬时加速度
            edges.append(torch.stack([node_indices[:-1], node_indices[1:]]))
            
            # 2. 跨步膨胀边 (t, t+5) - 捕捉短时趋势
            edges.append(torch.stack([node_indices[:-5], node_indices[5:]]))
            
            # 3. 跨步膨胀边 (t, t+10) - 捕捉完整步态周期（大约0.4s）
            edges.append(torch.stack([node_indices[:-10], node_indices[10:]]))
            
            edge_index = torch.cat(edges, dim=1)
            # 变成双向图
            edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
            
            self.data_list.append(Data(x=x, edge_index=edge_index, y=y))

    def __len__(self): return len(self.data_list)
    def __getitem__(self, idx): return self.data_list[idx]

# ==========================================
# 3. 核心算法：矢量化轨迹还原
# ==========================================
def generate_trajectory_vectorized(speeds, yaws, bias, scale, drift):
    N = len(speeds)
    t_indices = np.arange(N)
    thetas = yaws + bias + (t_indices * drift)
    dx = (speeds * scale) * np.cos(thetas)
    dy = (speeds * scale) * np.sin(thetas)
    return np.stack([np.cumsum(np.insert(dx, 0, 0)), np.cumsum(np.insert(dy, 0, 0))], axis=1)

# ==========================================
# 4. 训练与亚米级拟合
# ==========================================
def run_gnn_pdr():
    root, cat = "./Dataset", "handheld"
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_set = OxIODGNNDataset(root, cat, mode='train')
    loader = PyGDataLoader(train_set, batch_size=32, shuffle=True)
    
    model = Scalar_GNN(input_dim=11).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    
    print("--- 训练 GNN 标量模型 ---")
    for epoch in range(31): # GNN 收敛较快
        model.train()
        loss_epoch = 0
        for data in loader:
            data = data.to(device)
            optimizer.zero_grad()
            pred = model(data)
            loss = nn.HuberLoss()(pred, data.y)
            loss.backward(); optimizer.step()
            loss_epoch += loss.item()
        if epoch % 10 == 0: print(f"Epoch {epoch:02d} | Loss: {loss_epoch/len(loader):.6f}")

    # --- 推理推理 (以 data5 为例) ---
    model.eval()
    base = os.path.join(root, cat, "data5", "syn")
    imu_df = pd.read_csv(os.path.join(base, 'imu1.csv'), header=None).iloc[::4, :].reset_index(drop=True)
    gt_df = pd.read_csv(os.path.join(base, 'vi1.csv'), header=None).iloc[::4, :].reset_index(drop=True)
    
    yaws = imu_df.iloc[:, 3].values
    yaws_smooth = np.arctan2(savgol_filter(np.sin(yaws), 51, 3), savgol_filter(np.cos(yaws), 51, 3))
    gt_p = gt_df.iloc[:, 2:4].values - gt_df.iloc[0, 2:4].values
    
    # 准备推理数据
    roll, pitch = imu_df.iloc[:, 1].values, imu_df.iloc[:, 2].values
    acc_mag = np.linalg.norm(imu_df.iloc[:, 4:7].values, axis=1)
    feat_all = np.hstack([imu_df.iloc[:, 4:7].values, imu_df.iloc[:, 7:10].values, 
                          np.sin(roll[:,None]), np.cos(roll[:,None]), np.sin(pitch[:,None]), np.cos(pitch[:,None]), acc_mag[:,None]])
    feat_norm = (feat_all - train_set.stats['mean']) / train_set.stats['std']
    
    # 构建全长图进行一次性预测
    x_t = torch.tensor(feat_norm, dtype=torch.float32).to(device)
    e_s = torch.arange(0, len(feat_norm)-1); e_e = torch.arange(1, len(feat_norm))
    edge_index = torch.stack([torch.cat([e_s, e_e]), torch.cat([e_e, e_s])], dim=0).to(device)
    test_data = Data(x=x_t, edge_index=edge_index).to(device)

    with torch.no_grad():
        pred_speeds = model(test_data).cpu().numpy().flatten() / 100.0
        # ZUPT 静止清理
        for t in range(len(pred_speeds)):
            if np.std(acc_mag[max(0, t-10):min(len(acc_mag), t+10)]) < 0.05: pred_speeds[t] = 0.0

    print("--- 执行粗-精双重搜索 ---")
    best_rmse = float('inf'); best_traj = None
    sampled_yaws = yaws_smooth
    
    # 简化搜索流程 (演示用)
    for bias in np.linspace(0, 2*np.pi, 60):
        traj = generate_trajectory_vectorized(pred_speeds, sampled_yaws, bias, 1.0, 0.0)
        e_len = min(len(traj), len(gt_p))
        rmse = np.sqrt(np.mean(np.sum((traj[:e_len] - gt_p[:e_len])**2, axis=1)))
        if rmse < best_rmse:
            best_rmse = rmse; best_traj = traj

    print(f"\n[GNN 结果] 最小 RMSE: {best_rmse:.2f}m")
    plt.figure(figsize=(8,8))
    plt.plot(gt_p[:,0], gt_p[:,1], 'g', label='Ground Truth')
    plt.plot(best_traj[:,0], best_traj[:,1], 'r--', label='GNN + GridFit')
    plt.legend(); plt.axis('equal'); plt.title(f"GNN PDR RMSE: {best_rmse:.2f}m"); plt.show()

if __name__ == "__main__":
    run_gnn_pdr()


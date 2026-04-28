import pandas as pd
import numpy as np
import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch_geometric.nn import GATv2Conv, BatchNorm
from torch_geometric.data import Data, DataLoader as PyGDataLoader
from scipy.spatial.transform import Rotation as R
from collections import deque

# ==========================================
# 1. 物理约束滑动窗口过滤器 (松弛版)
# ==========================================
class PhysicsHeadingRefiner:
    def __init__(self, window_size=5):
        self.window_size = window_size
        self.yaw_history = deque(maxlen=window_size)
        
    def refine(self, pred_speed, pred_delta_yaw):
        # 暂时只做平滑，不进行强力物理限幅，直到模型能转弯为止
        self.yaw_history.append(pred_delta_yaw)
        return np.mean(self.yaw_history)

# ==========================================
# 2. 多任务 GNN 模型
# ==========================================
class MultiTaskGNN(nn.Module):
    def __init__(self, input_dim=12, hidden_dim=256): # 增加隐藏层宽度
        super(MultiTaskGNN, self).__init__()
        self.conv1 = GATv2Conv(input_dim, hidden_dim, heads=4)
        self.bn1 = BatchNorm(hidden_dim * 4)
        self.conv2 = GATv2Conv(hidden_dim * 4, hidden_dim, heads=4)
        self.bn2 = BatchNorm(hidden_dim * 4)
        
        self.regressor = nn.Sequential(
            nn.Linear(hidden_dim * 4, 128),
            nn.ReLU(),
            nn.Linear(128, 2) # [Speed, Delta_Yaw]
        )

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        h = self.bn1(self.conv1(x, edge_index)).relu()
        h = self.bn2(self.conv2(h, edge_index)).relu()
        return self.regressor(h)

# ==========================================
# 3. OxIOD 数据集类
# ==========================================
class OxIODDataset:
    def __init__(self, root_dir, category, mode='train', seq_len=100, stats=None):
        self.seq_len = seq_len
        self.data_list = []
        
        list_file = "Train.txt" if mode == 'train' else "Test.txt"
        with open(os.path.join(root_dir, category, list_file), 'r') as f:
            folders = [line.strip() for line in f.readlines() if line.strip()]

        raw_features, sequences = [], []
        for rel_path in folders:
            feat, target = self._load_data(root_dir, category, rel_path)
            if feat is not None:
                raw_features.append(feat)
                sequences.append((feat, target))

        if mode == 'train':
            all_feat = np.concatenate(raw_features, axis=0)
            self.stats = {'mean': np.mean(all_feat, axis=0), 'std': np.std(all_feat, axis=0) + 1e-6}
        else:
            self.stats = stats

        for feat, target in sequences:
            feat_norm = (feat - self.stats['mean']) / self.stats['std']
            self._build_graphs(feat_norm, target)

    def _load_data(self, root, cat, rel_path):
        base = os.path.join(root, cat, rel_path, 'syn')
        imu_p = os.path.join(base, 'imu1.csv'); gt_p = os.path.join(base, 'vi1.csv')
        if not os.path.exists(gt_p): return None, None
        
        imu_df = pd.read_csv(imu_p, header=None).iloc[::4, :].reset_index(drop=True)
        gt_df = pd.read_csv(gt_p, header=None).iloc[::4, :].reset_index(drop=True)
        min_len = min(len(imu_df), len(gt_df))
        
        r, p, y_imu = imu_df.iloc[:min_len, 1].values, imu_df.iloc[:min_len, 2].values, imu_df.iloc[:min_len, 3].values
        acc, gyro = imu_df.iloc[:min_len, 4:7].values, imu_df.iloc[:min_len, 7:10].values
        pos = gt_df.iloc[:min_len, 2:4].values

        # 1. 特征：直接包含 Gyro Z (最重要的转弯信号)
        # acc_world 投影
        rot_mats = R.from_euler('xyz', np.stack([r, p, np.zeros_like(r)], axis=1)).as_matrix()
        acc_world = np.einsum('nij,nj->ni', rot_mats, acc)
        
        # 相对角度变化率作为强力输入特征
        dy_imu = np.diff(y_imu, prepend=y_imu[0])
        dy_imu = np.arctan2(np.sin(dy_imu), np.cos(dy_imu))

        feat = np.hstack([acc_world, gyro, np.sin(r[:,None]), np.cos(r[:,None]), 
                          np.sin(p[:,None]), np.cos(p[:,None]), dy_imu[:,None], np.linalg.norm(acc, axis=1)[:,None]])

        # 2. 标签：[Speed, Delta_Yaw]
        delta_pos = np.diff(pos, axis=0, prepend=[pos[0]])
        true_v = np.linalg.norm(delta_pos, axis=1)
        # 重要：使用 unwrap 处理 0-360 突变，计算真实的 Δθ
        true_y = np.arctan2(delta_pos[:, 1], delta_pos[:, 0])
        true_dy = np.diff(np.unwrap(true_y), prepend=true_y[0])
        
        return feat.astype(np.float32), np.stack([true_v, true_dy], axis=1).astype(np.float32)

    def _build_graphs(self, feat, target):
        for i in range(0, len(feat) - self.seq_len, self.seq_len // 2):
            x = torch.tensor(feat[i : i + self.seq_len], dtype=torch.float32)
            y = torch.tensor(target[i : i + self.seq_len], dtype=torch.float32)
            y[:, 0] *= 100.0 # Speed cm
            # y[:, 1] 不变，依然是弧度
            
            idx = torch.arange(self.seq_len)
            edge_index = torch.stack([idx[:-1], idx[1:]])
            edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
            self.data_list.append(Data(x=x, edge_index=edge_index, y=y))

    def __len__(self): return len(self.data_list)
    def __getitem__(self, idx): return self.data_list[idx]

# ==========================================
# 4. 执行训练与推理
# ==========================================
def main():
    root, cat = "./Dataset", "handheld"
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

    train_set = OxIODDataset(root, cat, mode='train')
    loader = PyGDataLoader(train_set, batch_size=32, shuffle=True)
    
    model = MultiTaskGNN(input_dim=12).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4) # 略微调低学习率

    print(">>> 正在训练 (重点关注航向角误差)...")
    for epoch in range(51):
        model.train()
        l_sum = 0
        for data in loader:
            data = data.to(device)
            optimizer.zero_grad()
            pred = model(data)
            
            # 这里的 500.0 是关键！由于 Delta_Yaw 数值极小，必须大幅增加权重
            loss_v = nn.MSELoss()(pred[:, 0], data.y[:, 0])
            loss_y = nn.MSELoss()(pred[:, 1], data.y[:, 1]) * 1000.0 
            
            loss = loss_v + loss_y
            loss.backward(); optimizer.step()
            l_sum += loss.item()
        if epoch % 10 == 0: print(f"Epoch {epoch:02d} | Loss: {l_sum/len(loader):.4f}")

    # --- 推理 data5 ---
    model.eval()
    test_feat, test_target = train_set._load_data(root, cat, "data5")
    feat_norm = (test_feat - train_set.stats['mean']) / train_set.stats['std']
    
    x_t = torch.tensor(feat_norm, dtype=torch.float32).to(device)
    idx = torch.arange(len(x_t))
    edge_index = torch.stack([idx[:-1], idx[1:]]).to(device)
    
    with torch.no_grad():
        preds = model(Data(x=x_t, edge_index=edge_index)).cpu().numpy()
        pred_v = preds[:, 0] / 100.0
        pred_dy = preds[:, 1]

    # --- 轨迹重建 ---
    refiner = PhysicsHeadingRefiner(window_size=5)
    # 获取初始角度：从真值前10帧的平均位移矢量计算
    gt_df = pd.read_csv(os.path.join(root, cat, "data5/syn/vi1.csv"), header=None)
    gt_p = gt_df.iloc[:, 2:4].values - gt_df.iloc[0, 2:4].values
    init_yaw = np.arctan2(gt_p[10, 1], gt_p[10, 0]) 
    
    curr_yaw = init_yaw
    traj = [[0, 0]]
    for i in range(len(pred_v)):
        safe_dy = refiner.refine(pred_v[i], pred_dy[i])
        curr_yaw += safe_dy # 这里的累加决定了轨迹是否会转弯
        traj.append([traj[-1][0] + pred_v[i]*np.cos(curr_yaw), 
                     traj[-1][1] + pred_v[i]*np.sin(curr_yaw)])
    
    traj = np.array(traj)

    plt.figure(figsize=(8, 8))
    plt.plot(gt_p[:, 0], gt_p[:, 1], 'g-', label='Ground Truth')
    plt.plot(traj[:, 0], traj[:, 1], 'r--', label='GNN Trajectory')
    plt.axis('equal'); plt.legend(); plt.title("Autonomous Turn Test"); plt.show()

if __name__ == "__main__":
    main()
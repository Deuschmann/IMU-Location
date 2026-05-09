import pandas as pd
import numpy as np
import os
import torch
import torch.nn as nn
from torch_geometric.nn import GATv2Conv, BatchNorm
from torch_geometric.loader import DataLoader as PyGDataLoader
from torch_geometric.data import Data
import matplotlib.pyplot as plt

# ==========================================
# 1. 核心特征提取 (严格的实时因果版本)
# ==========================================
def extract_imu_features_realtime(acc, gyro):
    acc_mag = np.linalg.norm(acc, axis=1, keepdims=True)
    
    # 方差计算只看过去 (center=False)
    acc_var = pd.Series(acc_mag.flatten()).rolling(window=10, min_periods=1).var().fillna(0).values.reshape(-1, 1)
    
    # 实时重力估计 (一阶低通滤波器 EMA，绝对不看未来)
    alpha = 0.05 
    gravity = np.zeros_like(acc)
    gravity[0] = acc[0]
    for i in range(1, len(acc)):
        gravity[i] = alpha * acc[i] + (1 - alpha) * gravity[i-1]
        
    gravity_unit = gravity / (np.linalg.norm(gravity, axis=1, keepdims=True) + 1e-6)
    
    # 坐标系对齐
    acc_v = np.sum(acc * gravity_unit, axis=1, keepdims=True) 
    acc_h = np.linalg.norm(acc - acc_v * gravity_unit, axis=1, keepdims=True) 
    gyro_v = np.sum(gyro * gravity_unit, axis=1, keepdims=True) 
    
    feat = np.hstack([acc, gyro, acc_mag, np.linalg.norm(gyro, axis=1, keepdims=True), acc_v, acc_h, gyro_v, acc_var])
    return feat.astype(np.float32)

# ==========================================
# 2. Stage 1: 局部去噪 MLP
# ==========================================
class Stage1MLP(nn.Module):
    def __init__(self, input_dim=12, window_size=11):
        super(Stage1MLP, self).__init__()
        self.flatten_dim = input_dim * window_size
        self.net = nn.Sequential(
            nn.Linear(self.flatten_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        # 使用 Softplus 保证输出为正，且无物理上限
        self.out_layer = nn.Sequential(nn.Linear(32, 1), nn.Softplus())
        
    def forward(self, x):
        h = self.net(x.reshape(x.size(0), -1))
        return h, self.out_layer(h) 

# ==========================================
# 3. Stage 2: 乘法缩放 GNN
# ==========================================
class Stage2GNN(nn.Module):
    def __init__(self, stage1_model):
        super(Stage2GNN, self).__init__()
        self.stage1 = stage1_model
        for param in self.stage1.parameters(): param.requires_grad = False
            
        self.conv1 = GATv2Conv(32, 32, heads=2, concat=False)
        self.bn1 = BatchNorm(32)
        self.conv2 = GATv2Conv(32, 16, heads=1)
        self.scale_head = nn.Linear(16, 1)

    def forward(self, data):
        latent, base_step = self.stage1(data.x)
        h = torch.relu(self.bn1(self.conv1(latent, data.edge_index)))
        h = torch.relu(self.conv2(h, data.edge_index))
        scale_val = self.scale_head(h)
        
        # 使用 torch.exp 进行无界正向缩放
        return base_step * torch.exp(scale_val)

# ==========================================
# 4. K-Hop 因果图数据集构建
# ==========================================
class PDRDataset:
    def __init__(self, root_dir, mode='train', window_size=11, seq_len=100, stats=None):
        self.data_list = []
        raw_feats = []
        categories = ["handheld", "slow walking"]

        for cat in categories:
            list_f = "Train.txt" if mode == 'train' else "Test.txt"
            p = os.path.join(root_dir, cat, list_f)
            if not os.path.exists(p): continue
            with open(p, 'r') as f:
                folders = [l.strip() for l in f.readlines() if l.strip()]

            for folder in folders:
                base = os.path.join(root_dir, cat, folder, 'syn')
                if not os.path.exists(os.path.join(base, 'vi1.csv')): continue
                
                imu = pd.read_csv(os.path.join(base, 'imu1.csv'), header=None).iloc[::4]
                gt = pd.read_csv(os.path.join(base, 'vi1.csv'), header=None).iloc[::4]
                min_l = min(len(imu), len(gt))
                
                feat = extract_imu_features_realtime(imu.iloc[:min_l, 4:7].values, imu.iloc[:min_l, 7:10].values)
                
                # 目标放大 100 倍 (米变厘米)，完美解决梯度消失
                dist = np.insert(np.linalg.norm(np.diff(gt.iloc[:min_l, 2:4].values, axis=0), axis=1), 0, 0).astype(np.float32)
                dist = dist * 100.0  
                
                raw_feats.append(feat)
                
                if mode == 'train':
                    for i in range(window_size - 1, len(feat) - seq_len, 50):
                        nodes_x = [feat[t-window_size+1 : t+1] for t in range(i, i+seq_len)]
                        src, dst = [], []
                        for t in range(seq_len):
                            for n in range(max(0, t - 5), t):
                                src.append(n); dst.append(t) # 保证因果
                                
                        self.data_list.append(Data(
                            x=torch.tensor(np.array(nodes_x), dtype=torch.float32), 
                            edge_index=torch.tensor([src, dst], dtype=torch.long) if len(src)>0 else torch.empty((2,0), dtype=torch.long), 
                            y=torch.tensor(dist[i : i+seq_len], dtype=torch.float32).view(-1, 1)
                        ))

        if mode == 'train':
            all_f = np.concatenate(raw_feats)
            self.stats = {'mean': all_f.mean(0), 'std': all_f.std(0)+1e-6}
        else: self.stats = stats

        m, s = torch.tensor(self.stats['mean']), torch.tensor(self.stats['std'])
        for d in self.data_list: d.x = (d.x - m) / s
    def __len__(self): return len(self.data_list)
    def __getitem__(self, idx): return self.data_list[idx]

# ==========================================
# 5. 主程序: 训练、推理与多维量化评估
# ==========================================
def main():
    root = "./Dataset"
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    window_size = 11
    
    print(">>> 正在加载并处理因果数据集...")
    train_set = PDRDataset(root, mode='train', window_size=window_size)
    train_loader = PyGDataLoader(train_set, batch_size=32, shuffle=True)
    
    s1_model = Stage1MLP(window_size=window_size).to(device)
    opt_s1 = torch.optim.Adam(s1_model.parameters(), lr=1e-3)
    
    print("\n>>> 训练 Stage 1 (提取基准特征)...")
    for epoch in range(30):
        s1_model.train()
        for data in train_loader:
            _, pred = s1_model(data.x.to(device))
            loss = nn.L1Loss()(pred, data.y.to(device)) # L1 Loss
            opt_s1.zero_grad(); loss.backward(); opt_s1.step()

    model = Stage2GNN(s1_model).to(device)
    opt_s2 = torch.optim.AdamW(model.parameters(), lr=1e-3)
    print("\n>>> 训练 Stage 2 (GNN 步长精校准 - 因果图)...")
    for epoch in range(31):
        model.train(); l_sum = 0
        for data in train_loader:
            data = data.to(device)
            out = model(data)
            
            loss_step = nn.L1Loss()(out, data.y)
            loss_curve = nn.L1Loss()(torch.cumsum(out, dim=0), torch.cumsum(data.y, dim=0))
            loss = loss_step + 2.0 * loss_curve # 强调曲线贴合
            
            opt_s2.zero_grad(); loss.backward(); opt_s2.step()
            l_sum += loss.item()
        if epoch % 10 == 0: print(f"Stage 2 Epoch {epoch} | Loss: {l_sum/len(train_loader):.6f}")

    # ================= 真实数据推理 =================
    s1_model.eval()
    model.eval()
    
    test_p = os.path.join(root, "handheld", "data5", "syn")
    print(f"\n>>> 正在生成测试文件轨迹: {test_p}")
    imu_df = pd.read_csv(os.path.join(test_p, 'imu1.csv'), header=None).iloc[::4].reset_index(drop=True)
    gt_df = pd.read_csv(os.path.join(test_p, 'vi1.csv'), header=None).iloc[::4].reset_index(drop=True)
    
    feat = extract_imu_features_realtime(imu_df.iloc[:, 4:7].values, imu_df.iloc[:, 7:10].values)
    feat_norm = (feat - train_set.stats['mean']) / train_set.stats['std']
    
    nodes_x = [feat_norm[t-window_size+1 : t+1] for t in range(window_size-1, len(feat_norm))]
    x_t = torch.tensor(np.array(nodes_x), dtype=torch.float32).to(device)
    
    src, dst = [], []
    for t in range(len(x_t)):
        for n in range(max(0, t - 5), t):
            src.append(n); dst.append(t)
    edge_t = torch.tensor([src, dst], dtype=torch.long).to(device)
    
    with torch.no_grad():
        # 提取模型距离，除以 100 恢复为“米”
        gnn_preds = model(Data(x=x_t, edge_index=edge_t)).cpu().numpy().flatten() / 100.0

    # ================= 终极 2D 轨迹生成 =================
    # 1. 提取并对齐真值轨迹点 (Ground Truth)
    gt_p = gt_df.iloc[:, 2:4].values
    gt_dist_all = np.linalg.norm(np.diff(gt_p, axis=0), axis=1)
    gt_dist_all = np.insert(gt_dist_all, 0, 0)
    gt_dist = gt_dist_all[window_size - 1 : window_size - 1 + len(gnn_preds)]
    
    gt_traj = gt_p - gt_p[0] # 从 (0,0) 开始计算
    gt_traj_aligned = gt_traj[window_size - 1 : window_size - 1 + len(gnn_preds)]
    gt_traj_aligned = gt_traj_aligned - gt_traj_aligned[0] # 以截断后的起点重置为 (0,0)
    
    # 2. 计算每一帧的真实航向角 (True Yaw)
    gt_dx = np.diff(gt_traj_aligned[:, 0], prepend=gt_traj_aligned[1, 0] - gt_traj_aligned[0, 0])
    gt_dy = np.diff(gt_traj_aligned[:, 1], prepend=gt_traj_aligned[1, 1] - gt_traj_aligned[0, 1])
    gt_true_yaw_aligned = np.arctan2(gt_dy, gt_dx)
    
    # 3. 使用【预测距离 + 真实航向角】生成理论完美轨迹
    traj_cheat = np.stack([
        np.cumsum(gnn_preds * np.cos(gt_true_yaw_aligned)),
        np.cumsum(gnn_preds * np.sin(gt_true_yaw_aligned))
    ], axis=1)

    # 4. 提取手机原始 Yaw，并进行最优初始角度对齐
    imu_raw_yaw = np.unwrap(imu_df.iloc[window_size - 1 : window_size - 1 + len(gnn_preds), 3].values)
    
    best_rmse = float('inf')
    best_traj_real = None
    best_off = 0
    
    angles = np.linspace(0, 2 * np.pi, 360)
    for off in angles:
        yaw_temp = imu_raw_yaw + off
        traj_temp = np.stack([
            np.cumsum(gnn_preds * np.cos(yaw_temp)),
            np.cumsum(gnn_preds * np.sin(yaw_temp))
        ], axis=1)
        
        rmse = np.sqrt(np.mean(np.sum((traj_temp - gt_traj_aligned)**2, axis=1)))
        if rmse < best_rmse:
            best_rmse = rmse
            best_traj_real = traj_temp
            best_off = off

    # ================= 多维度量化评估与报告 =================
    print("\n" + "="*50)
    print(">>> 📊 深度学习 PDR 模型多维度量化评估报告 <<<")
    print("="*50)

    # [1] 距离量化
    dist_mae = np.mean(np.abs(gnn_preds - gt_dist))
    dist_rmse = np.sqrt(np.mean((gnn_preds - gt_dist)**2))
    total_pred_dist = np.sum(gnn_preds)
    total_true_dist = np.sum(gt_dist)
    dist_error_ratio = np.abs(total_pred_dist - total_true_dist) / total_true_dist * 100

    print(f"\n[1] 步长与距离量化 (Model Core Performance):")
    print(f"    - 逐帧平均绝对误差 (MAE):  {dist_mae:.4f} 米/帧")
    print(f"    - 逐帧均方根误差 (RMSE): {dist_rmse:.4f} 米/帧")
    print(f"    - 累积总距离误差:        {np.abs(total_pred_dist - total_true_dist):.2f} 米 (真实 {total_true_dist:.2f}m vs 预测 {total_pred_dist:.2f}m)")
    print(f"    - 累积总距离误差率:      {dist_error_ratio:.2f} %  <-- (低于 5% 为业内顶尖)")

    # [2] 航向角量化
    best_aligned_yaw = imu_raw_yaw + best_off
    # 计算角度差，限制在 [-pi, pi]
    yaw_diff = np.abs(np.arctan2(np.sin(gt_true_yaw_aligned - best_aligned_yaw), np.cos(gt_true_yaw_aligned - best_aligned_yaw)))
    yaw_mae_deg = np.rad2deg(np.mean(yaw_diff))
    yaw_max_deg = np.rad2deg(np.max(yaw_diff))

    print(f"\n[2] 航向角误差量化 (Hardware/Sensor Limit):")
    print(f"    - 平均航向角误差:        {yaw_mae_deg:.2f} 度")
    print(f"    - 最大航向角漂移:        {yaw_max_deg:.2f} 度  <-- (说明陀螺仪漂移有多严重)")

    # [3] 定位点误差量化
    error_ceiling = np.linalg.norm(traj_cheat - gt_traj_aligned, axis=1)
    error_real = np.linalg.norm(best_traj_real - gt_traj_aligned, axis=1)

    print(f"\n[3] 2D 轨迹点误差量化 (Final Trajectory Error):")
    print(f"  > 模型上限 (完美 Yaw + 预测距离):")
    print(f"    - 平均定位误差 (ATE):    {np.mean(error_ceiling):.2f} 米")
    print(f"    - 90% 误差分位数 (CEP90):{np.percentile(error_ceiling, 90):.2f} 米")
    print(f"  > 工程现实 (原生 Yaw + 预测距离):")
    print(f"    - 平均定位误差 (ATE):    {np.mean(error_real):.2f} 米")
    print(f"    - 90% 误差分位数 (CEP90):{np.percentile(error_real, 90):.2f} 米")
    print("="*50)

    # ================= 图形可视化展示 =================
    
    # --- 图 1: 2D 最终轨迹地图 ---
    plt.figure(figsize=(10, 10))
    plt.plot(gt_traj_aligned[:,0], gt_traj_aligned[:,1], 'g', label='Ground Truth', linewidth=5, alpha=0.5)
    plt.plot(traj_cheat[:,0], traj_cheat[:,1], 'b--', label=f'GNN Dist + True Yaw (ATE: {np.mean(error_ceiling):.2f}m)', linewidth=2)
    plt.plot(best_traj_real[:,0], best_traj_real[:,1], 'r-.', label=f'GNN Dist + Raw Yaw (ATE: {np.mean(error_real):.2f}m)', linewidth=2)
    plt.scatter(0, 0, color='black', marker='*', s=250, label='Start Point', zorder=5)
    plt.title("Final 2D PDR Trajectory Mapping", fontsize=14)
    plt.xlabel("X (meters)", fontsize=12)
    plt.ylabel("Y (meters)", fontsize=12)
    plt.axis('equal') 
    plt.legend(fontsize=11, loc='upper right')
    plt.grid(True, linestyle='--', alpha=0.7)

    # --- 图 2: 误差 CDF 与 漂移趋势 ---
    sorted_err_ceiling = np.sort(error_ceiling)
    sorted_err_real = np.sort(error_real)
    p = 1. * np.arange(len(error_ceiling)) / (len(error_ceiling) - 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # 子图 1: CDF
    ax1.plot(sorted_err_ceiling, p, 'b-', linewidth=3, label=f'Model Ceiling')
    ax1.plot(sorted_err_real, p, 'r--', linewidth=2, label=f'Raw IMU Yaw')
    ax1.axhline(0.9, color='gray', linestyle=':', label='90% CDF Line')
    ax1.set_title("CDF of Positioning Error", fontsize=13)
    ax1.set_xlabel("Positioning Error (meters)")
    ax1.set_ylabel("Cumulative Probability (CDF)")
    ax1.legend(loc='lower right')
    ax1.grid(True)

    # 子图 2: 航向角漂移随时间的变化
    ax2.plot(np.rad2deg(yaw_diff), 'purple', linewidth=1.5, alpha=0.6, label='Instant Yaw Error')
    trend = pd.Series(np.rad2deg(yaw_diff)).rolling(window=200, min_periods=1).mean()
    ax2.plot(trend, 'black', linewidth=3, label='Drift Trend (Moving Avg)')
    ax2.set_title("Heading (Yaw) Error Over Time", fontsize=13)
    ax2.set_xlabel("Time (Frames)")
    ax2.set_ylabel("Absolute Yaw Error (Degrees)")
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    # 统一显示所有的图表
    plt.show()

if __name__ == "__main__":
    main()
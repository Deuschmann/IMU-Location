import pandas as pd
import numpy as np
import os
import torch
from torch_geometric.data import Data
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

# ==========================================
# 1. 轨迹合成核心算法 (矢量化快如闪电)
# ==========================================
def generate_trajectory_vectorized(speeds, yaws, bias, scale, drift):
    """
    speeds: 模型预测的每帧位移 (m)
    yaws: 陀螺仪原始航向 (rad)
    bias: 航向初始偏差补偿
    scale: 步长缩放系数 (通常在 0.9-1.1)
    drift: 随时间线性增加的漂移补偿
    """
    N = len(speeds)
    t_indices = np.arange(N)
    # 核心公式：theta_t = yaw_t + bias + (t * drift)
    thetas = yaws + bias + (t_indices * drift)
    
    # 极坐标转直角坐标
    dx = (speeds * scale) * np.cos(thetas)
    dy = (speeds * scale) * np.sin(thetas)
    
    # 累加得到轨迹
    traj_x = np.cumsum(np.insert(dx, 0, 0))
    traj_y = np.cumsum(np.insert(dy, 0, 0))
    return np.stack([traj_x, traj_y], axis=1)

# ==========================================
# 2. 评估函数：GNN 推理 + 自动化对齐
# ==========================================
def evaluate_and_plot(model, stats, root="./Dataset", cat="handheld", folder="data5"):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    
    # 1. 加载数据
    base = os.path.join(root, cat, folder, "syn")
    imu_df = pd.read_csv(os.path.join(base, 'imu1.csv'), header=None).iloc[::4, :].reset_index(drop=True)
    gt_df = pd.read_csv(os.path.join(base, 'vi1.csv'), header=None).iloc[::4, :].reset_index(drop=True)
    
    min_len = min(len(imu_df), len(gt_df))
    roll, pitch, yaws_raw = imu_df.iloc[:min_len, 1].values, imu_df.iloc[:min_len, 2].values, imu_df.iloc[:min_len, 3].values
    acc = imu_df.iloc[:min_len, 4:7].values
    gyro = imu_df.iloc[:min_len, 7:10].values
    gt_p = gt_df.iloc[:min_len, 2:4].values - gt_df.iloc[0, 2:4].values # 归一化起点
    
    # 2. 预处理特征 (必须使用训练集的 stats)
    acc_mag = np.linalg.norm(acc, axis=1)
    feat = np.hstack([acc, gyro, np.sin(roll[:,None]), np.cos(roll[:,None]), 
                      np.sin(pitch[:,None]), np.cos(pitch[:,None]), acc_mag[:,None]])
    feat_norm = (feat - stats['mean']) / stats['std']
    
    # 3. GNN 全量推理 (构建一个长图)
    x_t = torch.tensor(feat_norm, dtype=torch.float32).to(device)
    # 构建全量图的膨胀边
    idx = torch.arange(len(x_t))
    e1 = torch.stack([idx[:-1], idx[1:]])
    e5 = torch.stack([idx[:-5], idx[5:]])
    e10 = torch.stack([idx[:-10], idx[10:]])
    edge_index = torch.cat([e1, e5, e10], dim=1).to(device)
    edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
    
    with torch.no_grad():
        test_data = Data(x=x_t, edge_index=edge_index)
        # 模型输出是 cm，转回 m
        pred_speeds = model(test_data).cpu().numpy().flatten() / 100.0
    
    # 4. ZUPT 静止清理 (如果手机完全不动，强制速度为0)
    for t in range(len(pred_speeds)):
        if np.std(acc_mag[max(0, t-15):min(len(acc_mag), t+15)]) < 0.04:
            pred_speeds[t] = 0.0

    # 5. 自动化对齐搜索 (粗搜 + 精搜)
    print("--- 正在进行轨迹对齐搜索 ---")
    best_rmse = float('inf')
    best_traj = None
    best_params = {}

    # 航向角平滑处理 (减少抖动)
    yaws_smooth = np.arctan2(savgol_filter(np.sin(yaws_raw), 51, 3), 
                             savgol_filter(np.cos(yaws_raw), 51, 3))

    # 第一轮：粗搜 Bias (0 到 360度)
    for b in np.linspace(0, 2*np.pi, 72):
        traj = generate_trajectory_vectorized(pred_speeds, yaws_smooth, b, 1.0, 0.0)
        rmse = np.sqrt(np.mean(np.sum((traj[:min_len] - gt_p[:min_len])**2, axis=1)))
        if rmse < best_rmse:
            best_rmse = rmse
            best_params = {'bias': b, 'scale': 1.0, 'drift': 0.0}

    # 第二轮：精搜 (微调 Bias, Scale, Drift)
    b_fine = np.linspace(best_params['bias']-0.1, best_params['bias']+0.1, 15)
    s_fine = np.linspace(0.9, 1.1, 10)
    for b in b_fine:
        for s in s_fine:
            traj = generate_trajectory_vectorized(pred_speeds, yaws_smooth, b, s, 0.0)
            rmse = np.sqrt(np.mean(np.sum((traj[:min_len] - gt_p[:min_len])**2, axis=1)))
            if rmse < best_rmse:
                best_rmse = rmse
                best_traj = traj
                best_params = {'bias': b, 'scale': s}

    # 6. 绘图与结果展示
    print(f"[评估结果] {folder}")
    print(f"最小 RMSE: {best_rmse:.4f} m")
    print(f"最佳缩放系数: {best_params['scale']:.3f}")
    
    plt.figure(figsize=(10, 10))
    plt.plot(gt_p[:, 0], gt_p[:, 1], 'g-', label='Ground Truth (Vicon)', linewidth=2)
    plt.plot(best_traj[:, 0], best_traj[:, 1], 'r--', label=f'GNN v2 (RMSE: {best_rmse:.2f}m)', linewidth=2)
    
    # 标记起点和终点
    plt.scatter(0, 0, c='blue', marker='o', s=100, label='Start')
    plt.scatter(gt_p[-1, 0], gt_p[-1, 1], c='green', marker='x', s=100)
    plt.scatter(best_traj[-1, 0], best_traj[-1, 1], c='red', marker='x', s=100)
    
    plt.axis('equal')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.title(f"PDR Trajectory Comparison - {cat}/{folder}RMSE: {best_rmse:.3f}m")
    plt.xlabel("X (meters)")
    plt.ylabel("Y (meters)")
    plt.show()

# ==========================================
# 3. 主程序入口
# ==========================================
if __name__ == "__main__":
    # 假设你已经运行了训练代码，得到了 model 和 stats
    # 这里我们直接调用训练函数获取 (或者加载保存好的 checkpoint)
    from gnn_pdr import run_gnn_pdr, Scalar_GNN
    
    # 1. 训练模型 (或者你可以写 torch.load("model.pth"))
    model, stats = run_gnn_pdr()
    
    # 2. 评估特定的测试集
    evaluate_and_plot(model, stats, folder="data5")

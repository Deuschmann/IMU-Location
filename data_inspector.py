import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
import os

def inspect_oxiod_data(root_dir, category, data_folder):
    # 1. 路径设置
    base_path = os.path.join(root_dir, category, data_folder, 'syn')
    imu_path = os.path.join(base_path, 'imu1.csv')
    gt_path = os.path.join(base_path, 'vi1.csv' if category != 'large scale' else 'tango1.csv')

    # 2. 读取原始数据
    imu_cols = ['time', 'roll', 'pitch', 'yaw', 'gr_x', 'gr_y', 'gr_z', 
                'g_x', 'g_y', 'g_z', 'acc_x', 'acc_y', 'acc_z', 'm_x', 'm_y', 'm_z']
    df_imu = pd.read_csv(imu_path, header=None, names=imu_cols).iloc[::4, :].reset_index(drop=True)
    df_gt = pd.read_csv(gt_path, header=None, names=['t','h','x','y','z','qx','qy','qz','qw']).iloc[::4, :].reset_index(drop=True)
    
    min_len = min(len(df_imu), len(df_gt))
    dt = 0.04

    # 3. 模拟 dataset_cls.py 的处理逻辑
    acc = df_imu[['acc_x', 'acc_y', 'acc_z']].values[:min_len]
    acc_mag = np.linalg.norm(acc, axis=1) # 合加速度
    yaws = df_imu['yaw'].values[:min_len]
    
    # 世界系原始位移 (GT)
    pos_xy_w = df_gt[['x', 'y']].values[:min_len]
    pos_xy_w -= pos_xy_w[0] # 起点归零
    
    dx_w = np.diff(pos_xy_w[:, 0], prepend=0)
    dy_w = np.diff(pos_xy_w[:, 1], prepend=0)

    # --- 核心转换逻辑 (机体系) ---
    # Forward (前进) 和 Lateral (横向)
    dx_b = dx_w * np.cos(yaws) + dy_w * np.sin(yaws)
    dy_b = -dx_w * np.sin(yaws) + dy_w * np.cos(yaws)

    # --- 逆转换逻辑 (验证数学公式是否写反) ---
    # 从 Body 还原回 World
    rev_dx_w = dx_b * np.cos(yaws) - dy_b * np.sin(yaws)
    rev_dy_w = dx_b * np.sin(yaws) + dy_b * np.cos(yaws)
    rev_pos_w = np.cumsum(np.stack([rev_dx_w, rev_dy_w], axis=1), axis=0)

    # 4. 开始可视化诊断
    plt.figure(figsize=(16, 10))

    # 子图 1: 加速度模长 (验证步频)
    plt.subplot(2, 2, 1)
    plt.plot(acc_mag[200:400], label='Acc Magnitude')
    plt.title("Step Pattern (Gait) - Is there a vibration?")
    plt.xlabel("Frames")
    plt.grid(True)
    plt.legend()

    # 子图 2: 机体系位移分布 (验证 Forward 逻辑)
    plt.subplot(2, 2, 2)
    plt.plot(dx_b[:1000], label='Body Forward (dx_b)', alpha=0.7)
    plt.plot(dy_b[:1000], label='Body Lateral (dy_b)', alpha=0.7)
    plt.axhline(0, color='black', linestyle='--')
    plt.title("Body-Frame Displacement (米)")
    plt.ylabel("Meters per frame")
    plt.legend()

    # 子图 3: 逆转换闭环测试 (验证旋转矩阵)
    plt.subplot(2, 2, 3)
    plt.plot(pos_xy_w[:, 0], pos_xy_w[:, 1], 'g-', label='Original GT (World)', linewidth=3)
    plt.plot(rev_pos_w[:, 0], rev_pos_w[:, 1], 'r--', label='Reconstructed from Body-Deltas', linewidth=1)
    plt.axis('equal')
    plt.title("Math Verification: World -> Body -> World")
    plt.legend()

    # 子图 4: 位移的统计直方图
    plt.subplot(2, 2, 4)
    plt.hist(dx_b, bins=50, color='blue', alpha=0.5, label='Forward Delta')
    plt.hist(dy_b, bins=50, color='red', alpha=0.5, label='Lateral Delta')
    plt.title("Displacement Distribution Histogram")
    plt.legend()

    plt.tight_layout()
    plt.show()

    # 打印数值统计
    print(f"--- 数据审计报告 ---")
    print(f"平均每帧前进位移: {np.mean(dx_b):.6f} 米")
    print(f"最大单步位移: {np.max(dx_b):.6f} 米")
    print(f"数据总帧数: {min_len}")
    print(f"还原误差 (MSE): {np.mean((pos_xy_w - rev_pos_w)**2):.10f}")

if __name__ == "__main__":
    # 请根据实际路径修改
    inspect_oxiod_data("./Dataset", "handheld", "data1")
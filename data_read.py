import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

def inspect_imu_physics(root_dir, category, sequence="data1"):
    """
    检查 IMU 数据的物理意义和尺度
    """
    imu_cols = ['time', 'roll', 'pitch', 'yaw', 'gr_x', 'gr_y', 'gr_z', 
                'g_x', 'g_y', 'g_z', 'acc_x', 'acc_y', 'acc_z', 'm_x', 'm_y', 'm_z']
    gt_cols = ['time', 'header', 'x', 'y', 'z', 'qx', 'qy', 'qz', 'qw']
    
    imu_path = os.path.join(root_dir, category, sequence, 'syn', 'imu1.csv')
    gt_path = os.path.join(root_dir, category, sequence, 'syn', 'vi1.csv' if category != 'large scale' else 'tango1.csv')
    
    if not os.path.exists(imu_path):
        print(f"未找到文件: {imu_path}")
        return

    # 1. 读取并下采样 (dt = 0.04s)
    df_imu = pd.read_csv(imu_path, header=None, names=imu_cols).iloc[::4, :].reset_index(drop=True)
    df_gt = pd.read_csv(gt_path, header=None, names=gt_cols).iloc[::4, :].reset_index(drop=True)
    min_len = min(len(df_imu), len(df_gt))
    
    acc = df_imu[['acc_x', 'acc_y', 'acc_z']].values[:min_len]
    acc_mag = np.linalg.norm(acc, axis=1)
    
    pos_w = df_gt[['x', 'y']].values[:min_len]
    dx_w = np.diff(pos_w[:, 0], prepend=pos_w[0, 0])
    dy_w = np.diff(pos_w[:, 1], prepend=pos_w[0, 1])
    dt = 0.04
    
    # --- 绘图检查 ---
    fig, axs = plt.subplots(3, 1, figsize=(10, 12))
    
    # 检查 1: 加速度模长是否包含重力？
    # 如果 acc_mag 一直在 9.8 左右浮动，说明重力未去除！神经网络直接吃这个会消化不良。
    axs[0].plot(acc_mag, label='Acc Magnitude', color='red')
    axs[0].axhline(y=9.81, color='blue', linestyle='--', label='Gravity (9.81)')
    axs[0].set_title("Check 1: Does Acc contain Gravity? (Should be around 9.8 if raw, 0 if linear)")
    axs[0].set_ylabel("m/s^2")
    axs[0].legend()
    
    # 检查 2: 每步位移 (Target) 有多大？
    # 0.04s 的采样率下，常人步行 (1.5m/s) 每帧位移大概是 0.06m。
    displacement_mag = np.sqrt(dx_w**2 + dy_w**2)
    axs[1].plot(displacement_mag, label='Displacement per frame (0.04s)')
    axs[1].set_title("Check 2: Target Displacement Scale (Meters per frame)")
    axs[1].set_ylabel("Meters")
    axs[1].legend()
    
    # 检查 3: 算出地面的真实速度（看看数据是否连续）
    # 如果速度极度抖动，说明真值本身对齐不好，模型也没法学。
    vel_x = dx_w / dt
    vel_y = dy_w / dt
    axs[2].plot(np.sqrt(vel_x**2 + vel_y**2), label='Ground Truth Velocity')
    axs[2].set_title("Check 3: Ground Truth Velocity (m/s)")
    axs[2].set_ylabel("m/s")
    axs[2].legend()
    
    plt.tight_layout()
    plt.show()

# 运行检查
inspect_imu_physics("./Dataset", "handheld", "data1")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

def diagnose_data(root, cat, seq):
    # 1. 加载数据
    base_path = f"{root}/{cat}/{seq}/syn"
    imu = pd.read_csv(f"{base_path}/imu1.csv", header=None).iloc[::4, :].reset_index(drop=True)
    gt = pd.read_csv(f"{base_path}/vi1.csv", header=None).iloc[::4, :].reset_index(drop=True)
    
    min_len = min(len(imu), len(gt))
    # 提取时间、加速度、航向角、GT位置
    t = np.arange(min_len) * 0.04 # 25Hz -> 0.04s per frame
    acc_lin = imu.iloc[:min_len, 4:7].values # 线性加速度
    sensor_yaw = imu.iloc[:min_len, 3].values # 传感器 Yaw
    pos_gt = gt.iloc[:min_len, 2:4].values # GT XY
    
    # 2. 计算 GT 的真实动态属性
    # 计算瞬时位移和速度
    delta_pos = np.diff(pos_gt, axis=0, prepend=[pos_gt[0]])
    v_gt = np.linalg.norm(delta_pos, axis=1) / 0.04 # 瞬时速率 m/s
    
    # 计算 GT 的真实航向角 (Direction of Motion)
    # 注意：只有在移动时航向才有意义，所以我们要对 delta_pos 做一点平滑
    gt_dx = pd.Series(delta_pos[:, 0]).rolling(10).mean().fillna(0).values
    gt_dy = pd.Series(delta_pos[:, 1]).rolling(10).mean().fillna(0).values
    gt_yaw = np.arctan2(gt_dy, gt_dx)
    
    # 3. 计算 IMU 的特征
    acc_mag = np.linalg.norm(acc_lin, axis=1)
    
    # --- 开始绘图 ---
    fig, axes = plt.subplots(3, 1, figsize=(15, 18))
    
    # 图 1：航向对决 (Sensor Yaw vs GT Motion Angle)
    # 这张图能告诉你为什么 PDR 会偏：传感器给的角度和实际走的方向差了多少？
    axes[0].plot(t, np.degrees(sensor_yaw), label='Sensor Reported Yaw', alpha=0.8)
    axes[0].plot(t, np.degrees(gt_yaw), label='GT Actual Motion Angle', linewidth=2)
    axes[0].set_title(f"Heading Analysis: {seq}")
    axes[0].set_ylabel("Degrees")
    axes[0].legend()
    axes[0].grid(True)
    
    # 图 2：能量映射 (Accel Magnitude vs GT Speed)
    # 这张图能告诉你 Weinberg 公式准不准：加速度大的时候，速度真的大吗？
    ax2_twin = axes[1].twinx()
    axes[1].plot(t, acc_mag, color='r', alpha=0.5, label='Accel Magnitude (m/s^2)')
    ax2_twin.plot(t, v_gt, color='g', linewidth=2, label='GT Speed (m/s)')
    axes[1].set_title("Energy vs Speed Mapping")
    axes[1].set_ylabel("Acceleration")
    ax2_twin.set_ylabel("Speed (m/s)")
    axes[1].legend(loc='upper left'); ax2_twin.legend(loc='upper right')
    
    # 图 3：误差散点图 (Sensor Yaw Error)
    yaw_err = sensor_yaw - gt_yaw
    # 角度差归一化到 [-pi, pi]
    yaw_err = (yaw_err + np.pi) % (2 * np.pi) - np.pi
    axes[2].fill_between(t, np.degrees(yaw_err), color='orange', alpha=0.3)
    axes[2].axhline(0, color='black', linestyle='--')
    axes[2].set_title("Heading Error Over Time")
    axes[2].set_ylabel("Degrees Error")
    axes[2].set_xlabel("Time (s)")
    
    plt.tight_layout()
    plt.show()

# 执行诊断
diagnose_data("./Dataset", "handheld", "data5")
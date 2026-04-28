import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, savgol_filter

class PDRBaselineApp:
    def __init__(self, root, cat):
        self.root = root
        self.cat = cat

    def run_v8_logic(self, seq_name="data5"):
        # 1. 加载
        base_path = f"{self.root}/{self.cat}/{seq_name}/syn"
        imu = pd.read_csv(f"{base_path}/imu1.csv", header=None).iloc[::4, :].reset_index(drop=True)
        gt = pd.read_csv(f"{base_path}/vi1.csv", header=None).iloc[::4, :].reset_index(drop=True)
        min_len = min(len(imu), len(gt))

        acc_lin = imu.iloc[:min_len, 4:7].values
        yaws_raw = imu.iloc[:min_len, 3].values
        gt_pos = gt.iloc[:min_len, 2:4].values - gt.iloc[0, 2:4].values

        # 2. 步次检测 (回归最稳健的 Weinberg)
        acc_mag = np.linalg.norm(acc_lin, axis=1)
        acc_mag_f = savgol_filter(acc_mag, 11, 3) # 使用 Savitzky-Golay 滤噪，零延迟
        peaks, _ = find_peaks(acc_mag_f, height=0.6, distance=14)

        # 3. 航向处理 (同样使用 Savitzky-Golay，拒绝相位滞后)
        sin_y = savgol_filter(np.sin(yaws_raw), 31, 3)
        cos_y = savgol_filter(np.cos(yaws_raw), 31, 3)
        yaws_smooth = np.arctan2(sin_y, cos_y)

        # 4. 轨迹推算 (使用固定步长 0.7，先看形状)
        curr_x, curr_y = 0.0, 0.0
        pred_pos = np.zeros((min_len, 2))
        step_ptr = 0
        for i in range(min_len):
            if step_ptr < len(peaks) and i == peaks[step_ptr]:
                L = 0.7 # 回归固定步长，排除能量模型的干扰
                theta = yaws_smooth[i]
                curr_x += L * np.cos(theta)
                curr_y += L * np.sin(theta)
                step_ptr += 1
            pred_pos[i] = [curr_x, curr_y]

        # 5. 全局最优对齐 (这是看清数据“本质形状”的唯一方法)
        # 我们寻找一个旋转角度 alpha，使得预测轨迹和 GT 的重合度最高
        best_rmse = float('inf')
        best_pred = pred_pos
        for angle in np.linspace(0, 2*np.pi, 360):
            c, s = np.cos(angle), np.sin(angle)
            rot_m = np.array([[c, -s], [s, c]])
            temp_pred = (rot_m @ pred_pos.T).T
            rmse = np.sqrt(np.mean(np.sum((temp_pred - gt_pos)**2, axis=1)))
            if rmse < best_rmse:
                best_rmse = rmse
                best_pred = temp_pred

        print(f"--- v8.0 形状对齐版: {seq_name} ---")
        print(f"最优全局对齐 RMSE: {best_rmse:.2f}m")

        # 6. 绘图
        plt.figure(figsize=(10, 10))
        plt.plot(gt_pos[:, 0], gt_pos[:, 1], 'g', label='Ground Truth', linewidth=3)
        plt.plot(best_pred[:, 0], best_pred[:, 1], 'r--', label='PDR v8.0 (Best-Fit Shape)', alpha=0.9)
        plt.title(f"PDR v8.0 Shape Analysis: {seq_name}\nBest RMSE: {best_rmse:.2f}m")
        plt.legend(); plt.axis('equal'); plt.grid(True); plt.show()

if __name__ == "__main__":
    app = PDRBaselineApp("./Dataset", "handheld")
    app.run_v8_logic("data5")
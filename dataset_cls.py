import pandas as pd
import numpy as np
import os
import torch
from torch.utils.data import Dataset
from scipy.signal import find_peaks

class MLPDatasetOxIOD(Dataset):
    """
    针对 MLP 点对点回归设计的 Dataset
    输入: 单帧 6 维特征 [acc_mag, sin_yaw, cos_yaw, roll, pitch, acc_z]
    标签: 单帧位移 [delta_x, delta_y] (单位：米)
    """
    def __init__(self, root_dir, category, mode='train', stats=None):
        self.root_dir = root_dir
        self.category = category
        self.mode = mode
        self.stats = stats # 外部传入的归一化参数

        # OxIOD 标准列名
        self.imu_cols = ['time', 'roll', 'pitch', 'yaw', 'gr_x', 'gr_y', 'gr_z', 
                          'g_x', 'g_y', 'g_z', 'acc_x', 'acc_y', 'acc_z', 
                          'm_x', 'm_y', 'm_z']
        
        self.features = [] # 存储全量特征 [N, 6]
        self.targets = []  # 存储全量位移 [N, 2]

        # 确定读取列表
        list_file = "Train.txt" if mode == 'train' else "Test.txt"
        list_path = os.path.join(root_dir, category, list_file)
        
        if not os.path.exists(list_path):
            raise FileNotFoundError(f"未找到列表文件: {list_path}")
            
        with open(list_path, 'r') as f:
            folders = [line.strip() for line in f.readlines() if line.strip()]

        print(f"[{mode.upper()}] 正在加载 {len(folders)} 个序列数据...")
        
        for rel_path in folders:
            base_path = os.path.join(root_dir, category, rel_path)
            imu_p = os.path.join(base_path, 'syn', 'imu1.csv')
            gt_filename = 'vi1.csv' if category != 'large scale' else 'tango1.csv'
            gt_p = os.path.join(base_path, 'syn', gt_filename)
            
            if os.path.exists(imu_p) and os.path.exists(gt_p):
                self._process_file(imu_p, gt_p)

        self.features = np.vstack(self.features).astype(np.float32)
        self.targets = np.vstack(self.targets).astype(np.float32)

        # 如果是训练集且没有提供 stats，计算并存储 stats
        if self.mode == 'train' and self.stats is None:
            self.stats = self._compute_stats()
            
        print(f"数据加载完成。总样本数: {len(self.features)}")

    def _process_file(self, imu_path, gt_path):
        # 1. 读取数据并执行 4 倍下采样 (dt=0.04s)
        df_imu = pd.read_csv(imu_path, header=None, names=self.imu_cols).iloc[::4, :].reset_index(drop=True)
        df_gt = pd.read_csv(gt_path, header=None).iloc[::4, :].reset_index(drop=True)
        
        min_len = min(len(df_imu), len(df_gt))
        
        # --- 2. 提取输入特征 (6维) ---
        # 线性加速度 (使用 gr 列)
        acc_lin = df_imu[['gr_x', 'gr_y', 'gr_z']].values[:min_len]
        acc_mag = np.linalg.norm(acc_lin, axis=1, keepdims=True)
        
        # 航向角处理 (分解为 sin/cos 以消除 0/360 度突变)
        yaws = df_imu['yaw'].values[:min_len]
        yaw_sin = np.sin(yaws).reshape(-1, 1)
        yaw_cos = np.cos(yaws).reshape(-1, 1)
        
        # 姿态角 (Roll, Pitch) + 垂直加速度 (acc_z)
        attitudes = df_imu[['roll', 'pitch']].values[:min_len]
        acc_z = acc_lin[:, 2:3]
        
        feat_single = np.hstack([acc_mag, yaw_sin, yaw_cos, attitudes, acc_z])

        # --- 3. 提取位移标签 (2维: delta_x, delta_y) ---
        # 世界坐标系下的位置
        pos_xy = df_gt.iloc[:min_len, 2:4].values
        # 计算每一帧相对于上一帧的位移 (m)
        delta_pos = np.diff(pos_xy, axis=0, prepend=[pos_xy[0]])

        self.features.append(feat_single)
        self.targets.append(delta_pos)

    def _compute_stats(self):
        """计算特征的均值和标准差"""
        return {
            'feat_mean': np.mean(self.features, axis=0),
            'feat_std': np.std(self.features, axis=0) + 1e-6
        }

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        feat = self.features[idx].copy()
        target = self.targets[idx].copy()

        # 对输入特征进行 Z-Score 归一化
        if self.stats is not None:
            feat = (feat - self.stats['feat_mean']) / self.stats['feat_std']
        
        # 目标值 (delta_x, delta_y) 不建议减均值归一化，因为位移是矢量。
        # 这里放大 10 倍是为了让 Loss 量级更好看，训练更稳定。
        target = target * 10.0 

        return torch.tensor(feat, dtype=torch.float32), torch.tensor(target, dtype=torch.float32)



####################################################################################
####################################################################################
####################################################################################
####################################################################################
####################################################################################
####################################################################################

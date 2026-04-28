import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import os

class OxIODWindowDataset(Dataset):
    def __init__(self, root_dir, category, mode='train', window_size=20, stride=1, stats=None):
        self.window_size = window_size
        self.stride = stride
        self.mode = mode
        
        self.features = []
        self.targets = []
        
        # 加载文件列表
        list_file = "Train.txt" if mode == 'train' else "Test.txt"
        list_path = os.path.join(root_dir, category, list_file)
        with open(list_path, 'r') as f:
            folders = [line.strip() for line in f.readlines() if line.strip()]

        print(f"[{mode.upper()}] 正在预处理 {len(folders)} 个序列...")
        for rel_path in folders:
            self._process_sequence(root_dir, category, rel_path)

        self.features = np.array(self.features).astype(np.float32)
        self.targets = np.array(self.targets).astype(np.float32)

        # 归一化处理
        if mode == 'train':
            self.stats = {
                'mean': np.mean(self.features, axis=(0, 1)),
                'std': np.std(self.features, axis=(0, 1)) + 1e-6
            }
        else:
            self.stats = stats

    def _rotate_to_horizontal(self, acc_body, roll, pitch):
        """
        物理预处理：将手机坐标系加速度投影到水平导航坐标系
        """
        # 构建旋转矩阵 (简化版投影)
        # cp, sp = cos(pitch), sin(pitch); cr, sr = cos(roll), sin(roll)
        acc_nav = np.zeros_like(acc_body)
        for i in range(len(acc_body)):
            r, p = roll[i], pitch[i]
            # 旋转矩阵 Rx(roll) * Ry(pitch)
            Rx = np.array([[1, 0, 0],
                           [0, np.cos(r), -np.sin(r)],
                           [0, np.sin(r), np.cos(r)]])
            Ry = np.array([[np.cos(p), 0, np.sin(p)],
                           [0, 1, 0],
                           [-np.sin(p), 0, np.cos(p)]])
            R = Ry @ Rx
            acc_nav[i] = R @ acc_body[i]
        return acc_nav

    def _process_sequence(self, root, cat, rel_path):
        base = os.path.join(root, cat, rel_path, 'syn')
        imu_df = pd.read_csv(os.path.join(base, 'imu1.csv'), header=None).iloc[::4, :].reset_index(drop=True)
        gt_f = 'vi1.csv' if cat != 'large scale' else 'tango1.csv'
        gt_df = pd.read_csv(os.path.join(base, gt_f), header=None).iloc[::4, :].reset_index(drop=True)
        
        min_len = min(len(imu_df), len(gt_df))
        
        # 1. 提取基础维度
        roll = imu_df.iloc[:min_len, 1].values
        pitch = imu_df.iloc[:min_len, 2].values
        yaw = imu_df.iloc[:min_len, 3].values
        acc_body = imu_df.iloc[:min_len, 4:7].values  # gr_x, gr_y, gr_z
        gyro = imu_df.iloc[:min_len, 7:10].values    # g_x, g_y, g_z
        
        # 2. 物理投影特征 (3维)
        acc_nav = self._rotate_to_horizontal(acc_body, roll, pitch)
        
        # 3. 组合 14 维特征
        # [acc_body(3), gyro(3), acc_nav(3), sin_cos_rp(4), acc_mag(1)]
        feat = np.hstack([
            acc_body, 
            gyro, 
            acc_nav,
            np.sin(roll)[:,None], np.cos(roll)[:,None],
            np.sin(pitch)[:,None], np.cos(pitch)[:,None],
            np.linalg.norm(acc_body, axis=1)[:,None]
        ])
        
        # 4. 提取位移 Label (delta_x, delta_y)
        pos = gt_df.iloc[:min_len, 2:4].values
        
        # 5. 滑动窗口切分
        for i in range(0, min_len - self.window_size, self.stride):
            window_feat = feat[i : i + self.window_size]
            # 标签是整个窗口结束相对于开始的位移
            delta_p = pos[i + self.window_size] - pos[i]
            
            self.features.append(window_feat)
            self.targets.append(delta_p)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        x = (self.features[idx] - self.stats['mean']) / self.stats['mean']
        y = self.targets[idx] * 10.0 # 放大 Label 量级以稳定训练
        return torch.tensor(x), torch.tensor(y)
import jittor as jt
from jittor.dataset import Dataset 
import pandas as pd
from pathlib import Path
import torch 
import numpy as np
import random

class CamelDataJt(Dataset): 
    def __init__(self, dataset_cfg=None, 
                 state=None):
        super(CamelDataJt, self).__init__() 
        
        self.dataset_cfg = dataset_cfg
        self.state = state

        # ----> data and label
        self.nfolds = self.dataset_cfg.nfold
        self.fold = self.dataset_cfg.fold
        self.feature_dir = self.dataset_cfg.data_dir
        self.csv_dir = self.dataset_cfg.label_dir + f'fold{self.fold}.csv'
        self.slide_data = pd.read_csv(self.csv_dir, index_col=0)

        self.shuffle = self.dataset_cfg.data_shuffle

        # ----> 划分数据集 
        if state == 'train':
            self.data = self.slide_data.loc[:, 'train'].dropna().reset_index(drop=True)
            self.label = self.slide_data.loc[:, 'train_label'].dropna().reset_index(drop=True)
        elif state == 'val':
            self.data = self.slide_data.loc[:, 'val'].dropna().reset_index(drop=True)
            self.label = self.slide_data.loc[:, 'val_label'].dropna().reset_index(drop=True)
        elif state == 'test':
            self.data = self.slide_data.loc[:, 'test'].dropna().reset_index(drop=True)
            self.label = self.slide_data.loc[:, 'test_label'].dropna().reset_index(drop=True)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        slide_id = self.data[idx]
        label = int(self.label[idx])
        full_path = Path(self.feature_dir) / f'{slide_id}.pt'
        
        # 为了保险，先用 torch 加载再转 numpy，这样最稳
        # 除非你确定 jt.load 能够直接读取你的 .pt 文件
        features_np = torch.load(str(full_path)).detach().cpu().numpy()
        features = jt.array(features_np)

        # ----> 内部 shuffle 逻辑
        if self.shuffle == True:
            # 这里的逻辑与原代码完全对齐
            num_features = features.shape[0]
            index = list(range(num_features))
            random.shuffle(index)
            features = features[index]

        return features, label
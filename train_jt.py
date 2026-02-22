"""
train_jt.py
TransMIL 训练入口 — Jittor 1.3.10 版本
替代了原版 train.py 中的 pytorch_lightning.Trainer。
"""

import argparse
from pathlib import Path
import numpy as np
import glob

import jittor as jt

# 开启 GPU（如需使用）
jt.flags.use_cuda = 1

from datasets import DataInterface
from models.model_interface_jt import ModelInterface, JittorTrainer
from utils.utils_jt import read_yaml, load_loggers, load_callbacks   # utils_jt.py


def make_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage',  default='train', type=str)
    parser.add_argument('--config', default='Camelyon/TransMIL.yaml', type=str)
    parser.add_argument('--gpus',   default=[0], type=int, nargs='+')
    parser.add_argument('--fold',   default=0,   type=int)
    args = parser.parse_args()
    return args


def main(cfg):

    # ------ 随机种子 ------
    jt.set_global_seed(cfg.General.seed)
    np.random.seed(cfg.General.seed)

    # ------ Loggers & Callbacks ------
    cfg.load_loggers = load_loggers(cfg)
    cfg.callbacks    = load_callbacks(cfg)

    # ------ DataModule ------
    DataInterface_dict = {
        'train_batch_size':  cfg.Data.train_dataloader.batch_size,
        'train_num_workers': cfg.Data.train_dataloader.num_workers,
        'test_batch_size':   cfg.Data.test_dataloader.batch_size,
        'test_num_workers':  cfg.Data.test_dataloader.num_workers,
        'dataset_name':      cfg.Data.dataset_name,
        'dataset_cfg':       cfg.Data,
    }
    dm = DataInterface(**DataInterface_dict)

    # ------ Model ------
    ModelInterface_dict = {
        'model':     cfg.Model,
        'loss':      cfg.Loss,
        'optimizer': cfg.Optimizer,
        'data':      cfg.Data,
        'log':       cfg.log_path,
    }
    model = ModelInterface(**ModelInterface_dict)

    # ------ Trainer ------
    trainer = JittorTrainer(
        max_epochs=cfg.General.epochs,
        callbacks=cfg.callbacks,
        loggers=cfg.load_loggers,
        check_val_every_n_epoch=1,
        seed=cfg.General.seed,
    )

    # ------ Train / Test ------
    if cfg.General.server == 'train':
        trainer.fit(model=model, datamodule=dm, grad_acc_steps=1)
    else:
        # 找到所有 checkpoint 文件（Jittor 保存为 .pkl）
        model_paths = sorted(Path(cfg.log_path).glob('epoch*.pkl'))
        for path in model_paths:
            print(f'Testing with checkpoint: {path}')
            trainer.test(model=model, datamodule=dm, checkpoint_path=str(path))


if __name__ == '__main__':
    args = make_parse()
    cfg  = read_yaml(args.config)

    # 更新配置
    cfg.config         = args.config
    cfg.General.gpus   = args.gpus
    cfg.General.server = args.stage
    cfg.Data.fold      = args.fold

    main(cfg)
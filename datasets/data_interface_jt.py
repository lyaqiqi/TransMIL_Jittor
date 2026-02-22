"""
data_interface_jt.py
DataInterface — Ported to Jittor 1.3.10
移除了 pytorch_lightning.LightningDataModule 依赖，
改用 jittor.dataset.DataLoader。
"""

import inspect
import importlib
from jittor.dataset import DataLoader


class DataInterface:

    def __init__(self, train_batch_size=64, train_num_workers=8,
                 test_batch_size=1, test_num_workers=1,
                 dataset_name=None, **kwargs):

        self.train_batch_size  = train_batch_size
        self.train_num_workers = train_num_workers
        self.test_batch_size   = test_batch_size
        self.test_num_workers  = test_num_workers
        self.dataset_name      = dataset_name
        self.kwargs            = kwargs

        # 缓存，避免重复实例化
        self._train_dataset = None
        self._val_dataset   = None
        self._test_dataset  = None

        self.load_data_module()

    # ------------------------------------------------------------------
    # 与原版一致：动态加载 Dataset 类
    # ------------------------------------------------------------------
    def load_data_module(self):
        camel_name = ''.join([i.capitalize() for i in self.dataset_name.split('_')])
        try:
            self.data_module = getattr(
                importlib.import_module(f'datasets.{self.dataset_name}'),
                camel_name
            )
        except Exception:
            raise ValueError('Invalid Dataset File Name or Invalid Class Name!')

    def instancialize(self, **other_args):
        # getargspec 在 Python 3.11+ 已移除，统一用 getfullargspec
        class_args = inspect.getfullargspec(self.data_module.__init__).args[1:]
        inkeys = self.kwargs.keys()
        args1 = {}
        for arg in class_args:
            if arg in inkeys:
                args1[arg] = self.kwargs[arg]
        args1.update(other_args)
        return self.data_module(**args1)

    # ------------------------------------------------------------------
    # setup：原版由 Lightning 自动调用；这里改为懒加载（首次请求时触发）
    # ------------------------------------------------------------------
    def _setup_fit(self):
        if self._train_dataset is None:
            self._train_dataset = self.instancialize(state='train')
        if self._val_dataset is None:
            self._val_dataset = self.instancialize(state='val')

    def _setup_test(self):
        if self._test_dataset is None:
            self._test_dataset = self.instancialize(state='test')

    # ------------------------------------------------------------------
    # DataLoader 接口（供 JittorTrainer 调用，接口名与原版完全一致）
    # ------------------------------------------------------------------
    def train_dataloader(self):
        self._setup_fit()
        return DataLoader(
            self._train_dataset,
            batch_size=self.train_batch_size,
            num_workers=self.train_num_workers,
            shuffle=True,
        )

    def val_dataloader(self):
        self._setup_fit()
        return DataLoader(
            self._val_dataset,
            batch_size=self.train_batch_size,
            num_workers=self.train_num_workers,
            shuffle=False,
        )

    def test_dataloader(self):
        self._setup_test()
        return DataLoader(
            self._test_dataset,
            batch_size=self.test_batch_size,
            num_workers=self.test_num_workers,
            shuffle=False,
        )
"""
model_interface_jt.py
TransMIL Model Interface — Ported to Jittor 1.3.10
替代了 PyTorch Lightning 的 LightningModule，改用手动训练循环。
"""

import sys
import numpy as np
import inspect
import importlib
import random
import pandas as pd
from pathlib import Path

# ---->
from MyOptimizer import create_optimizer   # optim_factory_jt.py
from MyLoss import create_loss             # loss_factory_jt.py
from utils.utils_jt import cross_entropy_torch  # utils_jt.py

# ---->
import jittor as jt
from jittor import nn

# ============================================================
# torchmetrics 替代：基于 jittor 的轻量指标实现
# 如果你安装了 jittor_metrics 包，可以直接 import；
# 否则使用下方的手动实现（已覆盖本项目所需的全部指标）。
# ============================================================
try:
    import jittor_metrics as jmetrics
    _USE_JMETRICS = True
except ImportError:
    _USE_JMETRICS = False

# ---------- 手动指标实现（fallback） ----------
def _to_numpy(x):
    if isinstance(x, jt.Var):
        return x.numpy()
    return np.array(x)

def compute_accuracy(preds, targets):
    """micro accuracy"""
    preds = _to_numpy(preds).flatten()
    targets = _to_numpy(targets).flatten()
    return float(np.mean(preds == targets))

def compute_cohen_kappa(preds, targets, n_classes):
    from sklearn.metrics import cohen_kappa_score
    return cohen_kappa_score(_to_numpy(preds).flatten(), _to_numpy(targets).flatten())

def compute_f1(preds, targets, n_classes, average='macro'):
    from sklearn.metrics import f1_score
    return f1_score(_to_numpy(targets).flatten(), _to_numpy(preds).flatten(),
                    average=average, zero_division=0)

def compute_recall(preds, targets, n_classes, average='macro'):
    from sklearn.metrics import recall_score
    return recall_score(_to_numpy(targets).flatten(), _to_numpy(preds).flatten(),
                        average=average, zero_division=0)

def compute_precision(preds, targets, n_classes, average='macro'):
    from sklearn.metrics import precision_score
    return precision_score(_to_numpy(targets).flatten(), _to_numpy(preds).flatten(),
                           average=average, zero_division=0)

def compute_specificity(preds, targets, n_classes):
    """macro specificity（每类的 TN/（TN+FP） 均值）"""
    from sklearn.metrics import confusion_matrix
    p = _to_numpy(preds).flatten()
    t = _to_numpy(targets).flatten()
    cm = confusion_matrix(t, p, labels=list(range(n_classes)))
    specificities = []
    for i in range(n_classes):
        tn = cm.sum() - cm[i, :].sum() - cm[:, i].sum() + cm[i, i]
        fp = cm[:, i].sum() - cm[i, i]
        specificities.append(tn / (tn + fp) if (tn + fp) > 0 else 0.0)
    return float(np.mean(specificities))

def compute_auroc(probs, targets, n_classes, average='macro'):
    from sklearn.metrics import roc_auc_score
    t = _to_numpy(targets).flatten()
    p = _to_numpy(probs)
    if n_classes == 2:
        # probs shape: (N, 2) 或 (N,)
        if p.ndim == 2:
            p = p[:, 1]
        return roc_auc_score(t, p)
    else:
        return roc_auc_score(t, p, multi_class='ovr', average=average)

def compute_all_metrics(preds, probs, targets, n_classes, prefix='val_'):
    """
    统一计算所有评估指标，返回 dict。
    preds:   (N,) 预测类别索引
    probs:   (N, C) 预测概率
    targets: (N,) 真实类别索引
    """
    result = {}
    result[f'{prefix}Accuracy']    = compute_accuracy(preds, targets)
    result[f'{prefix}CohenKappa']  = compute_cohen_kappa(preds, targets, n_classes)
    result[f'{prefix}F1Score']     = compute_f1(preds, targets, n_classes)
    result[f'{prefix}Recall']      = compute_recall(preds, targets, n_classes)
    result[f'{prefix}Precision']   = compute_precision(preds, targets, n_classes)
    if n_classes > 2:
        result[f'{prefix}Specificity'] = compute_specificity(preds, targets, n_classes)
    result[f'{prefix}AUROC']       = compute_auroc(probs, targets, n_classes)
    return result


# ============================================================
# ModelInterface：核心训练/验证/测试逻辑
# ============================================================
class ModelInterface(nn.Module):

    def __init__(self, model, loss, optimizer, **kargs):
        """
        参数与原版完全一致，兼容 train.py 的调用方式。
        model    : cfg.Model (addict.Dict)
        loss     : cfg.Loss  (addict.Dict)
        optimizer: cfg.Optimizer (addict.Dict)
        kargs    : data=cfg.Data, log=cfg.log_path
        """
        super(ModelInterface, self).__init__()

        # 保存超参（Lightning 的 save_hyperparameters 等价）
        self.hparams_model = model
        self.hparams_loss = loss
        self.hparams_optimizer = optimizer
        self.hparams_data = kargs.get('data', None)
        self.log_path = kargs['log']

        # 加载子模型
        self._load_model()

        # 损失函数 & 优化器配置（优化器在训练入口处通过 configure_optimizers 拿到）
        self.loss_fn = create_loss(loss)
        self.optimizer_cfg = optimizer

        self.n_classes = model.n_classes

        # 逐类别精度统计（与原版一致）
        self._reset_data_counter()

        # 随机打乱 seed 控制
        self.shuffle = kargs['data'].data_shuffle
        self.count = 0

    # ------------------------------------------------------------------
    # 内部工具
    # ------------------------------------------------------------------
    def _reset_data_counter(self):
        self.data = [{"count": 0, "correct": 0} for _ in range(self.n_classes)]

    def _print_class_acc(self):
        for c in range(self.n_classes):
            count   = self.data[c]["count"]
            correct = self.data[c]["correct"]
            acc = float(correct) / count if count > 0 else None
            print(f'class {c}: acc {acc}, correct {correct}/{count}')

    def configure_optimizers(self):
        """与原版接口一致，供 train.py 调用。"""
        return create_optimizer(self.optimizer_cfg, self.model_body)

    # ------------------------------------------------------------------
    # 单步前向（training / validation / test 共用）
    # ------------------------------------------------------------------
    def _forward_step(self, batch):
        data, label = batch
        results_dict = self.model_body(data=data, label=label)
        logits  = results_dict['logits']
        Y_prob  = results_dict['Y_prob']
        Y_hat   = results_dict['Y_hat']
        return logits, Y_prob, Y_hat, label

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def training_step(self, batch):
        """
        返回 loss（jt.Var），供训练循环调用 optimizer.step(loss)。
        """
        logits, Y_prob, Y_hat, label = self._forward_step(batch)
        loss = self.loss_fn(logits, label)

        # 逐类别精度统计
        Y_hat_int = int(Y_hat)
        Y_int     = int(label)
        self.data[Y_int]["count"]   += 1
        self.data[Y_int]["correct"] += int(Y_hat_int == Y_int)

        return loss

    def training_epoch_end(self):
        """每个 train epoch 结束后调用。"""
        self._print_class_acc()
        self._reset_data_counter()

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    def validation_step(self, batch):
        """返回单步结果 dict，供 validation_epoch_end 汇总。"""
        logits, Y_prob, Y_hat, label = self._forward_step(batch)

        Y_int = int(label)
        self.data[Y_int]["count"]   += 1
        self.data[Y_int]["correct"] += int(Y_hat.item() == Y_int)

        return {
            'logits': logits,
            'Y_prob': Y_prob,
            'Y_hat':  Y_hat,
            'label':  label,
        }

    def validation_epoch_end(self, val_step_outputs, epoch, tb_logger=None, csv_logger=None):
        """
        汇总验证集指标并写日志。
        epoch       : 当前 epoch 编号（用于 TensorBoard step）
        tb_logger   : tensorboardX.SummaryWriter（可选）
        csv_logger  : JittorCSVLogger（可选）
        返回 val_loss（float），供 EarlyStopping / ModelCheckpoint 使用。
        """
        # 拼接张量
        logits  = jt.concat([x['logits'] for x in val_step_outputs], dim=0)
        probs   = jt.concat([x['Y_prob'] for x in val_step_outputs], dim=0)
        preds   = jt.concat([x['Y_hat'].reshape(1) if x['Y_hat'].ndim == 0
                              else x['Y_hat'] for x in val_step_outputs], dim=0)
        targets = jt.concat([x['label'].reshape(1) if x['label'].ndim == 0
                              else x['label'] for x in val_step_outputs], dim=0)

        # 计算 val_loss
        val_loss = float(cross_entropy_torch(logits, targets).item())

        # 计算所有指标
        metrics = compute_all_metrics(preds, probs, targets, self.n_classes, prefix='val_')
        metrics['val_loss'] = val_loss

        # 打印
        print(f'\n[Val Epoch {epoch}]', '  '.join(f'{k}={v:.4f}' for k, v in metrics.items()))

        # 写 TensorBoard
        if tb_logger is not None:
            for k, v in metrics.items():
                tb_logger.add_scalar(k, v, global_step=epoch)

        # 写 CSV
        if csv_logger is not None:
            csv_logger.log_metrics({'epoch': epoch, **metrics})

        # 逐类别精度
        self._print_class_acc()
        self._reset_data_counter()

        # 随机 seed 控制
        if self.shuffle:
            self.count += 1
            random.seed(self.count * 50)

        return val_loss   # 供 EarlyStopping / ModelCheckpoint 使用

    # ------------------------------------------------------------------
    # Test
    # ------------------------------------------------------------------
    def test_step(self, batch):
        logits, Y_prob, Y_hat, label = self._forward_step(batch)

        Y_int = int(label)
        self.data[Y_int]["count"]   += 1
        self.data[Y_int]["correct"] += int(Y_hat.item() == Y_int)

        return {
            'logits': logits,
            'Y_prob': Y_prob,
            'Y_hat':  Y_hat,
            'label':  label,
        }

    def test_epoch_end(self, output_results):
        probs   = jt.concat([x['Y_prob'] for x in output_results], dim=0)
        preds   = jt.concat([x['Y_hat'].reshape(1) if x['Y_hat'].ndim == 0
                              else x['Y_hat'] for x in output_results], dim=0)
        targets = jt.concat([x['label'].reshape(1) if x['label'].ndim == 0
                              else x['label'] for x in output_results], dim=0)

        metrics = compute_all_metrics(preds, probs, targets, self.n_classes, prefix='test_')

        for k, v in metrics.items():
            print(f'{k} = {v:.4f}')

        self._print_class_acc()
        self._reset_data_counter()

        # 保存 CSV
        result = pd.DataFrame([{k: float(v) for k, v in metrics.items()}])
        result.to_csv(Path(self.log_path) / 'result.csv', index=False)

    # ------------------------------------------------------------------
    # 模型加载（与原版完全一致）
    # ------------------------------------------------------------------
    def _load_model(self):
        name = self.hparams_model.name
        if '_' in name:
            camel_name = ''.join([i.capitalize() for i in name.split('_')])
        else:
            camel_name = name
        try:
            Model = getattr(importlib.import_module(f'models.{name}'), camel_name)
        except Exception:
            raise ValueError('Invalid Module File Name or Invalid Class Name!')
        self.model_body = self._instancialize(Model)

    def _instancialize(self, Model, **other_args):
        # getargspec 在 Python 3.11+ 已删除，改用 getfullargspec
        class_args = inspect.getfullargspec(Model.__init__).args[1:]
        inkeys = self.hparams_model.keys()
        args1 = {}
        for arg in class_args:
            if arg in inkeys:
                args1[arg] = getattr(self.hparams_model, arg)
        args1.update(other_args)
        return Model(**args1)


# ============================================================
# 手写训练循环（替代 pytorch_lightning.Trainer）
# ============================================================
class JittorTrainer:
    """
    对标 Lightning Trainer 的核心行为：
      - fit(model, train_loader, val_loader)
      - test(model, test_loader)
    依赖 utils_jt.py 中的 EarlyStopping / ModelCheckpoint。
    """

    def __init__(self, max_epochs, callbacks=None, loggers=None,
                 check_val_every_n_epoch=1, seed=None):
        self.max_epochs = max_epochs
        self.callbacks  = callbacks or []
        self.loggers    = loggers   or []
        self.check_val_every_n_epoch = check_val_every_n_epoch

        if seed is not None:
            jt.set_global_seed(seed)
            random.seed(seed)
            np.random.seed(seed)

        # 分离两类 logger
        self.tb_logger  = None
        self.csv_logger = None
        for lg in self.loggers:
            # tensorboardX.SummaryWriter 判断
            if hasattr(lg, 'add_scalar'):
                self.tb_logger = lg
            # JittorCSVLogger 判断
            elif hasattr(lg, 'log_metrics'):
                self.csv_logger = lg

        # 分离两类 callback
        self.early_stop_cb   = None
        self.checkpoint_cb   = None
        for cb in self.callbacks:
            from utils.utils_jt import EarlyStopping, ModelCheckpoint   # utils_jt.py
            if isinstance(cb, EarlyStopping):
                self.early_stop_cb = cb
            elif isinstance(cb, ModelCheckpoint):
                self.checkpoint_cb = cb

    # ------ fit ------
    def fit(self, model, datamodule, grad_acc_steps=1):
        train_loader = datamodule.train_dataloader()
        val_loader   = datamodule.val_dataloader()
        optimizer    = model.configure_optimizers()

        for epoch in range(1, self.max_epochs + 1):
            print(f'\n===== Epoch {epoch}/{self.max_epochs} =====')

            # ---------- Train ----------
            model.train()
            step_count = 0
            base_opt = getattr(optimizer, 'base_optimizer', optimizer)

            for batch_idx, batch in enumerate(train_loader):
                loss = model.training_step(batch)

                # 每个 batch 单独反向传播，计算图独立，不会产生形状冲突
                base_opt.backward(loss / grad_acc_steps)
                step_count += 1

                if step_count % grad_acc_steps == 0:
                    # 参数更新
                    base_opt.step()
                    # Lookahead 慢权重更新
                    if optimizer is not base_opt:
                        for group in optimizer.param_groups:
                            group['lookahead_step'] += 1
                            if group['lookahead_step'] % group['lookahead_k'] == 0:
                                optimizer.update_slow(group)

                if batch_idx % 50 == 0:
                    print(f'  [Train] step {batch_idx}, loss={float(loss.item()):.4f}')

            # 处理最后不足 grad_acc_steps 的剩余 batch
            if step_count % grad_acc_steps != 0:
                base_opt.step()
                if optimizer is not base_opt:
                    for group in optimizer.param_groups:
                        group['lookahead_step'] += 1
                        if group['lookahead_step'] % group['lookahead_k'] == 0:
                            optimizer.update_slow(group)

            model.training_epoch_end()


            # ---------- Validation ----------
            if epoch % self.check_val_every_n_epoch == 0:
                model.eval()
                val_outputs = []
                with jt.no_grad():
                    for batch in val_loader:
                        out = model.validation_step(batch)
                        val_outputs.append(out)

                val_loss = model.validation_epoch_end(
                    val_outputs, epoch,
                    tb_logger=self.tb_logger,
                    csv_logger=self.csv_logger
                )

                if self.checkpoint_cb is not None:
                    self.checkpoint_cb.step(epoch, val_loss, model.model_body)

                if self.early_stop_cb is not None:
                    if self.early_stop_cb.step(val_loss):
                        print('Early stopping triggered, training stopped.')
                        break

        if self.csv_logger is not None:
            self.csv_logger.close()

    # ------ test ------
    def test(self, model, datamodule, checkpoint_path=None):
        """
        checkpoint_path: 若提供则从该路径加载权重再测试。
        """
        if checkpoint_path is not None:
            state_dict = jt.load(checkpoint_path)
            model.model_body.load_parameters(state_dict)
            print(f'Loaded checkpoint: {checkpoint_path}')

        test_loader = datamodule.test_dataloader()
        model.eval()
        test_outputs = []
        with jt.no_grad():
            for batch in test_loader:
                out = model.test_step(batch)
                test_outputs.append(out)

        model.test_epoch_end(test_outputs)
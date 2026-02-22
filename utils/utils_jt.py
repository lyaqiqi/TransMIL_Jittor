from pathlib import Path

# ----> read yaml
import yaml
from addict import Dict
def read_yaml(fpath=None):
    with open(fpath, mode="r") as file:
        yml = yaml.load(file, Loader=yaml.Loader)
        return Dict(yml)

# ---> load Loggers
from tensorboardX import SummaryWriter
import os
import csv

# ==========================================
# 1. 简单的 CSV Logger 替代类
# 用于替代 PyTorch Lightning 的 CSVLogger
# ==========================================
class JittorCSVLogger:
    def __init__(self, save_dir, name, version):
        # 拼接与 Lightning 相同的目录结构: save_dir/name/version
        self.log_dir = os.path.join(save_dir, str(name), str(version))
        os.makedirs(self.log_dir, exist_ok=True)
        self.csv_path = os.path.join(self.log_dir, 'metrics.csv')
        self.file = open(self.csv_path, 'a', newline='')
        self.writer = None

    def log_metrics(self, metrics_dict, step=None):
        """兼容 Lightning 的 log_metrics 接口"""
        if step is not None:
            metrics_dict = step
            
        # 动态获取字典的 keys 作为表头
        if self.writer is None:
            self.writer = csv.DictWriter(self.file, fieldnames=metrics_dict.keys())
            self.writer.writeheader()
            
        self.writer.writerow(metrics_dict)
        self.file.flush() # 确保实时写入文件

    def close(self):
        self.file.close()

def load_loggers(cfg):

    log_path = cfg.General.log_path
    Path(log_path).mkdir(exist_ok=True, parents=True)
    log_name = Path(cfg.config).parent 
    version_name = Path(cfg.config).name
    cfg.log_path = Path(log_path) / log_name / version_name / f'fold{cfg.Data.fold}'
    print(f'---->Log dir: {cfg.log_path}')
    
    # 基础保存路径
    base_save_dir = os.path.join(log_path, str(log_name))
    
    #---->TensorBoard
    # TensorboardX 的 SummaryWriter 直接接收拼接好的 log_dir
    tb_log_dir = os.path.join(base_save_dir, version_name, f'fold{cfg.Data.fold}')
    tb_logger = SummaryWriter(log_dir=tb_log_dir)
    # 注意: tensorboardX 不原生支持 log_graph 和 default_hp_metric 这样复杂的 Lightning 参数，
    # 如果后续你需要记录网络结构(log_graph)，可以在训练循环中手动调用 tb_logger.add_graph(model, input_to_model)

    #---->CSV
    csv_logger = JittorCSVLogger(
        save_dir=base_save_dir, 
        name=version_name, 
        version=f'fold{cfg.Data.fold}'
    )
    
    return [tb_logger, csv_logger]


import os
import jittor as jt

# ==========================================
# 1. 模拟 PyTorch Lightning 的 EarlyStopping
# ==========================================
class EarlyStopping:
    def __init__(self, monitor='val_loss', min_delta=0.00, patience=10, verbose=True, mode='min'):
        self.monitor = monitor
        self.min_delta = min_delta
        self.patience = patience
        self.verbose = verbose
        self.mode = mode
        
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def step(self, current_metric):
        """在每个 epoch 验证结束后调用"""
        score = -current_metric if self.mode == 'min' else current_metric
        
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print("Early stopping triggered!")
        else:
            self.best_score = score
            self.counter = 0
            
        return self.early_stop

# ==========================================
# 2. 模拟 PyTorch Lightning 的 ModelCheckpoint
# ==========================================
class ModelCheckpoint:
    def __init__(self, monitor='val_loss', dirpath='', filename='{epoch:02d}-{val_loss:.4f}',
                 verbose=True, save_last=True, save_top_k=1, mode='min', save_weights_only=True):
        self.monitor = monitor
        self.dirpath = dirpath
        self.filename = filename
        self.verbose = verbose
        self.save_last = save_last
        self.save_top_k = save_top_k  # 此处简化为只保留1个(即最优模型)
        self.mode = mode
        self.save_weights_only = save_weights_only
        
        self.best_score = None
        self.best_model_path = None
        os.makedirs(self.dirpath, exist_ok=True)
        
    def step(self, epoch, current_metric, model):
        """在每个 epoch 验证结束后调用"""
        # 1. 保存最新的 last 模型 (Jittor的惯用权重后缀是 .pkl)
        if self.save_last:
            last_path = os.path.join(self.dirpath, "last.pkl")
            # Jittor 获取参数字典用 model.parameters()，不过一般直接 save 就行
            jt.save(model.parameters() if self.save_weights_only else model, last_path)

        # 2. 判断并保存最佳模型
        score = -current_metric if self.mode == 'min' else current_metric
        
        if self.best_score is None or score > self.best_score:
            if self.verbose:
                best_current = (-self.best_score if self.mode == 'min' else self.best_score) if self.best_score is not None else None
                best_disp = f"{best_current:.4f}" if best_current is not None else "None"
                print(f"Validation {self.monitor} improved from {best_disp} to {current_metric:.4f}. Saving model...")
                
            self.best_score = score
            
            # 手动替换模板字符串中的变量
            formatted_filename = self.filename.replace('{epoch:02d}', f"{epoch:02d}")
            formatted_filename = formatted_filename.replace('{val_loss:.4f}', f"{current_metric:.4f}")
            if not formatted_filename.endswith('.pkl'):
                formatted_filename += '.pkl'
                
            filepath = os.path.join(self.dirpath, formatted_filename)
            
            # 删除旧的最佳模型
            if self.best_model_path and os.path.exists(self.best_model_path):
                os.remove(self.best_model_path)
                
            self.best_model_path = filepath
            
            # Jittor 保存模型
            jt.save(model.parameters() if self.save_weights_only else model, filepath)

# ==========================================
# 3. 转换后的 load_callbacks 函数
# ==========================================
def load_callbacks(cfg):
    Mycallbacks =[]
    
    # Make output path
    output_path = cfg.log_path
    output_path.mkdir(exist_ok=True, parents=True)

    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        min_delta=0.00,
        patience=cfg.General.patience,
        verbose=True,
        mode='min'
    )
    Mycallbacks.append(early_stop_callback)

    if cfg.General.server == 'train':
        Mycallbacks.append(ModelCheckpoint(
            monitor='val_loss',
            dirpath=str(cfg.log_path),
            filename='{epoch:02d}-{val_loss:.4f}',
            verbose=True,
            save_last=True,
            save_top_k=1,
            mode='min',
            save_weights_only=True
        ))
        
    return Mycallbacks

#---->val loss
import jittor as jt
from jittor import nn

def cross_entropy_torch(x, y):
    """
    Jittor 版本的交叉熵函数。
    x: 预测的 logits，形状通常为 (Batch, Classes)
    y: 真实的标签索引，形状通常为 (Batch,)
    """
    
    # ==========================================
    # 方法一：直接使用内置算子（强烈推荐，性能最高且完全等价）
    # ==========================================
    loss = nn.cross_entropy_loss(x, y)
    return loss

    # ==========================================
    # 方法二：如果你必须手动推导计算过程（向量化写法，替代原来的 for 循环）
    # ==========================================
    # 1. 对类别维度(dim=1)进行 softmax
    # x_softmax = nn.softmax(x, dim=1)
    # 
    # 2. 获取 batch_size 并生成索引
    # batch_size = x.shape
    # batch_indices = jt.arange(batch_size)
    # 
    # 3. 提取对应真实标签 y 的概率 (相当于原代码的 x_softmax])
    # target_probs = x_softmax
    # 
    # 4. 计算 log 并求负对数似然平均值
    # x_log = jt.log(target_probs)
    # loss = -jt.sum(x_log) / batch_size
    # return loss
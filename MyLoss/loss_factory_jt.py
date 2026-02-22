__author__ = 'shaozc (Ported to Jittor)'

import jittor as jt
from jittor import nn

def create_loss(args, w1=1.0, w2=0.5):
    """
    根据配置动态生成 Jittor 的 Loss 实例。
    """
    # 兼容传入的 args 是对象(有属性)还是字典(Dict)的情况
    if hasattr(args, 'base_loss'):
        conf_loss = args.base_loss
    elif isinstance(args, dict) and 'base_loss' in args:
        conf_loss = args
    elif isinstance(args, str):
        conf_loss = args
    else:
        conf_loss = "CrossEntropyLoss" # 默认 fallback

    loss = None
    
    # 判断 Jittor 的 nn 模块中是否包含该 loss (例如 CrossEntropyLoss)
    if hasattr(nn, conf_loss): 
        loss = getattr(nn, conf_loss)() 
    else:
        # 基于“按需移植”策略，我们移除了原来未使用的 pytorch_toolbelt 和其他自定义 Loss。
        # 如果未来你需要用 dice_loss 等，可以在这里手动实现或导入 Jittor 版本的模块。
        raise NotImplementedError(
            f"Loss '{conf_loss}' 目前未在 Jittor 中实现或未被导入。"
            f"根据按需移植策略，当前仅支持内置的 jittor.nn 损失函数 (如 CrossEntropyLoss)。"
        )
        
    return loss

import argparse
def make_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base-loss', default='CrossEntropyLoss', type=str)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = make_parse()
    myloss = create_loss(args)
    
    # Jittor 测试数据生成
    data = jt.randn(2, 3)
    # Jittor 中标签通常需要是整数类型，对应 PyTorch 的 torch.long
    label = jt.randint(0, 3, shape=(2,))
    
    loss = myloss(data, label)
    print(f" Test Loss successfully computed: {loss.item():.4f}")
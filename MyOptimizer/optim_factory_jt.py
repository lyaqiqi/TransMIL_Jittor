""" Optimizer Factory w/ Custom Weight Decay (Ported to Jittor)
Pruned for Lazy Porting strategy.
"""
import jittor as jt
from jittor import optim

# 只导入我们实际用到的/已经移植好的优化器组件
from .radam_jt import RAdam
from .lookahead_jt import Lookahead

def add_weight_decay(model, weight_decay=1e-5, skip_list=()):
    """
    智能权重衰减机制：对 1D 张量(如 LayerNorm/BatchNorm 权重) 和 bias 禁用 weight_decay。
    这个机制在 Jittor 中同样非常有用。
    """
    decay =[]
    no_decay =[]
    
    # Jittor 支持 named_parameters 迭代
    for name, param in model.named_parameters():
        # 在 Jittor 中，判断变量是否不需要梯度通常用 is_stop_grad()
        # 这里使用 getattr 兼容可能存在的不同 Jittor 版本或 PyTorch 别名
        if getattr(param, 'is_stop_grad', lambda: False)() or not getattr(param, 'requires_grad', True):
            continue  # frozen weights (跳过被冻结的权重)
            
        if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list:
            no_decay.append(param)
        else:
            decay.append(param)
            
    return [{'params': no_decay, 'weight_decay': 0.}, {'params': decay, 'weight_decay': weight_decay}]

def create_optimizer(args, model, filter_bias_and_bn=True):
    """
    根据配置动态生成 Jittor 的 Optimizer 实例。
    """
    # 兼容传入的 args 是对象(有属性)还是字典(Dict)的情况
    opt_lower = args.opt.lower() if hasattr(args, 'opt') else args.lower()
    weight_decay = getattr(args, 'weight_decay', 0) if hasattr(args, 'weight_decay') else args.get('weight_decay', 0)
    
    if weight_decay and filter_bias_and_bn:
        skip = {}
        if hasattr(model, 'no_weight_decay'):
            skip = model.no_weight_decay()
        parameters = add_weight_decay(model, weight_decay, skip)
        weight_decay = 0.
    else:
        parameters = model.parameters()

    # 提取基础参数
    lr = getattr(args, 'lr', 1e-3) if hasattr(args, 'lr') else args.get('lr', 1e-3)
    opt_args = dict(lr=lr, weight_decay=weight_decay)
    
    opt_eps = getattr(args, 'opt_eps', None) if hasattr(args, 'opt_eps') else args.get('opt_eps')
    if opt_eps is not None:
        opt_args['eps'] = opt_eps
        
    opt_betas = getattr(args, 'opt_betas', None) if hasattr(args, 'opt_betas') else args.get('opt_betas')
    if opt_betas is not None:
        opt_args['betas'] = opt_betas

    # 解析 Lookahead 包装器 (例如: lookahead_radam -> lookahead 和 radam)
    opt_split = opt_lower.split('_')
    opt_lower = opt_split[-1] 
    
    # --- 按需实例化优化器 ---
    if opt_lower == 'radam':
        optimizer = RAdam(parameters, **opt_args)
    elif opt_lower == 'adam':
        optimizer = optim.Adam(parameters, **opt_args)
    elif opt_lower == 'adamw':
        optimizer = optim.AdamW(parameters, **opt_args)
    elif opt_lower == 'sgd':
        momentum = getattr(args, 'momentum', 0) if hasattr(args, 'momentum') else args.get('momentum', 0)
        opt_args.pop('eps', None) # SGD 没有 eps
        optimizer = optim.SGD(parameters, momentum=momentum, **opt_args)
    else:
        raise ValueError(f"Optimizer '{opt_lower}' 目前未被移植。基于按需移植策略，目前仅支持 radam, adam, adamw, sgd。")

    if len(opt_split) > 1 and opt_split[0] == 'lookahead':
        optimizer = Lookahead(optimizer)

    return optimizer
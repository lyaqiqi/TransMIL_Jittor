import math
import jittor as jt
from jittor.optim import Optimizer 

class RAdam(Optimizer):

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
        # Jittor 基类只接受 params 和 lr，不接受 defaults 字典
        # 对应原版: super(RAdam, self).__init__(params, defaults)
        super(RAdam, self).__init__(params, lr)

        # betas/eps/weight_decay 原版存在 defaults 里由基类管理
        # Jittor 中基类不支持，改为直接存在实例属性上
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay

        # step() 中用于存储每个参数的动量状态，对应原版的 self.state[p]
        self.state = {}

        # 与原版完全一致，无需修改
        self.buffer = [[None, None, None] for ind in range(10)]

    def step(self, loss=None):
        # 对应原版: loss = closure() if closure else None
        # Jittor 中 pre_step 负责反向传播，填充 pg["grads"]
        self.pre_step(loss)

        for pg in self.param_groups:
            # betas/eps/weight_decay 存在 self 上（__init__ 里设置的）
            # lr 优先从 pg 里取，回退到 self.lr（和 SGD 的写法一致）
            lr           = pg.get('lr', self.lr)
            beta1, beta2 = pg.get('betas', self.betas)
            eps          = pg.get('eps', self.eps)
            weight_decay = pg.get('weight_decay', self.weight_decay)

            for p, g in zip(pg['params'], pg['grads']):
                # 对应原版: if p.grad is None: continue
                if p.is_stop_grad():
                    continue

                # 对应原版: self.state[p]
                pid = id(p)
                if pid not in self.state:
                    # 对应原版: len(state) == 0 的分支
                    self.state[pid] = {
                        'step':       0,
                        'exp_avg':    jt.zeros_like(p),
                        'exp_avg_sq': jt.zeros_like(p),
                    }

                state      = self.state[pid]
                exp_avg    = state['exp_avg']
                exp_avg_sq = state['exp_avg_sq']

                # 对应原版:
                # exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                # exp_avg.mul_(beta1).add_(1 - beta1, grad)
                exp_avg_sq.update(exp_avg_sq * beta2 + (1 - beta2) * g * g)
                exp_avg.update(exp_avg * beta1 + (1 - beta1) * g)

                state['step'] += 1
                buffered = self.buffer[int(state['step'] % 10)]

                if state['step'] == buffered[0]:
                    N_sma, step_size = buffered[1], buffered[2]
                else:
                    buffered[0] = state['step']
                    beta2_t   = beta2 ** state['step']
                    N_sma_max = 2 / (1 - beta2) - 1
                    N_sma     = N_sma_max - 2 * state['step'] * beta2_t / (1 - beta2_t)
                    buffered[1] = N_sma

                    if N_sma >= 5:
                        step_size = lr * math.sqrt(
                            (1 - beta2_t) *
                            (N_sma - 4) / (N_sma_max - 4) *
                            (N_sma - 2) / N_sma *
                            N_sma_max   / (N_sma_max - 2)
                        ) / (1 - beta1 ** state['step'])
                    else:
                        step_size = lr / (1 - beta1 ** state['step'])
                    buffered[2] = step_size

                # 对应原版: p_data_fp32.add_(-weight_decay * lr, p_data_fp32)
                # Jittor 直接在 p 上操作，不需要 fp32 中间变量
                if weight_decay != 0 and weight_decay is not None:
                    p.update(p * (1 - weight_decay * lr))

                # 对应原版:
                # denom = exp_avg_sq.sqrt().add_(eps)
                # p_data_fp32.addcdiv_(-step_size, exp_avg, denom)
                # p_data_fp32.add_(-step_size, exp_avg)
                # p.data.copy_(p_data_fp32)
                if N_sma >= 5:
                    denom = jt.sqrt(exp_avg_sq) + eps
                    p.update(p - step_size * exp_avg / denom)
                else:
                    p.update(p - step_size * exp_avg)

        # 对应原版: return loss
        # Jittor 中 post_step 负责清零梯度等收尾工作
        self.post_step()

class PlainRAdam(Optimizer):

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
        super(PlainRAdam, self).__init__(params, lr)
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.state = {}

    def step(self, loss=None):
        self.pre_step(loss)

        for pg in self.param_groups:
            lr           = pg.get('lr', self.lr)
            beta1, beta2 = pg.get('betas', self.betas)
            eps          = pg.get('eps', self.eps)
            weight_decay = pg.get('weight_decay', self.weight_decay)

            for p, g in zip(pg['params'], pg['grads']):
                if p.is_stop_grad():
                    continue

                pid = id(p)
                if pid not in self.state:
                    self.state[pid] = {
                        'step':       0,
                        'exp_avg':    jt.zeros_like(p),
                        'exp_avg_sq': jt.zeros_like(p),
                    }

                state      = self.state[pid]
                exp_avg    = state['exp_avg']
                exp_avg_sq = state['exp_avg_sq']

                exp_avg_sq.update(exp_avg_sq * beta2 + (1 - beta2) * g * g)
                exp_avg.update(exp_avg * beta1 + (1 - beta1) * g)

                state['step'] += 1
                beta2_t   = beta2 ** state['step']
                N_sma_max = 2 / (1 - beta2) - 1
                N_sma     = N_sma_max - 2 * state['step'] * beta2_t / (1 - beta2_t)

                if weight_decay != 0:
                    p.update(p * (1 - weight_decay * lr))

                if N_sma >= 5:
                    step_size = lr * math.sqrt(
                        (1 - beta2_t) *
                        (N_sma - 4) / (N_sma_max - 4) *
                        (N_sma - 2) / N_sma *
                        N_sma_max   / (N_sma_max - 2)
                    ) / (1 - beta1 ** state['step'])
                    denom = jt.sqrt(exp_avg_sq) + eps
                    p.update(p - step_size * exp_avg / denom)
                else:
                    step_size = lr / (1 - beta1 ** state['step'])
                    p.update(p - step_size * exp_avg)

        self.post_step()
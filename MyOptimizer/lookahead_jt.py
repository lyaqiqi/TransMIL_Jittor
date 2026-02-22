import jittor as jt
from jittor.optim import Optimizer
from collections import defaultdict

class Lookahead:

    def __init__(self, base_optimizer, alpha=0.5, k=6):
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f'Invalid slow update rate: {alpha}')
        if not 1 <= k:
            raise ValueError(f'Invalid lookahead steps: {k}')

        defaults = dict(lookahead_alpha=alpha, lookahead_k=k, lookahead_step=0)

        self.base_optimizer = base_optimizer

        # 对应原版: self.param_groups = self.base_optimizer.param_groups
        # 共享引用，和原版完全一致
        self.param_groups = self.base_optimizer.param_groups

        # 对应原版: self.defaults = base_optimizer.defaults
        #           self.defaults.update(defaults)
        # Jittor 中 defaults 是只读 property，改为单独存储
        self.alpha = alpha
        self.k = k

        # 对应原版: self.state = defaultdict(dict)
        # 基类没有 self.state，自己定义，用于存储 slow_buffer
        self.state = defaultdict(dict)

        # 对应原版: 把 lookahead 超参注入各 param_group
        for name, default in defaults.items():
            for group in self.param_groups:
                group.setdefault(name, default)
    
    def update_slow(self, group):
        for fast_p in group['params']:
            # 对应原版: if fast_p.grad is None: continue
            if fast_p.is_stop_grad():
                continue

            # 对应原版: self.state[fast_p]
            # Jittor 中用 id(fast_p) 作为 key，因为 defaultdict 支持任意 key，
            # 用 id 更安全
            param_state = self.state[id(fast_p)]

            if 'slow_buffer' not in param_state:
                # 对应原版:
                # param_state['slow_buffer'] = torch.empty_like(fast_p.data)
                # param_state['slow_buffer'].copy_(fast_p.data)
                param_state['slow_buffer'] = fast_p.clone().detach()

            slow = param_state['slow_buffer']

            # 对应原版: slow.add_(group['lookahead_alpha'], fast_p.data - slow)
            slow.update(slow + group['lookahead_alpha'] * (fast_p - slow))

            # 对应原版: fast_p.data.copy_(slow)
            fast_p.update(slow)

    def sync_lookahead(self):
        for group in self.param_groups:
            self.update_slow(group)
    
    def step(self, loss=None):
        # 对应原版: loss = self.base_optimizer.step(closure)
        # RAdam.step(loss) 内部会调用 pre_step(loss) 完成反向传播和参数更新
        self.base_optimizer.step(loss)

        # 对应原版: lookahead_step 计数和慢权重插值，逻辑完全一致
        for group in self.param_groups:
            group['lookahead_step'] += 1
            if group['lookahead_step'] % group['lookahead_k'] == 0:
                self.update_slow(group)
    
    def state_dict(self):
        # 对应原版: fast_state_dict = self.base_optimizer.state_dict()
        # Jittor 基类的 state_dict 返回 {"defaults": self.defaults}
        fast_state_dict = self.base_optimizer.state_dict()

        # 对应原版:
        # slow_state = {(id(k) if isinstance(k, torch.Tensor) else k): v
        #               for k, v in self.state.items()}
        # 我们的 self.state 已经用 id(fast_p) 作为 key，直接使用即可
        slow_state = dict(self.state)

        return {
            'fast_state': fast_state_dict,
            'slow_state': slow_state,
        }
    
    def load_state_dict(self, state_dict):
        # 对应原版: self.base_optimizer.load_state_dict(fast_state_dict)
        # Jittor 基类的 load_state_dict 只认 {"defaults": ...} 格式
        if 'fast_state' in state_dict:
            self.base_optimizer.load_state_dict(state_dict['fast_state'])

        # 对应原版: 恢复慢权重状态
        if 'slow_state' not in state_dict:
            # 对应原版: 'Loading state_dict from optimizer without Lookahead applied.'
            print('Loading state_dict from optimizer without Lookahead applied.')
            self.state = defaultdict(dict)

            # 对应原版: reapply defaults to catch missing lookahead specific ones
            # 原版用 self.defaults.items()，我们用 self.alpha 和 self.k 直接注入
            defaults = dict(lookahead_alpha=self.alpha, lookahead_k=self.k, lookahead_step=0)
            for name, default in defaults.items():
                for group in self.param_groups:
                    group.setdefault(name, default)
        else:
            # 对应原版: super(Lookahead, self).load_state_dict(slow_state_dict)
            # 我们直接恢复 self.state
            self.state = defaultdict(dict, state_dict['slow_state'])

        # 问题2确认：load_state_dict 后引用不变，不需要重新赋值
        # 对应原版: self.param_groups = self.base_optimizer.param_groups
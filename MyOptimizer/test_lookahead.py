import jittor as jt
import math

# ============================================================
# 把你的类粘贴在这里，或者直接 import
# ============================================================
from radam_jt import RAdam
from lookahead_jt import Lookahead

# ============================================================
# 测试1：基本初始化检查
# ============================================================
print("=== 测试1：初始化检查 ===")
w = jt.array([0.5]).float()
b = jt.array([0.0]).float()

base_optimizer = RAdam([w, b], lr=0.01)
optimizer = Lookahead(base_optimizer, alpha=0.5, k=3)

print(f"alpha:              {optimizer.alpha}")
print(f"k:                  {optimizer.k}")
print(f"state:              {dict(optimizer.state)}")
print(f"param_groups 共享:  {optimizer.param_groups is base_optimizer.param_groups}")
print(f"lookahead_alpha:    {optimizer.param_groups[0]['lookahead_alpha']}")
print(f"lookahead_k:        {optimizer.param_groups[0]['lookahead_k']}")
print(f"lookahead_step:     {optimizer.param_groups[0]['lookahead_step']}")
print()

# ============================================================
# 测试2：训练过程，观察 loss 下降，以及慢权重在第 k 步插值
# ============================================================
print("=== 测试2：训练过程 ===")
w = jt.array([0.5]).float()
b = jt.array([0.0]).float()
x      = jt.array([2.0]).float()
target = jt.array([5.0]).float()

base_optimizer = RAdam([w, b], lr=0.01)
optimizer = Lookahead(base_optimizer, alpha=0.5, k=3)

for step in range(9):
    pred = w * x + b
    loss = ((pred - target) ** 2).mean()
    optimizer.step(loss)
    
    lookahead_step = optimizer.param_groups[0]['lookahead_step']
    slow_updated = (lookahead_step % 3 == 0)  # k=3，每3步做一次慢权重插值
    print(f"step {step+1}: loss={float(loss):.6f}  w={float(w):.4f}  b={float(b):.4f}"
          f"  lookahead_step={lookahead_step}  {'← slow update' if slow_updated else ''}")

print()

# ============================================================
# 测试3：slow_buffer 检查，确认慢权重被正确初始化和更新
# ============================================================
print("=== 测试3：slow_buffer 检查 ===")
for pid, s in optimizer.state.items():
    print(f"param id {pid}: slow_buffer={float(s['slow_buffer']):.4f}")
print()

# ============================================================
# 测试4：state_dict 和 load_state_dict 的保存/恢复
# ============================================================
print("=== 测试4：state_dict / load_state_dict ===")
saved = optimizer.state_dict()
print(f"state_dict keys:       {list(saved.keys())}")
print(f"fast_state keys:       {list(saved['fast_state'].keys())}")
print(f"slow_state 条目数:     {len(saved['slow_state'])}")

# 记录保存时的参数值
w_before = float(w)
b_before = float(b)

# 再训练几步改变参数
for _ in range(3):
    pred = w * x + b
    loss = ((pred - target) ** 2).mean()
    optimizer.step(loss)

print(f"继续训练后: w={float(w):.4f}  b={float(b):.4f}")

# 恢复
optimizer.load_state_dict(saved)
print(f"load_state_dict 后 slow_state 条目数: {len(optimizer.state)}")
print(f"param_groups 共享引用不变: {optimizer.param_groups is base_optimizer.param_groups}")
print()

# ============================================================
# 测试5：sync_lookahead 手动触发慢权重同步
# ============================================================
print("=== 测试5：sync_lookahead ===")
w_before_sync = float(w)
optimizer.sync_lookahead()
print(f"sync 前 w={w_before_sync:.4f}，sync 后 w={float(w):.4f}")
print("（如果 sync 有效，fast 权重会被拉向慢权重）")
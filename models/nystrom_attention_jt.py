import jittor as jt
from jittor import nn
import jittor.linalg as jl
from math import ceil

# helper functions
def exists(val):
    return val is not None

def moore_penrose_iter_pinv_jt(x, iters=6):
    abs_x = jt.abs(x)
    col = abs_x.sum(dim=-1)
    row = abs_x.sum(dim=-2)
    
    z = x.transpose(-1, -2) / (jt.max(col) * jt.max(row))

    n = x.shape[-1]
    I = jt.diag(jt.ones(n, dtype=x.dtype)).unsqueeze(0)

    for _ in range(iters):
        xz = x @ z
        z = 0.25 * z @ (13 * I - (xz @ (15 * I - (xz @ (7 * I - xz)))))

    return z

# main attention class
class NystromAttention_jt(nn.Module):
    def __init__(
        self,
        dim,
        dim_head = 64,
        heads = 8,
        num_landmarks = 256,
        pinv_iterations = 6,
        residual = True,
        residual_conv_kernel = 33,
        eps = 1e-8,
        dropout = 0.
    ):
        super().__init__()
        self.eps = eps
        inner_dim = heads * dim_head

        self.num_landmarks = num_landmarks
        self.pinv_iterations = pinv_iterations

        self.heads = heads
        self.scale = dim_head ** -0.5
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

        self.residual = residual
        if residual:
            kernel_size = residual_conv_kernel
            padding = residual_conv_kernel // 2
            self.res_conv = nn.Conv2d(heads, heads, (kernel_size, 1), padding = (padding, 0), groups = heads, bias = False)

    def execute(self, x, mask = None, return_attn = False):
        b, n, _ = x.shape
        h, m, iters, eps = self.heads, self.num_landmarks, self.pinv_iterations, self.eps

        remainder = n % m
        if remainder > 0:
            padding = m - (n % m)
            
            # 完全对齐原版 F.pad(x, (0, 0, padding, 0)) 的逻辑：在序列维度的最前面拼接零张量
            b = x.shape[0]
            d = x.shape[-1]
            zeros_pad = jt.zeros((b, padding, d), dtype=x.dtype)
            x = jt.concat([zeros_pad, x], dim=1)

            if exists(mask):
                # 完全对齐原版 F.pad(mask, (padding, 0)) 的逻辑：在掩码维度的最前面拼接零张量
                mask_pad = jt.zeros((b, padding), dtype=mask.dtype)
                mask = jt.concat([mask_pad, mask], dim=1)

        # derive query, keys, values
        # 1. 线性映射并切分成 q, k, v 三个张量 (Jittor 原生支持 chunk)
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)

        # 2. 头部分割与维度重排
        b = x.shape[0]
        # 使用新变量 n_padded 记录当前张量的长度，绝对不覆盖原始的真实长度 n 
        n_padded = x.shape[1] 
        d = q.shape[-1] // self.heads

        # 严格等价于 rearrange(t, 'b n (h d) -> b h n d')
        # 先 reshape 拆分特征维度，再 transpose 交换“序列长度”(1)和“头数”(2)的位置
        q = q.reshape(b, n_padded, self.heads, d).transpose(0, 2, 1, 3)
        k = k.reshape(b, n_padded, self.heads, d).transpose(0, 2, 1, 3)
        v = v.reshape(b, n_padded, self.heads, d).transpose(0, 2, 1, 3)

        # set masked positions to 0 in queries, keys, values
        if exists(mask):
            # 3. 掩码维度扩展：等价于 rearrange(mask, 'b n -> b () n')
            # 变成 [b, 1, n_padded]
            mask_expanded = mask.unsqueeze(1)
            
            # 4. 掩码置零操作
            # 再加一个维度变成 [b, 1, n_padded, 1]，方便利用广播机制与 [b, h, n_padded, d] 相乘
            mask_expanded = mask_expanded.unsqueeze(-1)
            q = q * mask_expanded
            k = k * mask_expanded
            v = v * mask_expanded

        # 5. 对 query 进行常规的缩放
        q = q * self.scale

        # generate landmarks by sum reduction, and then calculate mean using the mask

        l = ceil(n / m)
        
        # 忠实翻译：将序列维度拆分为 m 和 l，然后在 l 所在的维度（dim=3）求和
        b = q.shape[0]
        h = q.shape[1]
        d = q.shape[-1]
        
        q_landmarks = q.reshape(b, h, m, l, d).sum(dim=3)
        k_landmarks = k.reshape(b, h, m, l, d).sum(dim=3)

        # calculate landmark mask, and also get sum of non-masked elements in preparation for masked mean

        divisor = l
        if exists(mask):
            # 【修复 1】：强制将 bool 转换为 float32，防止求和时溢出或报错
            mask_landmarks_sum = mask.float32().reshape(b, 1, m, l).sum(dim=3)
            
            # 【修复 2】：清剿残余的 [..., None] 语法，用 unsqueeze(-1) 替代
            divisor = mask_landmarks_sum.unsqueeze(-1) + eps
            mask_landmarks = mask_landmarks_sum > 0
        
        # masked mean (if mask exists)
        q_landmarks /= divisor
        k_landmarks /= divisor

        # similarities
        einops_eq = '... i d, ... j d -> ... i j'
        sim1 = jl.einsum(einops_eq, q, k_landmarks)
        sim2 = jl.einsum(einops_eq, q_landmarks, k_landmarks)
        sim3 = jl.einsum(einops_eq, q_landmarks, k)

        # masking

        if exists(mask):
            # Jittor 中推荐使用 -1e9 代替系统极小值，防止 FP16 计算时产生 NaN 溢出
            mask_value = -1e9
            
            # 【核心修复】：由于前文没有对 mask 进行原地扩展，此时 mask 还是 [b, n]！
            # 必须将其显式扩展为 [b, 1, n]，否则在后续增加维度时会发生灾难性的错位广播
            if len(mask.shape) == 2:
                mask = mask.unsqueeze(1)
            
            # 1. 显式增加维度，彻底规避 Jittor 对 [..., None] 的解析 Bug
            m_n_1 = mask.unsqueeze(-1)            # 形状变成: [b, 1, n_padded, 1]
            ml_1_m = mask_landmarks.unsqueeze(-2) # 形状变成: [b, 1, 1, m]
            ml_m_1 = mask_landmarks.unsqueeze(-1) # 形状变成: [b, 1, m, 1]
            m_1_n = mask.unsqueeze(-2)            # 形状变成: [b, 1, 1, n_padded]

            # 2. 逻辑与 (&) 生成 2D 验证网格
            valid1 = m_n_1 & ml_1_m  # 广播后严丝合缝变成: [b, 1, n_padded, m]
            valid2 = ml_m_1 & ml_1_m # 广播后严丝合缝变成: [b, 1, m, m]
            valid3 = ml_m_1 & m_1_n  # 广播后严丝合缝变成: [b, 1, m, n_padded]

            # 3. 使用 jt.where 替代 masked_fill_
            sim1 = jt.where(valid1.broadcast(sim1.shape), sim1, mask_value)
            sim2 = jt.where(valid2.broadcast(sim2.shape), sim2, mask_value)
            sim3 = jt.where(valid3.broadcast(sim3.shape), sim3, mask_value)

        # eq (15) in the paper and aggregate values

        # 原逻辑：t.softmax(dim = -1)
        # 忠实翻译：Jittor 中调用原生的 jt.nn.softmax
        attn1, attn2, attn3 = map(lambda t: jt.nn.softmax(t, dim = -1), (sim1, sim2, sim3))
        attn2_inv = moore_penrose_iter_pinv_jt(attn2, iters)

        out = (attn1 @ attn2_inv) @ (attn3 @ v)

        # add depth-wise conv residual of values
        if self.residual:
            out += self.res_conv(v)

        # merge and combine heads

        # 动态获取当前的维度参数
        # [Batch大小, 头数, 序列长度, 头部维度] => [Batch大小, 序列长度, 总特征维度]
        b = out.shape[0]
        n_padded = out.shape[2]  # 这是 padding 后的当前序列长度
        d = out.shape[3]
        # 原生平替 rearrange(out, 'b h n d -> b n (h d)')
        # 步骤 1：转置 (transpose)，将序列维度提到头数维度之前 -> [b, n_padded, h, d]
        # 步骤 2：展平 (reshape)，合并最后两个维度 -> [b, n_padded, h * d]
        out = out.transpose(0, 2, 1, 3).reshape(b, n_padded, self.heads * d)

        # 通过线性输出投影层
        out = self.to_out(out)

        # 截断回输入时的原始序列长度 (对应原版代码中的 :n)
        # 注意：这里的 n_orig 是你在 forward/execute 开头保存的未 padding 的原始长度变量
        out = out[:, :n]

        if return_attn:
            attn = attn1 @ attn2_inv @ attn3
            return out, attn
        
        return out



class PreNorm_jt(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def execute(self, x, **kwargs):
        x = self.norm(x)
        return self.fn(x, **kwargs)

class FeedForward_jt(nn.Module):
    def __init__(self, dim, mult = 4, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mult, dim)
        )

    def execute(self, x):
        return self.net(x)

class Nystromformer_jt(nn.Module):
    def __init__(
        self, 
        *, 
        dim, 
        depth, 
        dim_head = 64, 
        heads = 8, 
        num_landmarks = 256, 
        pinv_iterations = 6, 
        attn_values_residual = True, 
        attn_values_residual_conv_kernel = 33, 
        attn_dropout = 0., 
        ff_dropout = 0.
    ):

        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm_jt(dim, NystromAttention_jt(dim = dim, dim_head = dim_head, heads = heads, num_landmarks = num_landmarks, pinv_iterations = pinv_iterations, residual = attn_values_residual, residual_conv_kernel = attn_values_residual_conv_kernel, dropout = attn_dropout)),
                PreNorm_jt(dim, FeedForward_jt(dim = dim, dropout = ff_dropout))
            ]))

    def execute(self, x, mask = None):
        for attn, ff in self.layers:
            x = attn(x, mask = mask) + x
            x = ff(x) + x
        return x
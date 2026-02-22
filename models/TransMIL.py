import jittor as jt
from jittor import nn
import numpy as np
from .nystrom_attention_jt import NystromAttention_jt 

class TransLayer(nn.Module):
    def __init__(self, norm_layer=nn.LayerNorm, dim=512):
        super().__init__()
        self.norm = norm_layer(dim)
        self.attn = NystromAttention_jt(
            dim=dim,
            dim_head=dim//8,
            heads=8,
            num_landmarks=dim//2,
            pinv_iterations=6,
            residual=True,
            dropout=0.1
        )

    def execute(self, x):
        x = x + self.attn(self.norm(x))

        return x

class PPEG(nn.Module):
    def __init__(self, dim=512):
        super(PPEG, self).__init__()
        # Jittor 的 Conv2d 参数顺序与 PyTorch 一致：
        # (in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        # 注意：padding 传入整数即可
        self.proj = nn.Conv2d(dim, dim, 7, stride=1, padding=7//2, groups=dim)
        self.proj1 = nn.Conv2d(dim, dim, 5, stride=1, padding=5//2, groups=dim)
        self.proj2 = nn.Conv2d(dim, dim, 3, stride=1, padding=3//2, groups=dim)

    def execute(self, x, H, W):
        # 获取形状，x shape: [B, N, C]
        B, N, C = x.shape
        
        # 切片操作，与 Numpy/PyTorch 语法一致
        cls_token = x[:, 0]
        feat_token = x[:, 1:]
        
        # 维度变换
        # .transpose(1, 2) 交换维度
        # .view() 或 .reshape() 重塑形状
        cnn_feat = feat_token.transpose(1, 2).view(B, C, H, W)
        
        # 卷积与残差相加
        # Jittor 会自动处理运算符重载
        x = self.proj(cnn_feat) + cnn_feat + self.proj1(cnn_feat) + self.proj2(cnn_feat)
        
        # 还原形状
        # flatten(2) 表示从第2维开始拉平到最后一维
        x = x.flatten(2).transpose(1, 2)
        
        # 拼接
        # cls_token.unsqueeze(1) 扩展维度变为 [B, 1, C]
        # jt.cat 接收一个 Tensor 列表进行拼接
        x = jt.cat([cls_token.unsqueeze(1), x], dim=1)
        
        return x

class TransMIL(nn.Module):
    def __init__(self, n_classes):
        super(TransMIL, self).__init__()
        # PPEG 模块
        self.pos_layer = PPEG(dim=512)
        # Sequential 用法一致
        self._fc1 = nn.Sequential(nn.Linear(1024, 512), nn.ReLU())
        # 在 Jittor 中，成员变量如果是 Var 且需要梯度，会自动被视为参数
        # 也可以显式使用 jt.Var，这里直接初始化随机张量
        self.cls_token = jt.randn(1, 1, 512)
        self.n_classes = n_classes
        self.layer1 = TransLayer(dim=512)
        self.layer2 = TransLayer(dim=512)
        
        self.norm = nn.LayerNorm(512)
        self._fc2 = nn.Linear(512, self.n_classes)

    def execute(self, **kwargs):
        # 1. 获取数据
        # Jittor 中 float() 可以用 float32() 替代，或者直接 cast
        h = kwargs['data']
        if h.dtype != 'float32':
            h = h.float32()
        # print(f"[DEBUG] 1. input h: {h.shape}")
        # 2. 第一次降维投影
        h = self._fc1(h) # [B, n, 512]
        # print(f"[DEBUG] 2. after _fc1: {h.shape}")
        
        # ----> pad 
        H = h.shape[1]
        _H, _W = int(np.ceil(np.sqrt(H))), int(np.ceil(np.sqrt(H)))
        add_length = _H * _W - H
        # print(f"[DEBUG] 3. H={H}, _H={_H}, _W={_W}, add_length={add_length}")
        
        # 拼接填充
        # jt.cat 用法一致，dim=1
        if add_length > 0:
            h = jt.cat([h, h[:, :add_length, :]], dim=1) # [B, N, 512]
        # print(f"[DEBUG] 4. after pad: {h.shape}")

        # ----> cls_token
        B = h.shape[0]
        # 扩展维度。Jittor 中 expand 行为类似，或者使用 broadcast
        # self.cls_token 是 [1,1,512]，扩展为 [B,1,512]
        cls_tokens = self.cls_token.expand(B, -1, -1)
        
        # 拼接 Class Token 到序列头部
        h = jt.cat([cls_tokens, h], dim=1)
        # print(f"[DEBUG] 5. after cls_token cat: {h.shape}")


        # ----> Translayer x1
        h = self.layer1(h) 
        # print(f"[DEBUG] 6. after layer1: {h.shape}")

        # ----> PPEG (传入重塑后的高宽)
        h = self.pos_layer(h, _H, _W) 
        # print(f"[DEBUG] 7. after pos_layer: {h.shape}")
        
        # ----> Translayer x2
        h = self.layer2(h) 
        # print(f"[DEBUG] 8. after layer2: {h.shape}")


        # ----> cls_token 提取
        # 切片操作一致
        h = self.norm(h)[:, 0]
        # print(f"[DEBUG] 9. after norm+cls: {h.shape}")

        # ----> predict
        logits = self._fc2(h) # [B, n_classes]
        # print(f"[DEBUG] 10. after _fc2: {h.shape}")
        
        # 获取预测类别
        # jt.argmax 返回索引 (indices)，用法与 torch.argmax 基本一致
        # 若需要保持维度可用 keepdims=True，这里不需要
        Y_hat = jt.argmax(logits, dim=1)[0]
        
        # Softmax 计算概率
        Y_prob = nn.softmax(logits, dim=1)
        
        results_dict = {'logits': logits, 'Y_prob': Y_prob, 'Y_hat': Y_hat}
        return results_dict

if __name__ == "__main__":
    # 1. 生成测试数据
    # Jittor 不需要 .cuda()，数据会自动管理
    # shape: [Batch=1, Sequence_Length=6000, Dim=1024]
    data = jt.randn((1, 6000, 1024))

    # 2. 初始化模型
    # 同样不需要 .cuda()
    model = TransMIL(n_classes=2)

    # 3. 切换到评估模式
    # 这会影响 Dropout 或 BatchNorm 等层的行为
    print(model.eval())

    # 打印模型结构（可选，对应 PyTorch 的 print(model)）
    # print(model) 

    # 4. 前向传播
    # 这里的调用方式与 PyTorch 完全一致，通过 **kwargs 传入 data
    results_dict = model(data=data)

    # 5. 打印结果
    print(results_dict)
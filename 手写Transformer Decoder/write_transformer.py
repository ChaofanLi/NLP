import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import tiktoken
import torch.nn.functional as F
import os
import math
import requests

# Hyperparameter
batch_size = 4  # 批次
context_length = 16  # 上下文长度
d_model = 64  # 模型维度
num_blocks = 8  # transformer block 数量
num_heads = 4  # 头数量
learning_rate = 1e-3  # 学习率
dropout = 0.1  # 随机丢弃率
max_iters = 5000  # 整体循环次数
eval_interval = 50
eval_inters = 20
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)
TORCH_SEED = 1337
torch.manual_seed(TORCH_SEED)


if not os.path.exists("dataset/sales_textbook.txt"):
    url = "https://huggingface.co/datasets/goendalf666/sales-textbook_for_convincing_and_selling/resolve/main/sales_textbook.txt?download=true"
    r = requests.get(url)
    with open("dataset/sales_textbook.txt", "wb") as f:
        f.write(r.content)

with open("dataset/sales_textbook.txt", "r") as f:
    text = f.read()

# 语料token化
encoding = tiktoken.get_encoding("cl100k_base")  # tokenizer模型
token_text = encoding.encode(text)  # 编码
max_token_value = max(token_text)  # 最大值
tokenized_text = torch.tensor(token_text, dtype=torch.long, device=device)
# 拆分训练集和测试集
train_size = int(len(tokenized_text) * 0.9)
train_data = tokenized_text[:train_size]
valid_data = tokenized_text[train_size:]


# 前馈神经网络
class FeedForward(nn.Module):
    def __init__(self, dropout=0.1):
        super().__init__()
        self.d_model=d_model
        self.linear1 = nn.Linear(self.d_model, 4 * self.d_model)
        self.Relu = nn.ReLU()
        self.linear2 = nn.Linear(4 * self.d_model, self.d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = self.linear1(x)
        out = self.Relu(out)
        out = self.linear2(out)
        out = self.dropout(out)
        return out


# 定义Scaled Dot Product Attention
class Attention(nn.Module):
    def __init__(self, head_size: int):
        super().__init__()
        self.d_model = d_model
        self.head_size = head_size  # 头的维度
        self.context_length = context_length
        self.drop_out = dropout
        # 三个线性层代表K,Q,V
        self.key_layer = nn.Linear(
            in_features=self.d_model, out_features=self.head_size, bias=False
        )
        self.query_layer = nn.Linear(
            in_features=self.d_model, out_features=self.head_size, bias=False
        )
        self.value_layer = nn.Linear(
            in_features=self.d_model, out_features=self.head_size, bias=False
        )
        # 下三角掩码矩阵，主对角线上方元素置0
        self.register_buffer(
            "tril", torch.tril(torch.ones((self.context_length, context_length)))
        )
        # drop out 层
        self.dropout_layer = nn.Dropout(self.drop_out)

    def forward(self, x):
        B, T, C = x.shape # Batch size, Time steps(current context_length), Channels(dimensions)
        assert T <= self.context_length
        assert C == self.d_model
        # 计算q,k,v
        q = self.query_layer(x)
        k = self.key_layer(x)
        v = self.value_layer(x)
        # Q @ K^T / sqrt(d_k)
        weights = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        # weights = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        # 应用掩码
        weights = weights.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        # 应用softmax
        weights = F.softmax(weights, dim=-1)
        # 应用dropout
        weights = self.dropout_layer(weights)

        # weights@V
        out = weights @ v
        return out


class MultiHeadAttention(nn.Module):
    def __init__(self, head_size: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_size = head_size
        self.d_model = d_model
        self.context_length = context_length
        self.dropout = dropout

        self.heads = nn.ModuleList(
            [Attention(head_size=self.head_size) for _ in range(self.num_heads)]
        )
        self.projection_layer = nn.Linear(
            in_features=self.d_model, out_features=self.d_model
        )
        self.dropout = nn.Dropout(self.dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.projection_layer(out)
        out = self.dropout(out)
        return out


class TransformerBlock(nn.Module):
    def __init__(self, num_heads: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.context_length = context_length
        self.head_size = d_model // num_heads
        self.num_heads = num_heads
        self.dropout = dropout

        self.multi_head_attention_layer = MultiHeadAttention(
            head_size=self.head_size
        )  # 多头注意力机制层
        self.feed_forward_layer = FeedForward()
        self.norm_layer_1 = nn.LayerNorm(normalized_shape=self.d_model)
        self.norm_layer_2 = nn.LayerNorm(normalized_shape=self.d_model)

    def forward(self, x):
        x = x + self.multi_head_attention_layer(
            self.norm_layer_1(x)
        )  # 存疑，归一化层是应该在外面还是在里面？
        x = x + self.feed_forward_layer(self.norm_layer_2(x))
        return x


class TransformerLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.d_model = d_model
        self.context_length = context_length
        self.num_heads = num_heads
        self.num_blocks = num_blocks
        self.dropout = dropout
        self.max_token_value = max_token_value
        self.token_embedding_lookup_table = nn.Embedding(
            num_embeddings=self.max_token_value + 1, embedding_dim=self.d_model
        )  # token和embedding的映射表
        self.transformer_blocks = nn.Sequential(
            *(
                [
                    TransformerBlock(num_heads=self.num_heads)
                    for _ in range(self.num_blocks)
                ]
                + [nn.LayerNorm(self.d_model)]
            )
        )  # transformer block层
        self.language_model_out_linear_layer = nn.Linear(
            in_features=self.d_model, out_features=self.max_token_value
        )  # 线性输出层

    def forward(self, idx, targets=None):
        B, T = idx.shape
        # 设置position-embedding 映射表
        position_embedding_lookup_table = torch.zeros(self.context_length, self.d_model)
        position = torch.arange(0, self.context_length, dtype=torch.float).unsqueeze(1)
        # PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
        # PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2).float()
            * (-math.log(10000.0) / self.d_model)
        )  # 位置编码分母
        position_embedding_lookup_table[:, 0::2] = torch.sin(
            position * div_term
        )  # 偶数位位置编码
        position_embedding_lookup_table[:, 1::2] = torch.cos(
            position * div_term
        )  # 奇数位位置编码
        position_embedding = position_embedding_lookup_table[:T, :].to(
            device
        )  # 转移到对应设备上
        x = (
            self.token_embedding_lookup_table(idx) + position_embedding
        )  # token embedding 和 position embedding叠加
        x = self.transformer_blocks(x)  # transformer block层
        logits = self.language_model_out_linear_layer(x)

        # 计算损失
        if targets is not None:
            B, T, C = logits.shape
            # 形状匹配
            logits_reshaped = logits.view(B * T, C)
            targets_reshaped = targets.view(B * T)
            # 交叉熵计算损失
            loss = F.cross_entropy(input=logits_reshaped, target=targets_reshaped)
        else:
            loss = None
        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            # 将idx裁剪为位置嵌入表的最大尺寸
            idx_crop = idx[:, -self.context_length]  # idx:(B,T)
            # 前向计算
            logits, loss = self(idx_crop)  # logits:(B,T,C)
            # 提取最后一个时间步输出
            logits_last_timestep = logits[:, -1, :]  # logits_last_timestep:(B,C)
            # 应用Softmax计算概率
            probs = F.softmax(input=logits_last_timestep, dim=-1)  # probs:(B,C)
            # 根据概率进行采样
            idx_next = torch.multinomial(input=probs, num_samples=1)  # idx_next:(B,1)
            # 将采样结果拼接到原来的序列中
            idx = torch.cat((idx, idx_next), dim=1)  # idx:(B,T+1)
        return idx


# 模型初始化
model = TransformerLanguageModel()
model = model.to(device)


# 获取批次数据
def get_batch(split: str):
    data = train_data if split == "train" else valid_data
    idxs = torch.randint(low=0, high=len(data) - context_length, size=(batch_size,))
    x = torch.stack([data[idx : idx + context_length] for idx in idxs]).to(device)
    y = torch.stack([data[idx + 1 : idx + context_length + 1] for idx in idxs]).to(
        device
    )
    return x, y


# 损失计算
@torch.no_grad()
def estimate_loss():
    # 损失评价
    out = {}  # 记录损失
    model.eval()
    for split in ["train", "valide"]:
        losses = torch.zeros(eval_inters)  # 初始化损失张量
        for k in range(eval_inters):  # 记录每次迭代的损失
            x_batch, y_batch = get_batch(split)
            logits, loss = model(x_batch, y_batch)
            losses[k] = loss.item()
        out[split] = losses.mean()  # 计算本次迭代损失均值
    model.train()
    return out


# 设置优化器
optimizer = torch.optim.AdamW(params=model.parameters(), lr=learning_rate)
tracked_losses = []
for step in range(max_iters):
    if step % eval_inters == 0 or step == max_iters - 1:
        losses = estimate_loss()
        tracked_losses.append(losses)
        print(
            "Step:",
            step,
            "Training Loss:",
            round(losses["train"].item(), 3),
            "Validation Loss:",
            round(losses["valide"].item(), 3),
        )

    xb, yb = get_batch("train")
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# 存储模型
torch.save(model, model.state_dict(), "model-ckpt.pt")

# 文本生成
model.eval()
start = "The salesperson"
start_idx = encoding.encode(start)  # 对输入进行tokenizer
x = (torch.tensor(start_idx, dtype=torch.long, device=device)[None, ...])
y = model.generate(x, max_new_tokens=100)
print("-" * 30)
print(encoding.decode(y[0].tolist()))
print("-" * 30)

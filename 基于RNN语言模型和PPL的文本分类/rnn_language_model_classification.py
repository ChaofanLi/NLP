# coding:utf8

import torch
import torch.nn as nn
import numpy as np
import math
import random
import os


# 搭建模型结构
class LanguageModel(nn.Module):
    def __init__(self, input_dim, vocab):
        super(LanguageModel, self).__init__()
        self.embedding = nn.Embedding(len(vocab) + 1, input_dim)
        self.layer = nn.RNN(input_dim, input_dim, num_layers=2, batch_first=True)
        self.classify = nn.Linear(input_dim, len(vocab) + 1)
        self.dropout = nn.Dropout(0.1)
        self.loss = nn.functional.cross_entropy

    def forward(self, x, y=None):
        x = self.embedding(x) # in:[batch_size,sentence_len],out:[batch_size,sentence_len,input_dim]
        x, _ = self.layer(x) # out:[batch_size,sentence_len,input_dim]
        x = x[:, -1, :] # out:[batch_size,input_dim]
        x = self.dropout(x) # out:[batch_size,input_dim]
        y_pred = self.classify(x) # out:[batch_size,len(vocab) + 1]
        if y is not None:
            return self.loss(y_pred, y)
        else:
            return torch.softmax(y_pred, dim=-1) # out:[batch_size,len(vocab) + 1]


# 构建字符集
def build_vocab_from_corpus(path):
    vocab = set()
    with open(path, encoding="utf8") as f:
        for index, char in enumerate(f.read()):
            vocab.add(char)
    vocab.add("<UNK>")  # 增加一个unk token用来处理未登录词
    writer = open("vocab.txt", "w", encoding="utf8")
    for char in sorted(vocab):
        writer.write(char + "\n")
    return vocab


# 构建词表
def build_vocab(vocab_path):
    vocab = {}
    with open(vocab_path, encoding="utf8") as f:
        for index, line in enumerate(f):
            char = line[:-1]  # 去掉结尾换行符
            vocab[char] = index + 1  # 留出0位给pad token
        vocab["\n"] = 1
    return vocab


# 加载语料
def load_corpus(path):
    return open(path, encoding="utf8").read()


# 随机生成训练样本
def build_sample(vocab, window_size, corpus):
    start = random.randint(0, len(corpus) - 1 - window_size)
    end = start + window_size
    window = corpus[start:end]
    target = corpus[end]
    x = [vocab.get(word, vocab["<UNK>"]) for word in window]  # 将字转换成序号
    y = vocab.get(target, vocab["<UNK>"])
    return x, y


# 构造数据集
def build_dataset(sample_length, vocab, window_size, corpus):
    dataset_x = []
    dataset_y = []
    for i in range(sample_length):
        x, y = build_sample(vocab, window_size, corpus)
        dataset_x.append(x)
        dataset_y.append(y)
    return torch.LongTensor(dataset_x), torch.LongTensor(dataset_y)


# 模型实例化
def build_model(vocab, char_dim):
    model = LanguageModel(char_dim, vocab)
    return model

# 模型训练
def train(corpus_path, save_weight=True):
    epoch_num = 10  # 训练轮数
    batch_size = 128  # 每次训练样本个数
    train_sample = 10000  # 每轮训练总共训练的样本总数
    char_dim = 128  # 每个字的维度
    window_size = 6  # 样本文本长度
    vocab = build_vocab("vocab.txt")  # 建立字表
    corpus = load_corpus(corpus_path)  # 加载语料
    model = build_model(vocab, char_dim)  # 建立模型
    if torch.cuda.is_available():
        model = model.cuda()
    optim = torch.optim.Adam(model.parameters(), lr=0.001)  # 建立优化器
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch in range(int(train_sample / batch_size)):
            x, y = build_dataset(batch_size, vocab, window_size, corpus)  # 构建一组训练样本
            if torch.cuda.is_available():
                x, y = x.cuda(), y.cuda()
            optim.zero_grad()  # 梯度归零
            loss = model(x, y)  # 计算loss
            watch_loss.append(loss.item())
            loss.backward()  # 计算梯度
            optim.step()  # 更新权重
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
    if not save_weight:
        return
    else:
        base_name = os.path.basename(corpus_path).replace("txt", "pth")
        model_path = os.path.join("model", base_name)
        torch.save(model.state_dict(), model_path)
        return


# 每个类型语料训练一个语言模型
def train_all():
    for path in os.listdir("corpus"):
        corpus_path = os.path.join("corpus", path)
        train(corpus_path)


if __name__ == "__main__":
    train_all()

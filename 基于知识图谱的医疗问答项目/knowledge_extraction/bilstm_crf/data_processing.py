# coding=utf-8
import re, os
from itertools import chain
from collections import Counter
from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
import numpy as np


# 定义NerDataProcessor类
class NerDataProcessor(object):
    def __init__(self, max_len, vocab_size):
        super(NerDataProcessor, self).__init__()
        self.max_len = max_len  # 序列的最大长度
        self.vocab_size = vocab_size  # 词汇表的大小
        self.word2id = {}  # 单词到索引的映射字典

        self.tags = []  # 标签列表
        self.tag2id = {}  # 标签到索引的映射字典
        self.id2tag = {}  # 索引到标签的映射字典,用作预测值解码

        self.class_nums = 0  # 类别数目
        self.sample_nums = 0  # 样本数目

    # 读取数据
    def read_data(self, path, is_training_data=True):
        """
        数据格式如下（分隔符为空格）：
        便 B_disease
        秘 I_disease
        两 O
        个 O
        多 O
        月 O
        """
        X = []  # 存储句子
        y = []  # 存储标签
        sentence = []  # 临时存储单个句子
        labels = []  # 临时存储单个句子的标签
        split_pattern = re.compile(r'[；;。，、？！\.\?,! ]')  # 分隔符模式

        # 打开文件并逐行读取
        with open(path, 'r', encoding='utf8') as f:
            for line in f.readlines():
                # 每行为一个字符和其标签，中间用tab或者空格隔开
                line = line.strip().split()# 去除首位空白字符并按空格拆分

                # 如果行为空或长度小于2，则表示句子结束
                if not line or len(line) < 2:
                    X.append(sentence.copy())  # 将完整句子添加到X
                    y.append(labels.copy())  # 将完整标签添加到y
                    sentence.clear()  # 清空临时句子存储
                    labels.clear()  # 清空临时标签存储
                    continue

                # 提取单词和标签
                word, tag = line[0], line[1]
                tag = tag if tag != 'o' else 'O'  # 将小写'o'转换为大写'O'

                # 如果匹配到分隔符并且句子长度达到最大长度
                if split_pattern.match(word) and len(sentence) >= self.max_len:
                    sentence.append(word)  # 添加分隔符到句子
                    labels.append(tag)  # 添加分隔符对应的标签
                    X.append(sentence.copy())  # 将完整句子添加到X
                    y.append(labels.copy())  # 将完整标签添加到y
                    sentence.clear()  # 清空临时句子存储
                    labels.clear()  # 清空临时标签存储
                else:
                    sentence.append(word)  # 添加单词到句子
                    labels.append(tag)  # 添加标签到标签列表

            # 处理最后一个句子（如果存在）
            if len(sentence):
                X.append(sentence.copy())  # 将最后一个句子添加到X
                sentence.clear()  # 清空临时句子存储
                y.append(labels.copy())  # 将最后一个标签添加到y
                labels.clear()  # 清空临时标签存储

        if is_training_data:
            # 获取所有标签，并创建标签到索引和索引到标签的映射
            self.tags = sorted(list(set(chain(*y))))  # 使用chain将所有标签链在一起，然后去重并排序
            self.tag2id = {tag: idx + 1 for idx, tag in enumerate(self.tags)}  # 为每个标签分配一个唯一的索引，索引从1开始
            self.id2tag = {idx + 1: tag for idx, tag in enumerate(self.tags)}  # 创建索引到标签的映射，索引从1开始

            # 为标签和索引映射添加padding
            self.tag2id['padding'] = 0  # 为padding标签分配索引0
            self.id2tag[0] = 'padding'  # 为索引0分配标签padding
            self.class_nums = len(self.id2tag)  # 计算类别数目，即标签总数
            self.sample_nums = len(X)  # 计算样本数目

            vocab = list(chain(*X))  # 使用chain将所有句子链在一起，形成词汇列表
            print("vocab length", len(set(vocab)))  # 打印词汇表长度（去重后的词汇数目）
            print(self.id2tag)  # 打印索引到标签的映射字典
            vocab = Counter(vocab).most_common(self.vocab_size - 2)  # 统计词频，取前vocab_size-2个最常见的词
            vocab = [v[0] for v in vocab]  # 提取词汇表中的词
            for index, word in enumerate(vocab):
                self.word2id[word] = index + 2  # 为每个词分配一个唯一的索引，索引从2开始

            # OOV（未登录词）为1，padding为0
            self.word2id['padding'] = 0  # 为padding词分配索引0
            self.word2id['OOV'] = 1  # 为OOV词分配索引1

        return X, y

    def encode(self, X, y):
        """将训练样本映射成数字，并进行padding
        将标签进行 one-hot 编码"""

        # 将每个句子中的每个单词映射为索引，如果单词不在字典中，则使用索引1（OOV）
        X = [[self.word2id.get(word, 1) for word in x] for x in X]
        # 对句子进行填充，使其长度统一为 max_len，填充值为0（padding）
        X = pad_sequences(X, maxlen=self.max_len, value=0)

        # 将每个句子的每个标签映射为索引，如果标签不在字典中，则使用索引0（padding）
        y = [[self.tag2id.get(tag, 0) for tag in t] for t in y]
        # 对标签进行填充，使其长度统一为 max_len，填充值为0（padding）
        y = pad_sequences(y, maxlen=self.max_len, value=0)

        # 定义一个将标签索引转换为 one-hot 编码的函数
        def label_to_one_hot(index: []):
            data = []  # 存储 one-hot 编码后的数据
            for line in index:
                data_line = []  # 存储单个句子的 one-hot 编码
                for i, index in enumerate(line):
                    line_line = [0] * self.class_nums  # 创建一个全为0的列表，长度为类别数
                    line_line[index] = 1  # 将对应索引的位置置为1
                    data_line.append(line_line)  # 将 one-hot 编码添加到句子列表
                data.append(data_line)  # 将句子列表添加到数据列表
            return np.array(data)  # 将数据转换为 numpy 数组

        # 将标签数据转换为 one-hot 编码
        y = label_to_one_hot(index=y)
        print(y.shape)  # 打印 one-hot 编码后的标签数据的形状

        return X, y  # 返回处理后的句子和标签

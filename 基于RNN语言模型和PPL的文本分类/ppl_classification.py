# coding:utf8

import torch
import math
import os
from rnn_language_model_classification import build_model, build_vocab


# 加载训练好的模型
def load_trained_language_model(path):
    char_dim = 128  # 每个字的维度,与训练时保持一直
    window_size = 6  # 样本文本长度,与训练时保持一直
    vocab = build_vocab("vocab.txt")  # 加载字表
    model = build_model(vocab, char_dim)  # 加载模型
    model.load_state_dict(torch.load(path))  # 加载训练好的模型权重
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()
    model.window_size = window_size
    model.vocab = vocab
    return model


# 计算文本ppl
def calc_perplexity(sentence, model):
    prob = 0
    with torch.no_grad():
        for i in range(1, len(sentence)):
            start = max(0, i - model.window_size)
            window = sentence[start:i]
            x = [model.vocab.get(char, model.vocab["<UNK>"]) for char in window]
            x = torch.LongTensor([x])
            target = sentence[i]
            target_index = model.vocab.get(target, model.vocab["<UNK>"])
            if torch.cuda.is_available():
                x = x.cuda()
            # model(x).shape   [1, 4263]
            pred_prob_distribute = model(x)[0]
            target_prob = pred_prob_distribute[target_index]
            prob += math.log(target_prob, 10)
    return 2 ** (prob * (-1 / len(sentence)))


# 加载所有训练好的所有模型
def load_models():
    model_paths = os.listdir(os.path.dirname(os.path.abspath(__file__)) + "/model")
    class_to_model = {}
    for model_path in model_paths:
        class_name = model_path.replace(".pth", "")
        model_path = os.path.join("model", model_path)
        class_to_model[class_name] = load_trained_language_model(model_path)
    return class_to_model


# 使用不同的语言模型计算句子的ppl
def text_classification_based_on_language_model(class_to_model, sentence):
    ppl = []
    for class_name, class_lm in class_to_model.items():
        ppl.append([class_name, calc_perplexity(sentence, class_lm)])
    ppl = sorted(ppl, key=lambda x: x[1])
    print(sentence)
    print(ppl[0: 3])
    print("==================")
    return ppl

if __name__ == "__main__":
    sentence = ["在全球货币体系出现危机的情况下",
                "点击进入双色球玩法经典选号图表",
                "慢时尚服饰最大的优点是独特",
                "做处女座朋友的人真的很难",
                "网戒中心要求家长全程陪护",
                "在欧巡赛扭转了自己此前不利的状态",
                "选择独立的别墅会比公寓更适合你",
                ]

    class_to_model = load_models()
    for s in sentence:
        text_classification_based_on_language_model(class_to_model, s)

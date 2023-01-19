#!/usr/bin/env python
# -*- coding:utf-8 -*-
# author:HanZhou
# datetime:2022/5/6 14:53
# software: PyCharm
import json
import os
import random
import gc
import matplotlib.pyplot as plt
# import pandas as pd
import numpy as np
import torch
from keras_preprocessing.sequence import pad_sequences

from torch.utils.data import TensorDataset, RandomSampler, DataLoader, SequentialSampler
from tqdm import trange, tqdm
from torch.optim import AdamW
from transformers import BertForMaskedLM, BertConfig, BertForSequenceClassification  # , AdamW
from transformers import BertTokenizer, BertModel, TrainingArguments
from transformers import logging
import tokenizers
# from pytorch_pretrained_bert import BertAdam

import method

logging.set_verbosity_error()

data_path = './data/text_classification/chemprot/'  # 数据集路径
# bert_model = './Bert/bert-base-uncased/'  # 预训练模型的文件


label_num = method.label2num(data_path)  # 获取文件中的句子总共的类别数目
num_token = len(label_num)  # 分类样本数


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False


def data_read(bert_model, data_type):
    """
    将文本分类任务中的数据进行读取预处理
    :param data_type: 文本的类型：train，test，dev，用于构建地址
    :return: input_ids, labels, attention_masks
    """
    sentencses = []
    labels = []
    path = data_path + data_type + '.txt'
    print("%s data loading------" % data_type)
    with open(path, 'r', encoding='utf-8') as file:
        for line in tqdm(file.readlines()):
            dic = json.loads(line)
            sent = '[CLS] ' + dic['text'] + ' [SEP]'  # 获取句子
            label_token = dic['label']  # 获取句子标签
            label = int(label_num[label_token])  # 根据字典将标签转为整型

            sentencses.append(sent)
            labels.append(label)

    tokenizer = BertTokenizer.from_pretrained(bert_model, do_lower_case=True)
    tokenized_sents = [tokenizer.tokenize(sent) for sent in sentencses]

    # 定义句子最大长度（512）
    MAX_LEN = 128

    # 将分割后的句子转化成数字  word-->idx
    input_ids = [tokenizer.convert_tokens_to_ids(sent) for sent in tokenized_sents]

    # 做PADDING,这里使用keras的包做pad
    # 大于128做截断，小于128做PADDING
    input_ids = pad_sequences(input_ids, maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")

    # 建立mask
    attention_masks = []
    for seq in input_ids:
        seq_mask = [float(i > 0) for i in seq]
        attention_masks.append(seq_mask)

    return input_ids, labels, attention_masks


def data_load(bert_model, file_type, device, batch_size):
    """
    将Bert初始处理过的数据转换成为Pytorch识别的数据
    :param file_type: 文件类型，用于识别应该读取的文件
    :param device: 是否使用GPU
    :param batch_size: 批处理的尺寸
    :return: DataLoader
    """
    inputs, labels, masks = data_read(bert_model, file_type)  # 获取数据

    # 将数据集转化成tensor
    inputs = torch.tensor(inputs).to(device)
    labels = torch.tensor(labels).to(device)
    masks = torch.tensor(masks).to(device)

    # 生成dataloader
    data = TensorDataset(inputs, masks, labels)
    sampler = RandomSampler(data)
    dataloader = DataLoader(data, sampler=sampler, batch_size=batch_size, shuffle=False)

    return dataloader


def re_param(model, param_path):
    params_list = torch.load(param_path)  # 加载客户端的参数

    for name, params in model.named_parameters():  # 更新客户端的训练结果
        if name in params_list.keys():
            print("re_param!")
            # print(name)
            params.data = params_list[name].data

    return model


def share_param(model, param_path):
    params_dict = torch.load(param_path)  # 加载客户端的参数
    params_list = []
    for value in params_dict.values():
        params_list.append(value)

    for layer in model.bert.encoder.layer:
        for i, p in enumerate(layer.parameters()):
            p.data = params_list[i]

    return model


def model_test(model, test_dataloader, device, model_type):
    """
    对模型的准确度在特定数据集上进行测试
    :param model: 待测试的模型
    :param test_dataloader: 评测数据集
    :param device: 是否使用GPU
    :param model_type: 测试模型种类
    """
    model.eval()
    n_true = 0
    n_total = 0
    for batch in test_dataloader:
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, b_labels = batch
        with torch.no_grad():
            logits = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)[0]
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()

        pred_flat = np.argmax(logits, axis=1).flatten()
        labels_flat = label_ids.flatten()
        n_true += np.sum(pred_flat == labels_flat)
        n_total += len(labels_flat)

    accuracy = n_true / n_total
    print(model_type + "Accuracy: {}".format(accuracy))

    return accuracy


def train_classier(learning_rate, bert_model, epochs, device, batch_size, change_param=None, share=None):
    """
    在文本分类任务上对模型进行训练
    :param learning_rate: 学习率
    :param bert_model:待训练模型
    :param epochs: 训练轮次
    :param device: 是否使用GPU
    :param batch_size: 批处理数据大小
    :param change_param: 是否重构某些参数
    :param share: 是否根据已训练参数共享修改其余层参数
    """

    # learning_rate = 2e-5  # bigger
    # 设置随机数种子
    setup_seed(24)

    modelConfig = BertConfig.from_pretrained(bert_model)
    modelConfig.num_labels = num_token  # 设置分类模型的输出个数

    model = BertForSequenceClassification.from_pretrained(bert_model, config=modelConfig)

    if change_param:
        # 根据已经保存的文件重新构建模型的某些参数
        model = re_param(model, change_param)

    if share:
        # 根据已经保存某一层参数共享构建其余层的参数
        model = share_param(model, share)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
    )
    # print(optimizer.param_groups)

    train_dataloader = data_load(bert_model, 'train', device, batch_size)
    validation_dataloader = data_load(bert_model, 'dev', device, batch_size)
    model.to(device)
    # 训练开始

    val_acc = []  # 训练过程中的验证集精度
    # 测试训练结果
    test_acc = []
    test_dataloader = data_load(bert_model, 'test', device, batch_size)

    for i in range(epochs):
        print("Epochs:%d/%d" % ((i + 1), epochs))
        # 训练开始
        model.train()
        tr_loss = 0
        nb_tr_steps = 0
        for batch in tqdm(train_dataloader):
            batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask, labels = batch
            optimizer.zero_grad()
            # 取第一个位置，BertForSequenceClassification第一个位置是Loss，第二个位置是[CLS]的logits
            loss = model(input_ids, token_type_ids=None, attention_mask=input_mask, labels=labels)[0]

            loss.backward()
            optimizer.step()

            tr_loss += loss.item()
            nb_tr_steps += 1

        print("Train loss: {}".format(tr_loss / nb_tr_steps))
        # 模型评估
        v_acc = model_test(model, validation_dataloader, device, 'Val')
        val_acc.append(v_acc)
        t_acc = model_test(model, test_dataloader, device, 'Test')
        test_acc.append(t_acc)

    return val_acc, test_acc


# def find_lr(bert_model):
#     epochs = 15
#     batch_size = 64  # 32
#     device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
#
#     legend = []
#     Val_Acc = []
#     Test_Acc = []
#     Max_Acc = []
#
#     for x in range(2, 9):
#         learn_rate = x / (10 ** 5)
#         print("DownStream Learning Rate:" + str(learn_rate) + "训练中------")
#         # bert_model0 = './model/skip-mlm-new/'  # 预训练模型的文件
#         val_acc0, test_acc0 = train_classier(learn_rate, bert_model, epochs, device, batch_size, True)
#
#         max_acc = (max(val_acc0) + max(test_acc0)) / 2
#
#         legend.append(str(bert_model)+str(learn_rate))
#         Val_Acc.append(val_acc0)
#         Test_Acc.append(test_acc0)
#         Max_Acc.append(max_acc)
#
#     # method.plot_res('Val Acc', legend, Val_Acc)
#     # method.plot_res('Test Acc', legend, Test_Acc)
#     # method.get_test_acc(legend, Val_Acc, Test_Acc)
#     #
#     # print("(Max(val_acc)+Max(test_acc))/2:", end=" ")
#     # print(max(Max_Acc), end=" ")
#     # print("Learning Rate: %d e-5" % (2 + Max_Acc.index(max(Max_Acc))))
#     return legend, Val_Acc, Test_Acc, Max_Acc
#
#
# def find_pretrain_downstream_learn_date():
#     res_list = []
#     for x in range(2, 9):
#         learn_rate = x / (10 ** 5)
#         model_save = './outputs/learnRateTest/' + str(learn_rate) + '/'
#         print("Pretrain Learning Rate:" + str(learn_rate) + "训练中------")
#
#         res = find_lr(model_save)
#         res_list.append(res)
#     print("\n")
#     print("--------------------------------------------------------")
#     print("\n")
#     for i, r in enumerate(res_list):
#         print("Further Pretraining Learning Rate: %de-5 结果如下：" % (i+2))
#         print("--------------------------------------------------------")
#         method.get_test_acc(r[0], r[1], r[2])
#
#         print("(Max(val_acc)+Max(test_acc))/2:", end=" ")
#         print(max(r[3]), end=" ")
#         print("Learning Rate: %de-5" % (2 + r[3].index(max(r[3]))))
#         print("--------------------------------------------------------")
#         print("\n")


def main():
    epochs = 15
    batch_size = 128  # 32
    learn_rate = 5e-5
    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

    print("Bert模型训练中------")
    bert_model0 = './model/bert-base-uncased/'  # 预训练模型的文件
    val_acc0, test_acc0 = train_classier(learn_rate, bert_model0, epochs, device, batch_size)

    print("SciBert模型训练中------")
    bert_model1 = './model/SciBert/'  # 预训练模型的文件
    val_acc1, test_acc1 = train_classier(learn_rate, bert_model1, epochs, device, batch_size)

    print("FedAvg模型训练中------")
    bert_model2 = './outputs/fed_avg/'  # 预训练模型的文件
    val_acc2, test_acc2 = train_classier(learn_rate, bert_model2, epochs, device, batch_size)

    legend = ['Bert', 'SciBert', 'FedAvg']
    Val_Acc = [val_acc0, val_acc1, val_acc2]
    Test_Acc = [test_acc0, test_acc1, test_acc2]

    method.plot_res('Val Acc', legend, Val_Acc)
    method.plot_res('Test Acc', legend, Test_Acc)
    method.get_test_acc(legend, Val_Acc, Test_Acc)


if __name__ == "__main__":
    main()
    # test_bug()
    # find_lr()
    # find_pretrain_downstream_learn_date()

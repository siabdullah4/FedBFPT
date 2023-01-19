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

from torch.utils.data import TensorDataset, RandomSampler, DataLoader, SequentialSampler, random_split
from tqdm import trange, tqdm
from torch.optim import AdamW
from transformers import BertForMaskedLM, BertConfig, BertForSequenceClassification  # , AdamW
from transformers import BertTokenizer, BertModel, TrainingArguments
from transformers import logging
import tokenizers
# from pytorch_pretrained_bert import BertAdam

import method
import utils

logging.set_verbosity_error()

data_path = './data/text_classification/acl-arc/'  # 数据集路径
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


def data_load(bert_model, file_type, device, batch_size, split_zie=-0.1):
    """
    将Bert初始处理过的数据转换成为Pytorch识别的数据
    :param bert_model:模型存储地址
    :param file_type: 文件类型，用于识别应该读取的文件
    :param device: 是否使用GPU
    :param batch_size: 批处理的尺寸
    :param split_zie: 分批出来部分数据集用于训练
    :return: DataLoader
    """
    inputs, labels, masks = data_read(bert_model, file_type)  # 获取数据

    # 将数据集转化成tensor
    inputs = torch.tensor(inputs).to(device)
    labels = torch.tensor(labels).to(device)
    masks = torch.tensor(masks).to(device)

    # 生成dataloader
    data = TensorDataset(inputs, masks, labels)

    if split_zie != -0.1:
        #  随机划分部分数据集用于训练
        data_size = int(split_zie * len(data))
        print("Choose " + str(split_zie) + " Data for " + file_type + "ing!")
        data, remain_data = random_split(dataset=data, lengths=[data_size, len(data) - data_size])

    sampler = RandomSampler(data)
    dataloader = DataLoader(data, sampler=sampler, batch_size=batch_size, shuffle=False)

    return dataloader


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


def train_classier(learning_rate, bert_model, epochs, device, batch_size,
                   hidden_layers=12, change_param=None, share=None, seed=40, drop_layer=3):
    """
    在文本分类任务上对模型进行训练
    :param learning_rate: 学习率
    :param bert_model:待训练模型
    :param epochs: 训练轮次
    :param device: 是否使用GPU
    :param batch_size: 批处理数据大小
    :param hidden_layers: Bert中间层的数目
    :param change_param: 是否重构某些参数
    :param share: 是否根据已训练参数共享修改其余层参数
    :param seed: 随机数种子
    :param drop_layer: 跳过的层
    """

    # learning_rate = 2e-5  # bigger
    # 设置随机数种子
    setup_seed(seed)

    param_container = utils.create_container()  # 制作本地参数容器

    modelConfig = BertConfig.from_pretrained(bert_model)
    modelConfig.num_hidden_layers = hidden_layers  # 相当于构建一个小模型，transformer层只有六层
    modelConfig.num_labels = num_token  # 设置分类模型的输出个数

    model = BertForSequenceClassification.from_pretrained(bert_model, config=modelConfig)

    if hidden_layers != 12:
        model = utils.rebuild_model(model, param_container, layer_length=hidden_layers - drop_layer,
                                    drop_layer=drop_layer, ori_layer=drop_layer - 1)

    if change_param:
        # 根据已经保存的文件重新构建模型的某些参数
        model = utils.re_param(model, change_param)

    if share:
        # 根据已经保存某一层参数共享构建其余层的参数
        model = utils.share_param(model, share)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
    )
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.7)
    # print(optimizer.param_groups)

    model.to(device)
    # 训练开始

    val_acc = []  # 训练过程中的验证集精度
    # 测试训练结果
    test_acc = []

    train_dataloader = data_load(bert_model, 'train', device, batch_size)
    validation_dataloader = data_load(bert_model, 'dev', device, batch_size)
    test_dataloader = data_load(bert_model, 'test', device, batch_size)

    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name)

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

        scheduler.step()
        print("Train loss: {}".format(tr_loss / nb_tr_steps))
        # 模型评估
        v_acc = model_test(model, validation_dataloader, device, 'Val')
        val_acc.append(v_acc)
        t_acc = model_test(model, test_dataloader, device, 'Test')
        test_acc.append(t_acc)

    return val_acc, test_acc


def down_classier(bert_model, train_dataloader, device, learning_rate, optimizer, scheduler):
    bert_model.to(device)

    bert_model.train()
    tr_loss = 0
    nb_tr_steps = 0
    for batch in tqdm(train_dataloader):
        batch = tuple(t.to(device) for t in batch)
        input_ids, input_mask, labels = batch
        optimizer.zero_grad()
        # 取第一个位置，BertForSequenceClassification第一个位置是Loss，第二个位置是[CLS]的logits
        loss = bert_model(input_ids, token_type_ids=None, attention_mask=input_mask, labels=labels)[0]

        loss.backward()
        optimizer.step()

        tr_loss += loss.item()
        nb_tr_steps += 1

    scheduler.step()
    print("Train loss: {}".format(tr_loss / nb_tr_steps))

    return bert_model


def find_lr():
    seed = 40
    epochs = 15
    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

    print(seed)
    print(device)

    domain = 'Medicine'
    Pro3_6_e5 = './outputs/LayerModel/' + domain + '/cls_Pro3_6_e5/layer0_2.pt'

    model_path = './model/bert-base-uncased/'

    Index_list = []
    legend = []
    Val_Acc = []
    Test_Acc = []
    Max_Acc = []

    for x in range(2, 9):
        learn_rate = x / (10 ** 6)
        for i in range(6, 0, -1):
            batch_size = 2 ** i
            print("DownStream Learning Rate:" + str(learn_rate) + "训练中------")
            print("Batch Size:" + str(batch_size) + "训练中------")
            val_acc0, test_acc0 = train_classier(learn_rate, model_path, epochs, device, batch_size,
                                                 change_param=Pro3_6_e5, seed=seed)

            index = val_acc0.index(max(val_acc0))

            max_acc = test_acc0[index]

            Index_list.append(index)
            legend.append("Lr:" + str(learn_rate) + " BSize: " + str(batch_size))
            Val_Acc.append(val_acc0)
            Test_Acc.append(test_acc0)
            Max_Acc.append(max_acc)
            print("--------------------------------------------------------")

    best_index = Max_Acc.index(max(Max_Acc))
    print("Best 参数为：")
    print(legend[best_index])
    print('Highest Val Acc:%f,Epoch:%d' % (Val_Acc[best_index][Index_list[best_index]], Index_list[best_index]),
          end=', \t')
    print('Corresponding Test Acc:%f' % Test_Acc[best_index][Index_list[best_index]])
    print("Val Acc:")
    print(Val_Acc[best_index])
    print("Test ACC")
    print(Test_Acc[best_index])


def test_Pro3_6():
    epoch = 15
    batch_size = 32
    learn_rate = 6e-5
    seed = 40
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    print(epoch)
    print(batch_size)
    print(learn_rate)
    print(seed)
    print(device)

    domain = 'Computer'
    Pro3_6_e5 = './outputs/LayerModel/' + domain + '/cls_Pro3_6_e5/layer0_2.pt'
    cls_Pro3_6_e5_Skewed = './outputs/LayerModel/' + domain + '/cls_Pro3_6_e5_Skewed/layer0_2.pt'
    Bert_Center = './outputs/LayerModel/' + domain + '/Bert_Center/epoch_5/layer.pt'
    Bert_FL = './outputs/LayerModel/' + domain + '/Bert_FL/epoch_5/fed_avg.pt'

    Distill_Path = './outputs/Baseline/' + domain + '/DistillBert/epoch_5/fed_avg.pt'
    TinyBert_4_Path = './outputs/Baseline/' + domain + '/TinyBert_4/epoch_5/fed_avg.pt'
    TinyBert_6_Path = './outputs/Baseline/' + domain + '/TinyBert_6/epoch_5/fed_avg.pt'

    model_path = './model/bert-base-uncased/'
    model_path0 = './model/DistillBert/'
    model_path1 = './model/TinyBert_4/'
    model_path2 = './model/TinyBert_6/'

    print("Pro3_6_e5训练中------")
    val_acc0, test_acc0 = train_classier(learn_rate, model_path0, epoch, device, batch_size,
                                         change_param=cls_Pro3_6_e5_Skewed, seed=seed)

    Val_acc = [val_acc0]
    Test_acc = [test_acc0]

    legend = ['cls_Pro3_6_e5_Skewed']

    method.get_test_acc(legend, Val_acc, Test_acc)
    print("--------------------")
    for i in range(len(legend)):
        print(legend[i], end=" ")
        print("Max Test Acc: ")
        print(max(Test_acc[i]))
    print("**************************************************************")
    for i in range(len(legend)):
        print(legend[i])
        print("Val Acc:")
        print(Val_acc[i])
        print("Test Acc:")
        print(Test_acc[i])
        print("**************************************************************")


def main():
    epochs = 15
    batch_size = 128  # 32
    learn_rate = 5e-5
    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

    print("Bert模型训练中------")
    bert_model0 = './model/bert-base-uncased/'  # 预训练模型的文件
    val_acc0, test_acc0 = train_classier(learn_rate, bert_model0, epochs, device, batch_size)

    print("LayerBert_03模型训练中------")
    val_acc1, test_acc1 = train_classier(learn_rate, bert_model0, epochs, device, batch_size, hidden_layers=3)

    print("LayerBert_06模型训练中------")
    val_acc2, test_acc2 = train_classier(learn_rate, bert_model0, epochs, device, batch_size, hidden_layers=6)

    legend = ['Bert', 'LayerBert_03', 'LayerBert_06']
    Val_Acc = [val_acc0, val_acc1, val_acc2]
    Test_Acc = [test_acc0, test_acc1, test_acc2]

    method.plot_res('Val Acc', legend, Val_Acc)
    method.plot_res('Test Acc', legend, Test_Acc)
    method.get_test_acc(legend, Val_Acc, Test_Acc)
    print("--------------------")
    for i in range(len(legend)):
        print(legend[i], end=" ")
        print("Max Test Acc: ")
        print(max(Test_Acc[i]))


if __name__ == "__main__":
    # main()
    test_Pro3_6()
    # find_lr()

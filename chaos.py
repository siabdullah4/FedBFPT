#!/usr/bin/env python
# -*- coding:utf-8 -*-
# author:HanZhou
# datetime:2022/7/25 14:27
# software: PyCharm
import json
import os
import random
import time

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# os.environ["CUDA_VISIBLE_DEVICES"] = '3,4,5,6'
import natsort
import numpy as np
import torch
from tqdm.asyncio import tqdm
from transformers import BertTokenizerFast, LineByLineTextDataset, BertForMaskedLM, BertConfig, \
    BertForSequenceClassification

import utils
from downstream_train import data_load, num_token

import method


# from federated_train import rebuild_model


def create_text_classfication(start_length, domin_length, out_path):
    with open(out_path, "a") as f:
        # Chemistry
        chemistry_path = './data/sentence/Chemistry_test.txt'
        with open(chemistry_path, "r") as chemistry_file:
            for line in tqdm(chemistry_file.readlines()[start_length:start_length + domin_length]):
                tmp = {}
                tmp.update({'text': line})
                tmp.update({'label': 'Chemistry'})
                f.write(json.dumps(tmp) + '\n')

        # Economics
        economics_path = './data/sentence/Economics_test.txt'
        with open(economics_path, "r") as economics_file:
            for line in tqdm(economics_file.readlines()[start_length:start_length + domin_length]):
                tmp = {}
                tmp.update({'text': line})
                tmp.update({'label': 'Economics'})
                f.write(json.dumps(tmp) + '\n')

        # Physics
        physics_path = './data/sentence/Physics_test.txt'
        with open(physics_path, "r") as physics_file:
            for line in tqdm(physics_file.readlines()[start_length:start_length + domin_length]):
                tmp = {}
                tmp.update({'text': line})
                tmp.update({'label': 'Physics'})
                f.write(json.dumps(tmp) + '\n')

        # Biology
        biology_path = './data/sentence/Biology_test.txt'
        with open(biology_path, "r") as biology_file:
            for line in tqdm(biology_file.readlines()[start_length:start_length + domin_length]):
                tmp = {}
                tmp.update({'text': line})
                tmp.update({'label': 'Biology'})
                f.write(json.dumps(tmp) + '\n')

        # Computer
        computer_path = './data/sentence/Computer_test.txt'
        with open(computer_path, "r") as computer_file:
            for line in tqdm(computer_file.readlines()[start_length:start_length + domin_length]):
                tmp = {}
                tmp.update({'text': line})
                tmp.update({'label': 'Computer'})
                f.write(json.dumps(tmp) + '\n')

        # Medicine
        medicine_path = './data/sentence/Medicine_test.txt'
        with open(medicine_path, "r") as medicine_file:
            for line in tqdm(medicine_file.readlines()[start_length:start_length + domin_length]):
                tmp = {}
                tmp.update({'text': line})
                tmp.update({'label': 'Medicine'})
                f.write(json.dumps(tmp) + '\n')


def param_save():
    param_list = []
    model_path = './model/bert-base-uncased/'

    modelConfig = BertConfig.from_pretrained(model_path)
    model = BertForMaskedLM.from_pretrained(model_path, config=modelConfig)

    for name, param in model.named_parameters():
        if 'embeddings' in name:
            param.requires_grad = False
    for layer in model.bert.encoder.layer:
        for p in layer.parameters():
            p.requires_grad = False
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name)
            param_list.append(param)
    torch.save(param_list, './outputs/test/test.pt')


def re_param(model, param_path):
    model = model.cuda()
    params_list = torch.load(param_path)  # 加载客户端的参数
    for x in params_list:
        print(x)
    print("--------------------------------------")
    # for name, params in model.named_parameters():  # 更新客户端的训练结果
    #     print(name)

    return model


def train_first(model):
    for name, param in model.named_parameters():
        param.requires_grad = False
    for layer in model.bert.encoder.layer[:1]:
        for p in layer.parameters():
            p.requires_grad = True

    return model


def make_datasets(model_path, file_path, file_name, dataset_path):
    tokenizer = BertTokenizerFast.from_pretrained(model_path, do_lower_case=True)

    print("加载数据中------")
    T1 = time.time()
    # 通过LineByLineTextDataset接口 加载数据 #长度设置为128,
    train_dataset = LineByLineTextDataset(tokenizer=tokenizer, file_path=file_path, block_size=128)
    T2 = time.time()
    print('加载数据运行时间:%s 秒' % (T2 - T1))

    print("存储数据中------")
    T3 = time.time()
    torch.save(train_dataset, dataset_path + file_name + '_128.pt')
    T4 = time.time()
    print('存储数据运行时间:%s 秒' % (T4 - T3))


def creat_domin_txt(domain):
    num_workers = 6

    in_path = './data/corpus/' + domain + '/pdf_parses_10.jsonl'
    length = 12000

    for i in range(num_workers):
        # out_path = './data/sentence/Medicine/test_' + str(i) + '.txt'
        out_path = './data/sentence/' + domain + '/test_' + str(i) + '.txt'
        method.create_text(in_path, out_path, start=i * length, end=(i + 1) * length)


def create_domin_dataset():
    domain = 'Medicine'

    model_path = './model/bert-base-uncased/'
    file_path = './data/sentence/' + domain + '/'
    dataset_path = './data/datasets/' + domain + '/'

    # 先生成文本文档
    creat_domin_txt(domain)

    file_names = natsort.natsorted(os.listdir(file_path), alg=natsort.ns.PATH)
    for file in file_names:
        make_datasets(model_path, file_path + file, file[:-4], dataset_path)


def create_center_dataset():
    domain = 'Medicine'

    model_path = './model/bert-base-uncased/'
    file_path = './data/sentence/' + domain + '/'

    file_names = natsort.natsorted(os.listdir(file_path), alg=natsort.ns.PATH)

    center_txt = './data/sentence/Center/' + domain + '.txt'

    with open(center_txt, "w") as f:
        for file in file_names:
            with open(file_path + file, "r") as tmp:
                for line in tqdm(tmp.readlines()):
                    f.write(line)

    with open(center_txt, "r") as f:
        print(len(f.readlines()))

    tokenizer = BertTokenizerFast.from_pretrained(model_path, do_lower_case=True)

    print("加载数据中------")
    T1 = time.time()
    # 通过LineByLineTextDataset接口 加载数据 #长度设置为128,
    train_dataset = LineByLineTextDataset(tokenizer=tokenizer, file_path=center_txt, block_size=128)
    T2 = time.time()
    print('加载数据运行时间:%s 秒' % (T2 - T1))

    print("存储数据中------")
    T3 = time.time()
    torch.save(train_dataset, './data/datasets/Center/' + domain + '_128.pt')
    T4 = time.time()
    print('存储数据运行时间:%s 秒' % (T4 - T3))


def create_Skewed_Dataset():
    domain = 'Medicine'

    model_path = './model/bert-base-uncased/'
    center_path = './data/sentence/Center/' + domain + '.txt'
    skewd_txt_path = './data/sentence/Skewed/' + domain + '/'
    skewd_dataset_path = './data/datasets/Skewed/' + domain + '/'

    with open(center_path, 'r') as full_file:
        for line in tqdm(full_file.readlines()):
            res = np.random.normal(5, 2, 1)
            if res[0] < (10 / 6):
                file = 'txt_0.txt'
            elif res[0] < (20 / 6):
                file = 'txt_1.txt'
            elif res[0] < (30 / 6):
                file = 'txt_2.txt'
            elif res[0] < (40 / 6):
                file = 'txt_3.txt'
            elif res[0] < (50 / 6):
                file = 'txt_4.txt'
            else:
                file = 'txt_5.txt'

            with open(skewd_txt_path + file, 'a') as client_txt:
                client_txt.write(line)

    file_names = natsort.natsorted(os.listdir(skewd_txt_path), alg=natsort.ns.PATH)
    for file in file_names:
        make_datasets(model_path, skewd_txt_path + file, file[:-4], skewd_dataset_path)


def param_compare():
    model_path = './model/bert-base-uncased/'

    small_modelConfig = BertConfig.from_pretrained(model_path)
    small_modelConfig.num_hidden_layers = 6  # 相当于构建一个小模型，transformer层只有六层
    small_model = BertForMaskedLM.from_pretrained(model_path, config=small_modelConfig)

    big_modelConfig = BertConfig.from_pretrained(model_path)
    big_model = BertForMaskedLM.from_pretrained(model_path, config=big_modelConfig)

    for s_layer, b_layer in zip(small_model.bert.encoder.layer[:1], big_model.bert.encoder.layer[:1]):
        for s_p, b_p in zip(s_layer.parameters(), b_layer.parameters()):
            diff = s_p.data - b_p.data
            print(diff)


def cost_analysis():
    model_path = './model/bert-base-uncased/'
    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
    batch_size = 32
    learning_rate = 5e-5

    modelConfig = BertConfig.from_pretrained(model_path)
    modelConfig.num_hidden_layers = 12
    modelConfig.num_labels = num_token  # 设置分类模型的输出个数

    model = BertForSequenceClassification.from_pretrained(model_path, config=modelConfig)
    model.to(device)
    # model = utils.train_first(model)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
    )

    train_dataloader = data_load(model_path, 'train', device, batch_size)

    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name)

    forward_time = 0
    backward_time = 0

    with torch.autograd.profiler.profile() as prof:
        for batch in tqdm(train_dataloader):
            batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask, labels = batch
            optimizer.zero_grad()
            # 取第一个位置，BertForSequenceClassification第一个位置是Loss，第二个位置是[CLS]的logits
            t1 = time.time()
            loss = model(input_ids, token_type_ids=None, attention_mask=input_mask, labels=labels)[0]
            t2 = time.time()

            t3 = time.time()
            loss.backward()
            t4 = time.time()

            optimizer.step()

            forward_time += t2 - t1
            backward_time += t4 - t3

    print("正向传播时间为：%.4f S" % forward_time)
    print("反向传播时间为：%.4f S" % backward_time)

    # print(prof)


def merge_progressive():
    first_name = 'bert.encoder.layer.'
    last_name_list = [".attention.self.query.weight",
                      ".attention.self.query.bias",
                      ".attention.self.key.weight",
                      ".attention.self.key.bias",
                      ".attention.self.value.weight",
                      ".attention.self.value.bias",
                      ".attention.output.dense.weight",
                      ".attention.output.dense.bias",
                      ".attention.output.LayerNorm.weight",
                      ".attention.output.LayerNorm.bias",
                      ".intermediate.dense.weight",
                      ".intermediate.dense.bias",
                      ".output.dense.weight",
                      ".output.dense.bias",
                      ".output.LayerNorm.weight",
                      ".output.LayerNorm.bias"]

    file_path = './outputs/LayerModel/Biology/NewPro_0-11/'

    file_name = natsort.natsorted(os.listdir(file_path), alg=natsort.ns.PATH)

    sum_dict = {}
    bias = 0
    layer_num = [0, 1, 2]
    for name in file_name:
        epoch_number = int(name[6:])
        if epoch_number == 5 or epoch_number == 10 or epoch_number == 15 or epoch_number == 20:
            params_list = torch.load(file_path + name + '/fed_avg.pt')
            for layer in layer_num:
                for last_name in last_name_list:
                    new_name = first_name + str(layer + bias) + last_name
                    ori_name = first_name + str(layer) + last_name

                    sum_dict.update({new_name: params_list[ori_name]})
            bias += 3
    torch.save(sum_dict, file_path + 'layer0_11.pt')


def merge_layer():
    file_path = './outputs/LayerModel/Ablation/Computer/Pro3_6_e5_w_Map/'

    file_name = natsort.natsorted(os.listdir(file_path), alg=natsort.ns.PATH)

    sum_dict = {}

    for name in file_name:
        try:
            epoch_number = int(name[6:])
        except:
            continue
        if 2 < epoch_number < 5:
            params_list = torch.load(file_path + name + '/fed_avg.pt')
            sum_dict.update(params_list)

    torch.save(sum_dict, file_path + 'layer0_1.pt')


def rebuild():
    layer_list = np.random.randint(1, 12, size=5)  # 产生[drop_layer,12)之间的随机整数来选择模型接下来要拷贝的参数
    print(layer_list)
    layer_list = np.sort(layer_list)  # 对后续层进行排序
    print(layer_list)


def split_data():
    split_size = 0.02

    txt_path = './data/text_classification/'

    data_name = 'rct-20k'

    out_path = txt_path + 'tiny_' + data_name + '/'
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    file_name = natsort.natsorted(os.listdir(txt_path + data_name + '/'), alg=natsort.ns.PATH)

    for f_n in file_name:
        in_path = txt_path + data_name + '/' + f_n
        with open(in_path, 'r') as in_txt:
            for line in tqdm(in_txt.readlines()):
                p = random.random()  # 记录下来的概率
                if p < split_size:
                    with open(out_path + f_n, 'a') as out_txt:
                        out_txt.write(line)


def experiment_show():
    title = 'Rct-20k'
    # legend = ['DistillBert', 'TinyBert-4', 'TinyBert-6', 'Ours']
    legend = ['Bert', 'Bert-Center', 'Bert-FL',
              'DistillBert', 'TinyBert-4', 'TinyBert-6', 'Ours']

    datas = [[0.7579617834394905, 0.7818471337579618, 0.8168789808917197, 0.8184713375796179, 0.8136942675159236, 0.8200636942675159, 0.8232484076433121, 0.8248407643312102, 0.8248407643312102, 0.8248407643312102, 0.8232484076433121, 0.8264331210191083, 0.8232484076433121, 0.8232484076433121, 0.8248407643312102],
             [0.7866242038216561, 0.8073248407643312, 0.8121019108280255, 0.8168789808917197, 0.8152866242038217, 0.8168789808917197, 0.8168789808917197, 0.8152866242038217, 0.8152866242038217, 0.8152866242038217, 0.8136942675159236, 0.8184713375796179, 0.8152866242038217, 0.8168789808917197, 0.8168789808917197],
             [0.8105095541401274, 0.8136942675159236, 0.8184713375796179, 0.8248407643312102, 0.8232484076433121, 0.8136942675159236, 0.8264331210191083, 0.8184713375796179, 0.8152866242038217, 0.8184713375796179, 0.8184713375796179, 0.8168789808917197, 0.8184713375796179, 0.821656050955414, 0.8200636942675159],
             [0.7563694267515924, 0.7786624203821656, 0.7866242038216561, 0.785031847133758, 0.7882165605095541, 0.7818471337579618, 0.7818471337579618, 0.7834394904458599, 0.7786624203821656, 0.7818471337579618, 0.7802547770700637, 0.7802547770700637, 0.785031847133758, 0.7818471337579618, 0.7818471337579618],
             [0.7404458598726115, 0.7770700636942676, 0.7945859872611465, 0.7898089171974523, 0.8057324840764332, 0.804140127388535, 0.802547770700637, 0.7993630573248408, 0.7993630573248408, 0.7993630573248408, 0.7993630573248408, 0.7993630573248408, 0.7993630573248408, 0.7993630573248408, 0.7993630573248408],
             [0.7914012738853503, 0.8073248407643312, 0.7993630573248408, 0.8057324840764332, 0.8121019108280255, 0.8184713375796179, 0.821656050955414, 0.8152866242038217, 0.8200636942675159, 0.8184713375796179, 0.8232484076433121, 0.8232484076433121, 0.8232484076433121, 0.8232484076433121, 0.8232484076433121],
             [0.7961783439490446, 0.8089171974522293, 0.8312101910828026, 0.8152866242038217, 0.8184713375796179, 0.8152866242038217, 0.8200636942675159, 0.8184713375796179, 0.8184713375796179, 0.8168789808917197, 0.8184713375796179, 0.8184713375796179, 0.8184713375796179, 0.8200636942675159, 0.8200636942675159]]
    # 全部的线段风格
    styles = ['c:s', 'y:8', 'b:p', 'g:D', 'm:X', ':>', 'r:^', 'r:v']  # 其他可用风格 ':<',':H','k:o','k:*','k:*','k:*'

    plt.figure(figsize=(10, 7))
    # 设置字体
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams.update({'font.size': 22})
    plt.rc('legend', fontsize=20)

    # 正式的进行画图
    for i in range(len(datas)):
        x = np.arange(1, len(datas[i]) + 1)
        y = datas[i]
        plt.plot(x, y, styles[i], markersize=8, label=legend[i])

    # 设置图片的x,y轴的限制，和对应的标签
    # plt.xlim([0, 300])
    # plt.ylim([60, 78])
    plt.xlabel("Epochs")
    plt.xticks(np.linspace(1, 15, 8, endpoint=True))
    plt.ylabel("Accuracy")
    plt.title(title)

    # 设置图片的方格线和图例
    plt.grid()
    plt.legend(loc='lower right', framealpha=0.7, ncol=2)
    plt.tight_layout()
    # plt.show()

    # 如果想保存图片，请把plt.show注释掉，然后把下面这行代码打开注释
    plt.savefig("./images/" + title + ".pdf", dpi=800)


def parameters_size():
    model_path0 = './model/bert-base-uncased/'

    out_path = './outputs/params/'

    method.setup_seed(40)
    modelConfig_0 = BertConfig.from_pretrained(model_path0)
    # modelConfig_0.num_hidden_layers = 4  # 中间仅含有6层transformer layers
    model_0 = BertForMaskedLM.from_pretrained(model_path0, config=modelConfig_0)

    for name, param in model_0.named_parameters():
        param.requires_grad = False

    model_0 = utils.train_cls_layer(model_0)

    utils.layer_save(model_0, out_path + 'output.pt')


def main():
    # three()
    # param_compare()
    # cost_analysis()
    # merge_progressive()
    # merge_layer()
    # rebuild()
    # create_domin_dataset()
    # create_center_dataset()
    # create_Skewed_Dataset()
    experiment_show()
    # for i in range(6, 0, -1):
    #     batch_size = 2 ** i
    #     print(batch_size)

    # split_data()
    # parameters_size()


if __name__ == "__main__":
    main()

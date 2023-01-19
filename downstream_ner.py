#!/usr/bin/env python
# -*- coding:utf-8 -*-
# author:HanZhou
# datetime:2022/10/8 10:10
# software: PyCharm
import json
import random

import numpy as np
from torch.optim import AdamW
from torch.utils.data import DataLoader, RandomSampler, random_split
from tqdm import tqdm
from transformers import BertTokenizerFast, BertForTokenClassification, BertConfig, BertTokenizer
import torch
import method
import utils

data_path = './data/ner/SciERC/'  # 数据集路径
unique_tags, tag2id, id2tag = method.ner_label(data_path)


class NerDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


def data_read(data_type):
    sentencses = []
    labels = []
    path = data_path + data_type + '.txt'
    print("%s data loading------" % data_type)

    with open(path, 'r', encoding='utf-8') as file:
        tmp_words = []
        tmp_label = []
        for line in tqdm(file.readlines()):
            if 'DOCSTART' in line:
                continue
            else:
                if len(line) == 1:
                    sentencses.append(tmp_words)
                    labels.append(tmp_label)
                    tmp_words = []
                    tmp_label = []
                else:
                    tmp = line.split()
                    tmp_words.append(tmp[0])
                    tmp_label.append(tmp[-1])

    return sentencses, labels


def encode_tags(tags, encodings):
    labels = [[tag2id[tag] for tag in doc] for doc in tags]
    # print(labels)
    encoded_labels = []
    for doc_labels, doc_offset in zip(labels, encodings.offset_mapping):
        # 创建全由-100组成的矩阵
        doc_enc_labels = np.ones(len(doc_offset), dtype=int) * -100
        arr_offset = np.array(doc_offset)
        # set labels whose first offset position is 0 and the second is not 0
        if len(doc_labels) >= 510:  # 防止异常
            doc_labels = doc_labels[:510]
        doc_enc_labels[(arr_offset[:, 0] == 0) & (arr_offset[:, 1] != 0)] = doc_labels
        encoded_labels.append(doc_enc_labels.tolist())

    return encoded_labels


def data_load(model_path, batch_size, data_type, split_zie=-0.1):
    # model_path = './model/bert-base-uncased/'
    texts, tags = data_read(data_type)

    tokenizer = BertTokenizerFast.from_pretrained(model_path)
    # is_split_into_words表示已经分词好了
    encodings = tokenizer(texts, is_split_into_words=True, return_offsets_mapping=True, padding=True,
                          truncation=True, max_length=512)

    labels = encode_tags(tags, encodings)
    encodings.pop("offset_mapping")  # 训练不需要这个

    dataset = NerDataset(encodings, labels)

    if split_zie != -0.1:
        #  随机划分部分数据集用于训练
        data_size = int(split_zie * len(dataset))
        print("Choose " + str(split_zie) + " Data for Training!")
        dataset, remain_data = random_split(dataset=dataset, lengths=[data_size, len(dataset) - data_size])

    sampler = RandomSampler(dataset)
    dataloader = DataLoader(dataset, sampler=sampler, batch_size=batch_size)

    return dataloader


def token2span(token_list):
    """
    将token转换为(start, end, entity)的元组形式
    :param token_list:
    :return: 返回由元组构成的实体列表
    """
    span_list = []
    span_tuple = []
    merge_flag = False
    for i in range(len(token_list)):
        if token_list[i][0] == 'B':  # 如果起始字符是'B'
            if merge_flag:  # 如果还在合并过程中
                span_tuple.append(i - 1)  # 添加end终止位置
                span_tuple.append(token_list[i - 1][2:])  # 添加entity类型
                span_list.append(tuple(span_tuple))  # 添加entity元组
                span_tuple = []

            merge_flag = True
            span_tuple.append(i)  # 添加start起始位置

        elif token_list[i][0] == 'I':  # 如果起始字符是'I'
            if merge_flag:  # 如果还在合并过程中
                continue
            else:  # 如果是单个字符
                span_tuple.append(i)  # 添加start起始位置
                span_tuple.append(i)  # 添加end终止位置
                span_tuple.append(token_list[i][2:])  # 添加entity类型
                span_list.append(tuple(span_tuple))  # 添加entity元组
                span_tuple = []

        else:  # 如果是'O'
            if merge_flag:  # 如果还在合并过程中
                merge_flag = False
                span_tuple.append(i - 1)  # 添加end终止位置
                span_tuple.append(token_list[i - 1][2:])  # 添加entity类型
                span_list.append(tuple(span_tuple))  # 添加entity元组
                span_tuple = []

    return span_list


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = True


def model_test(model, test_dataloader, device, model_type):
    """
    对模型的准确度在特定数据集上进行测试
    :param model: 待测试的模型
    :param test_dataloader: 评测数据集
    :param device: 是否使用GPU
    :param model_type: 测试模型种类
    """
    model.eval()
    pred_token_list = []
    label_token_list = []

    for batch in test_dataloader:
        b_input_ids = batch['input_ids'].to(device)
        b_input_mask = batch['attention_mask'].to(device)
        b_labels = batch['labels'].to(device)
        with torch.no_grad():
            logits = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)[0]
        b_preds = np.argmax(logits.cpu().numpy(), axis=2).tolist()
        b_label_ids = b_labels.cpu().numpy().tolist()
        for i in range(len(b_label_ids)):
            pred = b_preds[i][1:-1]
            label = b_label_ids[i][1:-1]
            pred_list = [id2tag[x] for (x, y) in zip(pred, label) if y != -100]
            label_list = [id2tag[x] for x in label if x != -100]
            # print(pred_list)
            pred_token_list.extend(pred_list)
            label_token_list.extend(label_list)
    # print("-------------------")

    pred_span_set = set(token2span(pred_token_list))
    label_span_set = set(token2span(label_token_list))
    # print(pred_span_set)
    t_p = len(pred_span_set & label_span_set)

    print("t_p:" + str(t_p))
    if t_p == 0:
        F1 = 0
    else:
        precision = t_p / len(pred_span_set)
        recall = t_p / len(label_span_set)
        F1 = (2 * precision * recall) / (precision + recall)

    print(model_type + "F1: {}".format(F1))

    return F1


def ner_train(learning_rate, model_path, epochs, device, batch_size,
              change_param=None, seed=40):
    setup_seed(seed)

    modelConfig = BertConfig.from_pretrained(model_path)
    modelConfig.num_labels = len(unique_tags)  # 设置分类模型的输出个数
    model = BertForTokenClassification.from_pretrained(model_path, config=modelConfig)

    if change_param:
        # 根据已经保存的文件重新构建模型的某些参数
        model = utils.re_param(model, change_param)

    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.7)

    model.to(device)
    val_acc = []  # 训练过程中的验证集精度
    # 测试训练结果
    test_acc = []

    # 先导入数据
    train_dataloader = data_load(model_path, batch_size, 'train')
    validation_dataloader = data_load(model_path, batch_size, 'dev')
    test_dataloader = data_load(model_path, batch_size, 'test')

    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name)
    # model_test(model, validation_dataloader, device, 'Val ')
    for i in range(epochs):
        print("Epochs:%d/%d" % ((i + 1), epochs))
        # 训练开始
        model.train()
        tr_loss = 0
        nb_tr_steps = 0
        for batch in tqdm(train_dataloader):
            b_input_ids = batch['input_ids'].to(device)
            b_input_mask = batch['attention_mask'].to(device)
            b_labels = batch['labels'].to(device)
            optimizer.zero_grad()
            loss = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)[0]
            loss.backward()
            optimizer.step()

            tr_loss += loss.item()

            nb_tr_steps += 1

        scheduler.step()
        print("Train loss: {}".format(tr_loss / nb_tr_steps))
        # 模型评估
        acc = model_test(model, validation_dataloader, device, 'Val ')
        val_acc.append(acc)
        acc1 = model_test(model, test_dataloader, device, 'Test ')
        test_acc.append(acc1)

    return val_acc, test_acc


def find_lr():
    seed = 40
    epochs = 15
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print(seed)
    print(device)

    domain = 'Computer'
    Pro3_6_e5 = './outputs/LayerModel/' + domain + '/cls_Pro3_6_e5/layer0_2.pt'

    model_path = './model/bert-base-uncased/'

    Index_list = []
    legend = []
    Val_Acc = []
    Test_Acc = []
    Max_Acc = []

    for x in range(2, 9):
        learn_rate = x / (10 ** 5)
        print("DownStream Learning Rate:" + str(learn_rate) + "训练中------")
        for i in range(6, 0, -1):
            batch_size = 2 ** i
            print("Batch Size:" + str(batch_size) + "训练中------")
            val_acc0, test_acc0 = ner_train(learn_rate, model_path, epochs, device, batch_size,
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


def main():
    epochs = 15
    batch_size = 2
    learning_rate = 7e-4
    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
    print("-------------------------")
    print(batch_size)
    print(learning_rate)
    print(device)
    print("--------------------------")

    bert_model1 = './model/Computer/'  # 预训练模型的文件
    print("Computer SkipBert模型ner训练测试中------")
    val_acc3, test_acc3 = ner_train(bert_model1, epochs, device, batch_size, learning_rate)

    legend = ['SciERC']
    Val_Acc = [val_acc3]
    Test_Acc = [test_acc3]
    #
    # method.plot_res('NCBI-disease Val Acc', legend, Val_Acc)
    # method.plot_res('NCBI-disease Test Acc', legend, Test_Acc)
    method.get_test_acc(legend, Val_Acc, Test_Acc)
    print("--------------------")
    for i in range(len(legend)):
        print(legend[i], end=" ")
        print("Max Test Acc: ")
        print(max(Test_Acc[i]))


if __name__ == "__main__":
    # main()
    find_lr()

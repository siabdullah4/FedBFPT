#!/usr/bin/env python
# -*- coding:utf-8 -*-
# author:HanZhou
# datetime:2022/5/24 9:43
# software: PyCharm
import json
import os
import shutil

from torch.utils.data import DataLoader

os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3'

import random
import numpy as np
import spacy
import tokenizers
import torch
from torch.optim import AdamW
from tqdm import tqdm
from transformers import BertTokenizer, RobertaConfig, BertTokenizerFast, get_linear_schedule_with_warmup
from transformers import BertConfig
from transformers import BertForMaskedLM
from transformers import LineByLineTextDataset
from transformers import DataCollatorForLanguageModeling
from transformers import Trainer
from transformers import TrainingArguments

import time
import method


def hook(grad):
    grad[:30522] = 0.
    return grad


def bert_embeddings(bert_model):
    model_path = bert_model
    token_path = bert_model
    tokenizer = BertTokenizerFast.from_pretrained(token_path, do_lower_case=True)
    config = BertConfig.from_pretrained(model_path)
    model = BertForMaskedLM.from_pretrained(model_path, config=config)
    model.resize_token_embeddings(len(tokenizer))

    model = method.freeze_higher_layers(model, config)
    # model = method.freeze_lower_layers(model, config)

    for name, param in model.named_parameters():
        print(name)
        print(param.requires_grad)


def rebuild_model(model, layer_length):
    """
    重新构建模型的后续transformer层参数
    :param model:LayerBert模型,只有几层的transformer模型
    :param layer_length: 从大模型选择映射的transformer层数量
    """
    print("Rebuild Model......")
    ori_path = "./model/bert-base-uncased/"
    ori_modelConfig = BertConfig.from_pretrained(ori_path)
    ori_model = BertForMaskedLM.from_pretrained(ori_path, config=ori_modelConfig)

    layer_list = np.random.randint(1, 12, size=layer_length)  # 产生[1,12)之间的随机整数来选择模型接下来要拷贝的参数
    layer_param = []  # 记录下原始Bert模型参数
    for i in layer_list:
        params = []
        for layer in ori_model.bert.encoder.layer[i:i + 1]:
            for p in layer.parameters():
                params.append(p.data)
        layer_param.append(params)

    for j in range(len(layer_param)):
        for layer in model.module.bert.encoder.layer[j + 1:j + 2]:  # 第0层的transformer层参数不需要改变
            for p, d in zip(layer.parameters(), layer_param[j]):
                p.data = d

    return model


def huggingface_pretrain(model, save_path, train_dataset, learn_rate):
    """
    根据语料，对模型进行预训练
    :param model: 需要训练的模型
    :param save_path: 保存训练文件的地址
    :param train_dataset: 预训练数据集
    :param learn_rate: 学习率
    """
    # 加载tokenizer和模型
    token_path = './model/bert-base-uncased/'
    tokenizer = BertTokenizerFast.from_pretrained(token_path, do_lower_case=True)

    # for name, param in model.named_parameters():
    #     if param.requires_grad:
    #         print(name, end=" ")
    #         print(param.requires_grad)

    # MLM模型的数据DataCollator
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)
    # 训练参数
    pretrain_batch_size = 256
    num_train_epochs = 1

    # learn_rate = 6e-5  #

    # 过滤掉requires_grad = False的参数
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if p.requires_grad]}
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=learn_rate)

    training_args = TrainingArguments(
        output_dir=save_path, overwrite_output_dir=True,
        num_train_epochs=num_train_epochs, learning_rate=learn_rate,
        per_device_train_batch_size=pretrain_batch_size,
        fp16=True, save_total_limit=2)

    # 通过Trainer接口训练模型
    trainer = Trainer(
        model=model, args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        optimizers=(optimizer, None))

    # 开始训练
    print(method.time_beijing())
    trainer.train()
    # trainer.save_model(save_path)


def dp_pretrain(learning_rate, epochs, batch_size, model, file_path):
    """
    通过torch.nn.DataParallel实现Bert模型的mask预训练
    :param learning_rate: 学习率
    :param epochs: 训练轮次
    :param batch_size: batchsize
    :param model: 待训练的模型
    :param file_path: 预训练的文件地址
    :return: 训练完毕的模型进行返回
    """
    device_list = [1, 0, 2, 3]
    method.setup_seed(24)

    # 使用多卡
    bert_model = torch.nn.DataParallel(model, device_ids=device_list)
    bert_model.to(device_list[0])

    model_path = './model/bert-base-uncased/'  # Bert模型，包括词表等的地址
    tokenizer = BertTokenizerFast.from_pretrained(model_path, do_lower_case=True)
    print("加载数据中-------")
    datasets = torch.load(file_path)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)

    dataloader = DataLoader(datasets, batch_size=batch_size, collate_fn=data_collator)

    # 固定高层参数
    # bert_model = method.freeze_higher_layers(bert_model, bert_config)
    param_optimizer = list(bert_model.named_parameters())

    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [  # 剔除requires_grad == False 的参数
        {'params': [p for n, p in param_optimizer if (not any(nd in n for nd in no_decay) and p.requires_grad)],
         'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if (any(nd in n for nd in no_decay) and p.requires_grad)],
         'weight_decay': 0.0}
    ]

    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.7)

    for name, param in bert_model.named_parameters():
        if param.requires_grad:
            print(name)

    for epoch in range(epochs):
        for batch in tqdm(dataloader):
            input_ids = batch['input_ids'].to(device_list[0])
            attention_mask = batch['attention_mask'].to(device_list[0])
            labels = batch['labels'].to(device_list[0])
            result = bert_model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = result[0].mean()  # 解决dp返回时多个标量拼凑无法求导

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        scheduler.step()

    return bert_model


def main():
    file_path = './data/datasets/Biology_128.pt'  # 语料地址
    model_path = './model/bert-base-uncased/'  # Bert模型，包括词表等的地址
    model_save = "./outputs/test/"
    learn_rate = 5e-5

    modelConfig = BertConfig.from_pretrained(model_path)
    model = BertForMaskedLM.from_pretrained(model_path, config=modelConfig)
    # for name, param in model.named_parameters():
    #     print(param.data)
    #     break
    #
    # print("数据加载中------")
    # train_dataset = torch.load(file_path)
    # huggingface_pretrain(model, model_save, train_dataset, learn_rate)
    #
    # for name, param in model.named_parameters():
    #     print(param.data)
    #     break
    dp_pretrain(learn_rate, 5, 256, model, file_path)


if __name__ == '__main__':
    main()

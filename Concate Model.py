#!/usr/bin/env python
# -*- coding:utf-8 -*-
# author:HanZhou
# datetime:2022/10/13 14:50
# software: PyCharm
import torch
from tqdm import tqdm
from transformers import BertConfig, BertForSequenceClassification

import method
from downstream_train import data_load, model_test, num_token


def concate_layer(model):
    layer_path = './outputs/params/Biology/LayerBert_01/client_0.pt'

    layer_params = torch.load(layer_path)  # 加载客户端的参数
    params_list = []  # transformer层参数

    for name in layer_params:
        params_list.append(layer_params[name].data)

    for layer in model.bert.encoder.layer[:6]:  # 更改前六层的参数
        for i, p in enumerate(layer.parameters()):
            p.data = params_list[i]  # 替换transformer层的参数

    return model


def model_cls(model, learning_rate, epochs, device, batch_size):
    """
    根据拼凑更改的模型进行文本分类任务的训练
    :param model: 待训练模型
    :param learning_rate: 学习率
    :param epochs: 训练轮次
    :param device: 设备
    :param batch_size: batch_size
    """
    method.setup_seed(40)

    token_path = './model/bert-base-uncased/'

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
    )
    # print(optimizer.param_groups)

    train_dataloader = data_load(token_path, 'train', device, batch_size)
    validation_dataloader = data_load(token_path, 'dev', device, batch_size)
    model.to(device)
    # 训练开始

    val_acc = []  # 训练过程中的验证集精度
    # 测试训练结果
    test_acc = []
    test_dataloader = data_load(token_path, 'test', device, batch_size)

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


def main():
    epochs = 50
    batch_size = 32
    learning_rate = 5e-6
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_path = './model/bert-base-uncased/'

    modelConfig = BertConfig.from_pretrained(model_path)
    modelConfig.num_labels = num_token  # 设置分类模型的输出个数

    model = BertForSequenceClassification.from_pretrained(model_path, config=modelConfig)

    # 进行拼凑模型
    model = concate_layer(model)

    val_acc0, test_acc0 = model_cls(model, learning_rate, epochs, device, batch_size)

    Val_acc = [val_acc0]
    Test_acc = [test_acc0]

    legend = ['Concate Layer']
    method.plot_res('Val Acc', legend, Val_acc)
    method.plot_res('Test Acc', legend, Test_acc)

    method.get_test_acc(legend, Val_acc, Test_acc)
    print("--------------------")
    for i in range(len(legend)):
        print(legend[i], end=" ")
        print("Max Test Acc: ")
        print(max(Test_acc[i]))


if __name__ == "__main__":
    main()

#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2022/12/30 18:07
# @Author : HanZhou
import os
import shutil

import torch

from torch.utils.data import TensorDataset, RandomSampler, DataLoader
from transformers import BertConfig, BertForSequenceClassification

import method
import utils
from downstream_train import data_load, data_read, num_token, down_classier, model_test


def client_dataloader(bert_model, file_type, client_num, device, batch_size):
    inputs, labels, masks = data_read(bert_model, file_type)  # 获取数据
    data_num = int(len(inputs) / client_num)

    dataloader_list = []

    for i in range(client_num):
        client_inputs = inputs[i * data_num:(i + 1) * data_num]
        client_labels = labels[i * data_num:(i + 1) * data_num]
        client_masks = masks[i * data_num:(i + 1) * data_num]

        client_inputs = torch.tensor(client_inputs).to(device)
        client_labels = torch.tensor(client_labels).to(device)
        client_masks = torch.tensor(client_masks).to(device)

        # 生成dataloader
        client_data = TensorDataset(client_inputs, client_masks, client_labels)
        sampler = RandomSampler(client_data)
        dataloader = DataLoader(client_data, sampler=sampler, batch_size=batch_size, shuffle=False)

        dataloader_list.append(dataloader)

    return dataloader_list


def fed_classifier(model_path, device, batch_size, seed, epoch, learning_rate):
    layer_num = 6
    client_num = 6
    param_dict = './outputs/LayerModel/Chemprot/Pro3_6_e5/'

    # 清空已经更新过的无用参数，适用于客户端训练相同的参数
    shutil.rmtree(param_dict)
    os.makedirs(param_dict)

    param_container = utils.create_container()  # 制作本地参数容器

    dataloader_list = client_dataloader(model_path, 'train', client_num, device, batch_size)

    model_list = []
    optimizer_list = []
    scheduler_list = []

    for i in range(client_num):
        # layer_num = random.randint(3, 7)  # 产生模型架构层数，3-6之间

        print("模型transformer层数为：%d" % layer_num)
        method.setup_seed(seed)
        modelConfig = BertConfig.from_pretrained(model_path)
        modelConfig.num_hidden_layers = layer_num  # 相当于构建一个小模型，transformer层只有六层
        modelConfig.num_labels = num_token  # 设置分类模型的输出个数
        model = BertForSequenceClassification.from_pretrained(model_path, config=modelConfig)
        # model = utils.map9to3(model)  # 平均高层

        # model = utils.map_ori_layer(model, param_container, 11, layer_num)  # 匹配映射高层

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
        )
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.7)

        optimizer_list.append(optimizer)
        scheduler_list.append(scheduler)
        model_list.append(model)

    validation_dataloader = data_load(model_path, 'dev', device, batch_size)
    val_acc = []  # 训练过程中的验证集精度
    # 测试训练结果
    test_acc = []
    test_dataloader = data_load(model_path, 'test', device, batch_size)

    for i in range(epoch):
        # 记录本轮联邦存储的位置，如果文件夹不存在，则进行文件夹的创建
        epoch_save = param_dict + 'epoch_' + str(i + 1) + '/'
        if not os.path.exists(epoch_save):
            os.makedirs(epoch_save)

        drop_layer = 1  # 训练倒数第几层的参数

        for j in range(client_num):
            print("%d轮联邦%d号设备训练中------" % (i + 1, j))

            param_read = param_dict + 'epoch_' + str(i) + '/fed_avg.pt'
            param_save = epoch_save + 'client_' + str(j) + '.pt'

            if i != 0:
                # 如果不是第一轮联邦训练，则更新模型之前训练聚合得到的参数
                param_container = utils.update_container(param_container, param_read, map_num=layer_num)
                model_list[j] = utils.re_param(model_list[j], param_read)

            # 重构模型的lower层
            model_list[j] = utils.lower_build(model_list[j], param_container, layer_length=layer_num - drop_layer,
                                              drop_layer=drop_layer, ori_layer=12 - drop_layer)

            model_list[j] = utils.train_trans_layer(model_list[j], [layer_num - drop_layer])  # 训练特定的transformer layers
            model_list[j] = utils.train_cls_layer(model_list[j])  # 训练分类输出层

            for name, param in model_list[j].named_parameters():
                if param.requires_grad:
                    print("Train: " + name)

            # 进行分类训练
            model_list[j] = down_classier(model_list[j], dataloader_list[j], device, learning_rate,
                                          optimizer_list[j], scheduler_list[j])
            # 保存client更新的参数
            utils.layer_save(model_list[j], param_save)

        # 进行联邦聚合
        utils.federated_efficient_merge(epoch_save)

        # 进行大模型的测试
        method.setup_seed(seed)
        full_modelConfig = BertConfig.from_pretrained(model_path)
        full_modelConfig.num_labels = num_token  # 设置分类模型的输出个数

        full_model = BertForSequenceClassification.from_pretrained(model_path, config=full_modelConfig)
        # 根据已经保存的文件重新构建模型的某些参数
        full_model = utils.re_param(full_model, epoch_save + 'fed_avg.pt')
        full_model.to(device)

        v_acc = model_test(full_model, validation_dataloader, device, 'Val')
        val_acc.append(v_acc)
        t_acc = model_test(full_model, test_dataloader, device, 'Test')
        test_acc.append(t_acc)

    # 返回最后的测试结果
    return val_acc, test_acc


def center_classifier(model_path, device, batch_size, seed, epoch, learning_rate):
    layer_num = 12

    param_dict = './outputs/LayerModel/Chemprot/Center_Pro/'

    param_container = utils.create_container()  # 制作本地参数容器

    # 清空已经更新过的无用参数，适用于客户端训练相同的参数
    shutil.rmtree(param_dict)
    os.makedirs(param_dict)

    method.setup_seed(seed)
    modelConfig = BertConfig.from_pretrained(model_path)
    modelConfig.num_hidden_layers = layer_num  # 相当于构建一个小模型，transformer层只有六层
    modelConfig.num_labels = num_token  # 设置分类模型的输出个数
    model = BertForSequenceClassification.from_pretrained(model_path, config=modelConfig)

    train_dataloader = data_load(model_path, 'train', device, batch_size)
    validation_dataloader = data_load(model_path, 'dev', device, batch_size)
    val_acc = []  # 训练过程中的验证集精度
    # 测试训练结果
    test_acc = []
    test_dataloader = data_load(model_path, 'test', device, batch_size)

    drop_layer = 1

    # 重构模型的lower层
    # model = utils.lower_build(model, param_container, layer_length=layer_num - drop_layer,
    #                           drop_layer=drop_layer, ori_layer=12 - drop_layer)

    model = utils.train_trans_layer(model, [9, 10, 11])  # 训练特定的transformer layers
    model = utils.train_cls_layer(model)  # 训练分类输出层

    for name, param in model.named_parameters():
        if param.requires_grad:
            print("Train: " + name)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
    )
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.7)

    # 进行大模型的测试
    method.setup_seed(seed)
    full_modelConfig = BertConfig.from_pretrained(model_path)
    full_modelConfig.num_labels = num_token  # 设置分类模型的输出个数

    full_model = BertForSequenceClassification.from_pretrained(model_path, config=full_modelConfig)
    full_model.to(device)

    for i in range(epoch):
        print("Epochs:%d/%d" % ((i + 1), epoch))
        epoch_save = param_dict + 'epoch_' + str(i + 1) + '/'
        if not os.path.exists(epoch_save):
            os.makedirs(epoch_save)

        model = down_classier(model, train_dataloader, device, learning_rate, optimizer, scheduler)

        # 保存client更新的参数
        utils.layer_save(model, epoch_save + 'layer.pt')

        # 根据已经保存的文件重新构建模型的某些参数
        full_model = utils.re_param(full_model, epoch_save + 'layer.pt')

        v_acc = model_test(full_model, validation_dataloader, device, 'Val')
        val_acc.append(v_acc)
        t_acc = model_test(full_model, test_dataloader, device, 'Test')
        test_acc.append(t_acc)

    # 返回最后的测试结果
    return val_acc, test_acc


def main():
    model_path = './model/bert-base-uncased/'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    seed = 40
    batch_size = 32

    epoch = 15
    learning_rate = 5e-5

    # val_acc0, test_acc0 = fed_classifier(model_path, device, batch_size, seed, epoch, learning_rate)

    val_acc1, test_acc1 = center_classifier(model_path, device, batch_size, seed, epoch, learning_rate)

    Val_acc = [val_acc1]
    Test_acc = [test_acc1]

    legend = ['9-11 Chemprot']
    # method.plot_res('Val Acc', legend, Val_acc)
    method.plot_res('Test Acc', legend, Test_acc)

    method.get_test_acc(legend, Val_acc, Test_acc)
    print("--------------------")
    for i in range(len(legend)):
        print(legend[i], end=" ")
        print("Max Test Acc: ")
        print(max(Test_acc[i]))


if __name__ == "__main__":
    main()

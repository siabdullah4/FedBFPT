#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2023/1/6 15:29
# @Author : HanZhou
import os
import shutil

import numpy as np
import torch
from transformers import BertConfig, BertForSequenceClassification

import method
import utils
from BertAdapter import BertAdapterForSequenceClassification
from FedDown import client_dataloader
from downstream_train import num_token, data_load, down_classier, model_test


def fed_adapter(model_path, device, batch_size, seed, epoch, learning_rate):
    layer_num = 12
    client_num = 6
    param_dict = './outputs/LayerModel/Adapter/Chemprot/adapters_6/'

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
        modelConfig.adapter_nums = 6  # 高层含有adapters的数目
        modelConfig.has_adapter = True  # 设置开启adapter(diy)
        modelConfig.isLinear = False  # 非线性参数
        modelConfig.num_labels = num_token  # 设置分类模型的输出个数
        model = BertAdapterForSequenceClassification.from_pretrained(model_path, config=modelConfig)
        # model = utils.map9to3(model)  # 平均高层

        # model = utils.map_ori_layer(model, param_container, 11, layer_num)  # 匹配映射高层

        model.freeze_model(True)  # freeze all params
        # unfreeze adapter params
        adapter_param = ["adapter_fi", "adapter_se"]
        adapter_param_list = [p for n, p in model.named_parameters() if any(nd in n for nd in adapter_param)]
        for param in adapter_param_list:
            param.requires_grad = True

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

        for j in range(client_num):
            print("%d轮联邦%d号设备训练中------" % (i + 1, j))

            param_save = epoch_save + 'client_' + str(j) + '.pt'

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
        full_modelConfig.adapter_nums = 6  # 高层含有adapters的数目
        full_modelConfig.has_adapter = True  # 设置开启adapter(diy)
        full_modelConfig.isLinear = False  # 非线性参数

        full_model = BertAdapterForSequenceClassification.from_pretrained(model_path, config=full_modelConfig)

        full_model.freeze_model(True)  # freeze all params
        # unfreeze adapter params
        adapter_param = ["adapter_fi", "adapter_se"]
        adapter_param_list = [p for n, p in full_model.named_parameters() if any(nd in n for nd in adapter_param)]
        for param in adapter_param_list:
            param.requires_grad = True

        # 根据已经保存的文件重新构建模型的某些参数
        full_model = utils.re_adapter(full_model, epoch_save + 'fed_avg.pt')
        full_model.to(device)

        v_acc = model_test(full_model, validation_dataloader, device, 'Val')
        val_acc.append(v_acc)
        t_acc = model_test(full_model, test_dataloader, device, 'Test')
        test_acc.append(t_acc)

    # 返回最后的测试结果
    return val_acc, test_acc


def center_adapter_classifier(model_path, device, batch_size, seed, epoch, learning_rate):
    layer_num = 6

    param_dict = './outputs/LayerModel/Adapter/Chemprot/adapters_6/'
    # 清空已经更新过的无用参数，适用于客户端训练相同的参数
    shutil.rmtree(param_dict)
    os.makedirs(param_dict)

    param_container = utils.create_container()  # 制作本地参数容器

    method.setup_seed(seed)
    modelConfig = BertConfig.from_pretrained(model_path)
    modelConfig.num_hidden_layers = layer_num  # 相当于构建一个小模型，transformer层只有六层
    # modelConfig.adapter_nums = 6  # 高层含有adapters的数目
    modelConfig.has_adapter = True  # 设置开启adapter(diy)
    modelConfig.isLinear = False  # 非线性参数
    modelConfig.num_labels = num_token  # 设置分类模型的输出个数
    model = BertAdapterForSequenceClassification.from_pretrained(model_path, config=modelConfig)

    layer_list = np.random.randint(0, 12, size=layer_num)  # 产生[drop_layer,12)之间的随机整数来选择模型接下来要拷贝的参数
    layer_list = np.sort(layer_list)  # 对后续层进行排序
    print(layer_list)
    model = utils.re_transformers(model, param_container, layer_list)
    # model = utils.avg_transformers(model, param_container)  # 依次平均12层为6层

    model.freeze_model(True)  # freeze all params
    # unfreeze adapter params
    adapter_param = ["adapter_fi", "adapter_se"]
    adapter_param_list = [p for n, p in model.named_parameters() if any(nd in n for nd in adapter_param)]
    for param in adapter_param_list:
        param.requires_grad = True

    train_dataloader = data_load(model_path, 'train', device, batch_size)
    validation_dataloader = data_load(model_path, 'dev', device, batch_size)
    val_acc = []  # 训练过程中的验证集精度
    # 测试训练结果
    test_acc = []
    test_dataloader = data_load(model_path, 'test', device, batch_size)

    # model = utils.train_trans_layer(model, [0, 1, 2])  # 训练特定的transformer layers
    # model = utils.train_cls_layer(model)  # 训练分类输出层

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
    # full_modelConfig.num_hidden_layers = layer_num  # 相当于构建一个小模型，transformer层只有六层
    full_modelConfig.adapter_nums = 6  # 高层含有adapters的数目
    full_modelConfig.has_adapter = True  # 设置开启adapter(diy)
    full_modelConfig.isLinear = False  # 非线性参数

    full_model = BertAdapterForSequenceClassification.from_pretrained(model_path, config=full_modelConfig)

    full_model.freeze_model(True)  # freeze all params
    # unfreeze adapter params
    adapter_param = ["adapter_fi", "adapter_se"]
    adapter_param_list = [p for n, p in full_model.named_parameters() if any(nd in n for nd in adapter_param)]
    for param in adapter_param_list:
        param.requires_grad = True

    # full_model = utils.freeze_trans_layer(full_model, [0, 1, 2, 3, 4, 5])  # 固定低层adapters

    for name, param in full_model.named_parameters():
        if param.requires_grad:
            print(name)

    for i in range(epoch):
        epoch_save = param_dict + 'epoch_' + str(i + 1) + '/'

        if not os.path.exists(epoch_save):
            os.makedirs(epoch_save)
        print("Epochs:%d/%d" % ((i + 1), epoch))

        model = down_classier(model, train_dataloader, device, learning_rate, optimizer, scheduler)

        # 保存client更新的参数
        utils.layer_save(model, epoch_save + 'adapters.pt')

        # v_acc = model_test(model, validation_dataloader, device, 'Val')
        # val_acc.append(v_acc)
        # t_acc = model_test(model, test_dataloader, device, 'Test')
        # test_acc.append(t_acc)

        # 根据已经保存的文件重新构建模型的某些参数
        full_model = utils.re_adapter(full_model, epoch_save + 'adapters.pt')
        full_model.to(device)

        v_acc = model_test(full_model, validation_dataloader, device, 'Val')
        val_acc.append(v_acc)
        t_acc = model_test(full_model, test_dataloader, device, 'Test')
        test_acc.append(t_acc)

    # 返回最后的测试结果
    return val_acc, test_acc


def adapter_size():
    seed = 40
    model_path = './model/bert-base-uncased/'

    param_dict = './outputs/LayerModel/Adapter/Test/test.pt'

    method.setup_seed(seed)
    full_modelConfig = BertConfig.from_pretrained(model_path)
    full_modelConfig.num_labels = num_token  # 设置分类模型的输出个数
    # full_modelConfig.num_hidden_layers = layer_num  # 相当于构建一个小模型，transformer层只有六层
    full_modelConfig.adapter_nums = 6  # 高层含有adapters的数目
    full_modelConfig.has_adapter = True  # 设置开启adapter(diy)
    full_modelConfig.isLinear = False  # 非线性参数

    full_model = BertAdapterForSequenceClassification.from_pretrained(model_path, config=full_modelConfig)

    full_model.freeze_model(True)  # freeze all params
    # unfreeze adapter params

    # full_model = utils.freeze_trans_layer(full_model, [0, 1, 2, 3, 4, 5])  # 固定低层adapters

    for name, param in full_model.named_parameters():
        if 'adapter' in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    for name, param in full_model.named_parameters():
        if param.requires_grad:
            print(name)
            print(param.size())
            print(param.element_size())
            print(param.element_size() * param.nelement())
        else:
            print(name)
            print(param.element_size())

    utils.layer_save(full_model, param_dict)


def main():
    model_path = './model/bert-base-uncased/'
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    seed = 40
    batch_size = 32

    epoch = 15
    learning_rate = 5e-4

    # val_acc0, test_acc0 = fed_adapter(model_path, device, batch_size, seed, epoch, learning_rate)
    val_acc1, test_acc1 = center_adapter_classifier(model_path, device, batch_size, seed, epoch, learning_rate)

    Val_acc = [val_acc1]
    Test_acc = [test_acc1]

    legend = ['Random Adapters']
    # method.plot_res('Val Acc', legend, Val_acc)
    method.plot_res('Test Acc', legend, Test_acc)

    method.get_test_acc(legend, Val_acc, Test_acc)
    print("--------------------")
    for i in range(len(legend)):
        print(legend[i], end=" ")
        print("Max Test Acc: ")
        print(max(Test_acc[i]))


if __name__ == "__main__":
    # main()
    adapter_size()

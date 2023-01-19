#!/usr/bin/env python
# -*- coding:utf-8 -*-
# author:HanZhou
# datetime:2022/11/4 15:27
# software: PyCharm
import os
import shutil

import natsort
import numpy as np
import torch
from transformers import BertConfig, BertForMaskedLM

import method


def train_embeddings(model):
    for name, param in model.named_parameters():
        if 'embeddings' in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    return model


def train_low(model):
    for name, param in model.named_parameters():
        param.requires_grad = False
    for layer in model.bert.encoder.layer[:4]:
        for p in layer.parameters():
            p.requires_grad = True

    return model


def train_middle(model):
    for name, param in model.named_parameters():
        param.requires_grad = False
    for layer in model.bert.encoder.layer[4:8]:
        for p in layer.parameters():
            p.requires_grad = True

    return model


def train_high(model):
    for name, param in model.named_parameters():
        param.requires_grad = False
    for layer in model.bert.encoder.layer[8:]:
        for p in layer.parameters():
            p.requires_grad = True

    return model


def train_last(model):
    # 只训练最高层的encoder layer
    for name, param in model.named_parameters():
        if '11' in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    return model


def train_first(model):
    for name, param in model.named_parameters():
        param.requires_grad = False
    for layer in model.bert.encoder.layer[:1]:
        for p in layer.parameters():
            p.requires_grad = True

    return model


# def fed_add():
#     ori_path = './model/bert-base-uncased/'
#     fed_train_path = './outputs/fed/'
#     client_list = []
#     for file in os.listdir(fed_train_path):
#         client_list.append(fed_train_path + file + '/')
#
#     # 进行排序，来确定每个客户端训练的参数
#     client_list.sort()
#     model_list = []
#     param_list = []
#
#     # 构建模型列表
#     for i in range(len(client_list)):
#         modelConfig = BertConfig.from_pretrained(client_list[i])
#         model = BertForMaskedLM.from_pretrained(client_list[i], config=modelConfig)
#         model_list.append(model)
#
#     # 聚合低层参数
#     for layer0, layer1 in zip(model_list[0].bert.encoder.layer[:4],
#                               model_list[1].bert.encoder.layer[:4]):
#         for p0, p1 in zip(layer0.parameters(), layer1.parameters()):
#             param = (p0 + p1) / 2
#             param_list.append(param)
#
#     # 聚合中层参数
#     for layer2, layer3 in zip(model_list[2].bert.encoder.layer[4:8],
#                               model_list[3].bert.encoder.layer[4:8]):
#         for p2, p3 in zip(layer2.parameters(), layer3.parameters()):
#             param = (p2 + p3) / 2
#             param_list.append(param)
#
#     # 聚合高层参数
#     for layer4, layer5 in zip(model_list[4].bert.encoder.layer[8:],
#                               model_list[5].bert.encoder.layer[8:]):
#         for p4, p5 in zip(layer4.parameters(), layer5.parameters()):
#             param = (p4 + p5) / 2
#             param_list.append(param)
#
#     # 更新模型参数
#     modelConfig = BertConfig.from_pretrained(ori_path)
#     model = BertForMaskedLM.from_pretrained(ori_path, config=modelConfig)
#     num = 0
#     for layer in model.bert.encoder.layer:
#         for p in layer.parameters():
#             p.data = param_list[num]
#             num += 1
#
#     output_dir = './outputs/fed_avg/'
#     if not os.path.exists(output_dir):
#         os.makedirs(output_dir)
#
#     model_to_save = model.module if hasattr(model, 'module') else model
#     output_model_file = os.path.join(output_dir, WEIGHTS_NAME)
#     output_config_file = os.path.join(output_dir, CONFIG_NAME)
#
#     torch.save(model_to_save.state_dict(), output_model_file)
#     model_to_save.config.to_json_file(output_config_file)


def federated_global_merge():
    # 进行联邦聚合，不但客户端模型本身训练的参数会更新，没有训练的参数也会更新为其他客户端所训练的结果
    ori_path = './model/bert-base-uncased/'
    params_path = './outputs/params/client/'

    params_list = {}  # 以键值对的方式存储参数名称和参数值
    params_nums_list = {}  # 以键值对的方式存储参数名称和参数个数，即有几个客户端的参数相同需要聚合平均

    for file in os.listdir(params_path):
        params = torch.load(params_path + file)  # 加载一个客户端的参数
        for param_name in params:  # 遍历参数名称
            if param_name in params_list.keys():  # 如果已经存储有此参数
                param_value = params_list[param_name] + params[param_name]  # 加上其他客户端参数值
                param_num = params_nums_list[param_name] + 1  # 参数数目加一
                params_list.update({param_name: param_value})
                params_nums_list.update({param_name: param_num})
            else:
                param_value = params[param_name]
                param_num = 1
                params_list.update({param_name: param_value})
                params_nums_list.update({param_name: param_num})

    for param_name in params_list:  # 进行FedAVG更新
        param_value = params_list[param_name]
        param_num = params_nums_list[param_name]
        params_list.update({param_name: param_value / param_num})

    modelConfig = BertConfig.from_pretrained(ori_path)
    model = BertForMaskedLM.from_pretrained(ori_path, config=modelConfig)
    for name, params in model.named_parameters():
        if name in params_list.keys():
            params.data = params_list[name].data

    output_dir = './outputs/fed_avg/'
    method.model_save(model, output_dir)


def federated_efficient_merge(params_path):
    # 进行联邦聚合，仅客户端模型本身训练的参数会更新，各个客户端没有训练的参数保持不变

    params_list = {}  # 以键值对的方式存储参数名称和参数值
    params_nums_list = {}  # 以键值对的方式存储参数名称和参数个数，即有几个客户端的参数相同需要聚合平均

    file_name = natsort.natsorted(os.listdir(params_path), alg=natsort.ns.PATH)
    for file in file_name:
        params = torch.load(params_path + file)  # 加载一个客户端的参数
        for param_name in params:  # 遍历参数名称
            if param_name in params_list.keys():  # 如果已经存储有此参数
                param_value = params_list[param_name] + params[param_name]  # 加上其他客户端参数值
                param_num = params_nums_list[param_name] + 1  # 参数数目加一
                params_list.update({param_name: param_value})
                params_nums_list.update({param_name: param_num})
            else:
                param_value = params[param_name]
                param_num = 1
                params_list.update({param_name: param_value})
                params_nums_list.update({param_name: param_num})

    for param_name in params_list:  # 进行FedAVG更新
        param_value = params_list[param_name]
        param_num = params_nums_list[param_name]
        params_list.update({param_name: param_value / param_num})

    # 更新各个客户端已经训练的参数联邦后的结果，适用于各个客户端训练不同的参数
    # for file in file_name:
    #     params = torch.load(params_path + file)  # 加载一个客户端的参数
    #     for param_name in params:  # 遍历参数名称
    #         # 更新模型保存的参数
    #         params[param_name].data = params_list[param_name].data
    #     # 存储新的客户端参数结果
    #     torch.save(params, params_path + file)

    # 清空已经更新过的无用参数，适用于客户端训练相同的参数
    shutil.rmtree(params_path)
    os.makedirs(params_path)
    torch.save(params_list, params_path + 'fed_avg.pt')


def map_change(params_list, map_num):
    """
    将小模型与大模型进行映射
    :param params_list:原始小模型保存的参数
    :param map_num:需要跨越的层数
    :return:transformer 层进行更改过后的参数字典
    """
    new_param = {}
    for ori_name in params_list.keys():
        if 'bert.encoder.layer.' in ori_name:
            # 如果是transformer层，则进行映射
            name_list = ori_name.split('.')
            layer_num = int(name_list[3])  # 获取层数
            new_name = ori_name.replace(name_list[3], str(layer_num + map_num))

            new_param.update({new_name: params_list[ori_name]})
        else:
            # 否则不需要进行映射
            new_param.update({ori_name: params_list[ori_name]})

    return new_param


def re_param(model, param_path, map_num=-1):
    model = model.module if hasattr(model, 'module') else model
    params_list = torch.load(param_path)  # 加载客户端的参数

    if map_num != -1:  # 如果需要进行层数映射
        # 进行层数映射
        print("Change!")
        params_list = map_change(params_list, map_num)

    for name, params in model.named_parameters():  # 更新客户端的训练结果
        if name in params_list.keys():
            print("Re_param " + name + " !")
            # print(name)
            params.data = params_list[name].data

    return model


def re_adapter(model, param_path):
    model = model.module if hasattr(model, 'module') else model
    params_dict = torch.load(param_path)  # 加载客户端的参数

    param_list = []
    for p in params_dict.keys():
        param_list.append(params_dict[p])

    param_index = 0

    for name, params in model.named_parameters():
        if params.requires_grad:
            params.data = param_list[param_index]
            param_index += 1

    return model


def re_transformers(model, container, param_list):
    model = model.module if hasattr(model, 'module') else model

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

    params = []  # 记录下原始Bert模型参数
    for i in param_list:
        for last_name in last_name_list:
            name = first_name + str(i) + last_name
            print("Rebuild：" + name)
            params.append(container[name])

    param_index = 0
    for layer in model.bert.encoder.layer:  # 对所有的transformer层进行替换
        for n, p in layer.named_parameters():
            if 'adapter' in n:
                continue
            else:
                p.data = params[param_index]
                param_index += 1

    if param_index == len(params):
        print("Rebuild Success！")
    else:
        print("Rebuild Fail!")

    return model


def avg_transformers(model, container):
    model = model.module if hasattr(model, 'module') else model

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

    params = []
    for i in range(0, 12, 2):
        for last_name in last_name_list:
            pre_name = first_name + str(i) + last_name
            behind_name = first_name + str(i + 1) + last_name
            print("Avg: " + pre_name + " + " + behind_name)
            params.append((container[pre_name] + container[behind_name])/2)

    param_index = 0
    for layer in model.bert.encoder.layer:  # 对所有的transformer层进行替换
        for n, p in layer.named_parameters():
            if 'adapter' in n:
                continue
            else:
                p.data = params[param_index]
                param_index += 1

    if param_index == len(params):
        print("Avg Success！")
    else:
        print("Avg Fail!")

    return model


def share_param(model, param_path):
    model = model.module if hasattr(model, 'module') else model

    params_dict = torch.load(param_path)  # 加载客户端的参数
    params_list = []
    for value in params_dict.values():
        params_list.append(value)

    for layer in model.bert.encoder.layer:
        for i, p in enumerate(layer.parameters()):
            p.data = params_list[i]

    return model


def layer_save(model, save_path):
    model_to_save = model.module if hasattr(model, 'module') else model
    param_list = {}
    for name, param in model_to_save.named_parameters():
        if param.requires_grad:
            param_list.update({name: param})

    torch.save(param_list, save_path)


def rebuild_model(model, container, layer_length, drop_layer=1, ori_layer=-1):
    """
    重新构建模型的后续transformer层参数
    :param model:LayerBert模型,只有几层的transformer模型
    :param container:本地参数容器，保存有模型的参数
    :param layer_length: 从大模型选择映射的transformer层数量
    :param drop_layer: 跳过几层开始替换后续模型参数，默认跳过1层参数
    :param ori_layer:保持一层参数同parameter_container一致，用于训练
    """
    # print("Rebuild Model......")
    model = model.module if hasattr(model, 'module') else model

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

    if layer_length != 0:  # 如果需要进行重新构建模型
        layer_list = np.random.randint(drop_layer, 12, size=layer_length)  # 产生[drop_layer,12)之间的随机整数来选择模型接下来要拷贝的参数
        layer_list = np.sort(layer_list)  # 对后续层进行排序
        print("-----------------")
        print(layer_list)
        print("-----------------")
        layer_param = []  # 记录下原始Bert模型参数
        for i in layer_list:
            params = []
            for last_name in last_name_list:
                name = first_name + str(i) + last_name
                print("Rebuild：" + name)
                params.append(container[name])
            layer_param.append(params)

        for j in range(len(layer_param)):
            for layer in model.bert.encoder.layer[j + drop_layer:j + drop_layer + 1]:  # 跳过的transformer层参数不需要改变
                for p, d in zip(layer.parameters(), layer_param[j]):
                    p.data = d

    # 跳过层的最后一层与原始模型的相应层的结构对应
    if ori_layer != -1:
        print("Ori Layer：" + str(ori_layer))
        train_param = []
        for last_name in last_name_list:
            name = first_name + str(ori_layer) + last_name
            train_param.append(container[name])
        for layer in model.bert.encoder.layer[ori_layer:ori_layer + 1]:
            for p, d in zip(layer.parameters(), train_param):
                p.data = d

    return model


def lower_build(model, container, layer_length, drop_layer=1, ori_layer=-1):
    """
        重新构建模型的后续transformer层参数
        :param model:LayerBert模型,只有几层的transformer模型
        :param container:本地参数容器，保存有模型的参数
        :param layer_length: 从大模型选择映射的transformer层数量
        :param drop_layer: 跳过几层开始替换后续模型参数，默认跳过1层参数
        :param ori_layer:保持一层参数同parameter_container一致，用于训练
        """
    # print("Rebuild Model......")
    model = model.module if hasattr(model, 'module') else model

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

    if layer_length != 0:  # 如果需要进行重新构建模型
        layer_list = np.random.randint(0, 12 - drop_layer,
                                       size=layer_length)  # 产生[0,12-drop_layer)之间的随机整数来选择模型接下来要拷贝的参数
        layer_list = np.sort(layer_list)  # 对后续层进行排序
        print("-----------------")
        print(layer_list)
        print("-----------------")
        layer_param = []  # 记录下原始Bert模型参数
        for i in layer_list:
            params = []
            for last_name in last_name_list:
                name = first_name + str(i) + last_name
                # print("Rebuild：" + name)
                params.append(container[name])  # 记录这一层的参数
            layer_param.append(params)

        for j in range(len(layer_param)):
            for layer in model.bert.encoder.layer[j:j + 1]:  # 将低层参数进行改变
                for p, d in zip(layer.parameters(), layer_param[j]):
                    p.data = d

    # 跳过层的最后一层与原始模型的相应层的结构对应
    if ori_layer != -1:
        print("Ori Layer：" + str(ori_layer))
        train_param = []
        for last_name in last_name_list:
            name = first_name + str(ori_layer) + last_name
            train_param.append(container[name])
        for layer in model.bert.encoder.layer[ori_layer - 6:ori_layer - 5]:
            for p, d in zip(layer.parameters(), train_param):
                p.data = d

    return model


def map_ori_layer(model, container, ori_layer, layer_num):
    model = model.module if hasattr(model, 'module') else model

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

    train_param = []
    for last_name in last_name_list:
        name = first_name + str(ori_layer) + last_name
        train_param.append(container[name])
    for layer in model.bert.encoder.layer[ori_layer - layer_num:ori_layer - layer_num + 1]:
        for p, d in zip(layer.parameters(), train_param):
            p.data = d

    return model


def train_trans_layer(model, layer_list):
    model = model.module if hasattr(model, 'module') else model

    for name, param in model.named_parameters():
        param.requires_grad = False
    for l_n in layer_list:
        for layer in model.bert.encoder.layer[l_n:l_n + 1]:
            for p in layer.parameters():
                p.requires_grad = True

    return model


def train_cls_layer(model):
    model = model.module if hasattr(model, 'module') else model

    for name, param in model.named_parameters():
        if 'cls' in name:
            param.requires_grad = True

    return model


def freeze_trans_layer(model, layer_list):
    model = model.module if hasattr(model, 'module') else model

    for l_n in layer_list:
        for layer in model.bert.encoder.layer[l_n:l_n + 1]:
            for p in layer.parameters():
                p.requires_grad = False

    return model


def train_classifier_layer(model):
    for name, param in model.named_parameters():
        if 'classifier' in name:
            param.requires_grad = True

    return model


def create_container():
    ori_path = "./model/bert-base-uncased/"
    ori_modelConfig = BertConfig.from_pretrained(ori_path)
    ori_model = BertForMaskedLM.from_pretrained(ori_path, config=ori_modelConfig)

    param_list = {}
    # for layer in ori_model.bert.encoder.layer:
    #     for name, param in layer.named_parameters():
    #         param_list.update({name: param})
    for name, param in ori_model.named_parameters():
        if 'encoder' in name:
            param_list.update({name: param})

    # 返回初始的本地参数容器
    return param_list


def update_container(container, param_path, map_num=-1):
    params_list = torch.load(param_path)  # 加载聚合后的参数

    if map_num != -1:  # 如果需要进行层数映射
        # 进行层数映射
        print("Change!")
        params_list = map_change(params_list, map_num)

    for name in container:
        if name in params_list.keys():
            print("container " + name + " re_param!")
            # print(name)
            container[name].data = params_list[name].data

    return container


def map3to12(model, layer_list, container):
    """
    将原始模型的12层中根据layer——list映射到对应的model中，
    model的transformer 层只有三层
    :param model: 小模型只有三层
    :param layer_list: 需要映射的模型层数,共有三个
    :param container: 本地的参数容器
    """
    model = model.module if hasattr(model, 'module') else model

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

    param_dict = []
    for i in layer_list:
        for last_name in last_name_list:
            name = first_name + str(i) + last_name
            param_dict.append(container[name])

    param_index = 0
    for layer in model.bert.encoder.layer:
        for p in layer.parameters():
            p.data = param_dict[param_index]
            param_index += 1

    return model


def map9to3(model):
    ori_path = "./model/bert-base-uncased/"
    ori_modelConfig = BertConfig.from_pretrained(ori_path)
    ori_model = BertForMaskedLM.from_pretrained(ori_path, config=ori_modelConfig)

    first_name = 'bert.encoder.layer.'
    layer_param = {}

    for i, layer in enumerate(ori_model.bert.encoder.layer[3:]):
        layer_index = 3 + i // 3
        for n, p in layer.named_parameters():
            param_name = first_name + str(layer_index) + '.' + n
            if param_name in layer_param:  # 若存在则添加，不存在就新建
                layer_param.update({param_name: layer_param[param_name] + p.data})
            else:
                layer_param.update({param_name: p.data})
            if i % 3 == 2:  # 三层一平均
                layer_param.update({param_name: layer_param[param_name] / 3})

    for i, layer in enumerate(model.bert.encoder.layer[3:]):
        for n, p in layer.named_parameters():
            param_name = first_name + str(i + 3) + '.' + n
            print("Avg:" + param_name)
            p.data = layer_param[param_name]

    return model

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
    # only train the last encoder layer
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
#     # sort
#     client_list.sort()
#     model_list = []
#     param_list = []
#
#     # create model list
#     for i in range(len(client_list)):
#         modelConfig = BertConfig.from_pretrained(client_list[i])
#         model = BertForMaskedLM.from_pretrained(client_list[i], config=modelConfig)
#         model_list.append(model)
#
#     # merge shallower layers
#     for layer0, layer1 in zip(model_list[0].bert.encoder.layer[:4],
#                               model_list[1].bert.encoder.layer[:4]):
#         for p0, p1 in zip(layer0.parameters(), layer1.parameters()):
#             param = (p0 + p1) / 2
#             param_list.append(param)
#
#     # merge middle layers
#     for layer2, layer3 in zip(model_list[2].bert.encoder.layer[4:8],
#                               model_list[3].bert.encoder.layer[4:8]):
#         for p2, p3 in zip(layer2.parameters(), layer3.parameters()):
#             param = (p2 + p3) / 2
#             param_list.append(param)
#
#     # merge higher layers
#     for layer4, layer5 in zip(model_list[4].bert.encoder.layer[8:],
#                               model_list[5].bert.encoder.layer[8:]):
#         for p4, p5 in zip(layer4.parameters(), layer5.parameters()):
#             param = (p4 + p5) / 2
#             param_list.append(param)
#
#     # update
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
    # For federated aggregation, not only the parameters trained by the client model itself will be updated, 
    # but also the parameters that are not trained will be updated to the results of training by other clients
    ori_path = './model/bert-base-uncased/'
    params_path = './outputs/params/client/'

    params_list = {}  # Store parameter names and parameter values as key-value pairs
    params_nums_list = {}  # Store parameter names and number of parameters in the form of key-value pairs, that is, several clients with the same parameters need to be aggregated and averaged

    for file in os.listdir(params_path):
        params = torch.load(params_path + file)  # Load a client's parameters
        for param_name in params:  # Iterate through the parameter names
            if param_name in params_list.keys():  # If this parameter is already stored
                param_value = params_list[param_name] + params[param_name]  # plus other client parameter values
                param_num = params_nums_list[param_name] + 1  # The number of parameters is incremented by one
                params_list.update({param_name: param_value})
                params_nums_list.update({param_name: param_num})
            else:
                param_value = params[param_name]
                param_num = 1
                params_list.update({param_name: param_value})
                params_nums_list.update({param_name: param_num})

    for param_name in params_list:  # FedAVG
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
    # For federated aggregation, only the parameters trained by the client model itself are updated, and the parameters that are not trained by each client remain unchanged

    params_list = {}  # Store parameter names and parameter values as key-value pairs
    params_nums_list = {}  # Store parameter names and number of parameters in the form of key-value pairs, that is, several clients with the same parameters need to be aggregated and averaged
    file_name = natsort.natsorted(os.listdir(params_path), alg=natsort.ns.PATH)
    for file in file_name:
        params = torch.load(params_path + file)  # Load a client's parameters
        for param_name in params:  # Iterate through the parameter names
            if param_name in params_list.keys():  # If this parameter is already stored
                param_value = params_list[param_name] + params[param_name]  # plus other client parameter values
                param_num = params_nums_list[param_name] + 1  # The number of parameters is incremented by one
                params_list.update({param_name: param_value})
                params_nums_list.update({param_name: param_num})
            else:
                param_value = params[param_name]
                param_num = 1
                params_list.update({param_name: param_value})
                params_nums_list.update({param_name: param_num})

    for param_name in params_list:  # FedAVG
        param_value = params_list[param_name]
        param_num = params_nums_list[param_name]
        params_list.update({param_name: param_value / param_num})

    # The result of updating the federation of parameters trained by each client is applicable to the training of different parameters for each client
    # for file in file_name:
    #     params = torch.load(params_path + file)  # Load a client's parameters
    #     for param_name in params:  # Iterate through the parameter names
    #         # Update the parameters saved by the model
    #         params[param_name].data = params_list[param_name].data
    #     # Stores the new client parameter results
    #     torch.save(params, params_path + file)

    # Clear the already updated useless parameters, which is suitable for the client to train the same parameters
    shutil.rmtree(params_path)
    os.makedirs(params_path)
    torch.save(params_list, params_path + 'fed_avg.pt')

    
 def federated_merge_by_weight(params_path, weight_list):
    params_list = {}  # Store parameter names and parameter values as key-value pairs

    weight_sum = sum(weight_list)
    file_name = natsort.natsorted(os.listdir(params_path), alg=natsort.ns.PATH)
    for i, file in enumerate(file_name):
        params = torch.load(params_path + file)  # Load a client's parameters
        weight = weight_list[i]
        for param_name in params:  # Iterate through the parameter names
            if param_name in params_list.keys():  # If this parameter is already stored
                param_value = params_list[param_name] + weight*params[param_name]  # plus other client parameter values
                params_list.update({param_name: param_value})
            else:
                param_value = weight*params[param_name]
                params_list.update({param_name: param_value})

    for param_name in params_list:  # FedAvg
        param_value = params_list[param_name]
        params_list.update({param_name: param_value / weight_sum})

    # Clear the already updated useless parameters, which is suitable for the client to train the same parameters
    shutil.rmtree(params_path)
    os.makedirs(params_path)
    torch.save(params_list, params_path + 'fed_avg.pt')

def map_change(params_list, map_num):
    """
    Map small models to large models
    :param params_list:Parameters saved by the original small model
    :param map_num:The number of layers that need to be spanned
    :return:A dictionary of parameters after the transformer layer has been changed
    """
    new_param = {}
    for ori_name in params_list.keys():
        if 'bert.encoder.layer.' in ori_name:
            # If it is a transformer layer, it is mapped
            name_list = ori_name.split('.')
            layer_num = int(name_list[3])  # Gets the number of layers
            new_name = ori_name.replace(name_list[3], str(layer_num + map_num))

            new_param.update({new_name: params_list[ori_name]})
        else:
            # Otherwise, no mapping is required
            new_param.update({ori_name: params_list[ori_name]})

    return new_param


def re_param(model, param_path, map_num=-1):
    model = model.module if hasattr(model, 'module') else model
    params_list = torch.load(param_path)  #  Load the parameters of the client
    if map_num != -1:  # If layer mapping is required
       
        print("Change!")
        params_list = map_change(params_list, map_num)

    for name, params in model.named_parameters():  # update
        if name in params_list.keys():
            print("Re_param " + name + " !")
            # print(name)
            params.data = params_list[name].data

    return model


def re_adapter(model, param_path):
    model = model.module if hasattr(model, 'module') else model
    params_dict = torch.load(param_path)  # load parameter

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

    params = []  # Record the original Bert model parameters
    for i in param_list:
        for last_name in last_name_list:
            name = first_name + str(i) + last_name
            print("Rebuild：" + name)
            params.append(container[name])

    param_index = 0
    for layer in model.bert.encoder.layer:  # Replace all transformer layers
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
    for layer in model.bert.encoder.layer:  # Replace all transformer layers
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

    params_dict = torch.load(param_path)  # loading parameters from client
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
    Subsequent transformer layer parameters for rebuilding the model
    :param model:LayerBert, only have several transformer layers
    :param container:A local parameter container that holds the parameters of the model
    :param layer_length: Select the number of mapped transformer layers from the large model
    :param drop_layer: Skip a few layers to replace subsequent model parameters, and skip layer 1 parameter by default
    :param ori_layer:Keep a layer of parameters consistent with the parameter_container for training
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

    if layer_length != 0:  # If you need to rebuild the model
        layer_list = np.random.randint(drop_layer, 12, size=layer_length)  # Random integers between [drop_layer,12) are generated to select the parameters to be copied next by the model
        layer_list = np.sort(layer_list)  # Sorts subsequent layers
        print("-----------------")
        print(layer_list)
        print("-----------------")
        layer_param = []  # Record the original Bert model parameters
        for i in layer_list:
            params = []
            for last_name in last_name_list:
                name = first_name + str(i) + last_name
                print("Rebuild：" + name)
                params.append(container[name])
            layer_param.append(params)

        for j in range(len(layer_param)):
            for layer in model.bert.encoder.layer[j + drop_layer:j + drop_layer + 1]:  # The skipped transformer layer parameters do not need to be changed
                for p, d in zip(layer.parameters(), layer_param[j]):
                    p.data = d

    # The last layer of the skipped layer corresponds to the structure of the corresponding layer of the original model
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
        Subsequent transformer layer parameters for rebuilding the model
        :param model:LayerBert, only have several transformer layers
        :param container:A local parameter container that holds the parameters of the model
        :param layer_length: Select the number of mapped transformer layers from the large model
        :param drop_layer: Skip a few layers to replace subsequent model parameters, and skip layer 1 parameter by default
        :param ori_layer:Keep a layer of parameters consistent with the parameter_container for training
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

    if layer_length != 0:  # If you need to rebuild the model
        layer_list = np.random.randint(0, 12 - drop_layer,
                                       size=layer_length)  # Random integers between [0,12-drop_layer) are generated to select the parameters to be copied by the model next
        layer_list = np.sort(layer_list)  # Sorts subsequent layers
        print("-----------------")
        print(layer_list)
        print("-----------------")
        layer_param = []  # Record the original Bert model parameters
        for i in layer_list:
            params = []
            for last_name in last_name_list:
                name = first_name + str(i) + last_name
                # print("Rebuild：" + name)
                params.append(container[name])  # Record the parameters for this layer
            layer_param.append(params)

        for j in range(len(layer_param)):
            for layer in model.bert.encoder.layer[j:j + 1]:  # Change the low-level parameters
                for p, d in zip(layer.parameters(), layer_param[j]):
                    p.data = d

    # The last layer of the skipped layer corresponds to the structure of the corresponding layer of the original model
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

    # Returns the initial local parameter container
    return param_list


def update_container(container, param_path, map_num=-1):
    params_list = torch.load(param_path)  # Parameters after loading aggregations

    if map_num != -1:  # If layer mapping is required
        # Map the number of layers
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
    Map the 12 layers of the original model according to the layer-list to the corresponding model,
    small model have 3 transformer
    :param model: 
    :param layer_list: There are three model layers that need to be mapped
    :param container: local parameters pool
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
            if param_name in layer_param:  # Add if it exists, create a new one if it does not exist
                layer_param.update({param_name: layer_param[param_name] + p.data})
            else:
                layer_param.update({param_name: p.data})
            if i % 3 == 2:  # Three layers and one average
                layer_param.update({param_name: layer_param[param_name] / 3})

    for i, layer in enumerate(model.bert.encoder.layer[3:]):
        for n, p in layer.named_parameters():
            param_name = first_name + str(i + 3) + '.' + n
            print("Avg:" + param_name)
            p.data = layer_param[param_name]

    return model

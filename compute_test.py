#!/usr/bin/env python
# -*- coding:utf-8 -*-
# author:HanZhou
# datetime:2022/9/28 21:30
# software: PyCharm
import os
import time

from timeit import default_timer as timer

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import DataCollatorForLanguageModeling, BertConfig, BertForMaskedLM, BertTokenizerFast, \
    BertForSequenceClassification

import method
import utils
from BertAdapter import BertAdapterForSequenceClassification
from downstream_train import data_load, down_classier, num_token


def train_first(model):
    for name, param in model.named_parameters():
        param.requires_grad = False
    for layer in model.bert.encoder.layer[:1]:
        for p in layer.parameters():
            p.requires_grad = True

    return model


def get_optimizer(model, learning_rate):
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [  # except requires_grad == False
        {'params': [p for n, p in param_optimizer if (not any(nd in n for nd in no_decay) and p.requires_grad)],
         'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if (any(nd in n for nd in no_decay) and p.requires_grad)],
         'weight_decay': 0.0}
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=learning_rate)

    return optimizer


def mlm_train(model, dataloader, device, optimizer):
    for batch in tqdm(dataloader):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        result = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = result[0]

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def compute_cost():
    learning_rate = 5e-5
    batch_size = 256
    seed = 40
    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
    model_path0 = './model/bert-base-uncased/'
    file_path = './data/datasets/Computer/test_0_128.pt'
    # file_path = './data/datasets/Center/Biology_128.pt'

    print("Data loading-------")
    tokenizer = BertTokenizerFast.from_pretrained(model_path0, do_lower_case=True)
    datasets = torch.load(file_path)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)

    dataloader = DataLoader(datasets, batch_size=batch_size, collate_fn=data_collator)

    print("Bert--------")
    method.setup_seed(seed)
    modelConfig = BertConfig.from_pretrained(model_path0)

    model = BertForMaskedLM.from_pretrained(model_path0, config=modelConfig)
    model.to(device)

    optimizer = get_optimizer(model, learning_rate)

    for name, param in model.named_parameters():
        print(name)
    t1 = timer()
    mlm_train(model, dataloader, device, optimizer)
    t2 = timer()
    print("-----------------------")

    print("DistillBert--------")
    model_path1 = './model/DistillBert/'
    method.setup_seed(seed)
    modelConfig_1 = BertConfig.from_pretrained(model_path1)
    model_1 = BertForMaskedLM.from_pretrained(model_path1, config=modelConfig_1)
    # model_1 = train_first(model_1)
    model_1.to(device)

    optimizer_1 = get_optimizer(model_1, learning_rate)

    for name, param in model_1.named_parameters():
        if param.requires_grad:
            print(name)
    t3 = timer()
    mlm_train(model_1, dataloader, device, optimizer_1)
    t4 = timer()
    print("-----------------------")

    print("TinyBert_6--------")
    model_path2 = './model/TinyBert_6/'
    method.setup_seed(seed)
    modelConfig_2 = BertConfig.from_pretrained(model_path2)
    model_2 = BertForMaskedLM.from_pretrained(model_path2, config=modelConfig_2)
    # model_2 = train_first(model_2)
    model_2.to(device)

    optimizer_2 = get_optimizer(model_2, learning_rate)

    for name, param in model_2.named_parameters():
        if param.requires_grad:
            print(name)
    t5 = timer()
    mlm_train(model_2, dataloader, device, optimizer_2)
    t6 = timer()
    print("-----------------------")

    print("TinyBert_4--------")
    model_path3 = './model/TinyBert_4/'
    method.setup_seed(seed)
    modelConfig_3 = BertConfig.from_pretrained(model_path3)
    model_3 = BertForMaskedLM.from_pretrained(model_path3, config=modelConfig_3)
    # model_3 = train_first(model_3)
    model_3.to(device)

    optimizer_3 = get_optimizer(model_3, learning_rate)

    for name, param in model_3.named_parameters():
        if param.requires_grad:
            print(name)
    t7 = timer()
    mlm_train(model_3, dataloader, device, optimizer_3)
    t8 = timer()
    print("-----------------------")

    print("LayerModel_6--------")
    model_path4 = './model/bert-base-uncased/'
    method.setup_seed(seed)
    modelConfig_4 = BertConfig.from_pretrained(model_path4)
    modelConfig_4.num_hidden_layers = 6  # have 6 transformer layers
    model_4 = BertForMaskedLM.from_pretrained(model_path4, config=modelConfig_4)
    model_4 = train_first(model_4)  # only train first layer
    model_4 = utils.train_cls_layer(model_4)  # train output layer
    model_4.to(device)

    optimizer_4 = get_optimizer(model_4, learning_rate)

    for name, param in model_4.named_parameters():
        if param.requires_grad:
            print(name)
    t9 = timer()
    mlm_train(model_4, dataloader, device, optimizer_4)
    t10 = timer()
    print("-----------------------")

    print("Bert training cost: %s S" % (t2 - t1))
    print("DistillBert training cost: %s S" % (t4 - t3))
    print("TinyBert_6 training cost: %s S" % (t6 - t5))
    print("TinyBert_4 training cost: %s S" % (t8 - t7))
    print("FedBFPT training cost: %s S" % (t10 - t9))


def down_cost():
    model_path = './model/bert-base-uncased/'
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    seed = 40
    batch_size = 32

    learning_rate = 5e-4

    adapter_param = ["adapter_fi", "adapter_se"]

    train_dataloader = data_load(model_path, 'train', device, batch_size)

    method.setup_seed(seed)
    modelConfig_0 = BertConfig.from_pretrained(model_path)
    modelConfig_0.num_hidden_layers = 12  
    modelConfig_0.num_labels = num_token 
    model_0 = BertForSequenceClassification.from_pretrained(model_path, config=modelConfig_0)

    model_0.to(device)

    optimizer_0 = torch.optim.AdamW(
        model_0.parameters(),
        lr=learning_rate,
    )
    scheduler_0 = torch.optim.lr_scheduler.ExponentialLR(optimizer_0, gamma=0.7)

    print("Full Model=--------")
    for name, param in model_0.named_parameters():
        if param.requires_grad:
            print(name)
    t1 = timer()
    down_classier(model_0, train_dataloader, device, learning_rate, optimizer_0, scheduler_0)
    t2 = timer()
    print("-----------------------")

    method.setup_seed(seed)
    modelConfig_1 = BertConfig.from_pretrained(model_path)
    modelConfig_1.num_hidden_layers = 12  # local model transformr layers number
    # modelConfig.adapter_nums = 6  # the number of have adapters
    modelConfig_1.has_adapter = True  
    modelConfig_1.isLinear = False  
    modelConfig_1.num_labels = num_token  
    model_1 = BertAdapterForSequenceClassification.from_pretrained(model_path, config=modelConfig_1)

    model_1.to(device)

    model_1.freeze_model(True)  # freeze all params
    # unfreeze adapter params

    adapter_param_list = [p for n, p in model_1.named_parameters() if any(nd in n for nd in adapter_param)]
    for param in adapter_param_list:
        param.requires_grad = True

    optimizer_1 = torch.optim.AdamW(
        model_1.parameters(),
        lr=learning_rate,
    )
    scheduler_1 = torch.optim.lr_scheduler.ExponentialLR(optimizer_1, gamma=0.7)

    print("Full Model--------")
    for name, param in model_1.named_parameters():
        if param.requires_grad:
            print(name)
    t3 = timer()
    down_classier(model_1, train_dataloader, device, learning_rate, optimizer_1, scheduler_1)
    t4 = timer()
    print("-----------------------")

    method.setup_seed(seed)
    modelConfig_2 = BertConfig.from_pretrained(model_path)
    modelConfig_2.num_hidden_layers = 12  
    modelConfig_2.adapter_nums = 6  
    modelConfig_2.has_adapter = True  
    modelConfig_2.isLinear = False  
    modelConfig_2.num_labels = num_token  
    model_2 = BertAdapterForSequenceClassification.from_pretrained(model_path, config=modelConfig_2)

    model_2.to(device)

    model_2.freeze_model(True)  # freeze all params
    # unfreeze adapter params

    adapter_param_list = [p for n, p in model_2.named_parameters() if any(nd in n for nd in adapter_param)]
    for param in adapter_param_list:
        param.requires_grad = True

    optimizer_2 = torch.optim.AdamW(
        model_2.parameters(),
        lr=learning_rate,
    )
    scheduler_2 = torch.optim.lr_scheduler.ExponentialLR(optimizer_2, gamma=0.7)

    print("Full Model Higher Adapters--------")
    for name, param in model_2.named_parameters():
        if param.requires_grad:
            print(name)
    t5 = timer()
    down_classier(model_2, train_dataloader, device, learning_rate, optimizer_2, scheduler_2)
    t6 = timer()
    print("-----------------------")

    method.setup_seed(seed)
    modelConfig_3 = BertConfig.from_pretrained(model_path)
    modelConfig_3.num_hidden_layers = 6 
    # modelConfig_2.adapter_nums = 6 
    modelConfig_3.has_adapter = True 
    modelConfig_3.isLinear = False 
    modelConfig_3.num_labels = num_token 
    model_3 = BertAdapterForSequenceClassification.from_pretrained(model_path, config=modelConfig_3)

    model_3.to(device)

    model_3.freeze_model(True)  # freeze all params
    # unfreeze adapter params

    adapter_param_list = [p for n, p in model_3.named_parameters() if any(nd in n for nd in adapter_param)]
    for param in adapter_param_list:
        param.requires_grad = True

    optimizer_3 = torch.optim.AdamW(
        model_3.parameters(),
        lr=learning_rate,
    )
    scheduler_3 = torch.optim.lr_scheduler.ExponentialLR(optimizer_3, gamma=0.7)

    print("LayerBert_06 adapters--------")
    for name, param in model_3.named_parameters():
        if param.requires_grad:
            print(name)
    t7 = timer()
    down_classier(model_3, train_dataloader, device, learning_rate, optimizer_3, scheduler_3)
    t8 = timer()
    print("-----------------------")

    print("Full Model full parameters  training cost: %s S" % (t2 - t1))
    print("Full Model full adapters  training cost: %s S" % (t4 - t3))
    print("Full Model Higher adapters training cost: %s S" % (t6 - t5))
    print("LayerBert_06 adapters training cost: %s S" % (t8 - t7))


def different_length():
    learning_rate = 5e-5
    batch_size = 256
    seed = 40
    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
    model_path0 = './model/bert-base-uncased/'
    file_path = './data/datasets/Computer/test_0_128.pt'
    # file_path = './data/datasets/Center/Biology_128.pt'

    print("Data loading-------")
    tokenizer = BertTokenizerFast.from_pretrained(model_path0, do_lower_case=True)
    datasets = torch.load(file_path)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)

    dataloader = DataLoader(datasets, batch_size=batch_size, collate_fn=data_collator)

    print("LayerModel_4--------")
    model_path0 = './model/bert-base-uncased/'
    method.setup_seed(seed)
    modelConfig_0 = BertConfig.from_pretrained(model_path0)
    modelConfig_0.num_hidden_layers = 4 
    model_0 = BertForMaskedLM.from_pretrained(model_path0, config=modelConfig_0)
    model_0 = train_first(model_0)  
    model_0 = utils.train_cls_layer(model_0)  
    model_0.to(device)

    optimizer_0 = get_optimizer(model_0, learning_rate)

    for name, param in model_0.named_parameters():
        print(name + ": " + str(param.requires_grad))
    t1 = timer()
    mlm_train(model_0, dataloader, device, optimizer_0)
    t2 = timer()
    print("-----------------------")

    print("LayerModel_6--------")
    model_path1 = './model/bert-base-uncased/'
    method.setup_seed(seed)
    modelConfig_1 = BertConfig.from_pretrained(model_path1)
    modelConfig_1.num_hidden_layers = 6  
    model_1 = BertForMaskedLM.from_pretrained(model_path1, config=modelConfig_1)
    model_1 = train_first(model_1)  
    model_1 = utils.train_cls_layer(model_1) 
    model_1.to(device)

    optimizer_1 = get_optimizer(model_1, learning_rate)

    for name, param in model_1.named_parameters():
        print(name + ": " + str(param.requires_grad))
    t3 = timer()
    mlm_train(model_1, dataloader, device, optimizer_1)
    t4 = timer()
    print("-----------------------")

    print("LayerModel_8--------")
    model_path2 = './model/bert-base-uncased/'
    method.setup_seed(seed)
    modelConfig_2 = BertConfig.from_pretrained(model_path2)
    modelConfig_2.num_hidden_layers = 8  
    model_2 = BertForMaskedLM.from_pretrained(model_path2, config=modelConfig_2)
    model_2 = train_first(model_2)  
    model_2 = utils.train_cls_layer(model_2) 
    model_2.to(device)

    optimizer_2 = get_optimizer(model_2, learning_rate)

    for name, param in model_2.named_parameters():
        print(name + ": " + str(param.requires_grad))
    t5 = timer()
    mlm_train(model_2, dataloader, device, optimizer_2)
    t6 = timer()
    print("-----------------------")

    print("LayerModel_6--------")
    model_path3 = './model/bert-base-uncased/'
    method.setup_seed(seed)
    modelConfig_3 = BertConfig.from_pretrained(model_path3)
    modelConfig_3.num_hidden_layers = 6  
    model_3 = BertForMaskedLM.from_pretrained(model_path3, config=modelConfig_3)
    # model_1 = train_first(model_1)  
    # model_1 = utils.train_cls_layer(model_1)  
    model_3.to(device)

    optimizer_3 = get_optimizer(model_3, learning_rate)

    for name, param in model_3.named_parameters():
        print(name + ": " + str(param.requires_grad))
    t7 = timer()
    mlm_train(model_3, dataloader, device, optimizer_3)
    t8 = timer()
    print("-----------------------")

    print("LayerModel_4 training cost: %s S" % (t2 - t1))
    print("LayerModel_6 training cost: %s S" % (t4 - t3))
    print("LayerModel_8 training cost: %s S" % (t6 - t5))
    print("LayerModel_6Full training cost: %s S" % (t8 - t7))


def main():
    # compute_cost()
    # down_cost()
    different_length()


if __name__ == "__main__":
    main()

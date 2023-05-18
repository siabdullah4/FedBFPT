#!/usr/bin/env python
# -*- coding:utf-8 -*-
# author:HanZhou
# datetime:2022/5/24 9:43
# software: PyCharm
import json
import os
import shutil

from torch.utils.data import DataLoader

os.environ["CUDA_VISIBLE_DEVICES"] = '4,5,6,7' # the cuda device you can choose

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
    Subsequent transformer layer parameters for rebuilding the model
    :param model:LayerBert have less transformer layers
    :param layer_length: Select the number of mapped transformer layers from the large model
    """
    print("Rebuild Model......")
    ori_path = "./model/bert-base-uncased/"
    ori_modelConfig = BertConfig.from_pretrained(ori_path)
    ori_model = BertForMaskedLM.from_pretrained(ori_path, config=ori_modelConfig)

    layer_list = np.random.randint(1, 12, size=layer_length)  # Random integers between [1,12) are generated to select the parameters that the model will copy next
    layer_param = []  # Record the original Bert model parameters
    for i in layer_list:
        params = []
        for layer in ori_model.bert.encoder.layer[i:i + 1]:
            for p in layer.parameters():
                params.append(p.data)
        layer_param.append(params)

    for j in range(len(layer_param)):
        for layer in model.module.bert.encoder.layer[j + 1:j + 2]:  # The transformer layer parameters of layer 0 do not need to be changed
            for p, d in zip(layer.parameters(), layer_param[j]):
                p.data = d

    return model


def huggingface_pretrain(model, save_path, train_dataset, learn_rate):
    """
    Based on the corpus, the model is pre-trained
    :param model: The model that needs to be trained
    :param save_path: The address where the training file is saved
    :param train_dataset: Pre-training dataset
    :param learn_rate: learning rate 
    """
    # load tokenizer and model
    token_path = './model/bert-base-uncased/'
    tokenizer = BertTokenizerFast.from_pretrained(token_path, do_lower_case=True)

    # for name, param in model.named_parameters():
    #     if param.requires_grad:
    #         print(name, end=" ")
    #         print(param.requires_grad)

    # DataCollator for MLM
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)
    pretrain_batch_size = 256
    num_train_epochs = 1

    # learn_rate = 6e-5  #
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if p.requires_grad]}
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=learn_rate)

    training_args = TrainingArguments(
        output_dir=save_path, overwrite_output_dir=True,
        num_train_epochs=num_train_epochs, learning_rate=learn_rate,
        per_device_train_batch_size=pretrain_batch_size,
        fp16=True, save_total_limit=2)

    # training bu Trainer
    trainer = Trainer(
        model=model, args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        optimizers=(optimizer, None))

    # start training
    print(method.time_beijing())
    trainer.train()
    # trainer.save_model(save_path)


def dp_pretrain(learning_rate, epochs, batch_size, model, file_path, seed=24):
    """
    do MLM task for Bert us torch.nn.DataParallel
    :param learning_rate: 
    :param epochs: 
    :param batch_size: batchsize
    :param model: 
    :param file_path: The file address of the pre-training
    :param seed: seed 
    :return: The trained model is returned
    """
    device_list = [0, 1, 3, 2]
    method.setup_seed(seed)
    print("Seed: "+str(seed))

    # use multi cuda
    bert_model = torch.nn.DataParallel(model, device_ids=device_list)
    bert_model.to(device_list[0])

    model_path = './model/bert-base-uncased/'  # BERT model, including the address of the thesaurus, etc
    tokenizer = BertTokenizerFast.from_pretrained(model_path, do_lower_case=True)
    print("loading data for pretraining-------")
    datasets = torch.load(file_path)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)

    dataloader = DataLoader(datasets, batch_size=batch_size, collate_fn=data_collator)

    # freeze higher layers
    # bert_model = method.freeze_higher_layers(bert_model, bert_config)
    param_optimizer = list(bert_model.named_parameters())

    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [  # remove requires_grad == False parameters
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
            loss = result[0].mean()  # Solving dp return when multiple scalar patchwork cannot be derived

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        scheduler.step()

    return bert_model


def main():
    file_path = './data/datasets/Biology_128.pt'  # corpus address
    model_path = './model/bert-base-uncased/'  # BERT model, including the address of the thesaurus, etc
    model_save = "./outputs/test/"
    learn_rate = 5e-5

    modelConfig = BertConfig.from_pretrained(model_path)
    model = BertForMaskedLM.from_pretrained(model_path, config=modelConfig)
    # for name, param in model.named_parameters():
    #     print(param.data)
    #     break
    #
    # print("loading data------")
    # train_dataset = torch.load(file_path)
    # huggingface_pretrain(model, model_save, train_dataset, learn_rate)
    #
    # for name, param in model.named_parameters():
    #     print(param.data)
    #     break
    dp_pretrain(learn_rate, 5, 256, model, file_path)


if __name__ == '__main__':
    main()

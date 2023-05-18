#!/usr/bin/env python
# -*- coding:utf-8 -*-
# author:HanZhou
# datetime:2022/5/6 14:53
# software: PyCharm
import json
import os
import random
import gc
import matplotlib.pyplot as plt
# import pandas as pd
import numpy as np
import torch
from keras_preprocessing.sequence import pad_sequences

from torch.utils.data import TensorDataset, RandomSampler, DataLoader, SequentialSampler
from tqdm import trange, tqdm
from torch.optim import AdamW
from transformers import BertForMaskedLM, BertConfig, BertForSequenceClassification  # , AdamW
from transformers import BertTokenizer, BertModel, TrainingArguments
from transformers import logging
import tokenizers
# from pytorch_pretrained_bert import BertAdam

import method

logging.set_verbosity_error()

data_path = './data/text_classification/chemprot/'  # path of datasets
# bert_model = './Bert/bert-base-uncased/'  # model that pretrained


label_num = method.label2num(data_path)  # Gets the total number of categories for sentences in the file
num_token = len(label_num)  # The number of categorical samples


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False


def data_read(bert_model, data_type):
    """
    Preprocess the data in the text classification task
    :param data_type: Type of text: train，test，dev, to build the address
    :return: input_ids, labels, attention_masks
    """
    sentencses = []
    labels = []
    path = data_path + data_type + '.txt'
    print("%s data loading------" % data_type)
    with open(path, 'r', encoding='utf-8') as file:
        for line in tqdm(file.readlines()):
            dic = json.loads(line)
            sent = '[CLS] ' + dic['text'] + ' [SEP]'  # get sentence
            label_token = dic['label']  # get label
            label = int(label_num[label_token])  # Convert labels to integers according to the dictionary

            sentencses.append(sent)
            labels.append(label)

    tokenizer = BertTokenizer.from_pretrained(bert_model, do_lower_case=True)
    tokenized_sents = [tokenizer.tokenize(sent) for sent in sentencses]

    MAX_LEN = 128

    # word-->idx
    input_ids = [tokenizer.convert_tokens_to_ids(sent) for sent in tokenized_sents]

    # To do PADDING, here use keras' package to make pads GREATER THAN 128 FOR TRUNCATION, LESS THAN 128 FOR PADDING
    input_ids = pad_sequences(input_ids, maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")

    # do mask
    attention_masks = []
    for seq in input_ids:
        seq_mask = [float(i > 0) for i in seq]
        attention_masks.append(seq_mask)

    return input_ids, labels, attention_masks


def data_load(bert_model, file_type, device, batch_size):
    """
    Convert the data initially processed by Bert into data recognized by Pytorch
    :param file_type: 
    :param device: 
    :param batch_size: 
    :return: DataLoader
    """
    inputs, labels, masks = data_read(bert_model, file_type)  # 获取数据

    # datasets to tensor
    inputs = torch.tensor(inputs).to(device)
    labels = torch.tensor(labels).to(device)
    masks = torch.tensor(masks).to(device)

    # create dataloader
    data = TensorDataset(inputs, masks, labels)
    sampler = RandomSampler(data)
    dataloader = DataLoader(data, sampler=sampler, batch_size=batch_size, shuffle=False)

    return dataloader


def re_param(model, param_path):
    params_list = torch.load(param_path)  # load parameters of client

    for name, params in model.named_parameters():  # update resultes
        if name in params_list.keys():
            print("re_param!")
            # print(name)
            params.data = params_list[name].data

    return model


def share_param(model, param_path):
    params_dict = torch.load(param_path)  # load parameters of client
    params_list = []
    for value in params_dict.values():
        params_list.append(value)

    for layer in model.bert.encoder.layer:
        for i, p in enumerate(layer.parameters()):
            p.data = params_list[i]

    return model


def model_test(model, test_dataloader, device, model_type):
    """
    test 
    :param model: 
    :param test_dataloader: 
    :param device: 
    :param model_type: 
    """
    model.eval()
    n_true = 0
    n_total = 0
    for batch in test_dataloader:
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, b_labels = batch
        with torch.no_grad():
            logits = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)[0]
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()

        pred_flat = np.argmax(logits, axis=1).flatten()
        labels_flat = label_ids.flatten()
        n_true += np.sum(pred_flat == labels_flat)
        n_total += len(labels_flat)

    accuracy = n_true / n_total
    print(model_type + "Accuracy: {}".format(accuracy))

    return accuracy


def train_classier(learning_rate, bert_model, epochs, device, batch_size, change_param=None, share=None):
    """
    train
    :param learning_rate: 
    :param bert_model:
    :param epochs: 
    :param device: 
    :param batch_size: 
    :param change_param: Whether to refactor some parameters
    :param share: Whether to modify the remaining layer parameters based on trained parameter sharing
    """

    # learning_rate = 2e-5  # bigger
    setup_seed(24)

    modelConfig = BertConfig.from_pretrained(bert_model)
    modelConfig.num_labels = num_token  # Set the number of outputs of the classification model

    model = BertForSequenceClassification.from_pretrained(bert_model, config=modelConfig)

    if change_param:
        # Reconstruct some parameters of the model from the saved file
        model = re_param(model, change_param)

    if share:
        # The parameters for building the rest of the layers are shared based on the parameters of one layer that have been saved
        model = share_param(model, share)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
    )
    # print(optimizer.param_groups)

    train_dataloader = data_load(bert_model, 'train', device, batch_size)
    validation_dataloader = data_load(bert_model, 'dev', device, batch_size)
    model.to(device)

    val_acc = [] 
    test_acc = []
    test_dataloader = data_load(bert_model, 'test', device, batch_size)

    for i in range(epochs):
        print("Epochs:%d/%d" % ((i + 1), epochs))

        model.train()
        tr_loss = 0
        nb_tr_steps = 0
        for batch in tqdm(train_dataloader):
            batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask, labels = batch
            optimizer.zero_grad()
            # get first position, first position of BertForSequenceClassification is Loss，second position is [CLS]'logits
            loss = model(input_ids, token_type_ids=None, attention_mask=input_mask, labels=labels)[0]

            loss.backward()
            optimizer.step()

            tr_loss += loss.item()
            nb_tr_steps += 1

        print("Train loss: {}".format(tr_loss / nb_tr_steps))
        v_acc = model_test(model, validation_dataloader, device, 'Val')
        val_acc.append(v_acc)
        t_acc = model_test(model, test_dataloader, device, 'Test')
        test_acc.append(t_acc)

    return val_acc, test_acc


def main():
    epochs = 15
    batch_size = 128  # 32
    learn_rate = 5e-5
    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

    print("Bert------")
    bert_model0 = './model/bert-base-uncased/'  
    val_acc0, test_acc0 = train_classier(learn_rate, bert_model0, epochs, device, batch_size)

    print("SciBert------")
    bert_model1 = './model/SciBert/'  
    val_acc1, test_acc1 = train_classier(learn_rate, bert_model1, epochs, device, batch_size)

    print("FedAvg------")
    bert_model2 = './outputs/fed_avg/'  # 
    val_acc2, test_acc2 = train_classier(learn_rate, bert_model2, epochs, device, batch_size)

    legend = ['Bert', 'SciBert', 'FedAvg']
    Val_Acc = [val_acc0, val_acc1, val_acc2]
    Test_Acc = [test_acc0, test_acc1, test_acc2]

    method.plot_res('Val Acc', legend, Val_Acc)
    method.plot_res('Test Acc', legend, Test_Acc)
    method.get_test_acc(legend, Val_Acc, Test_Acc)


if __name__ == "__main__":
    main()
    # test_bug()
    # find_lr()
    # find_pretrain_downstream_learn_date()

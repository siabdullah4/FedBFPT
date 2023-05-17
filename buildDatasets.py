#!/usr/bin/env python
# -*- coding:utf-8 -*-
# author:HanZhou
# datetime:2023/4/17 10:12
# software: PyCharm
import os
import time

import natsort

import method
import torch
from transformers import BertTokenizerFast, LineByLineTextDataset


def clean_data(in_path, out_path):
    # json corpus to txt
    file_name = natsort.natsorted(os.listdir(in_path), alg=natsort.ns.PATH)
    for file in file_name:
        file_path = in_path + file
        method.create_text(file_path, out_path)


def save_dataset(bert_model, file_name):
    file_path = './data/sentence/'
    datasets_path = './data/datasets/'
    tokenizer = BertTokenizerFast.from_pretrained(bert_model, do_lower_case=True)
    print("loading data------")
    T1 = time.time()
    # loading data and set sentence length to 128,
    train_dataset = LineByLineTextDataset(tokenizer=tokenizer, file_path=file_path + file_name + '.txt', block_size=128)
    T2 = time.time()
    print('Loading data cost: %s second' % (T2 - T1))

    print("Saving data------")
    T3 = time.time()
    torch.save(train_dataset, datasets_path + file_name + '_ori_128.pt')
    T4 = time.time()
    print('Saving data cost: %s second' % (T4 - T3))


def main():
    domain = 'Economics'
    print("Cleaning corpus------")
    in_path = './data/corpus/'+domain+'/'
    out_path = './data/sentence/'+domain+'.txt'
    clean_data(in_path, out_path)

    print("Saving dataset------")
    bert_model = './model/bert-base-uncased/'

    save_dataset(bert_model, domain)


if __name__ == "__main__":
    main()

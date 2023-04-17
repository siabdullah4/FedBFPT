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
    # 语料json生成txt
    file_name = natsort.natsorted(os.listdir(in_path), alg=natsort.ns.PATH)
    for file in file_name:
        file_path = in_path + file
        method.create_text(file_path, out_path)


def save_dataset(bert_model, file_name):
    file_path = './data/sentence/'
    datasets_path = './data/datasets/'
    tokenizer = BertTokenizerFast.from_pretrained(bert_model, do_lower_case=True)
    print("加载数据中------")
    T1 = time.time()
    # 通过LineByLineTextDataset接口 加载数据 #长度设置为128,
    train_dataset = LineByLineTextDataset(tokenizer=tokenizer, file_path=file_path + file_name + '.txt', block_size=128)
    T2 = time.time()
    print('加载数据运行时间:%s 秒' % (T2 - T1))

    print("保存数据中------")
    T3 = time.time()
    torch.save(train_dataset, datasets_path + file_name + '_ori_128.pt')
    T4 = time.time()
    print('保存数据时间:%s 秒' % (T4 - T3))


def main():
    print("处理原始语料中------")
    in_path = './data/S2ORC/Medicine/'
    out_path = './data/sentence/Medicine.txt'
    clean_data(in_path, out_path)
    #

    print("保存数据dataset中------")
    bert_model = './model/bert-base-uncased/'
    file_path = 'Medicine'

    save_dataset(bert_model, file_path)


if __name__ == "__main__":
    main()

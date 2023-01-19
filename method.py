#!/usr/bin/env python
# -*- coding:utf-8 -*-
# author:HanZhou
# datetime:2022/5/10 9:29
# software: PyCharm
import json
import os
import random
import re

import datetime
import time

import natsort
import pytz as pytz
import numpy as np
import pdfplumber
import spacy
import tokenizers
from matplotlib import pyplot as plt
from tokenizers.implementations import ByteLevelBPETokenizer
from keras_preprocessing.sequence import pad_sequences
from tqdm import tqdm
from transformers import BertTokenizer, BertForMaskedLM, BertTokenizerFast, LineByLineTextDataset
from transformers import WEIGHTS_NAME, CONFIG_NAME
import torch
from torch.utils.data import TensorDataset, RandomSampler, DataLoader


def clean_pdf():
    sentences = []
    with pdfplumber.open("papers/test01.pdf") as pdf:
        for page in pdf.pages:  # 按页读取pdf文件
            text = page.extract_text()
            text = re.sub(r'[0-9]+', '', text)  # 去除数字
            text = text.strip('\n')  # 去掉换行符
            text = text.split('.')  # 按句子划分
            sentences.extend(text)

    for sentence in sentences:
        if len(sentence) < 20:
            sentences.pop(sentences.index(sentence))
        else:
            sentence = sentence.replace('\n', '')  # 去掉换行符
            print(sentence)
            print("----------------")


def read_tsv(path):
    word_list = []
    line_num = 0
    with open(path, 'r') as f:
        for line in f:
            line_num += 1
            text = line.split()
            if int(text[2]) < 300:  # 如果这个词频率过低
                continue
            else:
                word = text[0]
                if word.isalpha():
                    word_list.append(word)
                else:
                    continue

    for word in word_list:
        print(word)
    print(len(word_list))
    print(line_num)
    return word_list


def read_vocab(path):
    """
    读取文件词表
    :param path: 词表地址，返回词表列表
    :return:
    """
    vocab_list = []
    with open(path, 'r') as f:
        print("reading vocab......")
        for vocab in tqdm(f):
            vocab_list.append(vocab)

    return vocab_list


def check_token(ori_vocab, new_vocab):
    """
    将新老词表进行比较，将老词表中不存在的词追加在老词表末尾
    模型在加载新词表时，由于长度与原词表不一致，使用
    model.resize_token_embeddings(len(tokenizer))
    对模型Resize，使用原来的词语embeddings
    :param ori_vocab: 原词表的地址
    :param new_vocab: 新词表的地址
    """
    ori_token = read_vocab(ori_vocab)  # 读取词表
    new_token = read_vocab(new_vocab)
    res = []
    print("comparing token......")
    with open(ori_vocab, 'a') as file:
        for vocab in tqdm(new_token):
            if vocab not in ori_token:
                res.append(vocab)
                file.write(vocab)

    print('新增token数目：%d' % len(res))


def read_json(path):
    datas = []
    print("处理json数据中------")
    with open(path, 'r', encoding='utf-8') as file:
        for line in tqdm(file.readlines()):
            dic = json.loads(line)
            # print(dic.keys())
            datas.append(dic)
    return datas


def label2num(path):
    """
    提取文本分类任务训练中的所有标签类别
    :param path: 文本分类任务数据所在的文件夹的地址
    :return:句子所有类别的字典，可以获知句子的总体种类数，并根据这个得到标签的对应数字标签
    """
    label_num = {}  # 标签字典
    token_num = 0  # 标签编号
    files = os.listdir(path)  # 得到文件夹下的所有文件名称
    for file in files:  # 遍历文件夹
        position = path + file  # 构造绝对路径
        with open(position, 'r', encoding='utf-8') as file:
            for line in tqdm(file.readlines()):
                dic = json.loads(line)
                lab = dic['label']
                if lab in label_num.keys():
                    continue
                else:
                    label_num.update({lab: token_num})
                    token_num += 1
    return label_num


def create_text(in_path, out_path, start=0, end=-1):
    """
    根据语料jsonl文件生成对应的处理好的训练数据
    :param in_path: 语料文件的地址，语料文件为jsonl格式
    :param out_path: 输出txt文件的地址，输出的txt文件中每一行是一句语料文件中的正文内容
    :param start:文档起始位置
    :param end:需要构建的txt文档长度
    """
    with open(out_path, "w") as f:
        nlp = spacy.load("en_core_sci_sm")
        print("处理文章中......")
        with open(in_path, 'r', encoding='utf-8') as papers:
            for line in tqdm(papers.readlines()[start:end]):
                paper = json.loads(line)
                abstract_text = paper['abstract']  # 摘要列表
                body_text = paper['body_text']  # 主体列表
                for abstract in abstract_text:  # 所有的摘要内容
                    text0 = abstract['text']
                    doc = nlp(text0)
                    list_text = list(doc.sents)  # 进行划分句子
                    for sentence in list_text:  # 写入文件
                        f.write(str(sentence) + '\n')
                for body in body_text:
                    text1 = body['text']
                    doc = nlp(text1)
                    list_text = list(doc.sents)
                    for sentence in list_text:
                        f.write(str(sentence) + '\n')


def ner_label(path):
    tag_list = []
    files = natsort.natsorted(os.listdir(path), alg=natsort.ns.PATH)  # 文件夹下的文件名称  # 得到文件夹下的所有文件名称
    for file in files:  # 遍历文件夹
        position = path + file  # 构造绝对路径
        with open(position, 'r', encoding='utf-8') as doc:
            for line in tqdm(doc.readlines()):
                if 'DOCSTART' in line:
                    continue
                else:
                    if len(line) == 1:
                        continue
                    else:
                        tmp = line.split()
                        tag_list.append(tmp[-1])

    # unique_tags 代表有多少种标签，tag2id表示每种标签对应的id，id2tag表示每种id对应的标签。
    unique_tags = list(set(tag_list))
    unique_tags = sorted(unique_tags)  # 转列表并排序，保证每次的tag id对应关系一致
    tag2id = {tag: tag_id for tag_id, tag in enumerate(unique_tags)}
    id2tag = {tag_id: tag for tag, tag_id in tag2id.items()}

    return unique_tags, tag2id, id2tag


def train_token(filepath, save_path):
    """
    训练新的词表
    :param filepath: 语料地址，txt格式
    :param save_path: 保存训练好的vocab.txt格式的地址
    """
    # 创建分词器
    bwpt = tokenizers.BertWordPieceTokenizer()

    # 训练分词器
    bwpt.train(
        files=filepath,
        vocab_size=30000,  # 这里预设定的词语大小不是很重要
        min_frequency=10,
        limit_alphabet=1000
    )
    # 保存训练后的模型词表
    bwpt.save_model(save_path)

    # 加载刚刚训练的tokenizer
    tokenizer = BertTokenizer(vocab_file=save_path + 'vocab.txt')

    sequence0 = "Setrotech is a part of brain"
    tokens0 = tokenizer.tokenize(sequence0)
    print(tokens0)

    # v_size = len(tokenizer.vocab)  # 自己设置词汇大小
    # print(v_size)
    # model = BertForMaskedLM.from_pretrained("./Bert/bert-base-uncased")
    # model.resize_token_embeddings(len(tokenizer))


def add_token(path):
    model = "bert-base-uncased"
    tokenizer = BertTokenizer.from_pretrained(model, use_fast=True)
    model = BertForMaskedLM.from_pretrained(model)

    sequence0 = "Setrotech is a part of brain"
    tokens0 = tokenizer.tokenize(sequence0)
    print(tokens0)

    word_list = read_tsv(path)
    for word in tqdm(word_list):
        tokenizer.add_tokens(word)

    # 关键步骤，resize_token_embeddings输入的参数是tokenizer的新长度
    model.resize_token_embeddings(len(tokenizer))

    tokens1 = tokenizer.tokenize(sequence0)
    print(tokens1)

    tokenizer.save_pretrained("Pretrained_LMs/bert-base-cased")  # 还是保存到原来的bert文件夹下，这时候文件夹下多了三个文件


def plot_res(title, legend, datas):
    # 全部的线段风格
    styles = ['c:s', 'y:8', 'b:p', 'r:^', 'g:D', 'm:X', 'r:v', ':>']  # 其他可用风格 ':<',':H','k:o','k:*','k:*','k:*'

    plt.figure(figsize=(10, 7))
    # 设置字体
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams.update({'font.size': 22})
    plt.rc('legend', fontsize=15)

    # 正式的进行画图
    for i in range(len(datas)):
        x = np.arange(1, len(datas[i]) + 1)
        y = datas[i]
        plt.plot(x, y, styles[i], markersize=8, label=legend[i])

    # 设置图片的x,y轴的限制，和对应的标签
    # plt.xlim([0, 300])
    # plt.ylim([60, 78])
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title(title)

    # 设置图片的方格线和图例
    plt.grid()
    plt.legend(loc='lower right', framealpha=0.7)
    plt.tight_layout()
    plt.show()

    # 如果想保存图片，请把plt.show注释掉，然后把下面这行代码打开注释
    # plt.savefig("img.png", dpi=800)


def time_beijing():
    """
    获取当前北京时间
    :return: 返回当前北京时间
    """
    tz = pytz.timezone('Asia/Shanghai')  # 东八区
    t = datetime.datetime.fromtimestamp(int(time.time()), tz).strftime('%Y-%m-%d %H:%M:%S')

    return t


def create_dataloader(bert_model, file_path, device, batch_size):
    sentencses = []

    with open(file_path, 'r', encoding='utf-8') as file:
        print("读取数据------")
        for line in tqdm(file.readlines()):
            sent = '[CLS] ' + line + ' [SEP]'  # 获取句子
            sentencses.append(sent)

    tokenizer = BertTokenizerFast.from_pretrained(bert_model, do_lower_case=True)
    tokenized_sents = [tokenizer.tokenize(sent) for sent in sentencses]

    # 定义句子最大长度（512）
    MAX_LEN = 128

    # 将分割后的句子转化成数字  word-->idx
    input_ids = [tokenizer.convert_tokens_to_ids(sent) for sent in tokenized_sents]

    # 做PADDING,这里使用keras的包做pad
    # 大于128做截断，小于128做PADDING
    input_ids = pad_sequences(input_ids, maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")

    # 建立mask
    attention_masks = []
    for seq in input_ids:
        seq_mask = [float(i > 0) for i in seq]
        attention_masks.append(seq_mask)

    # 将数据集转化成tensor
    inputs = torch.tensor(input_ids).to(device)
    masks = torch.tensor(attention_masks).to(device)

    # 生成dataloader
    data = TensorDataset(inputs, masks)
    sampler = RandomSampler(data)
    dataloader = DataLoader(data, sampler=sampler, batch_size=batch_size)

    return dataloader


def get_test_acc(legend, val_acc, test_acc):
    for i, model_type in enumerate(legend):
        print(model_type, end='\t')
        index = val_acc[i].index(max(val_acc[i]))
        print('Highest Val Acc:%f,Epoch:%d' % (val_acc[i][index], index), end=', \t')
        print('Corresponding Test Acc:%f' % test_acc[i][index])


def freeze_lower_layers(model, config):
    for p in model.bert.embeddings.parameters():
        p.requires_grad = False
    for layer in model.bert.encoder.layer[
                 :config.num_hidden_layers - config.num_full_hidden_layers]:
        for p in layer.parameters():
            p.requires_grad = False
    try:
        for p in model.bert.shallow_skipping.linear.parameters():
            p.requires_grad = False
    except Exception as e:
        pass
    try:
        for p in model.bert.attn.parameters():
            p.requires_grad = False
    except Exception as e:
        pass

    model.bert.embeddings.dropout.p = 0.
    for layer in model.bert.encoder.layer[
                 :config.num_hidden_layers - config.num_full_hidden_layers]:
        for m in layer.modules():
            if isinstance(m, torch.nn.Dropout):
                m.p = 0.

    return model


def freeze_higher_layers(model, config):
    for layer in model.bert.encoder.layer[-config.num_full_hidden_layers:]:
        for p in layer.parameters():
            p.requires_grad = False
        for m in layer.modules():
            if isinstance(m, torch.nn.Dropout):
                m.p = 0.

    return model


def pre_data(tokenizer, filepath):
    print("加载预训练数据中------")
    T1 = time.time()
    # 通过LineByLineTextDataset接口 加载数据 #长度设置为128,
    train_dataset = LineByLineTextDataset(tokenizer=tokenizer, file_path=filepath, block_size=128)
    T2 = time.time()
    print('加载预训练数据运行时间:%.2f 秒' % (T2 - T1))
    print(time_beijing())
    print("---------------------------------------")

    return train_dataset


def model_save(model, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    model_to_save = model.module if hasattr(model, 'module') else model
    output_model_file = os.path.join(output_dir, WEIGHTS_NAME)
    output_config_file = os.path.join(output_dir, CONFIG_NAME)

    torch.save(model_to_save.state_dict(), output_model_file)
    model_to_save.config.to_json_file(output_config_file)


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False


def save_checkpoint(epochs, model, optimizer, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    model_to_save = model.module if hasattr(model, 'module') else model
    output_model_file = os.path.join(output_dir, WEIGHTS_NAME)
    output_config_file = os.path.join(output_dir, CONFIG_NAME)

    torch.save(model_to_save.state_dict(), output_model_file)
    model_to_save.config.to_json_file(output_config_file)

    torch.save(epochs, output_dir + "epochs.pth")
    torch.save(optimizer.state_dict(), output_dir + "optimizer.pth")


def get_subword_id(vocab_path):
    word_id = 0
    word_list = []

    with open(vocab_path, 'r', encoding='utf-8') as file:
        print("获取分词词表------")
        for line in tqdm(file.readlines()):
            if "#" in line:
                word_list.append(word_id)
            word_id += 1

    return word_list

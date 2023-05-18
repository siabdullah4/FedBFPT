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
        for page in pdf.pages:  # read pdf in pages
            text = page.extract_text()
            text = re.sub(r'[0-9]+', '', text)  # Remove the numbers
            text = text.strip('\n')  # Remove the '\n'
            text = text.split('.')  # split sentence
            sentences.extend(text)

    for sentence in sentences:
        if len(sentence) < 20:
            sentences.pop(sentences.index(sentence))
        else:
            sentence = sentence.replace('\n', '')  # Remove the '\n'
            print(sentence)
            print("----------------")


def read_tsv(path):
    word_list = []
    line_num = 0
    with open(path, 'r') as f:
        for line in f:
            line_num += 1
            text = line.split()
            if int(text[2]) < 300:  # If the word is too infrequently
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
    :param path: Thesaurus address, which returns a list of thesauruses
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
    Compare the old and new vocabulary and append words that do not exist in the old vocabulary to the end of the old vocabulary
    When the model loads a new vocabulary, it is used because the length is inconsistent with the original vocabulary
    model.resize_token_embeddings(len(tokenizer)) for model Resize, use the original word embeddings
    :param ori_vocab: The address of the original vocabulary
    :param new_vocab: The address of the new vocabulary
    """
    ori_token = read_vocab(ori_vocab)  
    new_token = read_vocab(new_vocab)
    res = []
    print("comparing token......")
    with open(ori_vocab, 'a') as file:
        for vocab in tqdm(new_token):
            if vocab not in ori_token:
                res.append(vocab)
                file.write(vocab)

    print('the number of new token: %d' % len(res))


def read_json(path):
    datas = []
    print("Process JSON data------")
    with open(path, 'r', encoding='utf-8') as file:
        for line in tqdm(file.readlines()):
            dic = json.loads(line)
            # print(dic.keys())
            datas.append(dic)
    return datas


def label2num(path):
    """
    Extract all label categories in the text classification task training
    :param path: The address of the folder where the text classification task data is located
    :return:The dictionary of all categories of sentences can get the overall number of types of sentences, and get the corresponding number labels of labels according to this
    """
    label_num = {}  # label dictionary
    token_num = 0  # label index
    files = os.listdir(path)  # Get the names of all files under the folder
    for file in files:  # Traverse the folder
        position = path + file  # Construct an absolute path
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
    Generate the corresponding processed training data according to the corpus JSONL file
    :param in_path: The address of the corpus file, which is in JSON format
    :param out_path: The address of the output txt file, each line in the output txt file is the body content of a sentence corpus file
    :param start:The starting position of the document
    :param end:The length of the txt document that needs to be built
    """
    with open(out_path, "a") as f: # append to the sentence, you can change to 'w'
        nlp = spacy.load("en_core_sci_sm")
        print("progress the article......")
        with open(in_path, 'r', encoding='utf-8') as papers:
            for line in tqdm(papers.readlines()[start:end]):
                paper = json.loads(line)
                abstract_text = paper['abstract']  # the list of abstract
                body_text = paper['body_text']  # the list of body
                for abstract in abstract_text:  # all abstract
                    text0 = abstract['text']
                    doc = nlp(text0)
                    list_text = list(doc.sents)  # split sentence
                    for sentence in list_text:  # write
                        f.write(str(sentence) + '\n')
                for body in body_text:
                    text1 = body['text']
                    doc = nlp(text1)
                    list_text = list(doc.sents)
                    for sentence in list_text:
                        f.write(str(sentence) + '\n')


def ner_label(path):
    tag_list = []
    files = natsort.natsorted(os.listdir(path), alg=natsort.ns.PATH)  # File names under folders # Get all file names under folders
    for file in files:  # Traverse the folder
        position = path + file  # Construct an absolute path
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

    # unique_tags represents how many kinds of tags there are, tag2id represents the ID corresponding to each tag, and id2tag represents the label corresponding to each ID
    unique_tags = list(set(tag_list))
    unique_tags = sorted(unique_tags)  # Go to the table and sort to ensure that the tag ID correspondence is consistent each time
    tag2id = {tag: tag_id for tag_id, tag in enumerate(unique_tags)}
    id2tag = {tag_id: tag for tag, tag_id in tag2id.items()}

    return unique_tags, tag2id, id2tag


def train_token(filepath, save_path):
    """
    train new vocab token
    :param filepath: Corpus address, in txt format
    :param save_path: Save the address of the trained vocab .txt format
    """
    # Create a tokenizer
    bwpt = tokenizers.BertWordPieceTokenizer()

    # Train the tokenizer
    bwpt.train(
        files=filepath,
        vocab_size=30000,  # The preset word size here is not very important
        min_frequency=10,
        limit_alphabet=1000
    )
    # Save the trained model vocabulary
    bwpt.save_model(save_path)

    # Load the tokenizer you just trained
    tokenizer = BertTokenizer(vocab_file=save_path + 'vocab.txt')

    sequence0 = "Setrotech is a part of brain"
    tokens0 = tokenizer.tokenize(sequence0)
    print(tokens0)

    # v_size = len(tokenizer.vocab)  # Set the vocabulary size yourself
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

    # The key step, resize_token_embeddings input parameter is the new length of tokenizer
    model.resize_token_embeddings(len(tokenizer))

    tokens1 = tokenizer.tokenize(sequence0)
    print(tokens1)

    tokenizer.save_pretrained("Pretrained_LMs/bert-base-cased")  # It is still saved to the original BERT folder, and there are three more files under the folder


def plot_res(title, legend, datas):
    # All line segment styles
    styles = ['c:s', 'y:8', 'b:p', 'r:^', 'g:D', 'm:X', 'r:v', ':>']  # Other styles available ':<',':H','k:o','k:*','k:*','k:*'

    plt.figure(figsize=(10, 7))
    # set font
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams.update({'font.size': 22})
    plt.rc('legend', fontsize=15)

    # plt 
    for i in range(len(datas)):
        x = np.arange(1, len(datas[i]) + 1)
        y = datas[i]
        plt.plot(x, y, styles[i], markersize=8, label=legend[i])

    # Set the x, y-axis limits of the image, and the corresponding labels
    # plt.xlim([0, 300])
    # plt.ylim([60, 78])
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title(title)

    # Set the checkered lines and legend for the picture
    plt.grid()
    plt.legend(loc='lower right', framealpha=0.7)
    plt.tight_layout()
    plt.show()

    # If you want to save the image, comment out plt.show, and then open the comment on the following line of code
    # plt.savefig("img.png", dpi=800)


def time_beijing():
    """
    :return: Returns the current Beijing time
    """
    tz = pytz.timezone('Asia/Shanghai')  # East 8th District
    t = datetime.datetime.fromtimestamp(int(time.time()), tz).strftime('%Y-%m-%d %H:%M:%S')

    return t


def create_dataloader(bert_model, file_path, device, batch_size):
    sentencses = []

    with open(file_path, 'r', encoding='utf-8') as file:
        print("loading data------")
        for line in tqdm(file.readlines()):
            sent = '[CLS] ' + line + ' [SEP]'  # get sentence
            sentencses.append(sent)

    tokenizer = BertTokenizerFast.from_pretrained(bert_model, do_lower_case=True)
    tokenized_sents = [tokenizer.tokenize(sent) for sent in sentencses]

    # max sentence length
    MAX_LEN = 128

    #  word-->idx
    input_ids = [tokenizer.convert_tokens_to_ids(sent) for sent in tokenized_sents]

    # To do PADDING, here use keras' package to make pads
    # GREATER THAN 128 FOR TRUNCATION, LESS THAN 128 FOR PADDING
    input_ids = pad_sequences(input_ids, maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")

    # do mask
    attention_masks = []
    for seq in input_ids:
        seq_mask = [float(i > 0) for i in seq]
        attention_masks.append(seq_mask)

    # dataset to tensor
    inputs = torch.tensor(input_ids).to(device)
    masks = torch.tensor(attention_masks).to(device)

    # generate dataloader
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
    print("loading data for pretraining------")
    T1 = time.time()
    train_dataset = LineByLineTextDataset(tokenizer=tokenizer, file_path=filepath, block_size=128)
    T2 = time.time()
    print('loading data for pretraining cost:%.2f second' % (T2 - T1))
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
        print("get token vocab------")
        for line in tqdm(file.readlines()):
            if "#" in line:
                word_list.append(word_id)
            word_id += 1

    return word_list

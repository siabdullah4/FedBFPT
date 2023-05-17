#!/usr/bin/env python
# -*- coding:utf-8 -*-
# author:HanZhou
# datetime:2022/10/13 14:50
# software: PyCharm
import torch
from tqdm import tqdm
from transformers import BertConfig, BertForSequenceClassification

import method
from downstream_train import data_load, model_test, num_token


def concate_layer(model):
    layer_path = './outputs/params/Biology/LayerBert_01/client_0.pt'

    layer_params = torch.load(layer_path)  # loading weights of model on clients
    params_list = []  # transformer parameters list

    for name in layer_params:
        params_list.append(layer_params[name].data)

    for layer in model.bert.encoder.layer[:6]:  # change 0-5th layers
        for i, p in enumerate(layer.parameters()):
            p.data = params_list[i]  # replace transformer

    return model


def model_cls(model, learning_rate, epochs, device, batch_size):
    """
    do cls task by concate model
    :param model: model
    :param learning_rate: learning_rate
    :param device: device
    :param device: device
    :param batch_size: batch_size
    """
    method.setup_seed(40)

    token_path = './model/bert-base-uncased/'

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
    )
    # print(optimizer.param_groups)

    train_dataloader = data_load(token_path, 'train', device, batch_size)
    validation_dataloader = data_load(token_path, 'dev', device, batch_size)
    model.to(device)
    # start training

    val_acc = []  # the accuracy of valid
    test_acc = []  # the accuracy of test
    test_dataloader = data_load(token_path, 'test', device, batch_size)

    for i in range(epochs):
        print("Epochs:%d/%d" % ((i + 1), epochs))
        model.train()
        tr_loss = 0
        nb_tr_steps = 0
        for batch in tqdm(train_dataloader):
            batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask, labels = batch
            optimizer.zero_grad()
            # first position of BertForSequenceClassification is Loss, second position is logits of [CLS]
            loss = model(input_ids, token_type_ids=None, attention_mask=input_mask, labels=labels)[0]

            loss.backward()
            optimizer.step()

            tr_loss += loss.item()
            nb_tr_steps += 1

        print("Train loss: {}".format(tr_loss / nb_tr_steps))
        # validate 
        v_acc = model_test(model, validation_dataloader, device, 'Val')
        val_acc.append(v_acc)
        t_acc = model_test(model, test_dataloader, device, 'Test')
        test_acc.append(t_acc)

    return val_acc, test_acc


def main():
    epochs = 50
    batch_size = 32
    learning_rate = 5e-6
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_path = './model/bert-base-uncased/'

    modelConfig = BertConfig.from_pretrained(model_path)
    modelConfig.num_labels = num_token  # set classification numbers
    model = BertForSequenceClassification.from_pretrained(model_path, config=modelConfig)

    # concate model
    model = concate_layer(model)

    val_acc0, test_acc0 = model_cls(model, learning_rate, epochs, device, batch_size)

    Val_acc = [val_acc0]
    Test_acc = [test_acc0]

    legend = ['Concate Layer']
    method.plot_res('Val Acc', legend, Val_acc)
    method.plot_res('Test Acc', legend, Test_acc)

    method.get_test_acc(legend, Val_acc, Test_acc)
    print("--------------------")
    for i in range(len(legend)):
        print(legend[i], end=" ")
        print("Max Test Acc: ")
        print(max(Test_acc[i]))


if __name__ == "__main__":
    main()

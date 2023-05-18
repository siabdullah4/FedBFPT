#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2023/1/17 22:45
# @Author : HanZhou
import os

import natsort
import torch

import method
from downstream_ner import ner_train
from downstream_train import train_classier


def baseline_test():
    epoch = 15
    batch_size = 32
    learn_rate = 4e-5
    seed = 40
    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

    print(epoch)
    print(batch_size)
    print(learn_rate)
    print(seed)
    print(device)

    domain = 'Biology'

    Distill_Path = './outputs/Baseline/' + domain + '/DistillBert/epoch_5/fed_avg.pt'
    TinyBert_4_Path = './outputs/Baseline/' + domain + '/TinyBert_4/epoch_5/fed_avg.pt'
    TinyBert_6_Path = './outputs/Baseline/' + domain + '/TinyBert_6/epoch_5/fed_avg.pt'

    # Pro3_6_e5 = './outputs/LayerModel/' + domain + '/Try_Pro3_6_e5/layer0_2.pt'
    cls_Pro3_6_e5 = './outputs/LayerModel/' + domain + '/cls_Pro3_6_e5/layer0_2.pt'

    print("DistillBert----")
    model_path0 = './model/DistillBert/'
    val_acc0, test_acc0 = ner_train(learn_rate, model_path0, epoch, device, batch_size,
                                    change_param=Distill_Path, seed=seed)
    #
    print("TinyBert_4----")
    model_path1 = './model/TinyBert_4/'
    val_acc1, test_acc1 = ner_train(learn_rate, model_path1, epoch, device, batch_size,
                                    change_param=TinyBert_4_Path, seed=seed)

    print("TinyBert_6----")
    model_path2 = './model/TinyBert_6/'
    val_acc2, test_acc2 = ner_train(learn_rate, model_path2, epoch, device, batch_size,
                                    change_param=TinyBert_6_Path, seed=seed)

    print("cls_Pro3_6_e5----")
    model_path3 = './model/bert-base-uncased/'
    val_acc3, test_acc3 = ner_train(learn_rate, model_path3, epoch, device, batch_size,
                                    change_param=cls_Pro3_6_e5, seed=seed)

    Val_acc = [val_acc0, val_acc1, val_acc2, val_acc3]
    Test_acc = [test_acc0, test_acc1, test_acc2, test_acc3]

    legend = ['DistillBert', 'TinyBert_4', 'TinyBert_6', 'Ours']
    method.plot_res('JNLPBA Val', legend, Val_acc)
    method.plot_res('JNLPBA Test', legend, Test_acc)

    method.get_test_acc(legend, Val_acc, Test_acc)
    print("--------------------")
    for i in range(len(legend)):
        print(legend[i], end=" ")
        print("Max Test Acc: ")
        print(max(Test_acc[i]))
    print("**************************************************************")
    for i in range(len(legend)):
        print(legend[i])
        print("Val Acc:")
        print(Val_acc[i])
        print("Test Acc:")
        print(Test_acc[i])
        print("**************************************************************")


def method_test():
    epoch = 15
    batch_size = 16
    learn_rate = 8e-5
    seed = 40
    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
    model_path = './model/bert-base-uncased/'

    print(epoch)
    print(batch_size)
    print(learn_rate)
    print(seed)
    print(device)

    domain = 'Computer'

    Bert_Center = './outputs/LayerModel/' + domain + '/Bert_Center/epoch_5/layer.pt'
    # Bert_Center = './outputs/LayerModel/Center/Biology_full_e5.pt'
    Bert_FL = './outputs/LayerModel/' + domain + '/Bert_FL/epoch_5/fed_avg.pt'
    cls_Pro3_6_e5 = './outputs/LayerModel/' + domain + '/cls_Pro3_6_e5/layer0_2.pt'
    # Pro3_6_e5 = './outputs/LayerModel/' + domain + '/Try_Pro3_6_e5/layer0_2.pt'

    print("Bert----")
    val_acc0, test_acc0 = ner_train(learn_rate, model_path, epoch, device, batch_size,
                                    seed=seed)

    print("Bert_Center----")
    val_acc1, test_acc1 = ner_train(learn_rate, model_path, epoch, device, batch_size,
                                    change_param=Bert_Center, seed=seed)

    print("Bert_FL----")
    val_acc2, test_acc2 = ner_train(learn_rate, model_path, epoch, device, batch_size,
                                    change_param=Bert_FL, seed=seed)

    #
    print("cls_Pro3_6_e5----")
    val_acc3, test_acc3 = ner_train(learn_rate, model_path, epoch, device, batch_size,
                                    change_param=cls_Pro3_6_e5, seed=seed)

    Val_acc = [val_acc0, val_acc1, val_acc2, val_acc3]
    Test_acc = [test_acc0, test_acc1, test_acc2, test_acc3]

    legend = ['Bert', 'Bert_Center', 'Bert_FL', 'Ours']
    method.plot_res('SciERC Val', legend, Val_acc)
    method.plot_res('SciERC Test', legend, Test_acc)

    method.get_test_acc(legend, Val_acc, Test_acc)
    print("--------------------")
    for i in range(len(legend)):
        print(legend[i], end=" ")
        print("Max Test Acc: ")
        print(max(Test_acc[i]))
    print("--------------------")
    for i in range(len(legend)):
        print(legend[i])
        print("Val Acc:")
        print(Val_acc[i])
        print("Test Acc:")
        print(Test_acc[i])
        print("*******************************")


def main():
    # method_test()
    baseline_test()


if __name__ == "__main__":
    main()

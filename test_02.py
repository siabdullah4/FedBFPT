#!/usr/bin/env python
# -*- coding:utf-8 -*-
# author:HanZhou
# datetime:2022/9/24 21:13
# software: PyCharm
import torch

import method
from downstream_ner import ner_train
from downstream_train import train_classier


def baseline_test():
    epoch = 15
    batch_size = 2
    learn_rate = 4e-6
    seed = 40
    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

    print(epoch)
    print(batch_size)
    print(learn_rate)
    print(seed)
    print(device)

    domain = 'Medicine'

    Distill_Path = './outputs/Baseline/' + domain + '/DistillBert/epoch_5/fed_avg.pt'
    TinyBert_4_Path = './outputs/Baseline/' + domain + '/TinyBert_4/epoch_5/fed_avg.pt'
    TinyBert_6_Path = './outputs/Baseline/' + domain + '/TinyBert_6/epoch_5/fed_avg.pt'

    cls_Pro3_6_e5 = './outputs/LayerModel/' + domain + '/cls_Pro3_6_e5/layer0_2.pt'

    print("DistillBert训练----")
    model_path0 = './model/DistillBert/'
    val_acc0, test_acc0 = train_classier(learn_rate, model_path0, epoch, device, batch_size,
                                         change_param=Distill_Path, seed=seed)

    print("TinyBert_4训练----")
    model_path1 = './model/TinyBert_4/'
    val_acc1, test_acc1 = train_classier(learn_rate, model_path1, epoch, device, batch_size,
                                         change_param=TinyBert_4_Path, seed=seed)

    print("TinyBert_6训练----")
    model_path2 = './model/TinyBert_6/'
    val_acc2, test_acc2 = train_classier(learn_rate, model_path2, epoch, device, batch_size,
                                         change_param=TinyBert_6_Path, seed=seed)

    print("cls_Pro3_6_e5训练----")
    model_path3 = './model/bert-base-uncased/'
    val_acc3, test_acc3 = train_classier(learn_rate, model_path3, epoch, device, batch_size,
                                         change_param=cls_Pro3_6_e5, seed=seed)

    Val_acc = [val_acc0, val_acc1, val_acc2, val_acc3]
    Test_acc = [test_acc0, test_acc1, test_acc2, test_acc3]

    legend = ['DistillBert', 'TinyBert-4', 'TinyBert-6', 'Ours']
    method.plot_res('Tiny-Rct Val', legend, Val_acc)
    method.plot_res('Tiny-Rct Test', legend, Test_acc)

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
    batch_size = 2
    learn_rate = 4e-6
    seed = 40
    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

    print(epoch)
    print(batch_size)
    print(learn_rate)
    print(seed)
    print(device)

    domain = 'Medicine'

    # Bert_Center = './outputs/LayerModel/Center/Biology_full_e5.pt'
    Bert_Center = './outputs/LayerModel/' + domain + '/Bert_Center/epoch_5/layer.pt'
    Bert_FL = './outputs/LayerModel/' + domain + '/Bert_FL/epoch_5/fed_avg.pt'
    Pro3_6_e5 = './outputs/LayerModel/' + domain + '/cls_Pro3_6_e5/layer0_2.pt'

    model_path = './model/bert-base-uncased/'

    print("Bert训练中------")
    val_acc0, test_acc0 = train_classier(learn_rate, model_path, epoch, device, batch_size,
                                         seed=seed)

    print("Bert_Center训练中------")
    val_acc1, test_acc1 = train_classier(learn_rate, model_path, epoch, device, batch_size,
                                         change_param=Bert_Center, seed=seed)

    print("Bert_Fl训练中------")
    val_acc2, test_acc2 = train_classier(learn_rate, model_path, epoch, device, batch_size,
                                         change_param=Bert_FL, seed=seed)

    print("Pro3_6_e5训练中------")
    val_acc3, test_acc3 = train_classier(learn_rate, model_path, epoch, device, batch_size,
                                         change_param=Pro3_6_e5, seed=seed)

    Val_acc = [val_acc0, val_acc1, val_acc2, val_acc3]
    Test_acc = [test_acc0, test_acc1, test_acc2, test_acc3]

    legend = ['Bert', 'Bert_Center', 'Bert_FL', 'Ours']
    method.plot_res('Tiny-Rct Val', legend, Val_acc)
    method.plot_res('Tiny-Rct Test', legend, Test_acc)

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


def skewed_test():
    epoch = 15
    batch_size = 2
    learn_rate = 4e-6
    seed = 40
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    model_path = './model/bert-base-uncased/'

    print(epoch)
    print(batch_size)
    print(learn_rate)
    print(seed)
    print(device)

    domain = 'Medicine'
    Pro3_6_e5 = './outputs/LayerModel/' + domain + '/cls_Pro3_6_e5/layer0_2.pt'
    skewed_Pro3_6_e5 = './outputs/LayerModel/' + domain + '/cls_Pro3_6_e5_Skewed/layer0_2.pt'

    print("cls_Pro3_6_e5训练----")

    val_acc0, test_acc0 = train_classier(learn_rate, model_path, epoch, device, batch_size,
                                         change_param=Pro3_6_e5, seed=seed)

    val_acc1, test_acc1 = train_classier(learn_rate, model_path, epoch, device, batch_size,
                                         change_param=skewed_Pro3_6_e5, seed=seed)

    Val_acc = [val_acc0, val_acc1]
    Test_acc = [test_acc0, test_acc1]

    legend = ['Uniform', 'Skewed']
    method.plot_res('Chemprot Val', legend, Val_acc)
    method.plot_res('Chemprot Test', legend, Test_acc)

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

    # baseline_test()
    # method_test()
    skewed_test()


if __name__ == "__main__":
    main()

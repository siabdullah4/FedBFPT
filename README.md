# FedBFPT
In this study, we have built upon previous research to investigate the role of the shallow layer in federated learning (FL) based Bert further pre-training.
To combat the limited computation and communication resources on the client side in FL, we proposed a novel framework, referred to as FedBFPT, which allows for training a single transformer layer of a global Bert model on the client side. Moreover, we proposed the Progressive Learning with Sampled Deep Layers (PL-SDL) method as a means of effectively and efficiently training the local Bert model with a focus on the shallower layers. Through experiments on a variety of corpora across multiple domains, including biology, computer science, and medicine,  we have demonstrated that our proposed FedBFPT in combination with PL-SDL, is capable of achieving accuracy levels comparable to traditional FL methods while significantly reducing computational and communication costs. Details can be seen:

## 1. Environment
1. We suggest you create a Conda environment called "FedBFPT" with Python 3.9 (any version >= 3.6 should work):
```python
conda create -n FedBFPT python=3.9
```
then activate this environment by do:
```python
conda activate FedBFPT
```
2. The requriements are showed on "requirements.txt", you can download them by run:
```python
pip install requirements.txt
```
## 2.Data
1. Corpus
Our corpus are download from "https://github.com/allenai/s2orc", you can download the specialized domain corpus you are interested in, then put them in the folder "./data/corpus/", such as: 
```
"./data/corpus/Biology/pdf_parses_50.jsonl"
```
2. Datasets
You can then run the "xxx.py" to transform the corpus into a dataset that can be used for training, then stored them in folder "./data/datasets/", such as :
```
"./data/datasets/Biology/test_0_128.pt"
```
3. Model
All baseline model files are downloaded from huggingface, you can get the specific download link from the corresponding folder, such as:
```
"./model/bert-base-uncased/link.txt"
```
we can only download the "config.json", "pytorch_model.bin" and "vocab.txt". In all baseline model, you have to download model is __"bert-base-uncased"__.


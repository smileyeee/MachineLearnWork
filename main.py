#!/usr/bin/env python
# coding: utf-8

# **目前PaddleNLP的通用数据处理流程如下：**
# 
# 加载数据集（内置数据集或者自定义数据集，数据集返回 原始数据）。
# 
# 定义 trans_func() ，包括tokenize，token to id等操作，并传入数据集的 map() 方法，将原始数据转为 feature 。
# 
# 根据上一步数据处理的结果定义 batchify 方法和 BatchSampler 。
# 
# 定义 DataLoader ， 传入 BatchSampler 和 batchify_fn() 。

# **1 、数据加载**
# 
# 使用的是自定义数据集

# In[ ]:


# 正式开始实验之前首先通过如下命令安装最新版本的 paddlenlp
get_ipython().system('pip install --upgrade paddlenlp')
get_ipython().system('pip install --upgrade paddlepaddle')


# In[ ]:


# 检查数据集所在路径
get_ipython().system('tree -L 3 /home/aistudio/data')


# In[ ]:


# 解压数据集的压缩文件
get_ipython().system('unzip -o data/data104940/train.zip -d data')


# In[ ]:


# 因为数据集太大了，所以我个人在跑的时候只使用了LCQMC这一个
get_ipython().system('cat ./data/train/LCQMC/train  > train.txt')
get_ipython().system('cat ./data/train/LCQMC/dev  > dev.txt')
get_ipython().system('cat ./data/train/LCQMC/test  > test.txt')


# In[5]:


# 这是一些要使用到的库
from functools import partial
import argparse
import os
import random
import time

import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F

import paddlenlp as ppnlp
from paddlenlp.data import Stack, Tuple, Pad
from paddlenlp.datasets import load_dataset
from paddlenlp.transformers import LinearDecayWithWarmup
# 在data文件中实现了三个比较重要的函数
from work.data import create_dataloader, read_text_pair, convert_example


# In[3]:


# 导入训练集，测试集
train_ds = load_dataset(read_text_pair, data_path="train.txt", is_test=False, lazy=False)
dev_ds = load_dataset(read_text_pair, data_path="dev.txt", is_test=False, lazy=False)
test_ds = load_dataset(read_text_pair, data_path="test.txt", is_test=False, lazy=False)


# 测试一下数据集有没有导入成功，以及查看数据的格式

# In[4]:


# 输出训练集的前 3 条样本
for idx, example in enumerate(train_ds):
    if idx <= 2:
        print(example)
# 输出测试集的前 3 条样本
for idx, example in enumerate(test_ds):
    if idx <= 2:
        print(example)


# **2、 数据预处理**
# 
# 通过 PaddleNLP 加载进来的数据集是原始的明文数据集，这部分我们来实现组 batch、tokenize 等预处理逻辑，将原始明文数据转换成网络训练的输入数据。

# In[5]:


# 因为是基于预训练模型 ERNIE-Gram 来进行，所以需要首先加载 ERNIE-Gram 的 tokenizer。
tokenizer = ppnlp.transformers.ErnieGramTokenizer.from_pretrained('ernie-gram-zh')


# 测试一下预处理的结果

# In[6]:


### 对训练集的第 1 条数据进行转换
input_ids, token_type_ids, label = convert_example(train_ds[0], tokenizer)
print(input_ids)
print(token_type_ids)


# In[7]:


'''
    用Python的partial方法定义一个转换函数trans_func, 第一个为基础函数，之后的两个是赋予的参数，
    trans_func就是一个已经设置好了两个参数的convert_example函数
'''
trans_func = partial(convert_example, tokenizer=tokenizer, max_seq_length=256)


# In[8]:


'''
    批处理函数，
'''
batchify_fn = lambda samples, fn=Tuple(
        Pad(axis=0, pad_val=tokenizer.pad_token_id),  # text_pair_input
        Pad(axis=0, pad_val=tokenizer.pad_token_type_id),  # text_pair_segment
        Stack(dtype="int64")  # label
    ): [data for data in fn(samples)]


# #### 定义 Dataloader
# 下面我们基于组 batchify_fn 函数和样本转换函数 trans_func 来构造训练集的 DataLoader, 支持多卡训练

# In[9]:


train_data_loader = create_dataloader(
        train_ds,
        mode='train',
        batch_size=32,
        batchify_fn=batchify_fn,
        trans_fn=trans_func)

dev_data_loader = create_dataloader(
        dev_ds,
        mode='dev',
        batch_size=128,
        batchify_fn=batchify_fn,
        trans_fn=trans_func)


# In[ ]:


# 我们基于 ERNIE-Gram 模型结构搭建 Point-wise 语义匹配网络
# 所以此处先定义 ERNIE-Gram 的 pretrained_model(预训练模型)
pretrained_model = ppnlp.transformers.ErnieGramModel.from_pretrained('ernie-gram-zh')

class QuestionMatching(nn.Layer):
    '''
    类构造方法：实例化时设置选用的预处理模型，dropout和
    '''
    def __init__(self, pretrained_model, dropout=None, rdrop_coef=0.0):
        super().__init__()
        self.ptm = pretrained_model # ptm为预训练模型
        self.dropout = nn.Dropout(dropout if dropout is not None else 0.1) 
         # 设置Dropout，指在深度学习网络的训练过程中，对于神经网络单元，按照一定的概率将其暂时从网络中丢弃。由于是随机丢弃，故而每一个mini-batch都在训练不同的网络。

        # 标签数目为2，1表示两个问句等价，0表示不等价。
        self.classifier = nn.Linear(self.ptm.config["hidden_size"], 2)  # linear线性变换层，
        self.rdrop_coef = rdrop_coef # 这是一个填补dropout缺陷的参数
        self.rdrop_loss = ppnlp.losses.RDropLoss()

    '''
    在模型训练时，不需要调用，只要在实例化一个对象中传入对应的参数就可以自动调用 forward 函数
    '''
    def forward(self,
                input_ids,
                token_type_ids=None,
                position_ids=None,
                attention_mask=None, 
                do_evaluate=False):  # 是否进行评价

        _, cls_embedding1 = self.ptm(input_ids, token_type_ids, position_ids, attention_mask)
        cls_embedding1 = self.dropout(cls_embedding1)
        logits1 = self.classifier(cls_embedding1)
        
        if self.rdrop_coef > 0 and not do_evaluate:
            _, cls_embedding2 = self.ptm(input_ids, token_type_ids, position_ids, attention_mask)
            cls_embedding2 = self.dropout(cls_embedding2)
            logits2 = self.classifier(cls_embedding2)  # 分类器（logit日志）
            kl_loss = self.rdrop_loss(logits1, logits2)  # KL散度
        else:
            kl_loss = 0.0

        return logits1, kl_loss  # 


# In[ ]:


# 实例化模型
model = QuestionMatching(pretrained_model, rdrop_coef=0.0)


# ### 2.4 模型训练 & 评估

# In[ ]:


epochs = 1  # 一次完整训练
num_training_steps = len(train_data_loader) * epochs  # 


lr_scheduler = LinearDecayWithWarmup(5e-5, num_training_steps, 0.0)

decay_params = [
        p.name for n, p in model.named_parameters()
        if not any(nd in n for nd in ["bias", "norm"])
    ]
    
optimizer = paddle.optimizer.AdamW(
        learning_rate=lr_scheduler,
        parameters=model.parameters(),
        weight_decay=0.0,
        apply_decay_param_fun=lambda x: x in decay_params)

criterion = paddle.nn.loss.CrossEntropyLoss() # 计算输入input和标签label间的交叉熵损失

metric = paddle.metric.Accuracy()  # 计算准确度


# In[ ]:


'''
评价函数
    参数为：模型，标准化函数， 度量， data_loader
    返回值为：计算出的准确度
'''
@paddle.no_grad()
def evaluate(model, criterion, metric, data_loader):
    model.eval()
    metric.reset()
    losses = []
    total_num = 0

    for batch in data_loader:
        input_ids, token_type_ids, labels = batch
        total_num += len(labels)
        logits, _ = model(input_ids=input_ids, token_type_ids=token_type_ids, do_evaluate=True)
        loss = criterion(logits, labels)
        losses.append(loss.numpy())
        correct = metric.compute(logits, labels)
        metric.update(correct)
        accu = metric.accumulate()

    print("dev_loss: {:.5}, accuracy: {:.5}, total_num:{}".format(np.mean(losses), accu, total_num))
    model.train()
    metric.reset()
    return accu


# In[19]:


global_step = 0  # 总步数
best_accuracy = 0.0  # 最优的准确度

tic_train = time.time()
for epoch in range(1, 1+ 1):  # 总共只执行一遍吧
    # 遍历训练集，每次取一个batch，batch_size的大小为32，
    for step, batch in enumerate(train_data_loader, start=1): # start表示起始下标位置
        input_ids, token_type_ids, labels = batch  
        logits1, kl_loss = model(input_ids=input_ids, token_type_ids=token_type_ids)
        correct = metric.compute(logits1, labels)  # 计算，会判断每个数据的结果是否与label相同，即是否正确
        metric.update(correct)  # 更新metric状态
        acc = metric.accumulate()  # 准确率

        ce_loss = criterion(logits1, labels)
        if kl_loss > 0:
            loss = ce_loss + kl_loss * args.rdrop_coef
        else:
            loss = ce_loss
            
        global_step += 1
        # 每10轮"global step %d, epoch: %d, batch: %d, loss: %.4f, ce_loss: %.4f., kl_loss: %.4f, accu: %.4f, speed: %.2f step/s"
        if global_step % 10 == 0:
            print(
                "训练步数：%d, 训练轮次: %d, 当前组: %d, 损失函数: %.4f, 交叉熵: %.4f., KL散度: %.4f, 准确率: %.4f, 速度: %.2f step/s"
                % (global_step, epoch, step, loss, ce_loss, kl_loss, acc,
                    10 / (time.time() - tic_train)))
            tic_train = time.time()

        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        optimizer.clear_grad()

        # 每100轮使用评价函数评价一下结果，获取其准确率
        if global_step % 200 == 0:
            accuracy = evaluate(model, criterion, metric, dev_data_loader)
            if accuracy > best_accuracy:
                save_dir = os.path.join("./checkpoint", "model_%d" % global_step)
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                save_param_path = os.path.join(save_dir, 'model_state.pdparams')
                paddle.save(model.state_dict(), save_param_path)
                tokenizer.save_pretrained(save_dir)
                best_accuracy = accuracy


# ### 2.5 模型预测
# 
# 接下来我们使用已经训练好的语义匹配模型对一些预测数据进行预测。

# In[1]:


get_ipython().system(' wget https://paddlenlp.bj.bcebos.com/models/text_matching/question_matching_rdrop0p0_baseline_model.tar')


# In[2]:


get_ipython().system(' tar -xvf question_matching_rdrop0p0_baseline_model.tar')


# In[3]:


get_ipython().system('head -3 "data/data104941/test_A"')


# In[6]:


get_ipython().system('$ unset CUDA_VISIBLE_DEVICES')

get_ipython().system(' export FLAGS_fraction_of_gpu_memory_to_use=0')
get_ipython().system('python -u     work/predict.py     --device cpu    --params_path "./checkpoint/model_200/model_state.pdparams"     --batch_size 128     --input_file "test.txt"     --result_file "result.csv"')


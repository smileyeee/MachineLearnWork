import paddle
import numpy as np

'''
参数：
    dataset：创建的数据集，
    mode：数据集类型
    batch_size:一次训练所抓取的数据样本数量
    batchify_fn: 批处理函数
    trans_fn: 转换函数
返回值：一个迭代 dataset 数据的迭代器
'''
def create_dataloader(dataset,
                      mode='train',
                      batch_size=1,
                      batchify_fn=None,
                      trans_fn=None):
    if trans_fn:
        dataset = dataset.map(trans_fn)  # map：对dataset的每条数据调用 trans_fn函数，返回包含每次 trans_fn 函数返回值的新列表

    shuffle = True if mode == 'train' else False # 一个flag，判断是否需要打乱排序
    if mode == 'train':  
        '''
        该方法用于加载数据集，分布式批采样器加载数据的一个子集。每个进程可以传递给DataLoader一个DistributedBatchSampler的实例，每个进程加载原始数据的一个子集。
        dataset用于生成下标，batch_size指定依次抓取的样本数，shuffle指定是否打乱排序
        返回样本下标数组的迭代器(用于访问数据集，可类比指针)
        '''
        batch_sampler = paddle.io.DistributedBatchSampler(
            dataset, batch_size=batch_size, shuffle=shuffle)
    else:
        '''
        该方法用于 paddle.io.DataLoader 中迭代式获取mini-batch的样本下标数组，
        前者：分布式批数据采样，后者：批数据采样
        '''
        batch_sampler = paddle.io.BatchSampler(
            dataset, batch_size=batch_size, shuffle=shuffle)
    '''
    返回一个迭代器，该迭代器根据 batch_sampler 给定的顺序/迭代一次/dataset
    collate_fn: 指定如何将样本列表组合为mini-batch数据；return_list: 以列表形式返回
    返回一个迭代 dataset 数据的迭代器，迭代器返回的数据中的每个元素都是一个Tensor（张量，结构与数组矩阵等类似，专门针对GPU来设计的，可以运行在GPU上来加快计算效率）。
    '''
    return paddle.io.DataLoader(
        dataset=dataset,
        batch_sampler=batch_sampler,
        collate_fn=batchify_fn,
        return_list=True) 



'''
参数：data_path: 数据文件路径；is_test：是否为测试数据（测试数据没有label）
返回值：一条数据的字典，含两个句子（和一个标签）
'''
def read_text_pair(data_path, is_test=False):
    # with as语句，执行前面的open操作，结果保存到变量f，该语句可保证执行完毕后自动关闭已经打开的文件。
    with open(data_path, 'r', encoding='utf-8') as f:
        n = 0
        # 遍历文件每一行
        for line in f:
            n = n+1
            # 先删除每行末尾的空白符，再以tab切分数据
            data = line.rstrip().split("\t")
            if is_test == False:
                if len(data) != 3: # 正确的数据应当有三个
                    continue
                # 使用yield返回数据而不使用return，函数会在遇到yield时返回，下次调用直接从yield处继续执行，局部变量会保存，就不需要反复执行全部操作了
                yield {'query1': data[0], 'query2': data[1], 'label': data[2]}
            else:
                if len(data) != 2:
                    continue
                yield {'query1': data[0], 'query2': data[1]}
            if n >= 32000:  # 只读32000条数据
                break;


'''
参数：
    example：一条数据（字典）；tokenizer：分词器函数；max_seq_length：序列最大长度；is_test：是否为测试集数据
返回值：
'''
def convert_example(example, tokenizer, max_seq_length=512, is_test=False):
    query, title = example["query1"], example["query2"]
    '''
    分词器将句子转为字码，
    '''
    encoded_inputs = tokenizer(
        text=query, text_pair=title, max_seq_len=max_seq_length)

    input_ids = encoded_inputs["input_ids"]
    token_type_ids = encoded_inputs["token_type_ids"]

    if not is_test:
        label = np.array([example["label"]], dtype="int64")
        return input_ids, token_type_ids, label
    else:
        return input_ids, token_type_ids



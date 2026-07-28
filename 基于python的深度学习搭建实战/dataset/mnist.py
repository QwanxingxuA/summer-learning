# MNIST数据相关库
import urllib.request
import gzip
import pickle
import os
import numpy as np

url = 'https://ossci-datasets.s3.amazonaws.com/mnist/' # 数据地址

# 需要的文件
key_file = { 
    'train_img':'train-images-idx3-ubyte.gz',
    'train_label':'train-labels-idx1-ubyte.gz',
    'test_img':'t10k-images-idx3-ubyte.gz',
    'test_label':'t10k-labels-idx1-ubyte.gz'
}

dataset_dir = os.path.dirname(os.path.abspath(__file__)) # 绝对路径目录名
save_file = dataset_dir+'/mnist.pkl'

train_num = 60000
test_num = 10000
img_dim = (1,28,28)
img_size = 784

# 下载文件
def download(file_name):
    file_path = dataset_dir + '/' + file_name

    if(os.path.exists(file_path)):
        return

    print('下载' + file_name + '...')
    urllib.request.urlretrieve(url + file_name,file_path)
    print('完成！')

# 下载数据集
def download_mnist():
    for v in key_file.values():
        download(v)

# 加载标签
def load_label(file_name):
    file_path = dataset_dir + '/' + file_name

    print('将' + file_name + '转为numpy')
    with gzip.open(file_path,'rb') as f:
        labels = np.frombuffer(f.read(),np.uint8,offset=8)
    print('完成')

# 加载图片
def load_img(file_name):
    file_path = dataset_dir + '/' + file_name

    print('将' + file_name + '转为numpy')
    with gzip.open(file_path,'rb') as f:
        data = np.frombuffer(f.read(),np.uint8,offset=16)
    data = data.reshape(-1,img_size)
    print('完成')

# 加载所有数据（转成numpy）
def convert_numpy():
    dataset = {}
    dataset['train_img'] = load_img(key_file['train_img'])
    dataset['test_img'] = load_img(key_file['train_img'])
    dataset['train_label'] = load_label(key_file['train_label'])
    dataset['test_label'] = load_label(key_file['test_label'])
    return dataset

# 初始化mnist数据集
def init_mnist():
    download_mnist()
    dataset = convert_numpy()
    print('创建pkl文件')
    with open(save_file,'wb') as f:
        pickle.dump(dataset,f,-1)
    print('完成')

# 独热编码
def change_one_hot_label(x):
    T = np.zeros((x.size,10))   
    for idx,row in enumerate(T):
        row[x[idx]] = 1         # x[idx]是第idx个标签对应的数如2，则第2号元素为1
    return T

# 加载mnist数据集(通过pkl加载)
def load_mnist(normalize=True,flatten=True,one_hot_label=False):
    if not os.path.exists(save_file):
        init_mnist()

    with open(save_file,'rb') as f:
        dataset = pickle.load(f)

    # 正则化
    if normalize:
        for key in ('train_img','test_img'):
            dataset[key] = dataset[key].astype(np.float32)
            dataset[key] = dataset[key]/255

    # 独热编码
    if one_hot_label:
        dataset['train_label'] = change_one_hot_label(dataset['train_label'])
        dataset['test_label'] = change_one_hot_label(dataset['test_label'])

    # 展平
    if not flatten:
        for key in ('train_img','test_img'):
            dataset[key] = dataset[key].reshape(-1,1,28,28)
                
    return (dataset['train_img'],dataset['train_label']),(dataset['test_img'],dataset['test_label'])

if __name__ == '__main__':
    init_mnist()
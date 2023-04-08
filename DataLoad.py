import pickle               # 读取和保存文件的库
import numpy as np


save_file = "./mnist.pkl"

train_num = 60000       # 训练数据个数
test_num = 10000        # 测试数据个数
img_dim = (1, 28, 28)   # 输入数据的维度
img_size = 784          # 数据数据的 size


def load_mnist():
    with open(save_file, 'rb') as f:
        dataset = pickle.load(f)

    # 正则化处理
    for key in ('train_img', 'test_img'):
        dataset[key] = dataset[key].astype(np.float32)
        dataset[key] /= 255.0

    # 将一维序列数据转化为多维数据
    for key in ('train_img', 'test_img'):
        dataset[key] = dataset[key].reshape(-1, 1, 28, 28)

    # 返回训练数据和测试数据
    return (dataset['train_img'], dataset['train_label']), (dataset['test_img'], dataset['test_label'])

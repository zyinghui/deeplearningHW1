import pickle

import numpy as np
from Layers import *
from collections import OrderedDict


class FullConnectedNet:
    """
    网络结构: affine - relu -
            affine - softmax
    """
    def __init__(self, input_dim=(1, 28, 28), hidden_size=100, output_size=10, weight_init_std=0.01):
        input_size = input_dim[1] * input_dim[2]
        self.hidden_size = hidden_size

        self.params = dict()

        # 第一层参数
        self.params['W1'] = np.random.randn(input_size, hidden_size) * weight_init_std
        self.params['b1'] = np.zeros(hidden_size)

        # 第二层参数, 同时也是输出层, 后接 softmax 的输出层激活函数
        self.params['W2'] = np.random.randn(hidden_size, output_size) * weight_init_std
        self.params['b2'] = np.zeros(output_size)

        self.layers = OrderedDict()
        self.layers['Affine1'] = Affine(self.params['W1'], self.params['b1'])
        self.layers['Relu1'] = Relu()
        self.layers['Affine2'] = Affine(self.params['W2'], self.params['b2'])
        self.last_layer = SoftmaxWithLoss()

    # 正向传播到 Affine 层
    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)

        return x

    # 正向传播到最后的 loss 层
    def loss(self, x, t):
        y = self.predict(x)
        return self.last_layer.forward(y, t)

    def gradient(self, x, t):
        self.loss(x, t)  # 正向传播

        # 反向传播求各层梯度
        douput = 1
        douput = self.last_layer.backward(douput)

        layers = list(self.layers.values())
        layers.reverse()

        for layer in layers:
            douput = layer.backward(douput)

        # 组装返回的结果
        grads = dict()
        grads['W1'], grads['b1'] = self.layers['Affine1'].dW, self.layers['Affine1'].db
        grads['W2'], grads['b2'] = self.layers['Affine2'].dW, self.layers['Affine2'].db

        return grads

    # 计算精确度
    def accuracy(self, x, t, batch_size=100):
        if t.ndim != 1:
            t = np.argmax(t, axis=1)

        acc = 0.0

        for i in range(int(x.shape[0] / batch_size)):
            tx = x[i * batch_size:(i + 1) * batch_size]
            tt = t[i * batch_size:(i + 1) * batch_size]
            y = self.predict(tx)
            y = np.argmax(y, axis=1)
            acc += np.sum(y == tt)

        return acc / x.shape[0]

    # 从文件中加载参数
    def load_params(self, file_name="params.pkl"):
        with open(file_name, 'rb') as f:
            params = pickle.load(f)

        for key, val in params.items():
            self.params[key] = val

        for i, key in enumerate(['Affine1', 'Affine2']):
            self.layers[key].W = self.params['W' + str(i + 1)]
            self.layers[key].b = self.params['b' + str(i + 1)]

    # 将参数保存到文件中
    def save_params(self, file_name="params.pkl"):
        params = {}
        for key, val in self.params.items():
            params[key] = val
        with open(file_name, 'wb') as f:
            pickle.dump(params, f)


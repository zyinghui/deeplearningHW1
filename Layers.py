import numpy as np


def relu(x):
    return np.maximum(0, x)


def softmax(x):
    if x.ndim == 2:
        x = x.T
        x = x - np.max(x, axis=0)
        y = np.exp(x) / np.sum(np.exp(x), axis=0)
        return y.T

    x = x - np.max(x)
    return np.exp(x) / np.sum(np.exp(x))


# 平均交叉熵误差
def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    if t.size == y.size:
        t = t.argmax(axis=1)

    batch_num = y.shape[0]

    return -np.sum(np.log(y[np.arange(batch_num), t] + 1e-7)) / batch_num


# 激活函数
class Relu:
    def __init__(self):
        self.is_negative = None

    def forward(self, input):
        self.is_negative = input <= 0

        output = input.copy()
        output[self.is_negative] = 0

        return output

    def backward(self, doutput):
        doutput[self.is_negative] = 0
        dinput = doutput

        return dinput


# softmax 层和 cross entropy error 层
class SoftmaxWithLoss:
    def __init__(self):
        self.y = None
        self.t = None
        self.loss = None

    def forward(self, input, t):
        self.y = softmax(input)
        self.t = t
        self.loss = cross_entropy_error(self.y, self.t)

        return self.loss

    def backward(self, doutput=1):
        batch_num = self.y.shape[0]

        # 标签数据是 one-hot 的情况
        if self.t.size == self.y.size:
            dinput = (self.y - self.t) / batch_num
        else:
            dinput = self.y.copy()
            dinput[np.arange(batch_num), self.t] -= 1
            dinput = dinput / batch_num

        return dinput

# 仿射层
class Affine:
    def __init__(self, W, b):
        self.W = W
        self.b = b

        self.x = None
        self.original_x_shape = None

        self.dW = None
        self.db = None

    def forward(self, x):
        self.original_x_shape = x.shape
        self.x = x.reshape(x.shape[0], -1)
        output = np.dot(self.x, self.W) + self.b

        return output

    def backward(self, doutput):
        dintput = np.dot(doutput, self.W.T)
        self.dW = np.dot(self.x.T, doutput)
        self.db = np.sum(doutput, axis=0)

        dintput = dintput.reshape(*self.original_x_shape)
        return dintput

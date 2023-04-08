import numpy as np


# 随机梯度下降法
class SGD:
    def __init__(self, lr=0.01, epochs=0, L2_lambda=0.001):
        self.lr = lr
        self.initial_lr = lr
        self.m = None
        self.beta1 = 0.9
        self.L2_lambda = L2_lambda
        self.decay = 0.01 / epochs
        self.decay_rate = 0.9

    def update(self, params, grads):
        if self.m is None:
            self.m = {}
            for key, val in params.items():
                self.m[key] = np.zeros_like(val)

        for key in params.keys():
            self.m[key] += (1 - self.beta1) * (grads[key] - self.m[key])
            params[key] -= self.lr * self.m[key] - self.L2_lambda * params[key]

    # 学习率指数衰减
    def update_lr(self, iterations=1):
        self.lr = self.initial_lr / (1 + self.decay * iterations)

# 训练器
class Trainer:
    def __init__(self, network,
                 x_train, t_train, x_test, t_test,
                 epochs=20, mini_batch_size=100, lr=0.01, verbose = True, L2_lambda = 0.001):
        self.network = network
        self.x_train = x_train
        self.t_train = t_train
        self.x_test = x_test
        self.t_test = t_test
        self.epochs = epochs
        self.batch_size = mini_batch_size
        self.lr = lr
        self.verbose = verbose
        self.L2_lambda = L2_lambda

        self.optimizer = SGD(lr=self.lr, epochs=self.epochs, L2_lambda=self.L2_lambda)

        self.train_size = x_train.shape[0]
        self.iter_per_epoch = max(self.train_size / mini_batch_size, 1)
        self.max_iter = int(epochs * self.iter_per_epoch)
        self.current_iter = 0
        self.current_epoch = 0

        self.train_loss_list = []
        self.test_loss_list = []
        self.train_acc_list = []
        self.test_acc_list = []

    def train_step(self):
        batch_index = np.random.choice(self.train_size, self.batch_size)

        x_batch = self.x_train[batch_index]
        t_batch = self.t_train[batch_index]

        # 反向传播
        grads = self.network.gradient(x_batch, t_batch)

        # 更新参数
        self.optimizer.update(self.network.params, grads)
        self.optimizer.update_lr(self.current_iter)

        # 计算损失函数值
        regularization = 0
        pa = np.hstack((self.network.params['W1'].reshape(-1), self.network.params['W2'].reshape(-1),
                        self.network.params['b1'], self.network.params['b2']))

        for value in pa:
            regularization += self.L2_lambda * pow(value, 2) / 2

        loss = self.network.loss(x_batch, t_batch) + regularization

        # 判断当前迭代是不是该 epoch 的最后一次迭代
        if self.current_iter % self.iter_per_epoch == 0:
            print("train loss:" + str(loss) + "  epoch:" + str(self.current_epoch) + '  lr:' + str(self.optimizer.lr))
            self.current_epoch += 1

            x_train_sample, t_train_sample = self.x_train, self.t_train
            x_test_sample, t_test_sample = self.x_test, self.t_test

            # 计算在测试集上的损失
            self.train_loss_list.append(loss)
            test_loss = self.network.loss(x_test_sample, t_test_sample)
            self.test_loss_list.append(test_loss)

            # 计算在训练集和测试集上的精度
            train_acc = self.network.accuracy(x_train_sample, t_train_sample)
            test_acc = self.network.accuracy(x_test_sample, t_test_sample)
            self.train_acc_list.append(train_acc)
            self.test_acc_list.append(test_acc)

            print("=== epoch:" + str(self.current_epoch)
                  + ", train acc:" + str(train_acc)
                  + ", test acc:" + str(test_acc)
                  + " ===")
        self.current_iter += 1

    def train(self):
        print('lr:' + str(self.lr) + '  hidden size:' + str(self.network.hidden_size)
              + '  L2 lambda:' + str(self.L2_lambda))

        for i in range(self.max_iter):
            self.train_step()

        test_acc = self.network.accuracy(self.x_test, self.t_test)

        print("训练已完成!")
        print("=============== Final Test Accuracy ===============")
        print("test acc:" + str(test_acc))

        return test_acc

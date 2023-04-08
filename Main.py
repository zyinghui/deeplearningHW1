from DataLoad import *
from Model import *
from Trainer import *
from Plot import *
import time


acc = []
best_acc = (0, 0, 0, 0)
candi_lr = [0.005, 0.01, 0.05]
candi_hidden_size = [100, 150, 200]
candi_L2_lambda = [0, 0.001, 0.01]

# 读入数据
(x_train, t_train), (x_test, t_test) = load_mnist()

# 设置遍历数据集的次数
max_epochs = 30
for lr in candi_lr:
    for hidden_size in candi_hidden_size:
        for L2_lambda in candi_L2_lambda:
            network = FullConnectedNet(input_dim=(1,28,28), hidden_size=hidden_size,
                                       output_size=10, weight_init_std=0.01)

            # 生成训练器的对象
            trainer = Trainer(network, x_train, t_train, x_test, t_test,
                              epochs=max_epochs, mini_batch_size=100, lr=lr, L2_lambda=L2_lambda)

            # 开始训练
            start_time = time.time()
            tmp_acc = trainer.train()
            acc.append([tmp_acc])
            end_time = time.time()
            print("time:{:.2f}".format(end_time - start_time))

            if tmp_acc > best_acc[0]:
                best_acc = (tmp_acc, lr, hidden_size, L2_lambda)
                # 保存训练出来的参数结果
                network.save_params("params.pkl")
                print("Saved Network Parameters!")
                print("===========================================")

            plot(tmp_acc, max_epochs, trainer)

print("best: acc=" + str(best_acc[0]) + "  lr=" + str(best_acc[1])
      + "  hidden size=" + str(best_acc[2]) + "  L2_lambda=" + str(best_acc[3]))

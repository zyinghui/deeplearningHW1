from DataLoad import *
from Model import *


(_, _), (x_test, t_test) = load_mnist()

hidden_size = 100
network = FullConnectedNet(input_dim=(1,28,28), hidden_size=hidden_size,
                                       output_size=10, weight_init_std=0.01)
network.load_params("params.pkl")

test_acc = network.accuracy(x_test, t_test)
print("test acc:" + str(test_acc))

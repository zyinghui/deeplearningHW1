import matplotlib.pylab as plt
import numpy as np


def plot(acc, max_epochs, trainer):
    x = np.arange(max_epochs)
    plt.plot(x, trainer.train_acc_list, marker='s', label='train', markevery=2)
    plt.plot(x, trainer.test_acc_list, marker='o', label='test', markevery=2)
    plt.xlabel("epochs")
    plt.ylabel("acc")
    plt.ylim(0, 1.0)
    plt.title('lr=' + str(trainer.lr) + ' h_s=' + str(trainer.network.hidden_size) + ' L2=' + str(trainer.L2_lambda))
    plt.legend(loc='lower right')
    plt.savefig('acc('+str(trainer.lr)+','+str(trainer.network.hidden_size)+','
                +str(trainer.L2_lambda)+','+str(acc)+')'+'.png')
    plt.show()

    plt.plot(x, trainer.train_loss_list, marker='s', label='train', markevery=2)
    plt.plot(x, trainer.test_loss_list, marker='o', label='test', markevery=2)
    plt.xlabel("epochs")
    plt.ylabel("loss")
    plt.title('lr=' + str(trainer.lr) + ' h_s=' + str(trainer.network.hidden_size) + ' L2=' + str(trainer.L2_lambda))
    plt.legend(loc='lower right')
    plt.savefig('loss('+str(trainer.lr)+','+str(trainer.network.hidden_size)+','+str(trainer.L2_lambda)+')'+'.png')
    plt.show()

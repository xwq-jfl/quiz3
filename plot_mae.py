import matplotlib.pyplot as plt
import numpy as np


class View_result():
    def __init__(self,trainlog_path,start=0,end=2000):
        self.trainlog_path = trainlog_path
        self.train_epoch,self.test_epoch,self.mae,self.mse,self.train_loss = self._load_log_data(self.trainlog_path)
        self.start = start
        self.end = end
    def _load_log_data(self,trainlog_path):
        train_epoch = []
        test_epoch = []
        mae = []
        mse = []
        loss = []
        with open(trainlog_path, 'r') as f:
            for line in f.readlines():
                if len(line.split(' ')) > 5:
                    if line.split(' ')[4] == 'test,':
                        test_epoch.append(int(line.split(' ')[3]))
                        mae.append(float(line.split(' ')[8].strip(',')))
                        mse.append(float(line.split(' ')[6]))
                    elif line.split(' ')[4] == 'Train,':
                        loss.append(float(line.split(' ')[6].strip(',')))
                        train_epoch.append(int(line.split(' ')[3]))
        return train_epoch,test_epoch,mae,mse,loss



    def plot_mae(self):
        test_epoch = self.test_epoch[self.start//5:self.end//5]
        mae = self.mae[self.start//5:self.end//5]
        plt.plot(test_epoch,mae)
        plt.xlabel('epoch')
        plt.ylabel('MAE')
        plt.show()
    def plot_mse(self):
        test_epoch = self.test_epoch[self.start // 5:self.end // 5]
        mse = self.mse[self.start // 5:self.end // 5]
        plt.plot(test_epoch,mse)
        plt.xlabel('epoch')
        plt.ylabel('MSE')
        plt.show()
    def plot_loss(self):
        train_epoch = self.train_epoch[self.start :self.end ]
        loss = self.train_loss[self.start :self.end ]
        plt.plot(train_epoch,loss)
        plt.xlabel('epoch')
        plt.ylabel('loss')

        plt.show()
trainlog_path = '/home/xwq/Bayesian-Crowd-Counting/log&model/0319-235922/train.log'
view_result = View_result(trainlog_path)
view_result.plot_mae()
view_result.plot_loss()
view_result.plot_mse()




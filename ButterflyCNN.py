import sys
import os
sys.path.append(os.pardir)  # 为了导入父目录的文件而进行的设定

import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

from collections import OrderedDict#能记住插入顺序的特殊字典
from dataset2.common_functions.Affine import Affine
from dataset2.common_functions.Convolution import Convolution
from dataset2.common_functions.Dropout import Dropout
from dataset2.common_functions.functions import *
from dataset2.common_functions.Pooling import Pooling
from dataset2.common_functions.Relu import Relu
from dataset2.common_functions.B_SoftmaxWithLoss import  B_SoftmaxWithLoss
from dataset2.common_functions.BatchNormalization import BatchNormalization


class ButterflyCNN:
    def __init__(self, input_dim=(3, 128, 128), output_size=100, dropout_ratio=0.3):
        self.input_dim = input_dim
        self.output_size = output_size
        self.dropout_ratio = dropout_ratio
        # 初始化参数
        self.params = {}
        self.__init_weights()
        # 创建网络层
        self.layers = OrderedDict()
        self.__build_layers()
        self.last_layer = B_SoftmaxWithLoss()

    def __init_weights(self, weight_init_std=0.01):
        # Conv1: 3 -> 32
        self.params['W1'] = weight_init_std * np.random.randn(32, 3, 3, 3)
        self.params['b1'] = np.zeros(32)
        # BN1 gamma and beta for 32 channels
        self.params['gamma1'] = np.ones(32)
        self.params['beta1'] = np.zeros(32)

        # Conv2: 32 -> 64
        self.params['W2'] = weight_init_std * np.random.randn(64, 32, 3, 3)
        self.params['b2'] = np.zeros(64)
        # BN2 gamma and beta for 64 channels
        self.params['gamma2'] = np.ones(64)
        self.params['beta2'] = np.zeros(64)

        # 计算全连接层输入尺寸 (经过2次pooling后 128->64->32)
        self.fc_input_size = 64 * 16 * 16
        # 第一个全连接层：降维
        self.params['W3'] = weight_init_std * np.random.randn(self.fc_input_size, 512)
        self.params['b3'] = np.zeros(512)
        # BN3 gamma and beta for 512 features
        self.params['gamma3'] = np.ones(512)
        self.params['beta3'] = np.zeros(512)

        # 第二个全连接层：输出层
        self.params['W4'] = weight_init_std * np.random.randn(512, self.output_size)
        self.params['b4'] = np.zeros(self.output_size)

    def __build_layers(self):
        # 卷积块1
        self.layers['Conv1'] = Convolution(self.params['W1'], self.params['b1'], stride=1, pad=1)
        # 添加 BN 层
        self.layers['BatchNorm1'] = BatchNormalization(self.params['gamma1'], self.params['beta1'])
        self.layers['Relu1'] = Relu()
        self.layers['Pool1'] = Pooling(2, 2, stride=2) # 128 -> 64

        # 卷积块2
        self.layers['Conv2'] = Convolution(self.params['W2'], self.params['b2'], stride=1, pad=1)
        # 添加 BN 层
        self.layers['BatchNorm2'] = BatchNormalization(self.params['gamma2'], self.params['beta2'])
        self.layers['Relu2'] = Relu()
        self.layers['Pool2'] = Pooling(2, 2, stride=2) # 64 -> 32

        # 全连接层
        self.layers['Affine1'] = Affine(self.params['W3'], self.params['b3'])
        # 添加 BN 层 (注意：对于全连接层，BN作用于最后一个维度)
        self.layers['BatchNorm3'] = BatchNormalization(self.params['gamma3'], self.params['beta3'])
        self.layers['Relu3'] = Relu()
        self.layers['Dropout1'] = Dropout(self.dropout_ratio)
        self.layers['Affine2'] = Affine(self.params['W4'], self.params['b4'])

    def predict(self, x, train_flg=False):
        for layer in self.layers.values():
            if hasattr(layer, 'forward'):
                # 注意：BatchNormalization 层需要 train_flg 参数
                if isinstance(layer, BatchNormalization):
                    x = layer.forward(x, train_flg)
                elif 'Dropout' in layer.__class__.__name__:
                    x = layer.forward(x, train_flg)
                else:
                    x = layer.forward(x)
        return x

    def loss(self, x, t, train_flg=False):
        y = self.predict(x, train_flg)
        return self.last_layer.forward(y, t, label_smoothing=0.1)

    def accuracy(self, x, t, batch_size=100):
        if t.ndim != 1:
            t = np.argmax(t, axis=1)
        acc = 0.0
        for i in range(0, len(x), batch_size):
            tx = x[i:i + batch_size]
            tt = t[i:i + batch_size]
            y = self.predict(tx, train_flg=False) # 预测时设置为 False
            y = np.argmax(y, axis=1)
            acc += np.sum(y == tt)
        return acc / x.shape[0] if x.shape[0] > 0 else 0.0

    def gradient(self, x, t):
        # 前向传播
        self.loss(x, t, train_flg=True) # 训练时设置为 True

        # 反向传播
        dout = 1
        dout = self.last_layer.backward(dout)
        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        # 计算梯度
        grads = {}
        grads['W1'] = self.layers['Conv1'].dW
        grads['b1'] = self.layers['Conv1'].db
        # 获取 BN1 的梯度
        grads['gamma1'] = self.layers['BatchNorm1'].dgamma
        grads['beta1'] = self.layers['BatchNorm1'].dbeta

        grads['W2'] = self.layers['Conv2'].dW
        grads['b2'] = self.layers['Conv2'].db
        # 获取 BN2 的梯度
        grads['gamma2'] = self.layers['BatchNorm2'].dgamma
        grads['beta2'] = self.layers['BatchNorm2'].dbeta

        grads['W3'] = self.layers['Affine1'].dW
        grads['b3'] = self.layers['Affine1'].db
        # 获取 BN3 的梯度
        grads['gamma3'] = self.layers['BatchNorm3'].dgamma
        grads['beta3'] = self.layers['BatchNorm3'].dbeta

        grads['W4'] = self.layers['Affine2'].dW
        grads['b4'] = self.layers['Affine2'].db

        return grads
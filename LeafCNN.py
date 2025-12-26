import sys, os
sys.path.append(os.pardir)  # 为了导入父目录的文件而进行的设定
from dataset1.common_functions.Affine import Affine
from dataset1.common_functions.Convolution import Convolution
from dataset1.common_functions.Dropout import Dropout
from dataset1.common_functions.functions import *
from dataset1.common_functions.Pooling import Pooling
from dataset1.common_functions.Relu import Relu
from dataset1.common_functions.SoftmaxWithLoss import SoftmaxWithLoss
from dataset1.common_functions.BatchNormalization import BatchNormalization
from collections import OrderedDict

class LeafCNN:

    def __init__(self, input_dim=(3, 128, 128), output_size=8, weight_init_std=0.01,
                 dropout_ratio=0.3,label_smoothing=0.0):
        self.input_dim = input_dim
        self.output_size = output_size
        self.dropout_ratio = dropout_ratio
        self.label_smoothing = label_smoothing

        # 初始化参数
        self.params = {}
        self.__init_weights(weight_init_std)

        # 创建网络层
        self.layers = OrderedDict()
        self.__build_layers()

        self.last_layer = SoftmaxWithLoss(label_smoothing=self.label_smoothing)

    def __init_weights(self, weight_init_std):
        # 增加模型容量：增加卷积层通道数
        # 卷积层1: 3->32 (从16增加到32)
        self.params['W1'] = weight_init_std * np.random.randn(32, 3, 3, 3)
        self.params['b1'] = np.zeros(32)
        # BN for Conv1 (gamma, beta shape matches number of channels: 32)
        self.params['gamma1'] = np.ones(32) # gamma 初始化为 1
        self.params['beta1'] = np.zeros(32) # beta 初始化为 0

        # 卷积层2: 32->64 (从32增加到64)
        self.params['W2'] = weight_init_std * np.random.randn(64, 32, 3, 3)
        self.params['b2'] = np.zeros(64)
        # BN for Conv2 (gamma, beta shape matches number of channels: 64)
        self.params['gamma2'] = np.ones(64)
        self.params['beta2'] = np.zeros(64)

        # 计算全连接层输入尺寸 (假设经过两次池化 128 -> 64 -> 32)
        self.fc_input_size = 64 * 16 * 16

        # 增加全连接层神经元数量
        # 全连接层1
        self.params['W3'] = weight_init_std * np.random.randn(self.fc_input_size, 256)  # 从128增加到256
        self.params['b3'] = np.zeros(256)
        # BN for Affine1 (gamma, beta shape matches number of units: 256)
        self.params['gamma3'] = np.ones(256)
        self.params['beta3'] = np.zeros(256)

        # 输出层
        self.params['W4'] = weight_init_std * np.random.randn(256, self.output_size)
        self.params['b4'] = np.zeros(self.output_size)

    def __build_layers(self):
        # 卷积块1
        self.layers['Conv1'] = Convolution(self.params['W1'], self.params['b1'], pad=1)
        # --- 插入 BN1 (for Conv1, expects 4D input) ---
        self.layers['BN1'] = BatchNormalization(self.params['gamma1'], self.params['beta1'])
        self.layers['Relu1'] = Relu()
        self.layers['Pool1'] = Pooling(2, 2, 2)

        # 卷积块2
        self.layers['Conv2'] = Convolution(self.params['W2'], self.params['b2'], pad=1)
        # --- 插入 BN2 (for Conv2, expects 4D input) ---
        self.layers['BN2'] = BatchNormalization(self.params['gamma2'], self.params['beta2'])
        self.layers['Relu2'] = Relu()
        self.layers['Pool2'] = Pooling(2, 2, 2)

        # 全连接层
        self.layers['Affine1'] = Affine(self.params['W3'], self.params['b3'])
        # --- 插入 BN3 (for Affine1, expects 2D input) ---
        self.layers['BN3'] = BatchNormalization(self.params['gamma3'], self.params['beta3'])
        self.layers['Relu3'] = Relu()

        if self.dropout_ratio > 0:
            self.layers['Dropout1'] = Dropout(self.dropout_ratio)

        self.layers['Affine2'] = Affine(self.params['W4'], self.params['b4'])

    def predict(self, x, train_flg=False):
        # 关键：将 train_flg 传递给所有层，特别是 BN 层
        for layer_name, layer in self.layers.items():
            if isinstance(layer, BatchNormalization):
                x = layer.forward(x, train_flg=train_flg) # 注意这里传递 train_flg
            elif hasattr(layer, 'forward'):
                if 'Dropout' in layer.__class__.__name__:
                    x = layer.forward(x, train_flg)
                else:
                    x = layer.forward(x)
        return x

    def loss(self, x, t, train_flg=False):
        y = self.predict(x, train_flg=train_flg) # 传递给 predict
        return self.last_layer.forward(y, t)

    def accuracy(self, x, t, batch_size=100):
        if t.ndim != 1:
            t = np.argmax(t, axis=1)

        if len(x) == 0:
            return 0.0

        acc = 0.0
        for i in range(0, len(x), batch_size):
            tx = x[i:i + batch_size]
            tt = t[i:i + batch_size]
            y = self.predict(tx, train_flg=False) # 推理时设为 False
            y = np.argmax(y, axis=1)
            acc += np.sum(y == tt)

        return acc / len(x)

    def gradient(self, x, t):
        # 前向传播
        self.loss(x, t, train_flg=True) # 训练时设为 True

        # 反向传播
        dout = 1
        dout = self.last_layer.backward(dout)

        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        # 计算梯度
        grads = {}
        grads['W1'], grads['b1'] = self.layers['Conv1'].dW, self.layers['Conv1'].db
        # --- 获取 BN1 的梯度 ---
        grads['gamma1'], grads['beta1'] = self.layers['BN1'].dgamma, self.layers['BN1'].dbeta
        grads['W2'], grads['b2'] = self.layers['Conv2'].dW, self.layers['Conv2'].db
        # --- 获取 BN2 的梯度 ---
        grads['gamma2'], grads['beta2'] = self.layers['BN2'].dgamma, self.layers['BN2'].dbeta
        grads['W3'], grads['b3'] = self.layers['Affine1'].dW, self.layers['Affine1'].db
        # --- 获取 BN3 的梯度 ---
        grads['gamma3'], grads['beta3'] = self.layers['BN3'].dgamma, self.layers['BN3'].dbeta
        grads['W4'], grads['b4'] = self.layers['Affine2'].dW, self.layers['Affine2'].db

        return grads
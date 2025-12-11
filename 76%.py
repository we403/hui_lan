import os
import numpy as np
import time
from PIL import Image, ImageEnhance
import matplotlib.pyplot as plt
import pickle # 用于保存模型

# 设置全局字体为支持中文的字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
# 解决保存图像时负号“-”显示为方块的问题
plt.rcParams['axes.unicode_minus'] = False

# ... 你其余的代码 ...
from collections import OrderedDict


# ==================== 基础层实现 ====================
class Convolution:
    def __init__(self, W, b, stride=1, pad=0):
        self.W = W
        self.b = b
        self.stride = stride
        self.pad = pad
        self.x = None
        self.col = None
        self.col_W = None
        self.dW = None
        self.db = None

    def forward(self, x):
        FN, C, FH, FW = self.W.shape
        N, C, H, W = x.shape
        out_h = 1 + int((H + 2 * self.pad - FH) / self.stride)
        out_w = 1 + int((W + 2 * self.pad - FW) / self.stride)

        col = im2col(x, FH, FW, self.stride, self.pad)
        col_W = self.W.reshape(FN, -1).T

        out = np.dot(col, col_W) + self.b
        out = out.reshape(N, out_h, out_w, -1).transpose(0, 3, 1, 2)

        self.x = x
        self.col = col
        self.col_W = col_W
        return out

    def backward(self, dout):
        FN, C, FH, FW = self.W.shape
        dout = dout.transpose(0, 2, 3, 1).reshape(-1, FN)

        self.db = np.sum(dout, axis=0)
        self.dW = np.dot(self.col.T, dout)
        self.dW = self.dW.transpose(1, 0).reshape(FN, C, FH, FW)

        dcol = np.dot(dout, self.col_W.T)
        dx = col2im(dcol, self.x.shape, FH, FW, self.stride, self.pad)
        return dx

# ==================== Batch Normalization Layer (修改后) ====================
class BatchNormalization:
    """
    https://arxiv.org/abs/1502.03167
    Batch Normalization for both fully connected layers (input shape: (N, D))
    and convolutional layers (input shape: (N, C, H, W)).
    For Conv layers, normalization is applied per channel.
    """
    def __init__(self, gamma, beta, momentum=0.9, running_mean=None, running_var=None):
        self.gamma = gamma # Shape: (D,) for FC or (C,) for Conv
        self.beta = beta   # Shape: (D,) for FC or (C,) for Conv
        self.momentum = momentum
        self.input_shape = None # Original input shape (N, C, H, W) or (N, D)

        # 测试时使用的均值和方差 (Shape: (D,) for FC or (C,) for Conv)
        self.running_mean = running_mean
        self.running_var = running_var

        # backward时使用的中间数据
        self.batch_size = None
        self.xc = None # x - mean (same shape as reshaped input)
        self.std = None # sqrt(variance + eps) (Shape: (D,) for FC or (C,) for Conv)
        self.xn = None # normalized x (same shape as reshaped input)
        self.dgamma = None
        self.dbeta = None

    def forward(self, x, train_flg=True):
        self.input_shape = x.shape
        if x.ndim == 4: # Convolutional layer input (N, C, H, W)
             # Store original shape
            N, C, H, W = x.shape
            # Reshape to (N, C, H*W) -> (N, H*W, C) -> (N*H*W, C)
            # This way, each row is a feature map element, columns are channels
            x_reshaped = x.transpose(0, 2, 3, 1).reshape(-1, C) # (N*H*W, C)
            out_reshaped = self.__forward(x_reshaped, train_flg) # (N*H*W, C)
            # Reshape back to (N*H*W, C) -> (N, H*W, C) -> (N, C, H, W)
            out = out_reshaped.reshape(N, H, W, C).transpose(0, 3, 1, 2) # (N, C, H, W)
        elif x.ndim == 2: # Fully connected layer input (N, D)
             out = self.__forward(x, train_flg) # (N, D)
        else:
            raise ValueError(f"Invalid input dimension for BatchNormalization: {x.ndim}. Expected 2 or 4.")
        return out

    def __forward(self, x, train_flg):
        # x has shape (N, D) where D is features (could be C for conv or C*H*W flattened)
        N, D = x.shape

        if self.running_mean is None: # First run initialization
            self.running_mean = np.zeros(D)
            self.running_var = np.zeros(D)

        if train_flg: # Training mode
            mu = np.mean(x, axis=0) # Mean over batch dim for each feature/channel (D,)
            xc = x - mu # Center (N, D)
            var = np.var(x, axis=0) # Variance over batch dim for each feature/channel (D,)
            std = np.sqrt(var + 1e-7) # Std deviation (D,)
            xn = xc / std # Normalize (N, D)

            self.batch_size = N
            self.xc = xc
            self.xn = xn
            self.std = std
            # Update running statistics (exponential moving average)
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * mu
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * var
        else: # Inference/Test mode
            # Use accumulated running statistics
            xc = x - self.running_mean # (N, D) - (D,) broadcasts correctly
            xn = xc / (np.sqrt(self.running_var + 1e-7)) # (N, D) / (D,) broadcasts correctly

        # Scale and shift: Broadcasting works because gamma/beta are (D,)
        # and xn is (N, D). Operation is applied element-wise along the feature dim.
        out = self.gamma * xn + self.beta # (D,) * (N, D) + (D,) broadcasts to (N, D)
        return out

    def backward(self, dout):
        if dout.ndim == 4: # Convolutional layer gradient (N, C, H, W)
            N, C, H, W = dout.shape
            # Reshape to match forward pass reshape
            dout_reshaped = dout.transpose(0, 2, 3, 1).reshape(-1, C) # (N*H*W, C)
            dx_reshaped = self.__backward(dout_reshaped) # (N*H*W, C)
            # Reshape back to original conv shape
            dx = dx_reshaped.reshape(N, H, W, C).transpose(0, 3, 1, 2) # (N, C, H, W)
        elif dout.ndim == 2: # Fully connected layer gradient (N, D)
            dx = self.__backward(dout) # (N, D)
        else:
            raise ValueError(f"Invalid dout dimension for BatchNormalization backward: {dout.ndim}. Expected 2 or 4.")
        return dx

    def __backward(self, dout):
        # dout has shape (N, D)
        N, D = dout.shape

        # Gradients w.r.t. beta and gamma
        dbeta = np.sum(dout, axis=0) # Sum over batch dim (D,)
        dgamma = np.sum(self.xn * dout, axis=0) # Sum over batch dim (D,)

        # Gradient of the normalized input (xn)
        dxn = self.gamma * dout # (D,) * (N, D) broadcasts to (N, D)

        # Gradient of standard deviation (std)
        # d(L)/d(std) = sum( d(L)/d(xn) * d(xn)/d(std) )
        # d(xn)/d(std) = -xc / std^2
        dstd = -np.sum((dxn * self.xc) / (self.std ** 2), axis=0) # (D,)

        # Gradient of variance (var)
        # d(L)/d(var) = d(L)/d(std) * d(std)/d(var)
        # d(std)/d(var) = 0.5 / sqrt(var + eps) = 0.5 / std
        dvar = 0.5 * dstd / self.std # (D,)

        # Gradient of centered input (xc)
        # Comes from two paths: direct (dxn/std) and via variance (2*(x-mu)*dvar/N)
        dxc1 = dxn / self.std # Direct path (N, D)
        dxc2 = 2.0 * self.xc * dvar / N # Via variance path (N, D) -> (D,) broadcast to (N, D)
        dxc = dxc1 + dxc2 # (N, D)

        # Gradient of mean (mu)
        # Comes from two paths: direct (sum(dxc)) and via variance (sum(2*(x-mu)*dvar/N))
        # But since sum(2*(x-mu)) = 0, the second path's contribution to d(mu) is 0.
        dmu = np.sum(dxc, axis=0) # (D,)

        # Final gradient w.r.t. input (x)
        # Comes from three paths: direct (dxc), via mean (-dmu/N), and via variance (2*(x-mu)*dvar/N)
        # The third path is already included in dxc2 part of dxc.
        dx1 = dxc # (N, D)
        dx2 = -dmu / N # (D,) broadcast to (N, D)
        dx = dx1 + dx2 # (N, D)

        self.dgamma = dgamma
        self.dbeta = dbeta

        return dx # (N, D)

class Pooling:
    def __init__(self, pool_h, pool_w, stride=1, pad=0):
        self.pool_h = pool_h
        self.pool_w = pool_w
        self.stride = stride
        self.pad = pad
        self.x = None
        self.arg_max = None

    def forward(self, x):
        N, C, H, W = x.shape
        out_h = int(1 + (H - self.pool_h) / self.stride)
        out_w = int(1 + (W - self.pool_w) / self.stride)

        col = im2col(x, self.pool_h, self.pool_w, self.stride, self.pad)
        col = col.reshape(-1, self.pool_h * self.pool_w)

        arg_max = np.argmax(col, axis=1)
        out = np.max(col, axis=1)
        out = out.reshape(N, out_h, out_w, C).transpose(0, 3, 1, 2)

        self.x = x
        self.arg_max = arg_max
        return out

    def backward(self, dout):
        dout = dout.transpose(0, 2, 3, 1)

        pool_size = self.pool_h * self.pool_w
        dmax = np.zeros((dout.size, pool_size))
        dmax[np.arange(self.arg_max.size), self.arg_max.flatten()] = dout.flatten()
        dmax = dmax.reshape(dout.shape + (pool_size,))

        dcol = dmax.reshape(dmax.shape[0] * dmax.shape[1] * dmax.shape[2], -1)
        dx = col2im(dcol, self.x.shape, self.pool_h, self.pool_w, self.stride, self.pad)
        return dx


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
        x = x.reshape(x.shape[0], -1)
        self.x = x
        out = np.dot(self.x, self.W) + self.b
        return out

    def backward(self, dout):
        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)
        dx = dx.reshape(*self.original_x_shape)
        return dx


class Relu:
    def __init__(self):
        self.mask = None

    def forward(self, x):
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0
        return out

    def backward(self, dout):
        dout[self.mask] = 0
        return dout


class Dropout:
    def __init__(self, dropout_ratio=0.3):  # 降低Dropout比例
        self.dropout_ratio = dropout_ratio
        self.mask = None

    def forward(self, x, train_flg=True):
        if train_flg:
            self.mask = np.random.rand(*x.shape) > self.dropout_ratio
            return x * self.mask
        else:
            return x * (1.0 - self.dropout_ratio)

    def backward(self, dout):
        return dout * self.mask


class SoftmaxWithLoss:
    def __init__(self):
        self.loss = None
        self.y = None
        self.t = None

    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)
        self.loss = cross_entropy_error(self.y, self.t)
        return self.loss

    def backward(self, dout=1):
        batch_size = self.t.shape[0]
        if self.t.size == self.y.size:
            dx = (self.y - self.t) / batch_size
        else:
            dx = self.y.copy()
            dx[np.arange(batch_size), self.t] -= 1
            dx = dx / batch_size
        return dx


# ==================== 工具函数 ====================
def im2col(input_data, filter_h, filter_w, stride=1, pad=0):
    N, C, H, W = input_data.shape
    out_h = (H + 2 * pad - filter_h) // stride + 1
    out_w = (W + 2 * pad - filter_w) // stride + 1

    img = np.pad(input_data, [(0, 0), (0, 0), (pad, pad), (pad, pad)], 'constant')
    col = np.zeros((N, C, filter_h, filter_w, out_h, out_w))

    for y in range(filter_h):
        y_max = y + stride * out_h
        for x in range(filter_w):
            x_max = x + stride * out_w
            col[:, :, y, x, :, :] = img[:, :, y:y_max:stride, x:x_max:stride]

    col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N * out_h * out_w, -1)
    return col


def col2im(col, input_shape, filter_h, filter_w, stride=1, pad=0):
    N, C, H, W = input_shape
    out_h = (H + 2 * pad - filter_h) // stride + 1
    out_w = (W + 2 * pad - filter_w) // stride + 1
    col = col.reshape(N, out_h, out_w, C, filter_h, filter_w).transpose(0, 3, 4, 5, 1, 2)

    img = np.zeros((N, C, H + 2 * pad + stride - 1, W + 2 * pad + stride - 1))
    for y in range(filter_h):
        y_max = y + stride * out_h
        for x in range(filter_w):
            x_max = x + stride * out_w
            img[:, :, y:y_max:stride, x:x_max:stride] += col[:, :, y, x, :, :]

    return img[:, :, pad:H + pad, pad:W + pad]


def softmax(x):
    if x.ndim == 2:
        x = x - x.max(axis=1, keepdims=True)
        y = np.exp(x)
        return y / np.sum(y, axis=1, keepdims=True)
    x = x - np.max(x)
    return np.exp(x) / np.sum(np.exp(x))


def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    if t.size == y.size:
        t = t.argmax(axis=1)

    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size


def apply_data_augmentation(image, is_training=True):
    if not is_training:
        return image

        # 1. 水平翻转（50%概率，树叶对称性好）✅ 保留
    if np.random.rand() > 0.5:
        image = image.transpose(Image.FLIP_LEFT_RIGHT)

        # 2. 轻微旋转（-5° to 5°，树叶角度变化小）✅ 保留
    if np.random.rand() < 0.2:
        angle = np.random.uniform(-5, 5)
        image = image.rotate(angle)

        # 3. 关键优化：裁剪比例从90%→93%（保留关键叶脉）
    if np.random.rand() < 0.2:
        crop_size = int(0.93 * min(image.size))
        left = np.random.randint(0, image.width - crop_size)
        top = np.random.randint(0, image.height - crop_size)
        image = image.crop((left, top, left + crop_size, top + crop_size))
        image = image.resize((128, 128))

        # 4. 亮度/对比度调整（-12% to +12%）✅ 保留
    if np.random.rand() < 0.3:
        enhancer = ImageEnhance.Brightness(image)
        image = enhancer.enhance(1 + np.random.uniform(-0.12, 0.12))
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(1 + np.random.uniform(-0.12, 0.12))

        # 5. ✅ 核心优化：用病害特征模拟替代噪声（10%概率）
    if np.random.rand() < 0.1:  # 10%概率（比之前1.5%更合理）
        img_array = np.array(image, dtype=np.float32)
        h, w = img_array.shape[:2]

        # 模拟病害特征：小斑点（Early_blight → Late_blight）
        center = (np.random.randint(0, w), np.random.randint(0, h))
        radius = np.random.randint(1, 3)
        color = np.array([0.3, 0.1, 0.0])  # 褐色（病害特征色）

        # 在图像上添加小斑点
        for i in range(-radius, radius + 1):
            for j in range(-radius, radius + 1):
                if i * i + j * j < radius * radius:
                    y, x = center[0] + i, center[1] + j
                    if 0 <= y < h and 0 <= x < w:
                        img_array[y, x] = color * 255  # 转换为0-255
        image = Image.fromarray(img_array.astype(np.uint8))

    return image

# ==================== 优化器 ====================
class Adam:
    """Adam优化器"""

    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.00005):  # 降低权重衰减
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.iter = 0
        self.m = None
        self.v = None

    def update(self, params, grads):
        if self.m is None:
            self.m = {}
            self.v = {}
            for key, val in params.items():
                self.m[key] = np.zeros_like(val)
                self.v[key] = np.zeros_like(val)

        self.iter += 1
        lr_t = self.lr * np.sqrt(1.0 - self.beta2 ** self.iter) / (1.0 - self.beta1 ** self.iter)

        for key in params.keys():
            # 添加L2正则化
            if self.weight_decay > 0:
                grads[key] += self.weight_decay * params[key]

            self.m[key] += (1 - self.beta1) * (grads[key] - self.m[key])
            self.v[key] += (1 - self.beta2) * (grads[key] ** 2 - self.v[key])

            params[key] -= lr_t * self.m[key] / (np.sqrt(self.v[key]) + self.eps)


# ==================== 改进的CNN模型 ====================
class LeafCNN:

    def __init__(self, input_dim=(3, 128, 128), output_size=8, weight_init_std=0.01, dropout_ratio=0.3):
        self.input_dim = input_dim
        self.output_size = output_size
        self.dropout_ratio = dropout_ratio

        # 初始化参数
        self.params = {}
        self.__init_weights(weight_init_std)

        # 创建网络层
        self.layers = OrderedDict()
        self.__build_layers()

        self.last_layer = SoftmaxWithLoss()

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
        self.fc_input_size = 64 * 32 * 32

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

# ==================== 数据加载器 ====================
class LeafDataLoader:
    def __init__(self, data_dir, img_size=128, batch_size=32):
        self.data_dir = data_dir
        self.img_size = img_size
        self.batch_size = batch_size
        self.classes = []
        self.class_to_idx = {}
        self._detect_classes()

    def _detect_classes(self):
        train_path = os.path.join(self.data_dir, 'train')
        if os.path.exists(train_path):
            self.classes = sorted([d for d in os.listdir(train_path)
                                   if os.path.isdir(os.path.join(train_path, d))])
        else:
            self.classes = sorted([d for d in os.listdir(self.data_dir)
                                   if os.path.isdir(os.path.join(self.data_dir, d))])

        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        print(f"检测到 {len(self.classes)} 个类别")

    def _load_image(self, image_path, is_training=True):
        try:
            image = Image.open(image_path).convert('RGB')

            # 应用数据增强
            if is_training:
                image = apply_data_augmentation(image)

            image = image.resize((self.img_size, self.img_size))
            img_array = np.array(image, dtype=np.float32) / 255.0

            # 归一化
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            img_array = (img_array - mean) / std

            return np.transpose(img_array, (2, 0, 1))
        except Exception as e:
            print(f"图像加载错误: {e}")
            return None

    def get_batches(self, data_type='train', max_images=None):
        path = os.path.join(self.data_dir, data_type)
        if not os.path.exists(path):
            path = self.data_dir

        images = []
        labels = []

        for class_name in self.classes:
            class_path = os.path.join(path, class_name)
            if not os.path.exists(class_path):
                continue

            for img_file in os.listdir(class_path):
                if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(class_path, img_file)
                    images.append(img_path)
                    labels.append(self.class_to_idx[class_name])

                    if max_images and len(images) >= max_images:
                        break
            if max_images and len(images) >= max_images:
                break

        if len(images) == 0:
            yield np.array([]), np.array([])
            return

        # 打乱数据
        indices = np.random.permutation(len(images))

        for i in range(0, len(indices), self.batch_size):
            batch_indices = indices[i:i + self.batch_size]
            batch_images = []
            batch_labels = []

            for idx in batch_indices:
                is_training = (data_type == 'train')
                img = self._load_image(images[idx], is_training)
                if img is not None:
                    batch_images.append(img)
                    batch_labels.append(labels[idx])

            if batch_images:
                yield np.array(batch_images), np.array(batch_labels)

# ==================== 改进的训练器 ====================
class CNNTrainer:
    def __init__(self, model, data_loader, learning_rate=0.001, weight_decay=0.00005):
        self.model = model
        self.data_loader = data_loader
        self.optimizer = Adam(lr=learning_rate, weight_decay=weight_decay)

        # 添加早停机制参数
        self.patience = 5  # 连续多少个epoch验证准确率没有提升就停止
        self.best_val_acc = 0.0
        self.patience_counter = 0
        self.min_delta = 0.001  # 最小改进阈值

        # 训练历史
        self.history = {
            'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': [],
            'epoch_times': []
        }

    def train_epoch(self, epoch):
        total_loss = 0
        total_acc = 0
        batch_count = 0

        for batch_images, batch_labels in self.data_loader.get_batches('train', max_images=3000):
            if len(batch_images) == 0:
                continue

            # 计算梯度并更新
            grads = self.model.gradient(batch_images, batch_labels)
            self.optimizer.update(self.model.params, grads)

            # 计算损失和准确率
            loss = self.model.loss(batch_images, batch_labels, train_flg=True)
            acc = self.model.accuracy(batch_images, batch_labels)

            total_loss += loss
            total_acc += acc
            batch_count += 1

            # 进度显示
            if batch_count % 10 == 0:
                print(f'Epoch {epoch} | Batch {batch_count} | Loss: {loss:.4f} | Acc: {acc:.4f}')

        avg_loss = total_loss / batch_count if batch_count > 0 else 0
        avg_acc = total_acc / batch_count if batch_count > 0 else 0

        return avg_loss, avg_acc

    def validate(self):
        total_loss = 0
        total_acc = 0
        batch_count = 0

        for batch_images, batch_labels in self.data_loader.get_batches('val', max_images=900):
            if len(batch_images) == 0:
                continue

            loss = self.model.loss(batch_images, batch_labels, train_flg=False) # 验证时设为 False
            acc = self.model.accuracy(batch_images, batch_labels)

            total_loss += loss
            total_acc += acc
            batch_count += 1

        avg_loss = total_loss / batch_count if batch_count > 0 else 0
        avg_acc = total_acc / batch_count if batch_count > 0 else 0

        return avg_loss, avg_acc

    def train(self, max_epochs=20):
        print("开始训练")
        print(f"最大epoch数: {max_epochs}, 早停耐心值: {self.patience}")
        start_time = time.time()

        for epoch in range(1, max_epochs + 1):
            epoch_start = time.time()

            # 训练
            train_loss, train_acc = self.train_epoch(epoch)
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)

            # 验证
            val_loss, val_acc = self.validate()
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)

            epoch_time = time.time() - epoch_start
            self.history['epoch_times'].append(epoch_time)

            print(f'Epoch {epoch}: 训练损失={train_loss:.4f}, 训练准确率={train_acc:.4f}, '
                  f'验证损失={val_loss:.4f}, 验证准确率={val_acc:.4f}, 时间={epoch_time:.1f}s')

            # 计算过拟合差距
            overfit_gap = train_acc - val_acc
            print(f'过拟合差距: {overfit_gap:.4f}')

            # 早停机制判断
            if val_acc > self.best_val_acc + self.min_delta:
                self.best_val_acc = val_acc
                self.patience_counter = 0
                print(f"验证准确率提升至 {val_acc:.4f}, 重置早停计数器")
            else:
                self.patience_counter += 1
                print(f"验证准确率未提升，早停计数器: {self.patience_counter}/{self.patience}")

            # 学习率调整（每5个epoch衰减一次）
            if epoch % 5 == 0:
                self.optimizer.lr *= 0.8
                print(f'学习率调整为: {self.optimizer.lr:.6f}')

            # 早停检查
            if self.patience_counter >= self.patience:
                print(f"早停触发: 连续 {self.patience} 个epoch验证准确率未提升")
                print(f"最佳验证准确率: {self.best_val_acc:.4f}")
                break

            # 如果达到最大epoch数但未触发早停
            if epoch == max_epochs:
                print(f"达到最大epoch数 {max_epochs}")
                print(f"最终验证准确率: {val_acc:.4f}")

        total_time = time.time() - start_time
        print(f"训练完成, 总时间: {total_time / 60:.1f}分钟, 实际训练epoch数: {epoch}")

    def evaluate(self):
        val_loss, val_acc = self.validate()
        print(f"最终评估: 验证损失={val_loss:.4f}, 验证准确率={val_acc:.4f}")

        # 性能评估
        if val_acc < 0.6:
            print("警告: 验证准确率低于60%，模型性能不佳")
        elif val_acc < 0.8:
            print("注意: 验证准确率在60%-80%之间，模型性能一般")
        else:
            print("优秀: 验证准确率超过80%，模型性能良好")

        return val_loss, val_acc

    def plot_training_history(self):
        if not self.history['train_loss']:
            return

        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.plot(self.history['train_loss'], label='训练损失')
        plt.plot(self.history['val_loss'], label='验证损失')
        plt.title('模型损失')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)

        plt.subplot(1, 2, 2)
        plt.plot(self.history['train_acc'], label='训练准确率')
        plt.plot(self.history['val_acc'], label='验证准确率')
        plt.title('模型准确率')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
        plt.show()



# ==================== 主函数 ====================
def main():
        # 配置参数
        DATA_DIR = "./dataset/dataset1"
        IMG_SIZE = 128
        BATCH_SIZE = 32
        MAX_EPOCHS = 20  # 修改为最大epoch数
        LEARNING_RATE = 0.0005
        WEIGHT_DECAY = 0.00005
        DROPOUT_RATIO = 0.3
        MODEL_SAVE_PATH = "trained_leaf_cnn_model.pkl" # 定义模型保存路径

        print("树叶分类CNN模型")

        # 创建数据加载器
        data_loader = LeafDataLoader(
            data_dir=DATA_DIR,
            img_size=IMG_SIZE,
            batch_size=BATCH_SIZE
        )

        # 创建改进的模型
        model = LeafCNN(
            input_dim=(3, IMG_SIZE, IMG_SIZE),
            output_size=len(data_loader.classes),
            dropout_ratio=DROPOUT_RATIO
        )

        # 创建训练器
        trainer = CNNTrainer(
            model=model,
            data_loader=data_loader,
            learning_rate=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY
        )

        # 开始训练，传入最大epoch数
        trainer.train(max_epochs=MAX_EPOCHS)

        # 最终评估
        final_val_loss, final_val_acc = trainer.evaluate() # 获取最终指标

        # 绘制训练历史
        trainer.plot_training_history()

        # ========== 保存模型 ==========
        print(f"\n正在保存模型到 '{MODEL_SAVE_PATH}'...")
        try:
            with open(MODEL_SAVE_PATH, 'wb') as f:
                pickle.dump(model.params, f)  # 仅保存参数字典
            print("模型保存成功!")
            print(f"保存的模型包含以下参数键: {list(model.params.keys())}")
        except Exception as e:
            print(f"模型保存失败: {e}")


if __name__ == "__main__":
    main()


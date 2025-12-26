import sys, os
sys.path.append(os.pardir)  # 为了导入父目录的文件而进行的设定
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
from dataset1.common_functions.functions import *
from PIL import Image
from dataset1.common_functions.L_dada_augmentor import DataAugmentor


class LeafDataLoader:
    def __init__(self, data_dir, img_size=128, batch_size=32):
        self.data_dir = data_dir
        self.img_size = img_size
        self.batch_size = batch_size
        self.classes = []
        self.class_to_idx = {}
        self._detect_classes()
        self.data_augmentor = DataAugmentor()

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
        # 读取图像
        img = Image.open(image_path)
        # 确保图像是 RGB 格式
        if img.mode != 'RGB':
            img = img.convert('RGB')
        # 调整大小并转换为numpy数组
        img_array = np.array(img.resize((self.img_size, self.img_size)))
        # 使用数据增强器增强图像
        if is_training:
            img_array = self.data_augmentor.augment_single_image(img_array)

        # 归一化
        img_array = img_array.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img_array = (img_array - mean) / std
        # 转换为模型输入格式
        img_array = img_array.transpose(2, 0, 1)
        return img_array

    def get_batches(self, data_type='train', max_images=None, epoch=None):
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
                # 关键修正：max_images检查在内层循环内
                if max_images and len(images) >= max_images:
                    break
            # 确保在class循环内检查max_images
            if max_images and len(images) >= max_images:
                break
        if len(images) == 0:
            yield np.array([]), np.array([])
            return

        # 设置当前epoch用于数据增强
        if data_type == 'train' and epoch is not None:
            self.data_augmentor.set_epoch(epoch)

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
                # _load_image 已返回 (C, H, W) 形状的图像
                # np.array(batch_images) 会自然形成 (N, C, H, W) 形状
                batch_images = np.array(batch_images)  # Ensure shape is (N, C, H, W)
                yield batch_images, np.array(batch_labels)
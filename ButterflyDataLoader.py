import sys, os
sys.path.append(os.pardir)  # 为了导入父目录的文件而进行的设定
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
from dataset2.common_functions.functions import *
from PIL import Image


class ButterflyDataLoader:
    def __init__(self, data_dir, img_size=64, batch_size=32, val_split=0.2):
        self.data_dir = data_dir
        self.img_size = img_size
        self.batch_size = batch_size
        self.val_split = val_split
        self.classes = []
        self.class_to_idx = {}
        self.train_files = []
        self.val_files = []
        self._prepare_data()

    def _prepare_data(self):
        train_path = os.path.join(self.data_dir, 'train')
        if not os.path.exists(train_path):
            raise FileNotFoundError(f"训练目录不存在: {train_path}")

        all_files = []
        self.classes = sorted([d for d in os.listdir(train_path) if os.path.isdir(os.path.join(train_path, d))])
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        print(f"检测到 {len(self.classes)} 个类别: {self.classes[:3]}...")

        for class_name in self.classes:
            class_path = os.path.join(train_path, class_name)
            files = []
            for f in os.listdir(class_path):
                if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                    # 确保文件名是字符串
                    if isinstance(f, bytes):
                        f = f.decode('utf-8')
                    files.append(os.path.join(class_path, f))
            all_files.extend([(f, self.class_to_idx[class_name]) for f in files])

        # 打乱数据
        np.random.shuffle(all_files)
        n_val = int(len(all_files) * self.val_split)
        val_exists = os.path.exists(os.path.join(self.data_dir, 'val'))
        if val_exists:
            print("检测到 val 目录，使用现有验证集")
            self._load_existing_val()
            self.train_files = all_files
        else:
            print(f"未检测到 val 目录，自动划分 {self.val_split * 100:.0f}% 作为验证集")
            self.val_files = all_files[:n_val]
            self.train_files = all_files[n_val:]
        print(f"训练样本: {len(self.train_files)}, 验证样本: {len(self.val_files)}")

    def _load_existing_val(self):
        val_path = os.path.join(self.data_dir, 'val')
        self.val_files = []
        for class_name in self.classes:
            class_path = os.path.join(val_path, class_name)
            if not os.path.exists(class_path):
                continue
            files = [os.path.join(class_path, f) for f in os.listdir(class_path) if
                     f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            self.val_files.extend([(f, self.class_to_idx[class_name]) for f in files])

    def _load_image(self, image_path, is_training=True):
        # 确保 image_path 是字符串
        if isinstance(image_path, bytes):
            image_path = image_path.decode('utf-8')
        elif not isinstance(image_path, str):
            print(f"错误: image_path 类型错误 {type(image_path)}, 期望字符串")
            return None

        # 修复路径格式
        image_path = os.path.normpath(image_path).replace('\\', '/')
        image_path = os.path.abspath(image_path)

        try:
            image = Image.open(image_path).convert('RGB')
            if is_training and hasattr(self, 'data_augmentor'):
                image = self.data_augmentor.augment_single_image(np.array(image))
                image = Image.fromarray(image.astype('uint8'))
            if image.mode != 'RGB':
                image = image.convert('RGB')
            image = image.resize((self.img_size, self.img_size))
            img_array = np.array(image, dtype=np.float32) / 255.0
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            img_array = (img_array - mean) / std
            return np.transpose(img_array, (2, 0, 1))
        except Exception as e:
            print(f"图像加载错误 {image_path}: {e}")
            return None

    def get_batches(self, data_type='train'):
        if data_type == 'train':
            files = self.train_files
            is_training = True
        else:
            files = self.val_files
            is_training = False

        if len(files) == 0:
            yield np.array([]), np.array([])
            return

        indices = np.random.permutation(len(files)) if data_type == 'train' else np.arange(len(files))
        for i in range(0, len(indices), self.batch_size):
            batch_indices = indices[i:i + self.batch_size]
            batch_images = []
            batch_labels = []

            for idx in batch_indices:
                # 确保我们获取的是文件路径，不是Image对象
                if isinstance(files[idx], tuple):
                    img_path, label = files[idx]
                else:
                    # 如果是单个元素，跳过这个样本
                    print(f"警告: 文件列表格式错误，跳过样本 {idx}")
                    continue

                # 修复：检查并转换类型
                if isinstance(img_path, bytes):
                    img_path = img_path.decode('utf-8')
                elif isinstance(img_path, Image.Image):
                    # 如果是Image对象，跳过
                    print(f"警告: 从文件列表中获取到Image对象，跳过: {idx}")
                    continue
                elif not isinstance(img_path, str):
                    print(f"警告: 无效的img_path类型 {type(img_path)}, 期望字符串")
                    continue

                img = self._load_image(img_path, is_training)
                if img is not None:
                    batch_images.append(img)
                    batch_labels.append(label)

            if batch_images:
                batch_labels = np.array(batch_labels)
                yield np.array(batch_images), batch_labels

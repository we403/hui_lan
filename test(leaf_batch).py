import os
import numpy as np
import pickle
from PIL import Image
import time
import argparse

# 加载模型（路径已修正为相对路径）
MODEL_PATH = "save/leaf_model_80%.pkl"
from dataset1.common_functions.functions import softmax
from dataset1.leaf_model.LeafCNN import LeafCNN
from dataset1.leaf_model.LeafDataLoader import LeafDataLoader
from dataset1.leaf_model.CNNTrainer import CNNTrainer
from dataset1.common_functions.SoftmaxWithLoss import SoftmaxWithLoss
from dataset1.common_functions.Affine import Affine
from dataset1.common_functions.Convolution import Convolution
from dataset1.common_functions.BatchNormalization import BatchNormalization
from dataset1.common_functions.Relu import Relu
from dataset1.common_functions.Pooling import Pooling
from dataset1.common_functions.Dropout import Dropout


try:
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    print("✅ 模型加载成功！")
except Exception as e:
    print(f"❌ 模型加载失败: {e}")
    model = None

# 类别名称（与训练一致）
class_names = [
    "Early_blight","Healthy",  "Late_blight", "Leaf Miner", "Magnesium Deficiency",
    "Nitrogen Deficiency", "Pottassium Deficiency", "Spotted Wilt Virus"
]


def softmax(x):
    """批量版softmax"""
    x = x - x.max(axis=1, keepdims=True)
    exp_x = np.exp(x)
    return exp_x / exp_x.sum(axis=1, keepdims=True)


def main(test_dir):
    if model is None:
        print("❌ 无法进行预测（模型加载失败）")
        return

    # 确保使用指定的路径格式
    if not os.path.exists(test_dir):
        print(f"❌ 测试目录不存在: {test_dir}")
        return

    # === 关键：收集所有图片路径（自动识别类别文件夹）===
    all_image_paths = []
    all_true_labels = []

    # 遍历test目录下的所有类别文件夹
    for class_folder in os.listdir(test_dir):
        class_path = os.path.join(test_dir, class_folder)
        if not os.path.isdir(class_path):
            continue

        # 收集该类别下所有图片
        for img_file in os.listdir(class_path):
            if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                all_image_paths.append(os.path.join(class_path, img_file))
                all_true_labels.append(class_folder)

    if not all_image_paths:
        print("❌ 测试集为空！")
        return

    print(f"正在加载 {len(all_image_paths)} 张测试图片...")
    start_time = time.time()

    # === 批量预处理（一次性加载所有图片）===
    batch_images = []
    for img_path in all_image_paths:
        img = Image.open(img_path).convert('RGB')
        img = img.resize((64, 64))
        img_array = np.array(img, dtype=np.float32) / 255.0

        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img_array = (img_array - mean) / std

        img_array = np.transpose(img_array, (2, 0, 1))
        batch_images.append(img_array)

    batch_images = np.array(batch_images)  # [N, 3, 128, 128]
    print(f"✅ 图片加载完成! 耗时: {time.time() - start_time:.2f}秒")

    # === 一次性预测 ===
    print("正在批量预测...")
    start_pred = time.time()
    predictions = model.predict(batch_images, train_flg=False)
    predictions = softmax(predictions)
    predicted_indices = np.argmax(predictions, axis=1)

    # === 准确率计算 ===
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}
    true_indices = np.array([class_to_idx[label] for label in all_true_labels])

    correct = np.sum(predicted_indices == true_indices)
    accuracy = correct / len(all_image_paths) * 100
    total_time = time.time() - start_time

    print(f"测试完成! 总图片: {len(all_image_paths)}")
    print(f"正确预测: {correct} 张 | 准确率: {accuracy:.2f}%")
    print(f"总耗时: {total_time:.2f}秒")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='树叶分类批量测试脚本')
    # 关键修改：统一路径为 competition/dataset/test
    parser.add_argument('--test_dir', type=str, default='competition/dataset/test',
                        help='测试数据集目录 (默认: competition/dataset/test)')
    args = parser.parse_args()

    main(args.test_dir)
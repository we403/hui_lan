# main_butterfly.py
import sys, os
sys.path.append(os.pardir)
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
from butterfly_model.ButterflyCNN import ButterflyCNN
from butterfly_model.ButterflyDataLoader import ButterflyDataLoader
from butterfly_model.CNNTrainer import CNNTrainer
from dataset2.common_functions.B_data_augmentor import DataAugmentor

def main():
    # 配置参数
    DATA_DIR = "../work/dataset/dataset2"
    IMG_SIZE = 64
    BATCH_SIZE = 16
    MAX_EPOCHS = 40
    LEARNING_RATE = 0.0001
    WEIGHT_DECAY = 0.0001
    DROPOUT_RATIO = 0.3
    BEST_MODEL_SAVE_PATH = "./best_butterfly_model.pkl"

    print("蝴蝶图像分类 - 增强版CNN训练脚本")
    print("=" * 50)

    # 创建数据加载器
    data_loader = ButterflyDataLoader(
        data_dir=DATA_DIR,
        img_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        val_split=0.2
    )

    # 创建数据增强器
    data_loader.data_augmentor = DataAugmentor()

    # 创建模型
    model = ButterflyCNN(
        input_dim=(3, IMG_SIZE, IMG_SIZE),
        output_size=len(data_loader.classes),
        dropout_ratio=DROPOUT_RATIO
    )

    # 创建训练器
    trainer = CNNTrainer(
        model=model,
        data_loader=data_loader,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        best_model_save_path=BEST_MODEL_SAVE_PATH
    )

    # 开始训练
    trainer.train(max_epochs=MAX_EPOCHS)

    # 确认最终模型已保存
    print(f"\n✅ 最终模型已成功保存至: {BEST_MODEL_SAVE_PATH}")

    # 最终评估
    final_val_loss, final_val_acc = trainer.evaluate()

    # 绘制训练历史
    trainer.plot_training_history()

    # 确认模型保存路径
    print(f"\n📊 模型文件已保存到: {BEST_MODEL_SAVE_PATH}")

if __name__ == "__main__":
    main()
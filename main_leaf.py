import sys, os
sys.path.append(os.pardir)  # 为了导入父目录的文件而进行的设定
import pickle# 用于保存模型
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
from leaf_model.LeafCNN import LeafCNN
from leaf_model.LeafDataLoader import LeafDataLoader
from leaf_model.CNNTrainer import CNNTrainer


def main():
    # 配置参数
    DATA_DIR = "../work/dataset/dataset1"
    IMG_SIZE = 64
    BATCH_SIZE = 32
    MAX_EPOCHS = 80
    LEARNING_RATE = 0.00005
    WEIGHT_DECAY = 0.00005
    DROPOUT_RATIO = 0.3
    LABEL_SMOOTHING = 0.5
    MODEL_SAVE_PATH = "./trained_leaf_model2.pkl"
    BEST_MODEL_SAVE_PATH = "best_leaf_model_params最新.pkl"

    print("树叶分类CNN模型")

    # 创建数据加载器
    data_loader = LeafDataLoader(
        data_dir=DATA_DIR,
        img_size=IMG_SIZE,
        batch_size=BATCH_SIZE
    )

    # 创建模型
    model = LeafCNN(
        input_dim=(3, IMG_SIZE, IMG_SIZE),
        output_size=len(data_loader.classes),
        dropout_ratio=DROPOUT_RATIO,
        label_smoothing=LABEL_SMOOTHING
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

    # 最终评估
    final_val_loss, final_val_acc = trainer.evaluate()

    # 获取最终指标
    trainer.plot_training_history()

    # 保存最终模型
    print(f"\n正在保存最终训练完成的模型到 '{MODEL_SAVE_PATH}'...")
    try:
        with open(MODEL_SAVE_PATH, 'wb') as f:
            pickle.dump(trainer.model, f)
        print("最终模型保存成功!")
    except Exception as e:
        print(f"最终模型保存失败: {e}")



if __name__ == "__main__":
    main()
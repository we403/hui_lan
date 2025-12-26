# CNNTrainer.py
import numpy as np
import pickle
import time
import sys, os

sys.path.append(os.pardir)
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
from dataset2.common_functions.optimizer import Adam


class CNNTrainer:
    def __init__(self, model, data_loader, learning_rate=0.001, weight_decay=0.00005,
                 best_model_save_path="butterfly_model（83%）.pkl"):
        self.model = model
        self.data_loader = data_loader
        self.optimizer = Adam(lr=learning_rate, weight_decay=weight_decay)
        # 添加早停机制参数
        self.patience = 4
        self.best_val_acc = 0.0
        self.patience_counter = 0
        self.min_delta = 0.001
        self.best_model_instance = None
        self.best_model_save_path = best_model_save_path
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'epoch_times': []
        }

    def train_epoch(self, epoch):
        # 设置数据增强器的当前epoch
        self.data_loader.data_augmentor.set_epoch(epoch)
        total_loss = 0
        total_acc = 0
        batch_count = 0
        for batch_images, batch_labels in self.data_loader.get_batches('train'):
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
        for batch_images, batch_labels in self.data_loader.get_batches('val'):
            if len(batch_images) == 0:
                continue
            loss = self.model.loss(batch_images, batch_labels, train_flg=False)
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
            # 保存最佳模型（使用深拷贝，确保保存的是整个模型实例）
            if val_acc > self.best_val_acc + self.min_delta:
                self.best_val_acc = val_acc
                self.patience_counter = 0
                # 保存整个模型实例
                import copy
                self.best_model_instance = copy.deepcopy(self.model)
                print(f"验证准确率提升至 {val_acc:.4f}, 重置早停计数器")
                print(f"正在保存当前最佳模型实例到 '{self.best_model_save_path}'...")
                try:
                    with open(self.best_model_save_path, 'wb') as f:
                        pickle.dump(self.best_model_instance, f)
                    print("最佳模型实例保存成功!")
                except Exception as e:
                    print(f"最佳模型实例保存失败: {e}")
            else:
                self.patience_counter += 1
                print(f"验证准确率未提升，早停计数器: {self.patience_counter}/{self.patience}")
            # 学习率调整（每5个epoch衰减一次）
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
        # 保存最终模型实例
        print(f"\n正在保存最终训练完成的模型实例到 '{self.best_model_save_path}'...")
        try:
            with open(self.best_model_save_path, 'wb') as f:
                pickle.dump(self.model, f)
            print("最终模型实例保存成功!")
        except Exception as e:
            print(f"最终模型实例保存失败: {e}")
        # 保存训练历史
        try:
            with open('training_history.pkl', 'wb') as f:
                pickle.dump(self.history, f)
            print("训练历史保存成功!")
        except Exception as e:
            print(f"训练历史保存失败: {e}")

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
        plt.title('损失曲线')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.subplot(1, 2, 2)
        plt.plot(self.history['train_acc'], label='训练准确率')
        plt.plot(self.history['val_acc'], label='验证准确率')
        plt.title('准确率曲线')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('butterfly_training_history.png', dpi=300, bbox_inches='tight')
        plt.show()
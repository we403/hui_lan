import gradio as gr
import pickle
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
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

# 加载模型
MODEL_PATH = "save/leaf_model_80%.pkl"  # 确保路径正确

try:
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    print("模型加载成功！")
    # 从模型中获取类别数量
    output_size = model.output_size
    print(f"检测到 {output_size} 个类别")

    # 手动指定类别名称（必须与训练时的顺序完全一致）
    class_names = [
        "Early_blight",
        "Healthy",
        "Late_blight",
        "Leaf_Miner",
        "Magnesium_Deficiency",
        "Nitrogen_Deficiency",
        "Potassium_Deficiency",
        "Spotted_Wilt_Virus"
    ]

    # 确保类别名称数量与模型输出匹配
    if len(class_names) != output_size:
        print(f"警告: 类别名称数量({len(class_names)})与模型输出数量({output_size})不匹配!")
        # 回退到默认名称
        class_names = [f"类别 {i + 1}" for i in range(output_size)]

except Exception as e:
    print(f"加载模型时出错: {e}")
    model = None
    class_names = ['Early_blight',
                   'Healthy',
                   'Late_blight',
                   'Leaf Miner',
                   'Magnesium Deficiency',
                   'Nitrogen Deficiency',
                   'Pottassium Deficiency',
                   'Spotted Wilt Virus']


def predict_image(image):
    if model is None:
        return None, "模型加载失败，请检查模型路径"
    img = Image.fromarray(image).convert('RGB')
    img = img.resize((64, 64))
    img_array = np.array(img, dtype=np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_array = (img_array - mean) / std

    # 转换为模型输入格式
    img_array = np.transpose(img_array, (2, 0, 1))
    img_array = np.expand_dims(img_array, axis=0)

    # 预测
    prediction = model.predict(img_array, train_flg=False)

    # 关键修复：应用softmax得到概率分布
    prediction = softmax(prediction)

    # 获取预测结果和置信度
    predicted_class = np.argmax(prediction)
    confidence = prediction[0][predicted_class] * 100

    # 创建概率条形图
    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(class_names, prediction[0] * 100)
    # 添加数值标签
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}%', xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')
    ax.set_ylabel('置信度 (%)')
    ax.set_title('类别置信度分布')
    plt.xticks(rotation=45)
    plt.tight_layout()

    # 保存图表到临时文件
    img_path = "save/temp_confidence.png"
    plt.savefig(img_path)
    plt.close()

    # 返回预测结果和置信度图
    return f"预测结果: {class_names[predicted_class]} (置信度: {confidence:.2f}%)", img_path


# 创建Gradio界面
if model is not None:
    interface = gr.Interface(
        fn=predict_image,
        inputs=gr.Image(type="numpy", label="上传树叶图片", height=300),
        outputs=[
            gr.Textbox(label="预测结果", lines=2, max_lines=2),
            gr.Image(label="置信度分布图", height=400)
        ],
        title="树叶病害分类系统",
        description="上传一张树叶图片，系统将预测其病害类别并显示置信度分布。"
    )
    print("预测界面已启动，请在浏览器中打开链接。")
    interface.launch()
else:
    print("模型加载失败，无法启动预测界面。")

# 第11章：AI之"眼"—— 卷积神经网络 (CNN)

## 🎯 学习目标

通过本章学习，你将能够：
- 理解CNN的核心概念和工作原理
- 掌握卷积层、池化层、全连接层的作用
- 完成MNIST手写数字识别完整项目
- 学会使用TensorFlow/Keras构建深度学习模型

## 📋 章节内容

### 11.1 计算机如何"看"图片：像素矩阵
### 11.2 核心武器：卷积层与池化层
### 11.3 CNN架构演进：从LeNet到AlexNet
### 11.4 代码实战：MNIST手写数字识别

## 🖼️ 实战项目：手写数字识别

### 项目背景
使用MNIST数据集训练CNN模型，识别0-9的手写数字图片。这是深度学习的"Hello World"项目。

### 项目流程
1. **数据加载**：加载MNIST数据集
2. **数据预处理**：归一化和数据增强
3. **模型构建**：设计CNN架构
4. **模型训练**：训练和验证模型
5. **性能评估**：测试集评估和混淆矩阵
6. **可视化**：训练过程和预测结果可视化

## 📁 文件结构

```
chapter11-cnn/
├── README.md                    # 本文件
├── mnist_cnn.ipynb             # 完整CNN项目notebook
├── cnn_tutorial.ipynb          # CNN概念教程
├── src/
│   ├── cnn_model.py            # CNN模型实现
│   ├── data_loader.py          # 数据加载和预处理
│   ├── train.py                # 训练脚本
│   ├── evaluate.py             # 评估脚本
│   └── visualization.py        # 可视化工具
├── models/
│   └── mnist_cnn_model.h5      # 训练好的模型
└── results/
    ├── training_history.png    # 训练历史
    ├── confusion_matrix.png    # 混淆矩阵
    └── sample_predictions.png  # 预测示例
```

## 🚀 快速开始

1. 运行完整CNN项目：
```bash
jupyter notebook mnist_cnn.ipynb
```

2. 或直接运行Python脚本：
```bash
python src/train.py
```

3. 模型评估：
```bash
python src/evaluate.py
```

## 🧠 核心概念

- **卷积层**：特征提取器，检测局部模式
- **池化层**：降采样，减少参数和计算量
- **激活函数**：引入非线性，增强模型表达能力
- **反向传播**：梯度计算和参数更新

## 📊 期待结果

- **准确率**：>99% (测试集)
- **训练时间**：约10-15分钟 (CPU)
- **模型大小**：<5MB
- **推理速度**：毫秒级

---

**让AI拥有"眼睛"** 👁️ 从像素到智能的神奇转换！

# 第12章：AI之"耳"与"口"—— 循环神经网络 (RNN)

## 🎯 学习目标

通过本章学习，你将能够：
- 理解RNN的核心概念和工作原理
- 掌握LSTM网络的结构和优势
- 完成电影评论情感分析项目
- 学会处理序列数据和自然语言

## 📋 章节内容

### 12.1 处理序列数据：为什么普通神经网络不够用？
### 12.2 拥有"记忆"的网络：RNN的基本结构
### 12.3 解决长期依赖问题：长短期记忆网络(LSTM)
### 12.4 应用场景：文本生成、情感分析
### 12.5 代码实战：构建情感分析模型

## 💬 实战项目：电影评论情感分析

### 项目背景
分析电影评论文本，判断评论情感是正面还是负面。这是自然语言处理的经典任务。

### 项目流程
1. **数据准备**：加载和预处理文本数据
2. **文本向量化**：将文本转换为数字表示
3. **模型构建**：设计LSTM网络架构
4. **模型训练**：训练情感分类模型
5. **性能评估**：评估模型分类性能
6. **实时预测**：对新评论进行情感分析

## 📁 文件结构

```
chapter12-rnn/
├── README.md                    # 本文件
├── sentiment_analysis.ipynb     # 完整情感分析项目
├── rnn_tutorial.ipynb          # RNN/LSTM教程
├── src/
│   ├── rnn_model.py            # RNN模型实现
│   ├── text_preprocessor.py    # 文本预处理
│   ├── data_loader.py          # 数据加载器
│   ├── train.py                # 训练脚本
│   └── predict.py              # 预测脚本
├── data/
│   ├── movie_reviews.csv       # 电影评论数据
│   └── word_embeddings.txt     # 词向量文件
├── models/
│   └── sentiment_lstm.h5       # 训练好的模型
└── results/
    ├── training_history.png    # 训练历史
    ├── confusion_matrix.png    # 混淆矩阵
    └── word_cloud.png          # 词云图
```

## 🚀 快速开始

1. 运行完整项目：
```bash
jupyter notebook sentiment_analysis.ipynb
```

2. 或分步运行：
```bash
python src/train.py
python src/predict.py
```

## 🧠 核心概念

- **RNN**：能处理序列数据的神经网络
- **LSTM**：解决梯度消失问题的特殊RNN
- **词嵌入**：将词语转换为向量表示
- **序列建模**：理解上下文关系

## 📊 期待结果

- **准确率**：>85% (测试集)
- **训练时间**：约20-30分钟
- **实时预测**：支持任意文本输入
- **可解释性**：词语重要性分析

---

**让AI理解人类的情感** 💭 从文字到情感的智能桥梁！

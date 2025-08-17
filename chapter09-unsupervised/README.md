# 第9章：发现隐藏的结构 —— 无监督学习

## 🎯 学习目标

通过本章学习，你将能够：
- 理解无监督学习的核心思想
- 掌握K-Means聚类算法的原理和实现
- 完成一个完整的客户分群项目
- 学会用手肘法确定最佳聚类数量
- 将聚类结果转化为商业洞察

## 📋 章节内容

### 9.1 "物以类聚"：K-均值聚类算法
### 9.2 "化繁为简"：主成分分析(PCA)
### 9.3 完整项目：客户分群分析

## 🎮 实战项目：商场客户分群

想象一下，你是一家购物中心的市场分析师。你需要从客户数据中找出不同类型的客户群体，以便针对性地设计营销活动。

### 项目流程
1. **数据探索**：了解客户收入和消费行为分布
2. **确定K值**：使用手肘法找到最佳聚类数量
3. **模型训练**：应用K-Means算法进行客户分群
4. **结果可视化**：直观展示分群结果
5. **商业解释**：为每个客户群体制定营销策略

## 📁 文件结构

```
chapter09-unsupervised/
├── README.md                    # 本文件
├── customer_segmentation.ipynb # 完整客户分群项目
├── kmeans_tutorial.ipynb       # K-Means算法教程
├── src/
│   ├── kmeans_model.py         # K-Means模型实现
│   ├── data_generator.py       # 数据生成器
│   ├── visualization.py        # 可视化工具
│   └── utils.py               # 工具函数
├── data/
│   ├── mall_customers.csv      # 商场客户数据
│   └── sample_data.py         # 示例数据生成
└── results/
    ├── cluster_analysis.png    # 聚类结果图
    └── elbow_method.png       # 手肘法图
```

## 🚀 快速开始

1. 安装依赖：
```bash
pip install -r ../requirements.txt
```

2. 运行客户分群项目：
```bash
jupyter notebook customer_segmentation.ipynb
```

## 💡 关键概念

- **无监督学习**：没有标准答案的学习方式
- **聚类**：将相似的数据点分为一组
- **K-Means**：基于距离的聚类算法
- **手肘法**：确定最佳聚类数量的方法
- **商业洞察**：将技术结果转化为业务价值

---

**发现数据中的隐藏模式** 🔍 让数据自己告诉我们答案！

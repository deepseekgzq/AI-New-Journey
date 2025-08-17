# 第7章：预测未来 —— 回归模型

## 🎯 学习目标

通过本章学习，你将能够：
- 理解线性回归的核心原理
- 掌握损失函数和梯度下降算法
- 完成波士顿房价预测完整项目
- 学会模型评估和结果解释

## 📋 章节内容

### 7.1 从一条直线开始：线性回归
### 7.2 如何评价"预测得准不准"：损失函数
### 7.3 寻找最优解的"下山"之旅：梯度下降法
### 7.4 代码实战：波士顿房价预测

## 🏠 实战项目：波士顿房价预测

### 项目背景
使用波士顿房屋数据集，基于房屋特征（如房间数、犯罪率、距离等）预测房价。

### 项目流程
1. **数据探索**：了解特征分布和相关性
2. **特征工程**：数据预处理和特征选择
3. **模型训练**：线性回归模型训练
4. **模型评估**：使用多种指标评估性能
5. **结果可视化**：预测效果可视化展示

## 📁 文件结构

```
chapter07-regression/
├── README.md                    # 本文件
├── boston_housing.ipynb         # 完整房价预测项目
├── linear_regression_tutorial.ipynb # 线性回归教程
├── src/
│   ├── linear_regression.py     # 线性回归实现
│   ├── gradient_descent.py      # 梯度下降算法
│   ├── data_preprocessing.py    # 数据预处理
│   └── visualization.py         # 可视化工具
├── data/
│   ├── boston_housing.csv       # 波士顿房价数据
│   └── data_description.md      # 数据说明
└── models/
    └── trained_model.pkl        # 训练好的模型
```

## 🚀 快速开始

1. 运行房价预测项目：
```bash
jupyter notebook boston_housing.ipynb
```

2. 或直接运行Python脚本：
```bash
python src/linear_regression.py
```

## 📊 核心概念

- **线性回归**：建立特征与目标的线性关系
- **损失函数**：衡量预测误差的函数（MSE）
- **梯度下降**：寻找最优参数的优化算法
- **模型评估**：R²、RMSE、MAE等指标

---

**预测未来从一条直线开始** 📈 让数据告诉我们答案！

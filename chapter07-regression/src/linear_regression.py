"""
第7章：线性回归完整实现
波士顿房价预测项目
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

class BostonHousingPredictor:
    """波士顿房价预测器"""
    
    def __init__(self):
        """初始化"""
        self.data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = None
        self.target_name = 'MEDV'
        
    def load_data(self):
        """加载波士顿房价数据"""
        print("📂 加载波士顿房价数据...")
        
        try:
            # 尝试加载sklearn内置数据集
            boston = load_boston()
            self.data = pd.DataFrame(boston.data, columns=boston.feature_names)
            self.data[self.target_name] = boston.target
            self.feature_names = boston.feature_names
            
            print(f"✅ 数据加载成功！")
            print(f"   - 样本数量: {len(self.data)}")
            print(f"   - 特征数量: {len(self.feature_names)}")
            print(f"   - 目标变量: {self.target_name} (房价中位数)")
            
        except ImportError:
            # 如果sklearn版本较新，可能不包含boston数据集
            print("⚠️ sklearn内置数据集不可用，生成模拟数据...")
            self.data = self._generate_sample_data()
            
        return self.data
    
    def _generate_sample_data(self):
        """生成模拟的房价数据"""
        np.random.seed(42)
        n_samples = 506
        
        # 生成特征数据
        features = {
            'CRIM': np.random.exponential(3.6, n_samples),  # 犯罪率
            'ZN': np.random.exponential(11.4, n_samples),   # 住宅用地比例
            'INDUS': np.random.normal(11.1, 6.9, n_samples),  # 非零售商用地比例
            'CHAS': np.random.binomial(1, 0.07, n_samples),   # 河边位置
            'NOX': np.random.normal(0.55, 0.12, n_samples),   # 氮氧化物浓度
            'RM': np.random.normal(6.3, 0.7, n_samples),      # 房间数
            'AGE': np.random.normal(68.6, 28.1, n_samples),   # 房屋年龄
            'DIS': np.random.exponential(3.8, n_samples),     # 距离中心距离
            'RAD': np.random.choice(range(1, 25), n_samples), # 交通便利性
            'TAX': np.random.normal(408, 169, n_samples),     # 税率
            'PTRATIO': np.random.normal(18.5, 2.2, n_samples), # 师生比
            'B': np.random.normal(356.7, 91.3, n_samples),    # 黑人比例
            'LSTAT': np.random.normal(12.7, 7.1, n_samples)   # 低收入人口比例
        }
        
        data = pd.DataFrame(features)
        self.feature_names = list(features.keys())
        
        # 基于特征生成目标变量（房价）
        price = (50 - 
                data['CRIM'] * 0.1 +
                data['RM'] * 8 -
                data['LSTAT'] * 0.5 -
                data['NOX'] * 15 +
                data['DIS'] * 0.8 +
                np.random.normal(0, 3, n_samples))
        
        data[self.target_name] = np.clip(price, 5, 50)  # 限制在合理范围
        
        return data
    
    def explore_data(self):
        """数据探索性分析"""
        print("\n🔍 数据探索分析...")
        
        # 基本信息
        print("\n📊 数据基本信息:")
        print(self.data.describe())
        
        # 检查缺失值
        missing_values = self.data.isnull().sum()
        if missing_values.sum() > 0:
            print("\n⚠️ 缺失值情况:")
            print(missing_values[missing_values > 0])
        else:
            print("\n✅ 无缺失值")
        
        # 创建可视化
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 1. 目标变量分布
        axes[0, 0].hist(self.data[self.target_name], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0, 0].set_title('房价分布', fontweight='bold')
        axes[0, 0].set_xlabel('房价中位数 (千美元)')
        axes[0, 0].set_ylabel('频数')
        
        # 2. 相关性热力图
        correlation_matrix = self.data.corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                   ax=axes[0, 1], fmt='.2f', square=True)
        axes[0, 1].set_title('特征相关性热力图', fontweight='bold')
        
        # 3. 房间数vs房价
        axes[0, 2].scatter(self.data['RM'], self.data[self.target_name], alpha=0.6)
        axes[0, 2].set_title('房间数 vs 房价', fontweight='bold')
        axes[0, 2].set_xlabel('平均房间数')
        axes[0, 2].set_ylabel('房价中位数')
        
        # 4. 犯罪率vs房价
        axes[1, 0].scatter(self.data['CRIM'], self.data[self.target_name], alpha=0.6, color='red')
        axes[1, 0].set_title('犯罪率 vs 房价', fontweight='bold')
        axes[1, 0].set_xlabel('犯罪率')
        axes[1, 0].set_ylabel('房价中位数')
        
        # 5. 与房价相关性最高的特征
        correlations = correlation_matrix[self.target_name].abs().sort_values(ascending=False)[1:6]
        axes[1, 1].barh(range(len(correlations)), correlations.values)
        axes[1, 1].set_yticks(range(len(correlations)))
        axes[1, 1].set_yticklabels(correlations.index)
        axes[1, 1].set_title('与房价相关性最高的5个特征', fontweight='bold')
        axes[1, 1].set_xlabel('相关系数(绝对值)')
        
        # 6. 房价箱线图
        axes[1, 2].boxplot(self.data[self.target_name])
        axes[1, 2].set_title('房价分布箱线图', fontweight='bold')
        axes[1, 2].set_ylabel('房价中位数')
        
        plt.tight_layout()
        plt.show()
        
        print("\n💡 数据洞察:")
        top_corr = correlation_matrix[self.target_name].abs().sort_values(ascending=False)[1:4]
        for feature, corr in top_corr.items():
            print(f"  • {feature}: 与房价相关性为 {corr:.3f}")
    
    def prepare_data(self, test_size=0.2, random_state=42):
        """准备训练和测试数据"""
        print("\n🔧 准备训练和测试数据...")
        
        # 分离特征和目标变量
        X = self.data[self.feature_names]
        y = self.data[self.target_name]
        
        # 划分训练集和测试集
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        # 特征标准化
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        print(f"✅ 数据准备完成!")
        print(f"   - 训练集大小: {len(self.X_train)}")
        print(f"   - 测试集大小: {len(self.X_test)}")
        print(f"   - 特征已标准化")
        
        return self.X_train_scaled, self.X_test_scaled, self.y_train, self.y_test
    
    def train_model(self):
        """训练线性回归模型"""
        print("\n🤖 训练线性回归模型...")
        
        # 创建并训练模型
        self.model = LinearRegression()
        self.model.fit(self.X_train_scaled, self.y_train)
        
        print("✅ 模型训练完成!")
        
        # 显示模型参数
        print(f"   - 截距: {self.model.intercept_:.4f}")
        print("   - 特征系数:")
        for feature, coef in zip(self.feature_names, self.model.coef_):
            print(f"     {feature}: {coef:.4f}")
        
        return self.model
    
    def evaluate_model(self):
        """评估模型性能"""
        print("\n📊 模型性能评估...")
        
        # 预测
        y_train_pred = self.model.predict(self.X_train_scaled)
        y_test_pred = self.model.predict(self.X_test_scaled)
        
        # 计算评估指标
        train_r2 = r2_score(self.y_train, y_train_pred)
        test_r2 = r2_score(self.y_test, y_test_pred)
        
        train_rmse = np.sqrt(mean_squared_error(self.y_train, y_train_pred))
        test_rmse = np.sqrt(mean_squared_error(self.y_test, y_test_pred))
        
        train_mae = mean_absolute_error(self.y_train, y_train_pred)
        test_mae = mean_absolute_error(self.y_test, y_test_pred)
        
        print("📈 评估结果:")
        print(f"训练集 R² 分数: {train_r2:.4f}")
        print(f"测试集 R² 分数: {test_r2:.4f}")
        print(f"训练集 RMSE: {train_rmse:.4f}")
        print(f"测试集 RMSE: {test_rmse:.4f}")
        print(f"训练集 MAE: {train_mae:.4f}")
        print(f"测试集 MAE: {test_mae:.4f}")
        
        # 过拟合检查
        if train_r2 - test_r2 > 0.1:
            print("⚠️ 可能存在过拟合")
        else:
            print("✅ 模型泛化性能良好")
        
        return {
            'train_r2': train_r2, 'test_r2': test_r2,
            'train_rmse': train_rmse, 'test_rmse': test_rmse,
            'train_mae': train_mae, 'test_mae': test_mae,
            'y_test_pred': y_test_pred
        }
    
    def visualize_results(self, evaluation_results):
        """可视化模型结果"""
        print("\n📈 可视化预测结果...")
        
        y_test_pred = evaluation_results['y_test_pred']
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. 预测值vs真实值散点图
        axes[0, 0].scatter(self.y_test, y_test_pred, alpha=0.6)
        axes[0, 0].plot([self.y_test.min(), self.y_test.max()], 
                       [self.y_test.min(), self.y_test.max()], 'r--', lw=2)
        axes[0, 0].set_xlabel('真实房价')
        axes[0, 0].set_ylabel('预测房价')
        axes[0, 0].set_title('预测值 vs 真实值', fontweight='bold')
        axes[0, 0].text(0.05, 0.95, f'R² = {evaluation_results["test_r2"]:.3f}', 
                       transform=axes[0, 0].transAxes, bbox=dict(boxstyle='round', facecolor='white'))
        
        # 2. 残差图
        residuals = self.y_test - y_test_pred
        axes[0, 1].scatter(y_test_pred, residuals, alpha=0.6)
        axes[0, 1].axhline(y=0, color='r', linestyle='--')
        axes[0, 1].set_xlabel('预测房价')
        axes[0, 1].set_ylabel('残差')
        axes[0, 1].set_title('残差分布图', fontweight='bold')
        
        # 3. 特征重要性
        feature_importance = abs(self.model.coef_)
        feature_names_sorted = [name for _, name in sorted(zip(feature_importance, self.feature_names), reverse=True)]
        importance_sorted = sorted(feature_importance, reverse=True)
        
        axes[1, 0].barh(range(len(feature_names_sorted)), importance_sorted)
        axes[1, 0].set_yticks(range(len(feature_names_sorted)))
        axes[1, 0].set_yticklabels(feature_names_sorted)
        axes[1, 0].set_xlabel('系数绝对值')
        axes[1, 0].set_title('特征重要性', fontweight='bold')
        
        # 4. 预测误差分布
        axes[1, 1].hist(residuals, bins=20, alpha=0.7, color='lightgreen', edgecolor='black')
        axes[1, 1].set_xlabel('预测误差')
        axes[1, 1].set_ylabel('频数')
        axes[1, 1].set_title('预测误差分布', fontweight='bold')
        axes[1, 1].axvline(x=0, color='r', linestyle='--')
        
        plt.tight_layout()
        plt.show()
        
        print("📊 可视化完成!")
    
    def predict_new_house(self, house_features):
        """预测新房子的价格"""
        print("\n🏠 预测新房子价格...")
        
        # 确保特征顺序正确
        if isinstance(house_features, dict):
            house_features = [house_features[feature] for feature in self.feature_names]
        
        # 标准化特征
        house_features_scaled = self.scaler.transform([house_features])
        
        # 预测价格
        predicted_price = self.model.predict(house_features_scaled)[0]
        
        print(f"💰 预测房价: ${predicted_price:.2f}千美元")
        
        return predicted_price
    
    def run_complete_project(self):
        """运行完整的房价预测项目"""
        print("🏠 开始波士顿房价预测项目...")
        print("=" * 60)
        
        # 1. 加载数据
        self.load_data()
        
        # 2. 数据探索
        self.explore_data()
        
        # 3. 准备数据
        self.prepare_data()
        
        # 4. 训练模型
        self.train_model()
        
        # 5. 评估模型
        evaluation_results = self.evaluate_model()
        
        # 6. 可视化结果
        self.visualize_results(evaluation_results)
        
        # 7. 示例预测
        print("\n🔮 示例预测:")
        example_house = {
            'CRIM': 0.1,      # 低犯罪率
            'ZN': 20.0,       # 住宅用地
            'INDUS': 5.0,     # 低工业比例
            'CHAS': 1,        # 靠近河边
            'NOX': 0.4,       # 低污染
            'RM': 7.0,        # 7个房间
            'AGE': 20.0,      # 房龄20年
            'DIS': 5.0,       # 距离市中心适中
            'RAD': 3,         # 交通便利
            'TAX': 300,       # 适中税率
            'PTRATIO': 15.0,  # 好的师生比
            'B': 390.0,       # 社区指标
            'LSTAT': 5.0      # 低收入人口比例低
        }
        
        predicted_price = self.predict_new_house(example_house)
        
        print("\n🎯 项目总结:")
        print(f"  • 数据集大小: {len(self.data)} 套房屋")
        print(f"  • 特征数量: {len(self.feature_names)} 个")
        print(f"  • 模型R²分数: {evaluation_results['test_r2']:.3f}")
        print(f"  • 平均绝对误差: {evaluation_results['test_mae']:.2f}千美元")
        print("\n🎉 波士顿房价预测项目完成!")
        
        return self

def run_gradient_descent_demo():
    """梯度下降算法演示"""
    print("\n" + "="*60)
    print("📈 梯度下降算法可视化演示")
    print("="*60)
    
    # 生成简单的线性数据
    np.random.seed(42)
    X = np.random.randn(100, 1)
    y = 4 + 3 * X.flatten() + np.random.randn(100) * 0.5
    
    # 手动实现梯度下降
    def gradient_descent(X, y, learning_rate=0.01, n_iterations=1000):
        m = len(y)
        theta = np.random.randn(2, 1)  # 随机初始化参数
        X_b = np.c_[np.ones((m, 1)), X]  # 添加偏置项
        
        cost_history = []
        theta_history = []
        
        for iteration in range(n_iterations):
            # 预测
            y_pred = X_b.dot(theta).flatten()
            
            # 计算损失
            cost = np.mean((y_pred - y) ** 2) / 2
            cost_history.append(cost)
            theta_history.append(theta.copy())
            
            # 计算梯度
            gradients = X_b.T.dot(y_pred - y) / m
            
            # 更新参数
            theta = theta - learning_rate * gradients.reshape(-1, 1)
        
        return theta, cost_history, theta_history
    
    # 运行梯度下降
    theta_final, cost_history, theta_history = gradient_descent(X, y)
    
    # 可视化结果
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # 1. 数据和拟合线
    axes[0].scatter(X, y, alpha=0.6, label='数据点')
    X_plot = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
    X_plot_b = np.c_[np.ones((100, 1)), X_plot]
    y_plot = X_plot_b.dot(theta_final).flatten()
    axes[0].plot(X_plot, y_plot, 'r-', linewidth=2, label='拟合直线')
    axes[0].set_xlabel('X')
    axes[0].set_ylabel('y')
    axes[0].set_title('线性回归结果', fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # 2. 损失函数变化
    axes[1].plot(cost_history, 'b-', linewidth=2)
    axes[1].set_xlabel('迭代次数')
    axes[1].set_ylabel('损失值')
    axes[1].set_title('梯度下降过程', fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print(f"✅ 梯度下降完成!")
    print(f"   - 最终参数: θ₀={theta_final[0][0]:.3f}, θ₁={theta_final[1][0]:.3f}")
    print(f"   - 最终损失: {cost_history[-1]:.6f}")
    print(f"   - 迭代次数: {len(cost_history)}")

def main():
    """主函数"""
    print("🏠 第7章：线性回归与房价预测")
    print("=" * 60)
    
    # 1. 运行完整项目
    predictor = BostonHousingPredictor()
    predictor.run_complete_project()
    
    # 2. 梯度下降演示
    run_gradient_descent_demo()
    
    print("\n🎓 学习总结:")
    print("  • 线性回归是预测连续值的基础算法")
    print("  • 梯度下降是寻找最优参数的有效方法")
    print("  • 特征工程和数据预处理很重要")
    print("  • 模型评估帮助我们了解预测性能")
    
    return predictor

if __name__ == "__main__":
    predictor = main()

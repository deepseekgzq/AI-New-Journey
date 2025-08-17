"""
第9章：K-Means聚类算法实现
完整的客户分群项目代码
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

class CustomerSegmentation:
    """客户分群分析类"""
    
    def __init__(self):
        """初始化"""
        self.data = None
        self.scaled_data = None
        self.kmeans = None
        self.labels = None
        self.optimal_k = None
        self.scaler = StandardScaler()
        
    def load_data(self, file_path=None):
        """加载数据"""
        if file_path is None:
            # 生成示例数据
            self.data = self.generate_sample_data()
        else:
            self.data = pd.read_csv(file_path)
        
        print(f"✅ 数据加载完成，共{len(self.data)}条记录")
        return self.data
    
    def generate_sample_data(self, n_samples=200):
        """生成示例客户数据"""
        np.random.seed(42)
        
        # 生成不同类型的客户群体
        # 群体1：高收入高消费
        group1_income = np.random.normal(80, 10, 50)
        group1_spending = np.random.normal(80, 8, 50)
        
        # 群体2：中等收入中等消费
        group2_income = np.random.normal(50, 8, 50)
        group2_spending = np.random.normal(50, 10, 50)
        
        # 群体3：高收入低消费（谨慎型）
        group3_income = np.random.normal(75, 8, 50)
        group3_spending = np.random.normal(30, 8, 50)
        
        # 群体4：低收入高消费（冲动型）
        group4_income = np.random.normal(25, 5, 50)
        group4_spending = np.random.normal(70, 10, 50)
        
        # 合并数据
        income = np.concatenate([group1_income, group2_income, group3_income, group4_income])
        spending = np.concatenate([group1_spending, group2_spending, group3_spending, group4_spending])
        
        # 确保数据在合理范围内
        income = np.clip(income, 15, 100)
        spending = np.clip(spending, 1, 100)
        
        # 创建DataFrame
        data = pd.DataFrame({
            'CustomerID': range(1, len(income) + 1),
            'AnnualIncome': income,
            'SpendingScore': spending
        })
        
        return data
    
    def explore_data(self):
        """数据探索性分析"""
        print("📊 数据基本信息：")
        print(self.data.describe())
        
        # 创建可视化
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. 收入分布
        axes[0, 0].hist(self.data['AnnualIncome'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0, 0].set_title('年收入分布', fontweight='bold')
        axes[0, 0].set_xlabel('年收入 (k$)')
        axes[0, 0].set_ylabel('客户数量')
        
        # 2. 消费分数分布
        axes[0, 1].hist(self.data['SpendingScore'], bins=20, alpha=0.7, color='lightcoral', edgecolor='black')
        axes[0, 1].set_title('消费分数分布', fontweight='bold')
        axes[0, 1].set_xlabel('消费分数 (1-100)')
        axes[0, 1].set_ylabel('客户数量')
        
        # 3. 散点图
        axes[1, 0].scatter(self.data['AnnualIncome'], self.data['SpendingScore'], 
                          alpha=0.6, c='green', s=50)
        axes[1, 0].set_title('收入 vs 消费分数', fontweight='bold')
        axes[1, 0].set_xlabel('年收入 (k$)')
        axes[1, 0].set_ylabel('消费分数')
        
        # 4. 箱线图
        data_for_box = [self.data['AnnualIncome'], self.data['SpendingScore']]
        axes[1, 1].boxplot(data_for_box, labels=['年收入', '消费分数'])
        axes[1, 1].set_title('数据分布箱线图', fontweight='bold')
        axes[1, 1].set_ylabel('数值')
        
        plt.tight_layout()
        plt.show()
        
        print("💡 从散点图可以看出，客户群体可能存在自然分群！")
    
    def find_optimal_k(self, max_k=10):
        """使用手肘法确定最佳K值"""
        # 准备数据
        X = self.data[['AnnualIncome', 'SpendingScore']]
        
        # 计算不同K值的WCSS（簇内平方和）
        wcss = []
        k_range = range(1, max_k + 1)
        
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(X)
            wcss.append(kmeans.inertia_)
        
        # 绘制手肘图
        plt.figure(figsize=(10, 6))
        plt.plot(k_range, wcss, 'bo-', linewidth=2, markersize=8)
        plt.title('手肘法确定最佳聚类数量', fontsize=14, fontweight='bold')
        plt.xlabel('聚类数量 (K)', fontsize=12)
        plt.ylabel('簇内平方和 (WCSS)', fontsize=12)
        plt.grid(True, alpha=0.3)
        
        # 标注每个点的数值
        for i, (k, w) in enumerate(zip(k_range, wcss)):
            plt.annotate(f'K={k}\\nWCSS={w:.0f}', (k, w), 
                        textcoords="offset points", xytext=(0,10), ha='center')
        
        plt.tight_layout()
        plt.show()
        
        # 自动检测手肘点（简单方法：找到变化率最大的点）
        diffs = np.diff(wcss)
        diffs2 = np.diff(diffs)
        elbow_k = np.argmax(diffs2) + 2  # +2因为diff操作减少了数组长度
        
        self.optimal_k = elbow_k
        print(f"📍 建议的最佳聚类数量：K = {elbow_k}")
        
        return wcss, elbow_k
    
    def fit_kmeans(self, n_clusters=None):
        """训练K-Means模型"""
        if n_clusters is None:
            if self.optimal_k is None:
                self.find_optimal_k()
            n_clusters = self.optimal_k
        
        # 准备数据
        X = self.data[['AnnualIncome', 'SpendingScore']]
        
        # 训练模型
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        self.labels = self.kmeans.fit_predict(X)
        
        # 添加聚类标签到数据
        self.data['Cluster'] = self.labels
        
        print(f"✅ K-Means训练完成，共分为{n_clusters}个群体")
        
        return self.labels
    
    def visualize_clusters(self):
        """可视化聚类结果"""
        if self.labels is None:
            print("❌ 请先训练模型！")
            return
        
        plt.figure(figsize=(12, 8))
        
        # 获取聚类中心
        centers = self.kmeans.cluster_centers_
        
        # 定义颜色和标签
        colors = ['red', 'blue', 'green', 'purple', 'orange', 'brown', 'pink', 'gray']
        cluster_names = ['群体A', '群体B', '群体C', '群体D', '群体E']
        
        # 绘制每个聚类
        for i in range(self.kmeans.n_clusters):
            cluster_data = self.data[self.data['Cluster'] == i]
            plt.scatter(cluster_data['AnnualIncome'], cluster_data['SpendingScore'],
                       c=colors[i], label=f'{cluster_names[i]}', alpha=0.7, s=60)
        
        # 绘制聚类中心
        plt.scatter(centers[:, 0], centers[:, 1], 
                   c='black', marker='x', s=200, linewidths=3, label='聚类中心')
        
        plt.title('客户分群结果', fontsize=16, fontweight='bold')
        plt.xlabel('年收入 (k$)', fontsize=12)
        plt.ylabel('消费分数 (1-100)', fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        
        print("🎨 不同颜色代表不同的客户群体！")
    
    def analyze_clusters(self):
        """分析聚类结果"""
        if self.labels is None:
            print("❌ 请先训练模型！")
            return
        
        print("📈 客户群体分析结果：")
        print("=" * 50)
        
        cluster_analysis = []
        
        for i in range(self.kmeans.n_clusters):
            cluster_data = self.data[self.data['Cluster'] == i]
            
            analysis = {
                'cluster': f'群体{chr(65+i)}',  # A, B, C, D...
                'size': len(cluster_data),
                'percentage': len(cluster_data) / len(self.data) * 100,
                'avg_income': cluster_data['AnnualIncome'].mean(),
                'avg_spending': cluster_data['SpendingScore'].mean(),
                'income_std': cluster_data['AnnualIncome'].std(),
                'spending_std': cluster_data['SpendingScore'].std()
            }
            
            cluster_analysis.append(analysis)
            
            # 打印分析结果
            print(f"\\n{analysis['cluster']} ({analysis['size']}人, {analysis['percentage']:.1f}%):")
            print(f"  • 平均年收入: {analysis['avg_income']:.1f}k$ (±{analysis['income_std']:.1f})")
            print(f"  • 平均消费分数: {analysis['avg_spending']:.1f} (±{analysis['spending_std']:.1f})")
            
            # 客户群体特征描述
            if analysis['avg_income'] > 60 and analysis['avg_spending'] > 60:
                char = "高价值核心客户 💎"
                strategy = "VIP服务、新品优先体验、专属折扣"
            elif analysis['avg_income'] > 60 and analysis['avg_spending'] < 40:
                char = "谨慎型富裕客户 🤔"
                strategy = "精准推送高品质商品，强调价值而非价格"
            elif analysis['avg_income'] < 40 and analysis['avg_spending'] > 60:
                char = "冲动型年轻客户 🎯"
                strategy = "推送时尚潮流、打折促销信息"
            elif 40 <= analysis['avg_income'] <= 60 and 40 <= analysis['avg_spending'] <= 60:
                char = "大众潜力客户 📈"
                strategy = "积分计划、满减活动，引导消费升级"
            else:
                char = "低频待激活客户 💤"
                strategy = "通过低价爆款商品吸引再次光顾"
            
            print(f"  • 特征: {char}")
            print(f"  • 营销策略: {strategy}")
        
        return cluster_analysis
    
    def generate_business_insights(self):
        """生成商业洞察报告"""
        if self.labels is None:
            print("❌ 请先训练模型！")
            return
        
        insights = self.analyze_clusters()
        
        print("\\n" + "="*60)
        print("📋 商业洞察总结报告")
        print("="*60)
        
        # 找出最大和最小的群体
        largest_cluster = max(insights, key=lambda x: x['size'])
        smallest_cluster = min(insights, key=lambda x: x['size'])
        highest_value = max(insights, key=lambda x: x['avg_income'] * x['avg_spending'])
        
        print(f"\\n🎯 关键发现:")
        print(f"• 最大客户群体: {largest_cluster['cluster']} ({largest_cluster['percentage']:.1f}%)")
        print(f"• 最小客户群体: {smallest_cluster['cluster']} ({smallest_cluster['percentage']:.1f}%)")
        print(f"• 最高价值群体: {highest_value['cluster']} (收入×消费分数最高)")
        
        print(f"\\n💡 营销建议:")
        print("• 重点关注高价值客户群体，提供个性化服务")
        print("• 针对不同群体制定差异化营销策略")
        print("• 通过数据驱动优化资源分配")
        print("• 定期重新分析客户群体变化")
        
        return insights
    
    def save_results(self, filename='customer_segmentation_results.csv'):
        """保存分析结果"""
        if self.data is not None:
            self.data.to_csv(filename, index=False)
            print(f"✅ 结果已保存到 {filename}")

def run_complete_project():
    """运行完整的客户分群项目"""
    print("🚀 开始客户分群项目...")
    print("="*50)
    
    # 1. 初始化项目
    segmentation = CustomerSegmentation()
    
    # 2. 加载数据
    print("\\n📁 步骤1: 加载数据")
    data = segmentation.load_data()
    print(f"数据预览:\\n{data.head()}")
    
    # 3. 数据探索
    print("\\n🔍 步骤2: 数据探索")
    segmentation.explore_data()
    
    # 4. 确定最佳K值
    print("\\n📐 步骤3: 确定最佳聚类数量")
    wcss, optimal_k = segmentation.find_optimal_k()
    
    # 5. 训练模型
    print("\\n🤖 步骤4: 训练K-Means模型")
    labels = segmentation.fit_kmeans(optimal_k)
    
    # 6. 可视化结果
    print("\\n📊 步骤5: 可视化聚类结果")
    segmentation.visualize_clusters()
    
    # 7. 分析结果
    print("\\n📈 步骤6: 分析客户群体")
    insights = segmentation.analyze_clusters()
    
    # 8. 生成商业洞察
    print("\\n💼 步骤7: 生成商业洞察")
    business_insights = segmentation.generate_business_insights()
    
    # 9. 保存结果
    print("\\n💾 步骤8: 保存结果")
    segmentation.save_results()
    
    print("\\n🎉 客户分群项目完成！")
    
    return segmentation

if __name__ == "__main__":
    # 运行完整项目
    project = run_complete_project()

"""
第4章：销售数据分析完整项目
Python数据科学实战案例
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

class SalesDataAnalyzer:
    """销售数据分析器"""
    
    def __init__(self):
        """初始化"""
        self.data = None
        self.monthly_sales = None
        self.product_analysis = None
        self.customer_analysis = None
        
    def generate_sample_data(self, n_records=10000):
        """生成模拟销售数据"""
        print("📊 生成模拟销售数据...")
        
        np.random.seed(42)
        
        # 时间范围：过去2年
        start_date = datetime.now() - timedelta(days=730)
        date_range = pd.date_range(start=start_date, periods=730, freq='D')
        
        # 产品类别和名称
        categories = ['电子产品', '服装', '家居用品', '书籍', '运动用品', '美妆护肤']
        products = {
            '电子产品': ['手机', '平板电脑', '耳机', '充电器', '智能手表'],
            '服装': ['T恤', '牛仔裤', '连衣裙', '外套', '运动鞋'],
            '家居用品': ['台灯', '收纳盒', '抱枕', '花瓶', '餐具'],
            '书籍': ['小说', '教材', '科普书', '传记', '工具书'],
            '运动用品': ['跑步鞋', '瑜伽垫', '哑铃', '泳衣', '运动服'],
            '美妆护肤': ['口红', '面膜', '洗面奶', '香水', '护手霜']
        }
        
        # 客户城市
        cities = ['北京', '上海', '广州', '深圳', '杭州', '南京', '武汉', '成都', '西安', '重庆']
        
        records = []
        
        for i in range(n_records):
            # 随机选择日期（工作日销量更高）
            date = np.random.choice(date_range)
            weekday = date.weekday()
            
            # 根据日期类型调整销量概率
            if weekday < 5:  # 工作日
                volume_multiplier = 1.2
            else:  # 周末
                volume_multiplier = 0.8
            
            # 季节性影响
            month = date.month
            if month in [11, 12, 1]:  # 购物旺季
                seasonal_multiplier = 1.5
            elif month in [6, 7, 8]:  # 夏季
                seasonal_multiplier = 1.2
            else:
                seasonal_multiplier = 1.0
            
            # 随机选择产品类别和产品
            category = np.random.choice(categories)
            product = np.random.choice(products[category])
            
            # 价格范围（根据类别）
            price_ranges = {
                '电子产品': (100, 5000),
                '服装': (50, 500),
                '家居用品': (20, 300),
                '书籍': (10, 100),
                '运动用品': (50, 800),
                '美妆护肤': (30, 400)
            }
            
            min_price, max_price = price_ranges[category]
            price = np.random.uniform(min_price, max_price)
            
            # 数量（大多数是1-3件）
            quantity = np.random.choice([1, 2, 3, 4, 5], p=[0.5, 0.25, 0.15, 0.07, 0.03])
            
            # 应用倍数
            quantity = int(quantity * volume_multiplier * seasonal_multiplier)
            quantity = max(1, quantity)  # 至少1件
            
            # 客户信息
            customer_id = f"C{np.random.randint(1000, 9999)}"
            city = np.random.choice(cities)
            
            # 销售渠道
            channel = np.random.choice(['线上', '线下'], p=[0.7, 0.3])
            
            # 计算总金额
            total_amount = price * quantity
            
            # 添加一些促销折扣
            discount = 0
            if np.random.random() < 0.15:  # 15%概率有折扣
                discount = np.random.uniform(0.05, 0.25)  # 5%-25%折扣
                total_amount *= (1 - discount)
            
            record = {
                'order_id': f"ORD{10000+i}",
                'date': date.strftime('%Y-%m-%d'),
                'category': category,
                'product': product,
                'price': round(price, 2),
                'quantity': quantity,
                'total_amount': round(total_amount, 2),
                'customer_id': customer_id,
                'city': city,
                'channel': channel,
                'discount': round(discount, 2)
            }
            
            records.append(record)
        
        self.data = pd.DataFrame(records)
        self.data['date'] = pd.to_datetime(self.data['date'])
        
        print(f"✅ 生成了 {len(self.data)} 条销售记录")
        print(f"   时间范围: {self.data['date'].min()} 到 {self.data['date'].max()}")
        print(f"   产品类别: {len(self.data['category'].unique())} 个")
        print(f"   城市数量: {len(self.data['city'].unique())} 个")
        
        return self.data
    
    def load_data(self, file_path=None):
        """加载销售数据"""
        if file_path and pd.io.common.file_exists(file_path):
            print(f"📂 从文件加载数据: {file_path}")
            self.data = pd.read_csv(file_path)
            self.data['date'] = pd.to_datetime(self.data['date'])
        else:
            print("📊 使用模拟数据...")
            self.generate_sample_data()
        
        return self.data
    
    def data_overview(self):
        """数据概览"""
        print("\\n🔍 数据基本信息:")
        print(f"数据形状: {self.data.shape}")
        print(f"时间范围: {self.data['date'].min()} 到 {self.data['date'].max()}")
        print(f"总销售额: ¥{self.data['total_amount'].sum():,.2f}")
        print(f"平均订单金额: ¥{self.data['total_amount'].mean():.2f}")
        
        print("\\n📋 数据类型:")
        print(self.data.dtypes)
        
        print("\\n📊 基本统计:")
        print(self.data.describe())
        
        # 检查缺失值
        missing_values = self.data.isnull().sum()
        if missing_values.sum() > 0:
            print("\\n⚠️ 缺失值:")
            print(missing_values[missing_values > 0])
        else:
            print("\\n✅ 无缺失值")
        
        # 显示前几行数据
        print("\\n📄 数据样本:")
        print(self.data.head())
    
    def sales_trend_analysis(self):
        """销售趋势分析"""
        print("\\n📈 销售趋势分析...")
        
        # 按月份汇总销售数据
        self.data['year_month'] = self.data['date'].dt.to_period('M')
        self.monthly_sales = self.data.groupby('year_month').agg({
            'total_amount': 'sum',
            'quantity': 'sum',
            'order_id': 'count'
        }).rename(columns={'order_id': 'order_count'})
        
        # 按日期汇总（用于日趋势）
        daily_sales = self.data.groupby('date').agg({
            'total_amount': 'sum',
            'quantity': 'sum',
            'order_id': 'count'
        }).rename(columns={'order_id': 'order_count'})
        
        # 可视化趋势
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        
        # 1. 月销售额趋势
        self.monthly_sales['total_amount'].plot(kind='line', ax=axes[0, 0], 
                                               marker='o', linewidth=2, markersize=6)
        axes[0, 0].set_title('月销售额趋势', fontweight='bold')
        axes[0, 0].set_ylabel('销售额 (¥)')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # 2. 月订单数量趋势
        self.monthly_sales['order_count'].plot(kind='line', ax=axes[0, 1], 
                                              marker='s', linewidth=2, markersize=6, color='orange')
        axes[0, 1].set_title('月订单数量趋势', fontweight='bold')
        axes[0, 1].set_ylabel('订单数量')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # 3. 日销售额分布（最近3个月）
        recent_data = daily_sales.tail(90)  # 最近90天
        recent_data['total_amount'].plot(kind='line', ax=axes[1, 0], 
                                        alpha=0.7, linewidth=1)
        axes[1, 0].set_title('日销售额趋势（最近3个月）', fontweight='bold')
        axes[1, 0].set_ylabel('日销售额 (¥)')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # 4. 季节性分析
        self.data['month'] = self.data['date'].dt.month
        monthly_avg = self.data.groupby('month')['total_amount'].mean()
        axes[1, 1].bar(monthly_avg.index, monthly_avg.values, 
                      color='skyblue', alpha=0.8, edgecolor='black')
        axes[1, 1].set_title('月度平均销售额', fontweight='bold')
        axes[1, 1].set_xlabel('月份')
        axes[1, 1].set_ylabel('平均销售额 (¥)')
        axes[1, 1].set_xticks(range(1, 13))
        axes[1, 1].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.show()
        
        # 计算增长率
        monthly_growth = self.monthly_sales['total_amount'].pct_change().dropna()
        avg_growth = monthly_growth.mean()
        
        print(f"📊 趋势分析结果:")
        print(f"  • 月均销售额: ¥{self.monthly_sales['total_amount'].mean():,.2f}")
        print(f"  • 月均增长率: {avg_growth:.2%}")
        print(f"  • 最高销售月: {self.monthly_sales['total_amount'].idxmax()}")
        print(f"  • 最低销售月: {self.monthly_sales['total_amount'].idxmin()}")
    
    def product_analysis(self):
        """产品分析"""
        print("\\n🛍️ 产品分析...")
        
        # 按产品类别分析
        category_stats = self.data.groupby('category').agg({
            'total_amount': ['sum', 'mean', 'count'],
            'quantity': 'sum'
        }).round(2)
        
        category_stats.columns = ['总销售额', '平均订单金额', '订单数量', '销售数量']
        category_stats = category_stats.sort_values('总销售额', ascending=False)
        
        # 按具体产品分析
        product_stats = self.data.groupby(['category', 'product']).agg({
            'total_amount': 'sum',
            'quantity': 'sum',
            'order_id': 'count'
        }).rename(columns={'order_id': '订单数'}).sort_values('total_amount', ascending=False)
        
        self.product_analysis = {
            'category_stats': category_stats,
            'product_stats': product_stats
        }
        
        # 可视化产品分析
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. 类别销售额饼图
        category_sales = category_stats['总销售额']
        axes[0, 0].pie(category_sales.values, labels=category_sales.index, 
                      autopct='%1.1f%%', startangle=90)
        axes[0, 0].set_title('各类别销售额占比', fontweight='bold')
        
        # 2. 类别销售额柱状图
        category_sales.plot(kind='bar', ax=axes[0, 1], 
                           color='lightblue', edgecolor='black', alpha=0.8)
        axes[0, 1].set_title('各类别销售额', fontweight='bold')
        axes[0, 1].set_ylabel('销售额 (¥)')
        axes[0, 1].tick_params(axis='x', rotation=45)
        axes[0, 1].grid(True, alpha=0.3, axis='y')
        
        # 3. 类别订单数量
        category_orders = category_stats['订单数量']
        category_orders.plot(kind='bar', ax=axes[1, 0], 
                            color='lightcoral', edgecolor='black', alpha=0.8)
        axes[1, 0].set_title('各类别订单数量', fontweight='bold')
        axes[1, 0].set_ylabel('订单数量')
        axes[1, 0].tick_params(axis='x', rotation=45)
        axes[1, 0].grid(True, alpha=0.3, axis='y')
        
        # 4. 类别平均订单金额
        avg_order_value = category_stats['平均订单金额']
        avg_order_value.plot(kind='bar', ax=axes[1, 1], 
                            color='lightgreen', edgecolor='black', alpha=0.8)
        axes[1, 1].set_title('各类别平均订单金额', fontweight='bold')
        axes[1, 1].set_ylabel('平均订单金额 (¥)')
        axes[1, 1].tick_params(axis='x', rotation=45)
        axes[1, 1].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.show()
        
        print("📊 产品分析结果:")
        print("\\n🏆 销售额排名:")
        for i, (category, amount) in enumerate(category_sales.items(), 1):
            print(f"  {i}. {category}: ¥{amount:,.2f}")
        
        print("\\n📦 热销产品TOP10:")
        top_products = product_stats.head(10)
        for i, ((category, product), row) in enumerate(top_products.iterrows(), 1):
            print(f"  {i}. {product} ({category}): ¥{row['total_amount']:,.2f}")
    
    def customer_analysis(self):
        """客户分析"""
        print("\\n👥 客户分析...")
        
        # 按客户分析
        customer_stats = self.data.groupby('customer_id').agg({
            'total_amount': ['sum', 'mean', 'count'],
            'date': ['min', 'max']
        })
        
        customer_stats.columns = ['总消费', '平均订单金额', '订单次数', '首次购买', '最后购买']
        
        # 计算客户生命周期（天数）
        customer_stats['生命周期'] = (customer_stats['最后购买'] - customer_stats['首次购买']).dt.days
        
        # 按城市分析
        city_stats = self.data.groupby('city').agg({
            'total_amount': 'sum',
            'order_id': 'count',
            'customer_id': 'nunique'
        }).rename(columns={'order_id': '订单数', 'customer_id': '客户数'})
        city_stats = city_stats.sort_values('total_amount', ascending=False)
        
        # 按渠道分析
        channel_stats = self.data.groupby('channel').agg({
            'total_amount': 'sum',
            'order_id': 'count'
        }).rename(columns={'order_id': '订单数'})
        
        self.customer_analysis = {
            'customer_stats': customer_stats,
            'city_stats': city_stats,
            'channel_stats': channel_stats
        }
        
        # 可视化客户分析
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        
        # 1. 客户消费分布
        customer_stats['总消费'].hist(bins=50, ax=axes[0, 0], alpha=0.7, color='skyblue', edgecolor='black')
        axes[0, 0].set_title('客户总消费分布', fontweight='bold')
        axes[0, 0].set_xlabel('总消费金额 (¥)')
        axes[0, 0].set_ylabel('客户数量')
        
        # 2. 客户订单次数分布
        customer_stats['订单次数'].hist(bins=30, ax=axes[0, 1], alpha=0.7, color='lightcoral', edgecolor='black')
        axes[0, 1].set_title('客户订单次数分布', fontweight='bold')
        axes[0, 1].set_xlabel('订单次数')
        axes[0, 1].set_ylabel('客户数量')
        
        # 3. 城市销售额排名
        top_cities = city_stats.head(10)
        top_cities['total_amount'].plot(kind='bar', ax=axes[0, 2], 
                                       color='lightgreen', alpha=0.8, edgecolor='black')
        axes[0, 2].set_title('城市销售额TOP10', fontweight='bold')
        axes[0, 2].set_ylabel('销售额 (¥)')
        axes[0, 2].tick_params(axis='x', rotation=45)
        
        # 4. 渠道对比
        channel_stats.plot(kind='bar', ax=axes[1, 0], 
                           color=['lightblue', 'lightcoral'], alpha=0.8, edgecolor='black')
        axes[1, 0].set_title('渠道对比', fontweight='bold')
        axes[1, 0].set_ylabel('数量')
        axes[1, 0].tick_params(axis='x', rotation=0)
        axes[1, 0].legend()
        
        # 5. 客户价值分层
        # 按总消费分为高、中、低价值客户
        customer_stats['价值分层'] = pd.cut(customer_stats['总消费'], 
                                        bins=[0, 500, 2000, float('inf')], 
                                        labels=['低价值', '中价值', '高价值'])
        value_distribution = customer_stats['价值分层'].value_counts()
        
        axes[1, 1].pie(value_distribution.values, labels=value_distribution.index, 
                      autopct='%1.1f%%', startangle=90, 
                      colors=['lightcoral', 'lightyellow', 'lightgreen'])
        axes[1, 1].set_title('客户价值分层', fontweight='bold')
        
        # 6. 复购率分析
        repeat_customers = customer_stats[customer_stats['订单次数'] > 1]
        repeat_rate = len(repeat_customers) / len(customer_stats)
        
        repeat_data = [repeat_rate, 1-repeat_rate]
        axes[1, 2].pie(repeat_data, labels=['复购客户', '单次客户'], 
                      autopct='%1.1f%%', startangle=90,
                      colors=['lightgreen', 'lightcoral'])
        axes[1, 2].set_title(f'客户复购率\\n({repeat_rate:.1%})', fontweight='bold')
        
        plt.tight_layout()
        plt.show()
        
        print("📊 客户分析结果:")
        print(f"  • 总客户数: {len(customer_stats):,}")
        print(f"  • 平均客户价值: ¥{customer_stats['总消费'].mean():.2f}")
        print(f"  • 客户复购率: {repeat_rate:.1%}")
        print(f"  • 平均订单次数: {customer_stats['订单次数'].mean():.1f}")
        print(f"  • 销售额最高城市: {city_stats.index[0]}")
    
    def advanced_analysis(self):
        """高级分析"""
        print("\\n🔬 高级分析...")
        
        # RFM分析（最近购买时间、购买频率、购买金额）
        current_date = self.data['date'].max()
        
        rfm = self.data.groupby('customer_id').agg({
            'date': lambda x: (current_date - x.max()).days,  # Recency
            'order_id': 'count',  # Frequency
            'total_amount': 'sum'  # Monetary
        }).rename(columns={
            'date': 'Recency',
            'order_id': 'Frequency', 
            'total_amount': 'Monetary'
        })
        
        # RFM评分（1-5分）
        rfm['R_Score'] = pd.qcut(rfm['Recency'], 5, labels=[5,4,3,2,1])  # 越近期越高分
        rfm['F_Score'] = pd.qcut(rfm['Frequency'].rank(method='first'), 5, labels=[1,2,3,4,5])
        rfm['M_Score'] = pd.qcut(rfm['Monetary'], 5, labels=[1,2,3,4,5])
        
        # 综合评分
        rfm['RFM_Score'] = rfm['R_Score'].astype(str) + rfm['F_Score'].astype(str) + rfm['M_Score'].astype(str)
        
        # 客户分群
        def rfm_segment(row):
            if row['RFM_Score'] in ['555', '554', '544', '545', '454', '455', '445']:
                return '冠军客户'
            elif row['RFM_Score'] in ['543', '444', '435', '355', '354', '345', '344', '335']:
                return '忠诚客户'
            elif row['RFM_Score'] in ['512', '511', '422', '421', '412', '411', '311']:
                return '潜在忠诚客户'
            elif row['RFM_Score'] in ['533', '532', '531', '523', '522', '521', '515', '514']:
                return '新客户'
            elif row['RFM_Score'] in ['155', '154', '144', '214', '215', '115', '114']:
                return '流失客户'
            else:
                return '其他客户'
        
        rfm['客户分群'] = rfm.apply(rfm_segment, axis=1)
        
        # 可视化RFM分析
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. RFM各维度分布
        axes[0, 0].scatter(rfm['Recency'], rfm['Frequency'], 
                          c=rfm['Monetary'], alpha=0.6, cmap='viridis')
        axes[0, 0].set_xlabel('最近购买天数')
        axes[0, 0].set_ylabel('购买频率')
        axes[0, 0].set_title('客户RFM分布（颜色表示消费金额）', fontweight='bold')
        
        # 2. 客户分群
        segment_counts = rfm['客户分群'].value_counts()
        axes[0, 1].pie(segment_counts.values, labels=segment_counts.index, 
                      autopct='%1.1f%%', startangle=90)
        axes[0, 1].set_title('客户分群分布', fontweight='bold')
        
        # 3. 各分群价值对比
        segment_value = rfm.groupby('客户分群')['Monetary'].mean().sort_values(ascending=False)
        segment_value.plot(kind='bar', ax=axes[1, 0], 
                          color='lightgreen', alpha=0.8, edgecolor='black')
        axes[1, 0].set_title('各客户群平均消费', fontweight='bold')
        axes[1, 0].set_ylabel('平均消费金额 (¥)')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # 4. 促销效果分析
        discount_effect = self.data.groupby('discount').agg({
            'total_amount': 'mean',
            'quantity': 'mean'
        })
        
        ax_twin = axes[1, 1].twinx()
        axes[1, 1].bar(discount_effect.index, discount_effect['total_amount'], 
                      alpha=0.7, color='lightblue', label='平均订单金额')
        ax_twin.plot(discount_effect.index, discount_effect['quantity'], 
                    'ro-', label='平均购买数量')
        
        axes[1, 1].set_xlabel('折扣率')
        axes[1, 1].set_ylabel('平均订单金额 (¥)')
        ax_twin.set_ylabel('平均购买数量')
        axes[1, 1].set_title('促销效果分析', fontweight='bold')
        axes[1, 1].legend(loc='upper left')
        ax_twin.legend(loc='upper right')
        
        plt.tight_layout()
        plt.show()
        
        print("📊 高级分析结果:")
        print("\\n👑 客户分群统计:")
        for segment, count in segment_counts.items():
            percentage = count / len(rfm) * 100
            avg_value = rfm[rfm['客户分群'] == segment]['Monetary'].mean()
            print(f"  • {segment}: {count}人 ({percentage:.1f}%), 平均消费¥{avg_value:.2f}")
        
        return rfm
    
    def generate_business_insights(self):
        """生成业务洞察"""
        print("\\n💡 业务洞察与建议...")
        
        insights = []
        
        # 销售趋势洞察
        if hasattr(self, 'monthly_sales') and self.monthly_sales is not None:
            growth = self.monthly_sales['total_amount'].pct_change().dropna().mean()
            if growth > 0.05:
                insights.append("📈 销售呈现良好增长趋势，建议加大营销投入")
            elif growth < -0.05:
                insights.append("📉 销售出现下滑，需要分析原因并制定应对策略")
        
        # 产品洞察
        if hasattr(self, 'product_analysis') and self.product_analysis is not None:
            top_category = self.product_analysis['category_stats'].index[0]
            insights.append(f"🏆 {top_category}是最大销售品类，建议重点发展")
            
            category_stats = self.product_analysis['category_stats']
            if len(category_stats) > 1:
                top_aov = category_stats['平均订单金额'].idxmax()
                insights.append(f"💰 {top_aov}的客单价最高，可重点推广高价值产品")
        
        # 客户洞察
        if hasattr(self, 'customer_analysis') and self.customer_analysis is not None:
            repeat_rate = len(self.customer_analysis['customer_stats'][
                self.customer_analysis['customer_stats']['订单次数'] > 1
            ]) / len(self.customer_analysis['customer_stats'])
            
            if repeat_rate < 0.3:
                insights.append("🔄 客户复购率较低，建议加强客户关系维护")
            else:
                insights.append("✨ 客户复购率良好，可以开发会员体系")
        
        # 渠道洞察
        if self.data['channel'].nunique() > 1:
            channel_revenue = self.data.groupby('channel')['total_amount'].sum()
            best_channel = channel_revenue.idxmax()
            insights.append(f"📱 {best_channel}渠道表现最佳，建议优化资源配置")
        
        # 地域洞察
        if self.data['city'].nunique() > 5:
            city_stats = self.data.groupby('city')['total_amount'].sum().sort_values(ascending=False)
            top_3_cities = city_stats.head(3).index.tolist()
            insights.append(f"🏙️ {', '.join(top_3_cities)}是核心市场，建议加大本地化运营")
        
        # 价格策略洞察
        discount_data = self.data[self.data['discount'] > 0]
        if len(discount_data) > 0:
            avg_discount_order = discount_data['total_amount'].mean()
            avg_normal_order = self.data[self.data['discount'] == 0]['total_amount'].mean()
            if avg_discount_order > avg_normal_order:
                insights.append("🎯 促销活动有效提升客单价，建议定期开展")
            else:
                insights.append("⚠️ 促销活动未能有效提升客单价，需要优化策略")
        
        print("📋 核心业务洞察:")
        for i, insight in enumerate(insights, 1):
            print(f"  {i}. {insight}")
        
        # 具体建议
        recommendations = [
            "建立客户分层运营策略，针对不同价值客户制定差异化服务",
            "优化产品组合，重点推广高毛利和高复购率产品",
            "建立客户生命周期管理体系，提高客户留存率",
            "利用数据分析定期优化营销活动ROI",
            "建立实时销售监控dashboard，及时发现业务异常"
        ]
        
        print("\\n🎯 行动建议:")
        for i, rec in enumerate(recommendations, 1):
            print(f"  {i}. {rec}")
        
        return insights, recommendations
    
    def run_complete_analysis(self):
        """运行完整的销售数据分析"""
        print("📊 开始销售数据分析项目...")
        print("=" * 60)
        
        # 1. 加载数据
        self.load_data()
        
        # 2. 数据概览
        self.data_overview()
        
        # 3. 销售趋势分析
        self.sales_trend_analysis()
        
        # 4. 产品分析
        self.product_analysis()
        
        # 5. 客户分析
        self.customer_analysis()
        
        # 6. 高级分析
        rfm_data = self.advanced_analysis()
        
        # 7. 业务洞察
        insights, recommendations = self.generate_business_insights()
        
        print("\\n🎯 分析总结:")
        print(f"  • 总销售额: ¥{self.data['total_amount'].sum():,.2f}")
        print(f"  • 总订单数: {len(self.data):,}")
        print(f"  • 活跃客户数: {self.data['customer_id'].nunique():,}")
        print(f"  • 产品类别数: {self.data['category'].nunique()}")
        print(f"  • 覆盖城市数: {self.data['city'].nunique()}")
        
        print("\\n🎉 销售数据分析项目完成!")
        
        return self

def numpy_tutorial():
    """NumPy基础教程"""
    print("\\n" + "="*60)
    print("🔢 NumPy基础教程")
    print("="*60)
    
    # 数组创建
    print("\\n1️⃣ 数组创建:")
    arr1 = np.array([1, 2, 3, 4, 5])
    arr2 = np.zeros((3, 4))
    arr3 = np.ones((2, 3))
    arr4 = np.random.random((2, 2))
    
    print(f"一维数组: {arr1}")
    print(f"零数组形状: {arr2.shape}")
    print(f"随机数组:\\n{arr4}")
    
    # 数组运算
    print("\\n2️⃣ 数组运算:")
    a = np.array([1, 2, 3])
    b = np.array([4, 5, 6])
    print(f"a + b = {a + b}")
    print(f"a * b = {a * b}")
    print(f"a @ b = {np.dot(a, b)}")  # 点积
    
    # 统计函数
    print("\\n3️⃣ 统计函数:")
    data = np.random.normal(100, 15, 1000)  # 生成正态分布数据
    print(f"均值: {np.mean(data):.2f}")
    print(f"标准差: {np.std(data):.2f}")
    print(f"最大值: {np.max(data):.2f}")
    print(f"最小值: {np.min(data):.2f}")

def pandas_tutorial():
    """Pandas基础教程"""
    print("\\n" + "="*60)
    print("🐼 Pandas基础教程")
    print("="*60)
    
    # DataFrame创建
    print("\\n1️⃣ DataFrame创建:")
    data = {
        'name': ['Alice', 'Bob', 'Charlie', 'David'],
        'age': [25, 30, 35, 28],
        'salary': [50000, 60000, 70000, 55000],
        'city': ['北京', '上海', '广州', '深圳']
    }
    df = pd.DataFrame(data)
    print(df)
    
    # 数据选择和过滤
    print("\\n2️⃣ 数据选择和过滤:")
    print("年龄大于28的员工:")
    print(df[df['age'] > 28])
    
    print("\\n薪资最高的员工:")
    print(df.loc[df['salary'].idxmax()])
    
    # 数据聚合
    print("\\n3️⃣ 数据聚合:")
    print(f"平均年龄: {df['age'].mean():.1f}")
    print(f"平均薪资: {df['salary'].mean():,.0f}")
    
    # 按城市分组
    city_stats = df.groupby('city').agg({
        'age': 'mean',
        'salary': 'mean'
    })
    print("\\n按城市统计:")
    print(city_stats)

def matplotlib_tutorial():
    """Matplotlib基础教程"""
    print("\\n" + "="*60)
    print("📊 Matplotlib基础教程")
    print("="*60)
    
    # 生成示例数据
    x = np.linspace(0, 10, 100)
    y1 = np.sin(x)
    y2 = np.cos(x)
    
    # 创建图表
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # 线图
    axes[0, 0].plot(x, y1, label='sin(x)', linewidth=2)
    axes[0, 0].plot(x, y2, label='cos(x)', linewidth=2)
    axes[0, 0].set_title('线图示例')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 散点图
    x_scatter = np.random.normal(0, 1, 100)
    y_scatter = 2 * x_scatter + np.random.normal(0, 0.5, 100)
    axes[0, 1].scatter(x_scatter, y_scatter, alpha=0.6)
    axes[0, 1].set_title('散点图示例')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 柱状图
    categories = ['A', 'B', 'C', 'D']
    values = [23, 45, 56, 78]
    axes[1, 0].bar(categories, values, color='lightblue', edgecolor='black')
    axes[1, 0].set_title('柱状图示例')
    
    # 直方图
    data_hist = np.random.normal(0, 1, 1000)
    axes[1, 1].hist(data_hist, bins=30, alpha=0.7, color='lightgreen', edgecolor='black')
    axes[1, 1].set_title('直方图示例')
    
    plt.tight_layout()
    plt.show()
    
    print("✅ 图表展示完成!")

def main():
    """主函数"""
    print("🐍 第4章：Python数据科学基础")
    print("=" * 60)
    
    # 1. 运行基础教程
    numpy_tutorial()
    pandas_tutorial()
    matplotlib_tutorial()
    
    # 2. 运行完整销售分析项目
    analyzer = SalesDataAnalyzer()
    analyzer.run_complete_analysis()
    
    print("\\n🎓 学习总结:")
    print("  • 掌握了NumPy的数组操作和数学运算")
    print("  • 学会了Pandas的数据处理和分析技能")
    print("  • 熟悉了Matplotlib的数据可视化方法")
    print("  • 完成了完整的销售数据分析项目")
    print("  • 具备了数据科学的基础编程能力")
    
    return analyzer

if __name__ == "__main__":
    analyzer = main()

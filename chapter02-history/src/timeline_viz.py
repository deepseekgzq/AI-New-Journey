"""
第2章：AI发展时间线可视化
展示AI发展历史的交互式时间线
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pandas as pd
import json
import seaborn as sns
from datetime import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

def load_timeline_data():
    """加载时间线数据"""
    try:
        with open('../data/ai_timeline.json', 'r', encoding='utf-8') as f:
            data = json.load(f)
        return pd.DataFrame(data['ai_timeline'])
    except FileNotFoundError:
        # 如果文件不存在，返回示例数据
        return create_sample_timeline_data()

def create_sample_timeline_data():
    """创建示例时间线数据"""
    sample_data = [
        {"year": 1950, "event": "图灵测试提出", "impact": 9, "category": "理论基础"},
        {"year": 1956, "event": "达特茅斯会议", "impact": 10, "category": "学科建立"},
        {"year": 1970, "event": "第一次AI冬天", "impact": -5, "category": "发展低潮"},
        {"year": 1997, "event": "深蓝战胜卡斯帕罗夫", "impact": 8, "category": "里程碑"},
        {"year": 2016, "event": "AlphaGo战胜李世石", "impact": 10, "category": "里程碑"},
        {"year": 2022, "event": "ChatGPT发布", "impact": 10, "category": "应用突破"}
    ]
    return pd.DataFrame(sample_data)

def plot_ai_timeline_static():
    """绘制静态AI发展时间线"""
    df = load_timeline_data()
    
    fig, ax = plt.subplots(figsize=(16, 10))
    
    # 创建颜色映射
    categories = df['category'].unique()
    colors = plt.cm.Set3(np.linspace(0, 1, len(categories)))
    color_map = dict(zip(categories, colors))
    
    # 绘制时间线主轴
    ax.axhline(y=0, color='black', linewidth=2, alpha=0.3)
    
    # 绘制事件点
    for idx, row in df.iterrows():
        x = row['year']
        y = row['impact']
        category = row['category']
        event = row['event']
        
        # 选择点的位置（正负交替显示避免重叠）
        y_pos = abs(y) if idx % 2 == 0 else -abs(y)
        
        # 绘制事件点
        ax.scatter(x, y_pos, s=abs(y)*20, c=[color_map[category]], 
                  alpha=0.7, edgecolors='black', linewidth=1)
        
        # 添加连接线
        ax.plot([x, x], [0, y_pos], color='gray', alpha=0.5, linewidth=1)
        
        # 添加事件标签
        ax.annotate(event, (x, y_pos), 
                   xytext=(10, 10 if y_pos > 0 else -10), 
                   textcoords='offset points',
                   bbox=dict(boxstyle='round,pad=0.3', 
                           facecolor=color_map[category], alpha=0.7),
                   fontsize=8, ha='left')
    
    # 设置图形属性
    ax.set_xlabel('年份', fontsize=14, fontweight='bold')
    ax.set_ylabel('影响力指数', fontsize=14, fontweight='bold')
    ax.set_title('AI发展时间线：从理论到现实', fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # 添加图例
    legend_elements = [plt.scatter([], [], c=[color_map[cat]], s=100, label=cat) 
                      for cat in categories]
    ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1, 1))
    
    # 添加AI冬天的阴影区域
    winter_periods = [(1970, 1980), (1987, 1993)]
    for start, end in winter_periods:
        ax.axvspan(start, end, alpha=0.2, color='blue', label='AI冬天' if start == 1970 else "")
    
    plt.tight_layout()
    plt.show()
    
    print("📈 AI发展历程：起起伏伏，但总体向上！")

def plot_ai_waves():
    """绘制AI发展的三次浪潮"""
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # 定义三次浪潮的数据
    waves = [
        {
            'name': '第一次浪潮\n(1950s-1970s)',
            'period': (1950, 1970),
            'peak': 1965,
            'technologies': ['感知器', '专家系统', '符号推理'],
            'color': '#FF6B6B'
        },
        {
            'name': '第二次浪潮\n(1980s-2000s)',
            'period': (1980, 2000),
            'peak': 1990,
            'technologies': ['机器学习', '支持向量机', '随机森林'],
            'color': '#4ECDC4'
        },
        {
            'name': '第三次浪潮\n(2010s-至今)',
            'period': (2010, 2024),
            'peak': 2020,
            'technologies': ['深度学习', 'CNN', 'RNN', 'Transformer'],
            'color': '#45B7D1'
        }
    ]
    
    # 绘制每次浪潮
    for i, wave in enumerate(waves):
        start, end = wave['period']
        peak = wave['peak']
        
        # 创建钟形曲线
        x = np.linspace(start, end, 100)
        # 使用高斯函数模拟浪潮形状
        sigma = (end - start) / 6
        y = np.exp(-0.5 * ((x - peak) / sigma) ** 2) * (i + 1)
        
        ax.fill_between(x, 0, y, alpha=0.3, color=wave['color'], label=wave['name'])
        ax.plot(x, y, color=wave['color'], linewidth=2)
        
        # 添加技术标签
        tech_y = max(y) * 0.7
        for j, tech in enumerate(wave['technologies']):
            tech_x = peak + (j - len(wave['technologies'])/2 + 0.5) * sigma * 0.5
            ax.text(tech_x, tech_y - j*0.1, tech, 
                   ha='center', va='center', fontsize=9,
                   bbox=dict(boxstyle='round,pad=0.2', facecolor=wave['color'], alpha=0.5))
    
    # 标记重要事件
    events = [
        (1956, 1.5, '达特茅斯会议'),
        (1997, 2.2, '深蓝胜利'),
        (2012, 2.8, 'AlexNet'),
        (2016, 3.0, 'AlphaGo'),
        (2022, 3.2, 'ChatGPT')
    ]
    
    for year, height, event in events:
        ax.annotate(event, (year, height), 
                   xytext=(0, 20), textcoords='offset points',
                   arrowprops=dict(arrowstyle='->', color='red'),
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                   fontsize=8, ha='center')
    
    ax.set_xlabel('年份', fontsize=12, fontweight='bold')
    ax.set_ylabel('发展热度', fontsize=12, fontweight='bold')
    ax.set_title('AI发展的三次浪潮', fontsize=16, fontweight='bold')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print("🌊 三次浪潮，一浪更比一浪高！")

def plot_driving_forces():
    """绘制推动AI发展的三驾马车"""
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    
    years = list(range(1990, 2025, 5))
    
    # 算力增长（摩尔定律）
    computing_power = [1 * (2 ** ((year-1990)/2)) for year in years]  # 每两年翻倍
    ax1.semilogy(years, computing_power, 'o-', color='#FF6B6B', linewidth=3, markersize=8)
    ax1.set_title('算力增长\n(摩尔定律)', fontsize=14, fontweight='bold')
    ax1.set_ylabel('相对算力', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.fill_between(years, computing_power, alpha=0.3, color='#FF6B6B')
    
    # 数据量增长
    data_growth = [0.01 * (10 ** ((year-1990)/10)) for year in years]  # 指数增长
    ax2.semilogy(years, data_growth, 's-', color='#4ECDC4', linewidth=3, markersize=8)
    ax2.set_title('数据量增长\n(数字化时代)', fontsize=14, fontweight='bold')
    ax2.set_ylabel('数据量 (相对单位)', fontsize=12)
    ax2.set_xlabel('年份', fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.fill_between(years, data_growth, alpha=0.3, color='#4ECDC4')
    
    # 算法突破（离散事件）
    algorithm_breakthroughs = {
        1995: 0.3, 2000: 0.4, 2005: 0.5, 2010: 0.7, 
        2015: 0.8, 2020: 0.95, 2024: 1.0
    }
    
    breakthrough_years = list(algorithm_breakthroughs.keys())
    breakthrough_values = list(algorithm_breakthroughs.values())
    
    ax3.step(breakthrough_years, breakthrough_values, where='post', 
             color='#45B7D1', linewidth=3)
    ax3.scatter(breakthrough_years, breakthrough_values, 
               color='#45B7D1', s=100, zorder=5)
    ax3.set_title('算法突破\n(阶段性跃升)', fontsize=14, fontweight='bold')
    ax3.set_ylabel('算法性能', fontsize=12)
    ax3.grid(True, alpha=0.3)
    ax3.fill_between(breakthrough_years, breakthrough_values, 
                     step='post', alpha=0.3, color='#45B7D1')
    
    # 标注关键算法
    algorithms = {
        1995: 'SVM', 2000: 'Random Forest', 2005: 'Deep Learning',
        2010: 'CNN', 2015: 'ResNet', 2020: 'Transformer', 2024: 'LLM'
    }
    
    for year, alg in algorithms.items():
        if year in algorithm_breakthroughs:
            ax3.annotate(alg, (year, algorithm_breakthroughs[year]),
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=8, ha='left')
    
    plt.suptitle('推动AI发展的三驾马车', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    print("🚗 三驾马车并驾齐驱，缺一不可！")

def create_interactive_timeline():
    """创建交互式时间线（使用Plotly）"""
    df = load_timeline_data()
    
    # 创建散点图
    fig = px.scatter(df, x='year', y='impact', 
                    color='category', size=abs(df['impact']),
                    hover_data=['event'], 
                    title='AI发展时间线（交互式）',
                    labels={'year': '年份', 'impact': '影响力指数'})
    
    # 添加时间线
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    
    # 更新布局
    fig.update_layout(
        width=1000, height=600,
        showlegend=True,
        font=dict(family="Arial", size=12)
    )
    
    return fig

def analyze_milestone_events():
    """分析里程碑事件"""
    milestones = [
        {
            'event': '深蓝 vs 卡斯帕罗夫 (1997)',
            'significance': '首次在复杂策略游戏中战胜人类',
            'technology': '暴力搜索 + 启发式算法',
            'impact': 8
        },
        {
            'event': 'AlphaGo vs 李世石 (2016)', 
            'significance': '在最复杂棋类游戏中获胜',
            'technology': '深度学习 + 强化学习',
            'impact': 10
        },
        {
            'event': 'ChatGPT发布 (2022)',
            'significance': 'AI真正走向大众化应用',
            'technology': '大语言模型 + RLHF',
            'impact': 10
        }
    ]
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    events = [m['event'] for m in milestones]
    impacts = [m['impact'] for m in milestones]
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    
    bars = ax.bar(events, impacts, color=colors, alpha=0.7, edgecolor='black')
    
    # 添加数值标签
    for bar, milestone in zip(bars, milestones):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f"{height}/10", ha='center', va='bottom', fontweight='bold')
        
        # 添加技术说明
        ax.text(bar.get_x() + bar.get_width()/2., height/2,
                milestone['technology'], ha='center', va='center',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8),
                fontsize=9)
    
    ax.set_ylabel('影响力指数', fontsize=12, fontweight='bold')
    ax.set_title('AI发展史上的里程碑事件', fontsize=16, fontweight='bold')
    ax.set_ylim(0, 11)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()
    
    # 打印详细分析
    print("🏆 里程碑事件分析：")
    for milestone in milestones:
        print(f"• {milestone['event']}")
        print(f"  意义：{milestone['significance']}")
        print(f"  技术：{milestone['technology']}")
        print(f"  影响力：{milestone['impact']}/10\n")

def run_all_visualizations():
    """运行所有可视化"""
    print("🎭 第2章：AI历史可视化演示")
    print("=" * 50)
    
    print("\n1. AI发展时间线")
    plot_ai_timeline_static()
    
    print("\n2. AI发展的三次浪潮")
    plot_ai_waves()
    
    print("\n3. 推动AI发展的三驾马车")
    plot_driving_forces()
    
    print("\n4. 里程碑事件分析")
    analyze_milestone_events()
    
    print("\n✨ 历史告诉我们：AI的发展并非一帆风顺，但每次低潮后都有更大的突破！")

if __name__ == "__main__":
    run_all_visualizations()

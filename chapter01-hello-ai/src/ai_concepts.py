"""
第1章：AI概念演示代码
AI基本概念的可视化实现
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import seaborn as sns
from matplotlib.patches import Circle
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

def setup_chinese_font():
    """设置中文字体"""
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
    plt.rcParams['axes.unicode_minus'] = False

def visualize_ai_applications():
    """可视化AI应用实例"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle('AI在生活中的应用实例', fontsize=16, fontweight='bold')
    
    # 应用实例数据
    applications = [
        ('图像识别', ['照片自动分类', '人脸解锁', '医学影像诊断']),
        ('语音助手', ['Siri', 'Alexa', '小爱同学']),
        ('推荐系统', ['淘宝商品推荐', 'Netflix电影推荐', '抖音视频推荐']),
        ('自动驾驶', ['路径规划', '障碍物识别', '交通标志识别']),
        ('机器翻译', ['Google翻译', '百度翻译', '有道翻译']),
        ('游戏AI', ['AlphaGo', '游戏NPC', '策略优化'])
    ]
    
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD']
    
    for i, (app_name, examples) in enumerate(applications):
        row, col = i // 3, i % 3
        ax = axes[row, col]
        
        # 创建饼图
        sizes = [1] * len(examples)
        ax.pie(sizes, labels=examples, colors=[colors[i]] * len(examples), 
               autopct='', startangle=90)
        ax.set_title(app_name, fontweight='bold', fontsize=12)
    
    plt.tight_layout()
    plt.show()
    print("💡 看到了吗？AI已经悄悄进入我们生活的方方面面！")

def visualize_ml_types():
    """可视化机器学习类型"""
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # 创建三个学习类型的区域
    types = {
        '监督学习\n(Supervised Learning)': {
            'position': (2, 6),
            'color': '#FF6B6B',
            'examples': ['邮件分类(垃圾/正常)', '房价预测', '图像识别', '语音识别']
        },
        '无监督学习\n(Unsupervised Learning)': {
            'position': (8, 6),
            'color': '#4ECDC4',
            'examples': ['客户分群', '异常检测', '数据降维', '关联规则']
        },
        '强化学习\n(Reinforcement Learning)': {
            'position': (5, 2),
            'color': '#45B7D1',
            'examples': ['游戏AI', '自动驾驶', '机器人控制', '推荐优化']
        }
    }
    
    # 绘制每种学习类型
    for ml_type, info in types.items():
        x, y = info['position']
        
        # 绘制圆圈
        circle = Circle((x, y), 1.5, color=info['color'], alpha=0.7)
        ax.add_patch(circle)
        
        # 添加标题
        ax.text(x, y + 0.3, ml_type, ha='center', va='center', 
                fontweight='bold', fontsize=11)
        
        # 添加例子
        for i, example in enumerate(info['examples']):
            ax.text(x, y - 0.3 - i*0.2, f"• {example}", ha='center', va='center', 
                    fontsize=8)
    
    # 添加说明文字
    ax.text(2, 8.5, "有标准答案\n(老师教学)", ha='center', va='center', 
            bbox=dict(boxstyle="round,pad=0.3", facecolor='#FFE5E5'), fontsize=10)
    
    ax.text(8, 8.5, "无标准答案\n(自主探索)", ha='center', va='center', 
            bbox=dict(boxstyle="round,pad=0.3", facecolor='#E5F9F6'), fontsize=10)
    
    ax.text(5, 0.5, "通过奖惩学习\n(试错学习)", ha='center', va='center', 
            bbox=dict(boxstyle="round,pad=0.3", facecolor='#E5F2FF'), fontsize=10)
    
    # 设置图形属性
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 9)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('机器学习的三种主要类型', fontsize=16, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.show()
    print("🎓 机器学习就像人类的三种学习方式：跟老师学、自己摸索、通过奖惩学习！")

def visualize_deep_learning():
    """可视化深度学习网络结构"""
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))
    
    # 绘制神经网络结构
    layers = [
        {'name': '输入层', 'neurons': 4, 'x': 1, 'color': '#FF6B6B'},
        {'name': '隐藏层1\n(边缘检测)', 'neurons': 6, 'x': 3, 'color': '#4ECDC4'},
        {'name': '隐藏层2\n(形状识别)', 'neurons': 6, 'x': 5, 'color': '#45B7D1'},
        {'name': '隐藏层3\n(对象识别)', 'neurons': 4, 'x': 7, 'color': '#96CEB4'},
        {'name': '输出层', 'neurons': 3, 'x': 9, 'color': '#FFEAA7'}
    ]
    
    # 绘制神经元
    neuron_positions = {}
    for layer in layers:
        positions = []
        for i in range(layer['neurons']):
            y = 4 + (i - layer['neurons']/2 + 0.5) * 0.8
            circle = Circle((layer['x'], y), 0.3, color=layer['color'], alpha=0.8)
            ax.add_patch(circle)
            positions.append((layer['x'], y))
        neuron_positions[layer['x']] = positions
        
        # 添加层标签
        ax.text(layer['x'], 1.5, layer['name'], ha='center', va='center', 
                fontweight='bold', fontsize=9)
    
    # 绘制连接线
    for i in range(len(layers) - 1):
        current_x = layers[i]['x']
        next_x = layers[i+1]['x']
        
        for pos1 in neuron_positions[current_x]:
            for pos2 in neuron_positions[next_x]:
                ax.plot([pos1[0], pos2[0]], [pos1[1], pos2[1]], 
                       'gray', alpha=0.3, linewidth=0.5)
    
    # 添加学习层次说明
    learning_levels = [
        (1, 6, "原始数据\n(像素)"),
        (3, 6, "基础特征\n(线条、边缘)"),
        (5, 6, "复合特征\n(形状、纹理)"),
        (7, 6, "高级概念\n(眼睛、鼻子)"),
        (9, 6, "最终结果\n(猫、狗、人)")
    ]
    
    for x, y, text in learning_levels:
        ax.text(x, y, text, ha='center', va='center', 
                bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8),
                fontsize=8)
    
    # 添加箭头表示信息流向
    for i in range(len(layers) - 1):
        x1, x2 = layers[i]['x'], layers[i+1]['x']
        ax.annotate('', xy=(x2-0.4, 2.8), xytext=(x1+0.4, 2.8),
                    arrowprops=dict(arrowstyle='->', lw=2, color='red'))
    
    ax.text(5, 2.5, "信息流向：从简单到复杂", ha='center', va='center', 
            fontsize=12, fontweight='bold', color='red')
    
    # 设置图形属性
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 7)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('深度学习：层层递进的特征学习', fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    plt.show()
    print("🔥 深度学习的魔力：每一层都在学习更高级的特征！")

def visualize_ai_ml_dl_relationship():
    """可视化AI、ML、DL的关系"""
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    
    # 绘制同心圆表示包含关系
    # AI (最大圆)
    ai_circle = Circle((5, 5), 4, color='#FFE5E5', alpha=0.7, linewidth=3, edgecolor='#FF6B6B')
    ax.add_patch(ai_circle)
    
    # ML (中等圆)
    ml_circle = Circle((5, 5), 2.8, color='#E5F9F6', alpha=0.8, linewidth=3, edgecolor='#4ECDC4')
    ax.add_patch(ml_circle)
    
    # DL (最小圆)
    dl_circle = Circle((5, 5), 1.5, color='#E5F2FF', alpha=0.9, linewidth=3, edgecolor='#45B7D1')
    ax.add_patch(dl_circle)
    
    # 添加标签
    ax.text(5, 8.5, '人工智能 (AI)', ha='center', va='center', 
            fontsize=16, fontweight='bold', color='#FF6B6B')
    ax.text(5, 7.5, '让机器表现出智能行为的所有方法', ha='center', va='center', 
            fontsize=12, color='#666')
    
    ax.text(5, 6.8, '机器学习 (ML)', ha='center', va='center', 
            fontsize=14, fontweight='bold', color='#4ECDC4')
    ax.text(5, 6.3, '让机器从数据中学习的方法', ha='center', va='center', 
            fontsize=11, color='#666')
    
    ax.text(5, 5.5, '深度学习 (DL)', ha='center', va='center', 
            fontsize=12, fontweight='bold', color='#45B7D1')
    ax.text(5, 5, '使用多层神经网络的ML方法', ha='center', va='center', 
            fontsize=10, color='#666')
    
    # 添加AI的其他方法示例
    other_ai_methods = [
        (1.5, 7, "专家系统"),
        (8.5, 7, "搜索算法"),
        (1.5, 3, "符号推理"),
        (8.5, 3, "进化算法")
    ]
    
    for x, y, method in other_ai_methods:
        ax.text(x, y, method, ha='center', va='center', 
                bbox=dict(boxstyle="round,pad=0.3", facecolor='#FFE5E5', alpha=0.8),
                fontsize=9)
    
    # 添加ML的其他方法示例  
    other_ml_methods = [
        (2.5, 4, "决策树"),
        (7.5, 4, "支持向量机"),
        (2.5, 6, "贝叶斯方法"),
        (7.5, 6, "集成学习")
    ]
    
    for x, y, method in other_ml_methods:
        ax.text(x, y, method, ha='center', va='center', 
                bbox=dict(boxstyle="round,pad=0.3", facecolor='#E5F9F6', alpha=0.8),
                fontsize=9)
    
    # 添加DL的具体方法
    dl_methods = [
        (4, 4, "CNN"),
        (6, 4, "RNN"),
        (4, 3.5, "GAN"),
        (6, 3.5, "Transformer")
    ]
    
    for x, y, method in dl_methods:
        ax.text(x, y, method, ha='center', va='center', 
                bbox=dict(boxstyle="round,pad=0.2", facecolor='#E5F2FF', alpha=0.9),
                fontsize=8)
    
    # 添加关系说明
    ax.text(5, 1, "包含关系：AI ⊃ ML ⊃ DL", ha='center', va='center', 
            fontsize=14, fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.5", facecolor='yellow', alpha=0.8))
    
    # 设置图形属性
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('AI、ML、DL的"套娃"关系图', fontsize=18, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.show()
    print("🎯 现在你明白了吧！深度学习是机器学习的一种，机器学习是人工智能的一种！")

def visualize_ai_capabilities():
    """可视化AI能力雷达图"""
    import numpy as np
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8), subplot_kw=dict(projection='polar'))
    
    # AI擅长的领域
    categories_good = ['图像识别', '语音识别', '文本处理', '数据分析', '模式识别', '游戏策略']
    scores_good = [95, 90, 85, 98, 95, 92]  # AI在这些领域的表现分数
    
    # 计算角度
    angles = np.linspace(0, 2 * np.pi, len(categories_good), endpoint=False)
    scores_good += scores_good[:1]  # 闭合图形
    angles = np.concatenate((angles, [angles[0]]))
    
    # 绘制AI擅长的领域
    ax1.plot(angles, scores_good, 'o-', linewidth=2, color='#4ECDC4')
    ax1.fill(angles, scores_good, alpha=0.25, color='#4ECDC4')
    ax1.set_xticks(angles[:-1])
    ax1.set_xticklabels(categories_good, fontsize=10)
    ax1.set_ylim(0, 100)
    ax1.set_title('AI擅长的领域', fontsize=14, fontweight='bold', color='#4ECDC4', pad=20)
    ax1.grid(True)
    
    # AI不擅长的领域
    categories_weak = ['常识推理', '创新思维', '情感理解', '道德判断', '因果推理', '跨领域迁移']
    scores_weak = [30, 25, 35, 20, 40, 45]  # AI在这些领域的表现分数
    
    # 计算角度
    angles_weak = np.linspace(0, 2 * np.pi, len(categories_weak), endpoint=False)
    scores_weak += scores_weak[:1]  # 闭合图形
    angles_weak = np.concatenate((angles_weak, [angles_weak[0]]))
    
    # 绘制AI不擅长的领域
    ax2.plot(angles_weak, scores_weak, 'o-', linewidth=2, color='#FF6B6B')
    ax2.fill(angles_weak, scores_weak, alpha=0.25, color='#FF6B6B')
    ax2.set_xticks(angles_weak[:-1])
    ax2.set_xticklabels(categories_weak, fontsize=10)
    ax2.set_ylim(0, 100)
    ax2.set_title('AI的局限性领域', fontsize=14, fontweight='bold', color='#FF6B6B', pad=20)
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()
    print("⚖️ 理性看待AI：既要看到它的强大，也要了解它的局限性！")

def run_all_demos():
    """运行所有演示"""
    print("🚀 开始AI概念可视化演示...")
    
    setup_chinese_font()
    
    print("\n1. AI应用实例")
    visualize_ai_applications()
    
    print("\n2. 机器学习类型")
    visualize_ml_types()
    
    print("\n3. 深度学习网络结构")
    visualize_deep_learning()
    
    print("\n4. AI、ML、DL关系图")
    visualize_ai_ml_dl_relationship()
    
    print("\n5. AI能力分析")
    visualize_ai_capabilities()
    
    print("\n🎉 所有演示完成！")

if __name__ == "__main__":
    run_all_demos()

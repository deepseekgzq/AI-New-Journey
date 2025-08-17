"""
第1章工具函数
提供通用的辅助功能
"""

import matplotlib.pyplot as plt
import numpy as np

def setup_matplotlib_chinese():
    """配置matplotlib支持中文显示"""
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    return True

def create_color_palette():
    """创建统一的颜色调色板"""
    colors = {
        'primary': '#FF6B6B',      # 红色
        'secondary': '#4ECDC4',    # 青色
        'accent': '#45B7D1',       # 蓝色
        'success': '#96CEB4',      # 绿色
        'warning': '#FFEAA7',      # 黄色
        'info': '#DDA0DD',         # 紫色
        'light': '#F8F9FA',        # 浅灰
        'dark': '#343A40'          # 深灰
    }
    return colors

def save_figure(fig, filename, dpi=300, bbox_inches='tight'):
    """保存图形到文件"""
    fig.savefig(filename, dpi=dpi, bbox_inches=bbox_inches, 
                facecolor='white', edgecolor='none')
    print(f"✅ 图形已保存到：{filename}")

def create_info_box(text, box_type='info'):
    """创建信息框样式"""
    colors = create_color_palette()
    
    box_styles = {
        'info': {'facecolor': colors['info'], 'alpha': 0.2},
        'success': {'facecolor': colors['success'], 'alpha': 0.2},
        'warning': {'facecolor': colors['warning'], 'alpha': 0.2},
        'error': {'facecolor': colors['primary'], 'alpha': 0.2}
    }
    
    return dict(boxstyle="round,pad=0.3", **box_styles.get(box_type, box_styles['info']))

def print_section_header(title, emoji="🎯"):
    """打印章节标题"""
    print(f"\n{emoji} {title}")
    print("=" * (len(title) + 3))

def print_subsection_header(title, emoji="📍"):
    """打印子章节标题"""
    print(f"\n{emoji} {title}")
    print("-" * (len(title) + 3))

def create_concept_summary():
    """创建概念总结"""
    summary = {
        "AI (人工智能)": {
            "定义": "让机器表现出智能行为的能力",
            "特点": "解决特定问题的智能系统",
            "误区": "不等于机器人或通用智能"
        },
        "ML (机器学习)": {
            "定义": "让机器从数据中学习规律的方法",
            "核心": "数据 → 模式 → 预测",
            "类型": "监督学习、无监督学习、强化学习"
        },
        "DL (深度学习)": {
            "定义": "使用多层神经网络的机器学习方法",
            "特点": "层层递进学习复杂特征",
            "应用": "图像识别、语音处理、自然语言"
        }
    }
    return summary

def display_concept_summary():
    """显示概念总结"""
    summary = create_concept_summary()
    
    print_section_header("第1章概念总结", "📚")
    
    for concept, details in summary.items():
        print_subsection_header(concept)
        for key, value in details.items():
            print(f"  • {key}: {value}")
    
    print_subsection_header("关系图", "🔗")
    print("  • AI ⊃ 机器学习 ⊃ 深度学习")
    print("  • 包含关系：大圈套小圈")
    print("  • 每一层都是上一层的实现方式")

if __name__ == "__main__":
    # 测试工具函数
    setup_matplotlib_chinese()
    colors = create_color_palette()
    print("颜色调色板:", colors)
    display_concept_summary()

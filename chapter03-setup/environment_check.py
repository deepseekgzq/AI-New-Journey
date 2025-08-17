"""
第3章：环境检测脚本
检查AI开发环境是否正确配置
"""

import sys
import subprocess
import importlib
import platform
from datetime import datetime

def print_header():
    """打印检测头部信息"""
    print("=" * 60)
    print("🔧 AI开发环境检测工具")
    print("=" * 60)
    print(f"检测时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"操作系统: {platform.system()} {platform.release()}")
    print(f"Python版本: {sys.version}")
    print("=" * 60)

def check_python_version():
    """检查Python版本"""
    print("\n📍 检查Python版本...")
    
    version = sys.version_info
    if version.major == 3 and version.minor >= 8:
        print(f"✅ Python {version.major}.{version.minor}.{version.micro} - 版本符合要求")
        return True
    else:
        print(f"❌ Python {version.major}.{version.minor}.{version.micro} - 需要Python 3.8+")
        return False

def check_package(package_name, import_name=None, min_version=None):
    """检查单个包是否安装"""
    if import_name is None:
        import_name = package_name
    
    try:
        module = importlib.import_module(import_name)
        version = getattr(module, '__version__', 'Unknown')
        
        # 检查最小版本要求
        if min_version and version != 'Unknown':
            from packaging import version as pkg_version
            if pkg_version.parse(version) < pkg_version.parse(min_version):
                print(f"⚠️ {package_name} {version} - 建议升级到{min_version}+")
                return False
        
        print(f"✅ {package_name} {version}")
        return True
    except ImportError:
        print(f"❌ {package_name} - 未安装")
        return False
    except Exception as e:
        print(f"⚠️ {package_name} - 检测异常: {str(e)}")
        return False

def check_ai_packages():
    """检查AI相关包"""
    print("\n📍 检查AI核心包...")
    
    packages = [
        # 基础数据科学包
        ('numpy', 'numpy', '1.21.0'),
        ('pandas', 'pandas', '1.3.0'),
        ('matplotlib', 'matplotlib', '3.4.0'),
        ('seaborn', 'seaborn', '0.11.0'),
        ('plotly', 'plotly', '5.0.0'),
        
        # 机器学习包
        ('scikit-learn', 'sklearn', '1.0.0'),
        ('xgboost', 'xgboost', '1.5.0'),
        
        # 深度学习包
        ('tensorflow', 'tensorflow', '2.8.0'),
        ('torch', 'torch', '1.11.0'),
        
        # 自然语言处理
        ('transformers', 'transformers', '4.15.0'),
        ('nltk', 'nltk', '3.7'),
        
        # 开发工具
        ('jupyter', 'jupyter', '1.0.0'),
        ('notebook', 'notebook', '6.4.0'),
        ('ipython', 'IPython', '7.30.0'),
    ]
    
    success_count = 0
    for package_info in packages:
        if len(package_info) == 3:
            package, import_name, min_version = package_info
        else:
            package, import_name = package_info
            min_version = None
        
        if check_package(package, import_name, min_version):
            success_count += 1
    
    print(f"\n📊 包检查结果: {success_count}/{len(packages)} 个包可用")
    return success_count, len(packages)

def check_jupyter():
    """检查Jupyter环境"""
    print("\n📍 检查Jupyter环境...")
    
    try:
        result = subprocess.run(['jupyter', '--version'], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print("✅ Jupyter已安装")
            print(f"版本信息:\n{result.stdout}")
            return True
        else:
            print("❌ Jupyter未正确安装")
            return False
    except subprocess.TimeoutExpired:
        print("⚠️ Jupyter检查超时")
        return False
    except FileNotFoundError:
        print("❌ 未找到Jupyter命令")
        return False
    except Exception as e:
        print(f"⚠️ Jupyter检查异常: {str(e)}")
        return False

def check_gpu_support():
    """检查GPU支持"""
    print("\n📍 检查GPU支持...")
    
    # 检查TensorFlow GPU支持
    try:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print(f"✅ TensorFlow检测到{len(gpus)}个GPU设备")
            for i, gpu in enumerate(gpus):
                print(f"  GPU {i}: {gpu.name}")
        else:
            print("ℹ️ TensorFlow未检测到GPU设备")
    except ImportError:
        print("⚠️ TensorFlow未安装，无法检查GPU支持")
    except Exception as e:
        print(f"⚠️ TensorFlow GPU检查异常: {str(e)}")
    
    # 检查PyTorch GPU支持
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✅ PyTorch CUDA可用，检测到{torch.cuda.device_count()}个GPU")
            for i in range(torch.cuda.device_count()):
                print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        else:
            print("ℹ️ PyTorch CUDA不可用")
    except ImportError:
        print("⚠️ PyTorch未安装，无法检查GPU支持")
    except Exception as e:
        print(f"⚠️ PyTorch GPU检查异常: {str(e)}")

def run_hello_ai():
    """运行第一个AI程序"""
    print("\n📍 运行Hello AI程序...")
    
    try:
        import numpy as np
        import matplotlib.pyplot as plt
        
        # 生成简单的数据
        x = np.linspace(0, 10, 100)
        y = np.sin(x) + np.random.normal(0, 0.1, 100)
        
        # 创建简单的线性回归
        from sklearn.linear_model import LinearRegression
        from sklearn.preprocessing import PolynomialFeatures
        
        # 多项式特征
        poly_features = PolynomialFeatures(degree=3)
        X_poly = poly_features.fit_transform(x.reshape(-1, 1))
        
        # 训练模型
        model = LinearRegression()
        model.fit(X_poly, y)
        
        # 预测
        y_pred = model.predict(X_poly)
        
        print("✅ Hello AI程序运行成功！")
        print(f"  • 生成了{len(x)}个数据点")
        print(f"  • 训练了多项式回归模型")
        print(f"  • 模型拟合分数: {model.score(X_poly, y):.4f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Hello AI程序运行失败: {str(e)}")
        return False

def provide_recommendations(python_ok, packages_success, packages_total, jupyter_ok, hello_ai_ok):
    """提供改进建议"""
    print("\n" + "=" * 60)
    print("💡 环境优化建议")
    print("=" * 60)
    
    if not python_ok:
        print("🔧 Python版本建议:")
        print("  • 升级到Python 3.8或更高版本")
        print("  • 推荐使用Anaconda进行Python环境管理")
    
    if packages_success < packages_total * 0.8:
        print("🔧 包安装建议:")
        print("  • 运行: pip install -r requirements.txt")
        print("  • 或使用conda: conda install package_name")
        print("  • 创建虚拟环境避免包冲突")
    
    if not jupyter_ok:
        print("🔧 Jupyter安装建议:")
        print("  • 安装Jupyter: pip install jupyter")
        print("  • 或使用conda: conda install jupyter")
    
    if not hello_ai_ok:
        print("🔧 Hello AI程序建议:")
        print("  • 确保已安装numpy, scikit-learn, matplotlib")
        print("  • 检查是否有包版本冲突")
    
    print("\n📚 学习资源:")
    print("  • Anaconda安装: https://www.anaconda.com/")
    print("  • Jupyter文档: https://jupyter.org/")
    print("  • Python官网: https://python.org/")

def main():
    """主函数"""
    print_header()
    
    # 执行各项检查
    python_ok = check_python_version()
    packages_success, packages_total = check_ai_packages()
    jupyter_ok = check_jupyter()
    check_gpu_support()
    hello_ai_ok = run_hello_ai()
    
    # 生成总结报告
    print("\n" + "=" * 60)
    print("📋 环境检测总结")
    print("=" * 60)
    
    total_score = 0
    max_score = 4
    
    if python_ok:
        print("✅ Python版本检查通过")
        total_score += 1
    else:
        print("❌ Python版本检查失败")
    
    if packages_success >= packages_total * 0.8:
        print("✅ AI包安装检查通过")
        total_score += 1
    else:
        print("❌ AI包安装检查失败")
    
    if jupyter_ok:
        print("✅ Jupyter环境检查通过")
        total_score += 1
    else:
        print("❌ Jupyter环境检查失败")
    
    if hello_ai_ok:
        print("✅ Hello AI程序运行通过")
        total_score += 1
    else:
        print("❌ Hello AI程序运行失败")
    
    print(f"\n🎯 总体评分: {total_score}/{max_score}")
    
    if total_score == max_score:
        print("🎉 恭喜！您的AI开发环境已完美配置！")
    elif total_score >= max_score * 0.75:
        print("👍 您的环境基本就绪，有少量问题需要解决")
    else:
        print("⚠️ 您的环境需要进一步配置")
    
    # 提供建议
    provide_recommendations(python_ok, packages_success, packages_total, jupyter_ok, hello_ai_ok)
    
    print("\n🚀 环境检测完成！准备开始您的AI学习之旅吧！")

if __name__ == "__main__":
    main()

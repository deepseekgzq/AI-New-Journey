"""
第11章：CNN手写数字识别完整实现
MNIST数据集手写数字识别项目
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

class MNISTCNNClassifier:
    """MNIST手写数字CNN分类器"""
    
    def __init__(self):
        """初始化"""
        self.model = None
        self.history = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.y_train_cat = None
        self.y_test_cat = None
        self.class_names = [str(i) for i in range(10)]
        
    def load_data(self):
        """加载MNIST数据集"""
        print("📂 加载MNIST数据集...")
        
        # 加载数据
        (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
        
        print(f"✅ 数据加载成功！")
        print(f"   - 训练集: {X_train.shape[0]} 张图片")
        print(f"   - 测试集: {X_test.shape[0]} 张图片")
        print(f"   - 图片尺寸: {X_train.shape[1]}×{X_train.shape[2]}")
        print(f"   - 类别数量: {len(np.unique(y_train))}")
        
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        
        return X_train, X_test, y_train, y_test
    
    def explore_data(self):
        """数据探索性分析"""
        print("\n🔍 数据探索分析...")
        
        # 数据基本信息
        print(f"训练集形状: {self.X_train.shape}")
        print(f"测试集形状: {self.X_test.shape}")
        print(f"像素值范围: {self.X_train.min()} - {self.X_train.max()}")
        
        # 类别分布
        unique, counts = np.unique(self.y_train, return_counts=True)
        print(f"类别分布: {dict(zip(unique, counts))}")
        
        # 可视化数据样本
        fig, axes = plt.subplots(3, 4, figsize=(12, 9))
        
        # 显示每个数字的示例
        for i in range(10):
            if i < 12:  # 只显示前12个
                row, col = i // 4, i % 4
                if row < 3:
                    # 找到第一个该数字的样本
                    idx = np.where(self.y_train == i)[0][0]
                    axes[row, col].imshow(self.X_train[idx], cmap='gray')
                    axes[row, col].set_title(f'数字 {i}', fontweight='bold')
                    axes[row, col].axis('off')
        
        # 隐藏多余的子图
        for i in range(10, 12):
            row, col = i // 4, i % 4
            if row < 3:
                axes[row, col].axis('off')
        
        plt.suptitle('MNIST数据集样本展示', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()
        
        # 类别分布柱状图
        plt.figure(figsize=(10, 6))
        plt.bar(unique, counts, color='skyblue', edgecolor='black', alpha=0.7)
        plt.title('训练集类别分布', fontsize=14, fontweight='bold')
        plt.xlabel('数字类别')
        plt.ylabel('样本数量')
        plt.xticks(unique)
        plt.grid(True, alpha=0.3)
        
        # 添加数值标签
        for i, count in enumerate(counts):
            plt.text(i, count + 50, str(count), ha='center', va='bottom')
        
        plt.tight_layout()
        plt.show()
        
        print("💡 数据特点:")
        print("  • 图片尺寸: 28×28像素")
        print("  • 灰度图像，像素值0-255")
        print("  • 10个类别，分布相对均匀")
        print("  • 训练集6万张，测试集1万张")
    
    def preprocess_data(self):
        """数据预处理"""
        print("\n🔧 数据预处理...")
        
        # 归一化像素值到0-1范围
        self.X_train = self.X_train.astype('float32') / 255.0
        self.X_test = self.X_test.astype('float32') / 255.0
        
        # 重塑数据为CNN输入格式 (样本数, 高, 宽, 通道数)
        self.X_train = self.X_train.reshape(-1, 28, 28, 1)
        self.X_test = self.X_test.reshape(-1, 28, 28, 1)
        
        # 将标签转换为one-hot编码
        self.y_train_cat = to_categorical(self.y_train, 10)
        self.y_test_cat = to_categorical(self.y_test, 10)
        
        print(f"✅ 数据预处理完成!")
        print(f"   - 像素值已归一化到 [0, 1]")
        print(f"   - 训练数据形状: {self.X_train.shape}")
        print(f"   - 测试数据形状: {self.X_test.shape}")
        print(f"   - 标签已转换为one-hot编码")
        
        return self.X_train, self.X_test, self.y_train_cat, self.y_test_cat
    
    def build_model(self):
        """构建CNN模型"""
        print("\n🏗️ 构建CNN模型...")
        
        model = models.Sequential([
            # 第一个卷积块
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
            layers.BatchNormalization(),
            layers.Conv2D(32, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # 第二个卷积块
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # 第三个卷积块
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.Dropout(0.25),
            
            # 展平和全连接层
            layers.Flatten(),
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(10, activation='softmax')
        ])
        
        # 编译模型
        model.compile(
            optimizer=optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        
        print("✅ CNN模型构建完成!")
        print(f"   - 总参数数量: {model.count_params():,}")
        
        # 显示模型结构
        print("\n📋 模型架构:")
        model.summary()
        
        return model
    
    def visualize_model_architecture(self):
        """可视化模型架构"""
        print("\n🎨 可视化模型架构...")
        
        # 创建一个示例输入来可视化特征映射
        sample_input = self.X_train[0:1]  # 取第一个样本
        
        # 获取每层的输出
        layer_outputs = []
        layer_names = []
        
        for layer in self.model.layers:
            if isinstance(layer, (layers.Conv2D, layers.MaxPooling2D)):
                layer_outputs.append(layer.output)
                layer_names.append(layer.name)
        
        # 创建激活模型
        if layer_outputs:
            activation_model = models.Model(inputs=self.model.input, outputs=layer_outputs)
            activations = activation_model.predict(sample_input)
            
            # 可视化前几层的特征映射
            fig, axes = plt.subplots(2, 4, figsize=(16, 8))
            
            for i, (activation, name) in enumerate(zip(activations[:8], layer_names[:8])):
                if i < 8:
                    row, col = i // 4, i % 4
                    
                    if len(activation.shape) == 4:  # 卷积层输出
                        # 显示第一个特征映射
                        feature_map = activation[0, :, :, 0]
                        axes[row, col].imshow(feature_map, cmap='viridis')
                        axes[row, col].set_title(f'{name}\\n{activation.shape[1:]}', fontsize=10)
                        axes[row, col].axis('off')
            
            plt.suptitle('CNN特征映射可视化', fontsize=16, fontweight='bold')
            plt.tight_layout()
            plt.show()
    
    def train_model(self, epochs=20, batch_size=128, validation_split=0.1):
        """训练CNN模型"""
        print(f"\\n🤖 训练CNN模型 (epochs={epochs}, batch_size={batch_size})...")
        
        # 设置回调函数
        callbacks_list = [
            callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=5,
                restore_best_weights=True
            ),
            callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=3,
                min_lr=1e-7
            )
        ]
        
        # 训练模型
        self.history = self.model.fit(
            self.X_train, self.y_train_cat,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=callbacks_list,
            verbose=1
        )
        
        print("✅ 模型训练完成!")
        
        return self.history
    
    def plot_training_history(self):
        """绘制训练历史"""
        print("\\n📈 可视化训练过程...")
        
        if self.history is None:
            print("❌ 没有训练历史记录")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # 准确率曲线
        ax1.plot(self.history.history['accuracy'], label='训练准确率', linewidth=2)
        ax1.plot(self.history.history['val_accuracy'], label='验证准确率', linewidth=2)
        ax1.set_title('模型准确率', fontweight='bold')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('准确率')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 损失曲线
        ax2.plot(self.history.history['loss'], label='训练损失', linewidth=2)
        ax2.plot(self.history.history['val_loss'], label='验证损失', linewidth=2)
        ax2.set_title('模型损失', fontweight='bold')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('损失值')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # 打印最佳性能
        best_val_acc = max(self.history.history['val_accuracy'])
        best_epoch = self.history.history['val_accuracy'].index(best_val_acc) + 1
        print(f"📊 最佳验证准确率: {best_val_acc:.4f} (Epoch {best_epoch})")
    
    def evaluate_model(self):
        """评估模型性能"""
        print("\\n📊 模型性能评估...")
        
        # 测试集评估
        test_loss, test_accuracy = self.model.evaluate(self.X_test, self.y_test_cat, verbose=0)
        print(f"✅ 测试集性能:")
        print(f"   - 准确率: {test_accuracy:.4f}")
        print(f"   - 损失值: {test_loss:.4f}")
        
        # 预测
        y_pred_proba = self.model.predict(self.X_test, verbose=0)
        y_pred = np.argmax(y_pred_proba, axis=1)
        
        # 分类报告
        print("\\n📋 详细分类报告:")
        print(classification_report(self.y_test, y_pred, target_names=self.class_names))
        
        # 混淆矩阵
        cm = confusion_matrix(self.y_test, y_pred)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=self.class_names, yticklabels=self.class_names)
        plt.title('混淆矩阵', fontsize=16, fontweight='bold')
        plt.xlabel('预测标签')
        plt.ylabel('真实标签')
        plt.tight_layout()
        plt.show()
        
        return test_accuracy, test_loss, y_pred, y_pred_proba
    
    def visualize_predictions(self, num_samples=12):
        """可视化预测结果"""
        print(f"\\n🔮 可视化预测结果 (显示{num_samples}个样本)...")
        
        # 获取预测
        y_pred_proba = self.model.predict(self.X_test, verbose=0)
        y_pred = np.argmax(y_pred_proba, axis=1)
        
        # 选择一些样本进行可视化
        indices = np.random.choice(len(self.X_test), num_samples, replace=False)
        
        fig, axes = plt.subplots(3, 4, figsize=(12, 9))
        axes = axes.ravel()
        
        for i, idx in enumerate(indices):
            if i < num_samples:
                # 显示图片
                axes[i].imshow(self.X_test[idx].reshape(28, 28), cmap='gray')
                
                # 预测结果
                true_label = self.y_test[idx]
                pred_label = y_pred[idx]
                confidence = y_pred_proba[idx][pred_label]
                
                # 设置标题颜色
                color = 'green' if true_label == pred_label else 'red'
                
                axes[i].set_title(f'真实: {true_label}\\n预测: {pred_label}\\n置信度: {confidence:.2f}', 
                                color=color, fontweight='bold')
                axes[i].axis('off')
        
        plt.suptitle('预测结果可视化', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()
        
        # 统计准确预测的数量
        correct_predictions = np.sum(y_pred == self.y_test)
        total_predictions = len(self.y_test)
        accuracy = correct_predictions / total_predictions
        
        print(f"📈 预测统计:")
        print(f"   - 正确预测: {correct_predictions}/{total_predictions}")
        print(f"   - 准确率: {accuracy:.4f}")
    
    def predict_single_image(self, image, show_image=True):
        """预测单张图片"""
        if image.shape != (28, 28):
            raise ValueError("图片尺寸必须是28x28")
        
        # 预处理图片
        if image.max() > 1.0:
            image = image.astype('float32') / 255.0
        
        image = image.reshape(1, 28, 28, 1)
        
        # 预测
        prediction_proba = self.model.predict(image, verbose=0)
        prediction = np.argmax(prediction_proba)
        confidence = prediction_proba[0][prediction]
        
        if show_image:
            plt.figure(figsize=(6, 4))
            plt.subplot(1, 2, 1)
            plt.imshow(image.reshape(28, 28), cmap='gray')
            plt.title('输入图片', fontweight='bold')
            plt.axis('off')
            
            plt.subplot(1, 2, 2)
            plt.bar(range(10), prediction_proba[0])
            plt.title(f'预测: {prediction} (置信度: {confidence:.3f})', fontweight='bold')
            plt.xlabel('数字类别')
            plt.ylabel('概率')
            plt.xticks(range(10))
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.show()
        
        return prediction, confidence, prediction_proba[0]
    
    def save_model(self, filepath='mnist_cnn_model.h5'):
        """保存模型"""
        if self.model is not None:
            self.model.save(filepath)
            print(f"✅ 模型已保存到 {filepath}")
        else:
            print("❌ 没有可保存的模型")
    
    def load_model(self, filepath='mnist_cnn_model.h5'):
        """加载模型"""
        try:
            self.model = tf.keras.models.load_model(filepath)
            print(f"✅ 模型已从 {filepath} 加载")
            return self.model
        except Exception as e:
            print(f"❌ 模型加载失败: {str(e)}")
            return None
    
    def run_complete_project(self):
        """运行完整的CNN项目"""
        print("🖼️ 开始MNIST手写数字识别项目...")
        print("=" * 60)
        
        # 1. 加载数据
        self.load_data()
        
        # 2. 数据探索
        self.explore_data()
        
        # 3. 数据预处理
        self.preprocess_data()
        
        # 4. 构建模型
        self.build_model()
        
        # 5. 可视化模型架构
        self.visualize_model_architecture()
        
        # 6. 训练模型
        self.train_model(epochs=10)  # 减少epochs以便快速演示
        
        # 7. 可视化训练过程
        self.plot_training_history()
        
        # 8. 评估模型
        test_acc, test_loss, y_pred, y_pred_proba = self.evaluate_model()
        
        # 9. 可视化预测结果
        self.visualize_predictions()
        
        # 10. 单张图片预测示例
        print("\\n🔍 单张图片预测示例:")
        sample_idx = np.random.randint(0, len(self.X_test))
        sample_image = self.X_test[sample_idx].reshape(28, 28)
        prediction, confidence, proba = self.predict_single_image(sample_image)
        
        print(f"真实标签: {self.y_test[sample_idx]}")
        print(f"预测结果: {prediction}")
        print(f"预测置信度: {confidence:.4f}")
        
        # 11. 保存模型
        self.save_model()
        
        print("\\n🎯 项目总结:")
        print(f"  • 数据集: MNIST (60,000训练 + 10,000测试)")
        print(f"  • 模型: CNN (卷积神经网络)")
        print(f"  • 测试准确率: {test_acc:.4f}")
        print(f"  • 模型参数: {self.model.count_params():,}")
        print("\\n🎉 MNIST手写数字识别项目完成!")
        
        return self

def demonstrate_cnn_concepts():
    """演示CNN核心概念"""
    print("\\n" + "="*60)
    print("🧠 CNN核心概念演示")
    print("="*60)
    
    # 1. 卷积操作演示
    print("\\n1️⃣ 卷积操作演示:")
    
    # 创建一个简单的5x5图像
    image = np.array([
        [0, 0, 1, 0, 0],
        [0, 1, 1, 1, 0],
        [1, 1, 1, 1, 1],
        [0, 1, 1, 1, 0],
        [0, 0, 1, 0, 0]
    ])
    
    # 边缘检测卷积核
    kernel = np.array([
        [-1, -1, -1],
        [-1,  8, -1],
        [-1, -1, -1]
    ])
    
    # 手动卷积计算
    def manual_convolution(image, kernel):
        result = np.zeros((3, 3))
        for i in range(3):
            for j in range(3):
                result[i, j] = np.sum(image[i:i+3, j:j+3] * kernel)
        return result
    
    conv_result = manual_convolution(image, kernel)
    
    # 可视化
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    
    axes[0].imshow(image, cmap='gray')
    axes[0].set_title('原始图像 (5×5)', fontweight='bold')
    axes[0].grid(True)
    
    axes[1].imshow(kernel, cmap='RdBu')
    axes[1].set_title('卷积核 (3×3)\\n边缘检测', fontweight='bold')
    
    axes[2].imshow(conv_result, cmap='gray')
    axes[2].set_title('卷积结果 (3×3)', fontweight='bold')
    
    for ax in axes:
        ax.set_xticks(range(ax.get_xlim()[1]))
        ax.set_yticks(range(ax.get_ylim()[1]))
    
    plt.tight_layout()
    plt.show()
    
    print("   ✅ 卷积操作将局部特征转换为特征映射")
    
    # 2. 池化操作演示
    print("\\n2️⃣ 池化操作演示:")
    
    # 创建一个4x4特征映射
    feature_map = np.random.randint(0, 10, (4, 4))
    
    # 最大池化
    def max_pooling(feature_map, pool_size=2):
        h, w = feature_map.shape
        result = np.zeros((h//pool_size, w//pool_size))
        for i in range(0, h, pool_size):
            for j in range(0, w, pool_size):
                result[i//pool_size, j//pool_size] = np.max(feature_map[i:i+pool_size, j:j+pool_size])
        return result
    
    pooled_result = max_pooling(feature_map)
    
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    
    im1 = axes[0].imshow(feature_map, cmap='viridis')
    axes[0].set_title('原始特征映射 (4×4)', fontweight='bold')
    plt.colorbar(im1, ax=axes[0])
    
    im2 = axes[1].imshow(pooled_result, cmap='viridis')
    axes[1].set_title('最大池化结果 (2×2)', fontweight='bold')
    plt.colorbar(im2, ax=axes[1])
    
    # 添加网格和数值
    for ax, data in zip(axes, [feature_map, pooled_result]):
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                ax.text(j, i, f'{data[i,j]:.0f}', ha='center', va='center', 
                       color='white', fontweight='bold')
    
    plt.tight_layout()
    plt.show()
    
    print("   ✅ 池化操作减少特征映射尺寸，保留重要信息")

def main():
    """主函数"""
    print("🖼️ 第11章：CNN与手写数字识别")
    print("=" * 60)
    
    # 1. 运行完整项目
    classifier = MNISTCNNClassifier()
    classifier.run_complete_project()
    
    # 2. 演示CNN概念
    demonstrate_cnn_concepts()
    
    print("\\n🎓 学习总结:")
    print("  • CNN擅长处理图像数据")
    print("  • 卷积层提取局部特征")
    print("  • 池化层减少计算量")
    print("  • 深度学习需要大量数据和计算资源")
    
    return classifier

if __name__ == "__main__":
    classifier = main()

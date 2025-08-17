"""
第12章：RNN情感分析完整实现
电影评论情感分类项目
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import re
import string
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

class SentimentAnalyzer:
    """情感分析器"""
    
    def __init__(self, max_features=20000, max_length=200):
        """初始化"""
        self.max_features = max_features  # 词汇表大小
        self.max_length = max_length      # 序列最大长度
        self.tokenizer = None
        self.model = None
        self.history = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
    def generate_sample_data(self, n_samples=10000):
        """生成模拟电影评论数据"""
        print("📝 生成模拟电影评论数据...")
        
        np.random.seed(42)
        
        # 正面评论的常用词汇
        positive_words = [
            'excellent', 'amazing', 'wonderful', 'fantastic', 'great', 'awesome',
            'brilliant', 'outstanding', 'perfect', 'incredible', 'superb', 'magnificent',
            'love', 'beautiful', 'impressive', 'stunning', 'remarkable', 'exceptional',
            'good', 'nice', 'enjoy', 'happy', 'pleased', 'satisfied', 'delighted'
        ]
        
        # 负面评论的常用词汇
        negative_words = [
            'terrible', 'awful', 'horrible', 'bad', 'worst', 'disgusting',
            'boring', 'stupid', 'waste', 'disappointing', 'pathetic', 'useless',
            'hate', 'annoying', 'frustrating', 'ridiculous', 'nonsense', 'garbage',
            'poor', 'weak', 'failed', 'mess', 'disaster', 'unbearable'
        ]
        
        # 中性词汇
        neutral_words = [
            'movie', 'film', 'story', 'plot', 'character', 'actor', 'actress',
            'director', 'scene', 'dialogue', 'music', 'sound', 'visual', 'effect',
            'action', 'drama', 'comedy', 'thriller', 'romance', 'adventure',
            'watch', 'see', 'think', 'feel', 'experience', 'show', 'performance'
        ]
        
        reviews = []
        labels = []
        
        for i in range(n_samples):
            # 随机决定情感
            sentiment = np.random.choice([0, 1])  # 0: 负面, 1: 正面
            
            if sentiment == 1:  # 正面评论
                # 选择更多正面词汇
                chosen_positive = np.random.choice(positive_words, 
                                                 size=np.random.randint(3, 8), replace=True)
                chosen_neutral = np.random.choice(neutral_words, 
                                                size=np.random.randint(5, 12), replace=True)
                chosen_negative = np.random.choice(negative_words, 
                                                 size=np.random.randint(0, 2), replace=True)
                words = np.concatenate([chosen_positive, chosen_neutral, chosen_negative])
            else:  # 负面评论
                # 选择更多负面词汇
                chosen_negative = np.random.choice(negative_words, 
                                                 size=np.random.randint(3, 8), replace=True)
                chosen_neutral = np.random.choice(neutral_words, 
                                                size=np.random.randint(5, 12), replace=True)
                chosen_positive = np.random.choice(positive_words, 
                                                  size=np.random.randint(0, 2), replace=True)
                words = np.concatenate([chosen_negative, chosen_neutral, chosen_positive])
            
            # 随机打乱词序
            np.random.shuffle(words)
            
            # 构建评论
            review = ' '.join(words)
            
            reviews.append(review)
            labels.append(sentiment)
        
        # 创建DataFrame
        data = pd.DataFrame({
            'review': reviews,
            'sentiment': labels
        })
        
        print(f"✅ 生成了 {len(data)} 条评论数据")
        print(f"   正面评论: {sum(labels)} 条")
        print(f"   负面评论: {len(labels) - sum(labels)} 条")
        
        return data
    
    def load_data(self, file_path=None):
        """加载电影评论数据"""
        print("📂 加载电影评论数据...")
        
        if file_path and pd.io.common.file_exists(file_path):
            data = pd.read_csv(file_path)
        else:
            data = self.generate_sample_data()
        
        print(f"数据形状: {data.shape}")
        print(f"正负样本分布:")
        print(data['sentiment'].value_counts())
        
        return data
    
    def preprocess_text(self, text):
        """文本预处理"""
        # 转换为小写
        text = text.lower()
        
        # 移除标点符号
        text = text.translate(str.maketrans('', '', string.punctuation))
        
        # 移除多余空格
        text = re.sub(r'\\s+', ' ', text).strip()
        
        return text
    
    def explore_data(self, data):
        """数据探索分析"""
        print("\\n🔍 数据探索分析...")
        
        # 文本长度分析
        data['review_length'] = data['review'].str.len()
        data['word_count'] = data['review'].str.split().str.len()
        
        # 可视化分析
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. 情感分布
        sentiment_counts = data['sentiment'].value_counts()
        labels = ['负面', '正面']
        axes[0, 0].pie(sentiment_counts.values, labels=labels, autopct='%1.1f%%',
                      colors=['lightcoral', 'lightblue'], startangle=90)
        axes[0, 0].set_title('情感分布', fontweight='bold')
        
        # 2. 评论长度分布
        axes[0, 1].hist([data[data['sentiment']==0]['review_length'],
                        data[data['sentiment']==1]['review_length']],
                       bins=50, alpha=0.7, label=['负面', '正面'],
                       color=['lightcoral', 'lightblue'])
        axes[0, 1].set_title('评论长度分布', fontweight='bold')
        axes[0, 1].set_xlabel('字符数')
        axes[0, 1].set_ylabel('频数')
        axes[0, 1].legend()
        
        # 3. 词数分布
        axes[1, 0].hist([data[data['sentiment']==0]['word_count'],
                        data[data['sentiment']==1]['word_count']],
                       bins=30, alpha=0.7, label=['负面', '正面'],
                       color=['lightcoral', 'lightblue'])
        axes[1, 0].set_title('词数分布', fontweight='bold')
        axes[1, 0].set_xlabel('词数')
        axes[1, 0].set_ylabel('频数')
        axes[1, 0].legend()
        
        # 4. 词频分析
        all_words = []
        for review in data['review']:
            all_words.extend(review.split())
        
        word_freq = Counter(all_words)
        top_words = word_freq.most_common(20)
        
        words, freqs = zip(*top_words)
        axes[1, 1].barh(range(len(words)), freqs)
        axes[1, 1].set_yticks(range(len(words)))
        axes[1, 1].set_yticklabels(words)
        axes[1, 1].set_title('高频词TOP20', fontweight='bold')
        axes[1, 1].set_xlabel('频数')
        
        plt.tight_layout()
        plt.show()
        
        print(f"📊 数据统计:")
        print(f"  • 平均评论长度: {data['review_length'].mean():.1f} 字符")
        print(f"  • 平均词数: {data['word_count'].mean():.1f} 词")
        print(f"  • 词汇总数: {len(set(all_words)):,}")
        print(f"  • 最高频词: {top_words[0][0]} (出现{top_words[0][1]}次)")
    
    def prepare_data(self, data):
        """准备训练数据"""
        print("\\n🔧 准备训练数据...")
        
        # 文本预处理
        data['review_clean'] = data['review'].apply(self.preprocess_text)
        
        # 分离特征和标签
        texts = data['review_clean'].values
        labels = data['sentiment'].values
        
        # 文本向量化
        self.tokenizer = Tokenizer(num_words=self.max_features, oov_token='<OOV>')
        self.tokenizer.fit_on_texts(texts)
        
        # 转换为序列
        sequences = self.tokenizer.texts_to_sequences(texts)
        
        # 填充序列
        X = pad_sequences(sequences, maxlen=self.max_length, padding='post', truncating='post')
        
        # 划分训练集和测试集
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, labels, test_size=0.2, random_state=42, stratify=labels
        )
        
        print(f"✅ 数据准备完成!")
        print(f"   训练集: {len(self.X_train)} 样本")
        print(f"   测试集: {len(self.X_test)} 样本")
        print(f"   词汇表大小: {len(self.tokenizer.word_index)}")
        print(f"   序列长度: {self.max_length}")
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def build_model(self, embedding_dim=128, lstm_units=64):
        """构建LSTM模型"""
        print("\\n🏗️ 构建LSTM模型...")
        
        model = Sequential([
            # 词嵌入层
            Embedding(input_dim=self.max_features, 
                     output_dim=embedding_dim, 
                     input_length=self.max_length,
                     mask_zero=True),
            
            # 双向LSTM层
            Bidirectional(LSTM(lstm_units, dropout=0.3, recurrent_dropout=0.3)),
            
            # 全连接层
            Dense(64, activation='relu'),
            Dropout(0.5),
            
            # 输出层
            Dense(1, activation='sigmoid')
        ])
        
        # 编译模型
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        
        print("✅ LSTM模型构建完成!")
        print(f"   总参数数量: {model.count_params():,}")
        
        # 显示模型结构
        print("\\n📋 模型架构:")
        model.summary()
        
        return model
    
    def train_model(self, epochs=20, batch_size=32, validation_split=0.2):
        """训练模型"""
        print(f"\\n🤖 训练LSTM模型 (epochs={epochs})...")
        
        # 设置回调函数
        callbacks = [
            EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)
        ]
        
        # 训练模型
        self.history = self.model.fit(
            self.X_train, self.y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=callbacks,
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
        
        # 测试集预测
        y_pred_proba = self.model.predict(self.X_test, verbose=0)
        y_pred = (y_pred_proba > 0.5).astype(int).flatten()
        
        # 计算准确率
        accuracy = accuracy_score(self.y_test, y_pred)
        
        print(f"✅ 测试集准确率: {accuracy:.4f}")
        
        # 分类报告
        print("\\n📋 详细分类报告:")
        target_names = ['负面', '正面']
        print(classification_report(self.y_test, y_pred, target_names=target_names))
        
        # 混淆矩阵
        cm = confusion_matrix(self.y_test, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=target_names, yticklabels=target_names)
        plt.title('混淆矩阵', fontsize=16, fontweight='bold')
        plt.xlabel('预测标签')
        plt.ylabel('真实标签')
        plt.tight_layout()
        plt.show()
        
        return accuracy, y_pred, y_pred_proba
    
    def predict_sentiment(self, text):
        """预测单条文本的情感"""
        if isinstance(text, str):
            text = [text]
        
        # 预处理文本
        processed_text = [self.preprocess_text(t) for t in text]
        
        # 转换为序列
        sequences = self.tokenizer.texts_to_sequences(processed_text)
        padded = pad_sequences(sequences, maxlen=self.max_length, 
                              padding='post', truncating='post')
        
        # 预测
        predictions = self.model.predict(padded, verbose=0)
        
        results = []
        for i, pred in enumerate(predictions):
            sentiment = "正面" if pred[0] > 0.5 else "负面"
            confidence = pred[0] if pred[0] > 0.5 else 1 - pred[0]
            results.append({
                'text': text[i],
                'sentiment': sentiment,
                'confidence': confidence,
                'score': pred[0]
            })
        
        return results[0] if len(results) == 1 else results
    
    def analyze_word_importance(self, text, top_n=10):
        """分析词语重要性"""
        print(f"\\n🔍 分析词语重要性: '{text[:50]}...'")
        
        # 预处理文本
        processed_text = self.preprocess_text(text)
        words = processed_text.split()
        
        # 计算每个词对预测结果的影响
        original_pred = self.predict_sentiment(text)['score']
        
        word_importance = []
        
        for i, word in enumerate(words):
            # 移除这个词后的文本
            modified_words = words[:i] + words[i+1:]
            modified_text = ' '.join(modified_words)
            
            if modified_text.strip():  # 确保不是空文本
                modified_pred = self.predict_sentiment(modified_text)['score']
                importance = abs(original_pred - modified_pred)
                word_importance.append((word, importance))
        
        # 排序并返回最重要的词
        word_importance.sort(key=lambda x: x[1], reverse=True)
        
        print(f"重要词语 TOP{top_n}:")
        for i, (word, importance) in enumerate(word_importance[:top_n], 1):
            print(f"  {i}. {word}: {importance:.4f}")
        
        return word_importance[:top_n]
    
    def demonstrate_predictions(self):
        """演示预测功能"""
        print("\\n🔮 情感分析演示...")
        
        # 示例文本
        sample_texts = [
            "This movie is absolutely fantastic! I loved every minute of it.",
            "Terrible film, waste of time and money. Very disappointing.",
            "The movie was okay, nothing special but not bad either.",
            "Amazing performance by the actors, brilliant story and direction.",
            "Boring plot, terrible acting, one of the worst movies ever."
        ]
        
        results = []
        for text in sample_texts:
            result = self.predict_sentiment(text)
            results.append(result)
            print(f"文本: {text}")
            print(f"情感: {result['sentiment']} (置信度: {result['confidence']:.3f})")
            print("-" * 50)
        
        return results
    
    def save_model(self, filepath='sentiment_lstm.h5'):
        """保存模型"""
        if self.model is not None:
            self.model.save(filepath)
            print(f"✅ 模型已保存到 {filepath}")
        else:
            print("❌ 没有可保存的模型")
    
    def load_model(self, filepath='sentiment_lstm.h5'):
        """加载模型"""
        try:
            self.model = tf.keras.models.load_model(filepath)
            print(f"✅ 模型已从 {filepath} 加载")
            return self.model
        except Exception as e:
            print(f"❌ 模型加载失败: {str(e)}")
            return None
    
    def run_complete_project(self):
        """运行完整的情感分析项目"""
        print("💬 开始电影评论情感分析项目...")
        print("=" * 60)
        
        # 1. 加载数据
        data = self.load_data()
        
        # 2. 数据探索
        self.explore_data(data)
        
        # 3. 准备数据
        self.prepare_data(data)
        
        # 4. 构建模型
        self.build_model()
        
        # 5. 训练模型
        self.train_model(epochs=10)  # 减少epochs用于演示
        
        # 6. 可视化训练过程
        self.plot_training_history()
        
        # 7. 评估模型
        accuracy, y_pred, y_pred_proba = self.evaluate_model()
        
        # 8. 演示预测
        self.demonstrate_predictions()
        
        # 9. 词语重要性分析
        sample_text = "This movie is absolutely amazing and fantastic!"
        self.analyze_word_importance(sample_text)
        
        # 10. 保存模型
        self.save_model()
        
        print("\\n🎯 项目总结:")
        print(f"  • 数据集: {len(data)} 条电影评论")
        print(f"  • 模型: 双向LSTM网络")
        print(f"  • 测试准确率: {accuracy:.4f}")
        print(f"  • 词汇表大小: {len(self.tokenizer.word_index)}")
        print("\\n🎉 电影评论情感分析项目完成!")
        
        return self

def demonstrate_rnn_concepts():
    """演示RNN核心概念"""
    print("\\n" + "="*60)
    print("🧠 RNN核心概念演示")
    print("="*60)
    
    print("\\n1️⃣ 序列数据的特点:")
    print("文本: 'I love this movie'")
    print("序列: [I] -> [love] -> [this] -> [movie]")
    print("特点: 每个词的理解依赖于前面的上下文")
    
    print("\\n2️⃣ RNN vs 普通神经网络:")
    print("普通NN: 输入 -> 隐藏层 -> 输出 (无记忆)")
    print("RNN: 输入 + 历史状态 -> 隐藏层 -> 输出 (有记忆)")
    
    print("\\n3️⃣ LSTM的优势:")
    print("• 解决梯度消失问题")
    print("• 能记住长期依赖关系")
    print("• 通过门控机制选择性记忆")
    
    # 简单的序列预测演示
    print("\\n4️⃣ 序列预测演示:")
    
    # 生成简单的数学序列
    sequence = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    print(f"原始序列: {sequence}")
    
    # 模拟RNN预测下一个数字
    def simple_rnn_predict(seq, window_size=3):
        if len(seq) < window_size:
            return None
        
        # 简单的线性预测（实际RNN会学习更复杂的模式）
        recent = seq[-window_size:]
        diff = recent[-1] - recent[-2] if len(recent) > 1 else 1
        predicted = recent[-1] + diff
        return predicted
    
    predicted = simple_rnn_predict(sequence)
    print(f"预测下一个数字: {predicted}")
    print(f"实际应该是: 11")
    
    print("\\n✅ RNN能够学习序列中的模式并进行预测！")

def main():
    """主函数"""
    print("💬 第12章：RNN与情感分析")
    print("=" * 60)
    
    # 1. 概念演示
    demonstrate_rnn_concepts()
    
    # 2. 运行完整项目
    analyzer = SentimentAnalyzer()
    analyzer.run_complete_project()
    
    print("\\n🎓 学习总结:")
    print("  • 理解了RNN和LSTM的工作原理")
    print("  • 掌握了文本预处理和向量化技术") 
    print("  • 完成了端到端的情感分析项目")
    print("  • 学会了序列建模和自然语言处理")
    print("  • 具备了处理文本数据的AI能力")
    
    return analyzer

if __name__ == "__main__":
    analyzer = main()

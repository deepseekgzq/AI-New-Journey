"""
第14章：泰坦尼克号生还预测完整项目
从数据到模型的端到端机器学习项目
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

class TitanicSurvivalPredictor:
    """泰坦尼克号生还预测完整项目"""
    
    def __init__(self):
        """初始化"""
        self.train_data = None
        self.test_data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.models = {}
        self.best_model = None
        self.feature_names = None
        self.scaler = StandardScaler()
        
    def load_data(self):
        """加载泰坦尼克号数据"""
        print("📂 加载泰坦尼克号数据集...")
        
        # 生成模拟的泰坦尼克号数据
        self.train_data = self._generate_titanic_data(n_samples=891, is_train=True)
        self.test_data = self._generate_titanic_data(n_samples=418, is_train=False)
        
        print(f"✅ 数据加载成功！")
        print(f"   - 训练集: {len(self.train_data)} 条记录")
        print(f"   - 测试集: {len(self.test_data)} 条记录")
        
        return self.train_data, self.test_data
    
    def _generate_titanic_data(self, n_samples=891, is_train=True):
        """生成模拟的泰坦尼克号数据"""
        np.random.seed(42 if is_train else 24)
        
        # 生成基础特征
        data = {
            'PassengerId': range(1 if is_train else 892, n_samples + (1 if is_train else 892)),
            'Pclass': np.random.choice([1, 2, 3], n_samples, p=[0.24, 0.21, 0.55]),
            'Sex': np.random.choice(['male', 'female'], n_samples, p=[0.65, 0.35]),
            'Age': np.random.normal(29.7, 14.5, n_samples),
            'SibSp': np.random.choice(range(9), n_samples, p=[0.68, 0.23, 0.06, 0.02, 0.005, 0.003, 0.001, 0.001, 0.001]),
            'Parch': np.random.choice(range(7), n_samples, p=[0.76, 0.13, 0.08, 0.005, 0.004, 0.001, 0.001]),
            'Fare': np.random.lognormal(2.5, 1.2, n_samples),
            'Embarked': np.random.choice(['S', 'C', 'Q'], n_samples, p=[0.72, 0.19, 0.09])
        }
        
        # 创建DataFrame
        df = pd.DataFrame(data)
        
        # 处理年龄（引入一些缺失值）
        missing_age_mask = np.random.random(n_samples) < 0.2
        df.loc[missing_age_mask, 'Age'] = np.nan
        
        # 确保年龄在合理范围内
        df['Age'] = np.clip(df['Age'], 0.42, 80)
        
        # 确保票价在合理范围内
        df['Fare'] = np.clip(df['Fare'], 0, 512)
        
        # 添加姓名（简化版）
        titles = ['Mr.', 'Mrs.', 'Miss.', 'Master.', 'Dr.', 'Rev.']
        df['Name'] = [f"{np.random.choice(titles)} {chr(65+i%26)}{chr(65+(i*7)%26)}" for i in range(n_samples)]
        
        # 添加船票号码（简化版）
        df['Ticket'] = [f"T{1000+i}" for i in range(n_samples)]
        
        # 添加船舱号码（部分缺失）
        cabin_mask = np.random.random(n_samples) < 0.23
        df['Cabin'] = [f"{np.random.choice(['A','B','C','D','E','F','G'])}{np.random.randint(1,200)}" 
                      if cabin_mask[i] else np.nan for i in range(n_samples)]
        
        # 生成生还标签（仅训练集）
        if is_train:
            # 基于特征生成较为真实的生还概率
            survival_prob = 0.5  # 基础概率
            
            # 性别影响（女性生还率更高）
            survival_prob += np.where(df['Sex'] == 'female', 0.3, -0.2)
            
            # 年龄影响（儿童生还率更高）
            survival_prob += np.where(df['Age'] < 16, 0.2, 0)
            survival_prob += np.where(df['Age'] > 60, -0.1, 0)
            
            # 船票等级影响（高等级生还率更高）
            pclass_effect = {1: 0.25, 2: 0.05, 3: -0.15}
            survival_prob += df['Pclass'].map(pclass_effect)
            
            # 家庭规模影响
            family_size = df['SibSp'] + df['Parch']
            survival_prob += np.where((family_size >= 1) & (family_size <= 3), 0.1, -0.05)
            
            # 票价影响
            survival_prob += np.where(df['Fare'] > df['Fare'].median(), 0.1, -0.05)
            
            # 登船港口影响
            embarked_effect = {'C': 0.1, 'Q': -0.05, 'S': 0}
            survival_prob += df['Embarked'].map(embarked_effect)
            
            # 添加一些随机性
            survival_prob += np.random.normal(0, 0.1, n_samples)
            
            # 确保概率在0-1范围内
            survival_prob = np.clip(survival_prob, 0, 1)
            
            # 生成二分类标签
            df['Survived'] = np.random.binomial(1, survival_prob)
        
        return df
    
    def exploratory_data_analysis(self):
        """探索性数据分析"""
        print("\\n🔍 探索性数据分析...")
        
        # 基本信息
        print("\\n📊 数据基本信息:")
        print(f"训练集形状: {self.train_data.shape}")
        print(f"测试集形状: {self.test_data.shape}")
        
        print("\\n📋 特征信息:")
        print(self.train_data.info())
        
        print("\\n📈 生还率统计:")
        survival_rate = self.train_data['Survived'].mean()
        print(f"总体生还率: {survival_rate:.3f}")
        
        # 可视化分析
        fig, axes = plt.subplots(3, 3, figsize=(20, 15))
        
        # 1. 生还率分布
        survival_counts = self.train_data['Survived'].value_counts()
        axes[0, 0].pie(survival_counts.values, labels=['未生还', '生还'], autopct='%1.1f%%', 
                      colors=['lightcoral', 'lightblue'])
        axes[0, 0].set_title('生还率分布', fontweight='bold')
        
        # 2. 性别与生还率
        sex_survival = pd.crosstab(self.train_data['Sex'], self.train_data['Survived'], normalize='index')
        sex_survival.plot(kind='bar', ax=axes[0, 1], color=['lightcoral', 'lightblue'])
        axes[0, 1].set_title('性别与生还率', fontweight='bold')
        axes[0, 1].set_xlabel('性别')
        axes[0, 1].set_ylabel('生还率')
        axes[0, 1].legend(['未生还', '生还'])
        axes[0, 1].tick_params(axis='x', rotation=0)
        
        # 3. 船票等级与生还率
        pclass_survival = pd.crosstab(self.train_data['Pclass'], self.train_data['Survived'], normalize='index')
        pclass_survival.plot(kind='bar', ax=axes[0, 2], color=['lightcoral', 'lightblue'])
        axes[0, 2].set_title('船票等级与生还率', fontweight='bold')
        axes[0, 2].set_xlabel('船票等级')
        axes[0, 2].set_ylabel('生还率')
        axes[0, 2].legend(['未生还', '生还'])
        axes[0, 2].tick_params(axis='x', rotation=0)
        
        # 4. 年龄分布
        axes[1, 0].hist([self.train_data[self.train_data['Survived']==0]['Age'].dropna(),
                        self.train_data[self.train_data['Survived']==1]['Age'].dropna()],
                       bins=30, alpha=0.7, label=['未生还', '生还'], color=['lightcoral', 'lightblue'])
        axes[1, 0].set_title('年龄分布', fontweight='bold')
        axes[1, 0].set_xlabel('年龄')
        axes[1, 0].set_ylabel('人数')
        axes[1, 0].legend()
        
        # 5. 票价分布
        axes[1, 1].hist([self.train_data[self.train_data['Survived']==0]['Fare'],
                        self.train_data[self.train_data['Survived']==1]['Fare']],
                       bins=50, alpha=0.7, label=['未生还', '生还'], color=['lightcoral', 'lightblue'])
        axes[1, 1].set_title('票价分布', fontweight='bold')
        axes[1, 1].set_xlabel('票价')
        axes[1, 1].set_ylabel('人数')
        axes[1, 1].legend()
        axes[1, 1].set_xlim(0, 200)  # 限制显示范围
        
        # 6. 家庭规模与生还率
        self.train_data['FamilySize'] = self.train_data['SibSp'] + self.train_data['Parch'] + 1
        family_survival = self.train_data.groupby('FamilySize')['Survived'].mean()
        axes[1, 2].bar(family_survival.index, family_survival.values, color='skyblue', alpha=0.7)
        axes[1, 2].set_title('家庭规模与生还率', fontweight='bold')
        axes[1, 2].set_xlabel('家庭规模')
        axes[1, 2].set_ylabel('生还率')
        
        # 7. 登船港口与生还率
        embarked_survival = pd.crosstab(self.train_data['Embarked'], self.train_data['Survived'], normalize='index')
        embarked_survival.plot(kind='bar', ax=axes[2, 0], color=['lightcoral', 'lightblue'])
        axes[2, 0].set_title('登船港口与生还率', fontweight='bold')
        axes[2, 0].set_xlabel('登船港口')
        axes[2, 0].set_ylabel('生还率')
        axes[2, 0].legend(['未生还', '生还'])
        axes[2, 0].tick_params(axis='x', rotation=0)
        
        # 8. 相关性热力图
        numeric_features = ['Survived', 'Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'FamilySize']
        correlation_matrix = self.train_data[numeric_features].corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, ax=axes[2, 1])
        axes[2, 1].set_title('特征相关性热力图', fontweight='bold')
        
        # 9. 缺失值分析
        missing_data = self.train_data.isnull().sum()
        missing_data = missing_data[missing_data > 0].sort_values(ascending=False)
        if len(missing_data) > 0:
            axes[2, 2].bar(range(len(missing_data)), missing_data.values, color='orange', alpha=0.7)
            axes[2, 2].set_title('缺失值统计', fontweight='bold')
            axes[2, 2].set_xlabel('特征')
            axes[2, 2].set_ylabel('缺失值数量')
            axes[2, 2].set_xticks(range(len(missing_data)))
            axes[2, 2].set_xticklabels(missing_data.index, rotation=45)
        else:
            axes[2, 2].text(0.5, 0.5, '无缺失值', ha='center', va='center', transform=axes[2, 2].transAxes)
            axes[2, 2].set_title('缺失值统计', fontweight='bold')
        
        plt.tight_layout()
        plt.show()
        
        print("\\n💡 数据洞察:")
        print(f"  • 女性生还率显著高于男性")
        print(f"  • 高等级船票乘客生还率更高") 
        print(f"  • 年龄和家庭规模对生还率有影响")
        print(f"  • 票价与生还率呈正相关")
    
    def feature_engineering(self):
        """特征工程"""
        print("\\n🔧 特征工程...")
        
        # 合并训练集和测试集进行统一处理
        all_data = pd.concat([self.train_data, self.test_data], ignore_index=True)
        
        # 1. 处理缺失值
        print("   处理缺失值...")
        
        # 年龄：用中位数填充
        all_data['Age'].fillna(all_data['Age'].median(), inplace=True)
        
        # 登船港口：用众数填充
        all_data['Embarked'].fillna(all_data['Embarked'].mode()[0], inplace=True)
        
        # 票价：用中位数填充
        all_data['Fare'].fillna(all_data['Fare'].median(), inplace=True)
        
        # 2. 创建新特征
        print("   创建新特征...")
        
        # 家庭规模
        all_data['FamilySize'] = all_data['SibSp'] + all_data['Parch'] + 1
        
        # 是否独自一人
        all_data['IsAlone'] = (all_data['FamilySize'] == 1).astype(int)
        
        # 年龄分组
        all_data['AgeGroup'] = pd.cut(all_data['Age'], bins=[0, 12, 18, 35, 60, 100], 
                                     labels=['Child', 'Teen', 'Adult', 'Middle', 'Senior'])
        
        # 票价分组
        all_data['FareGroup'] = pd.qcut(all_data['Fare'], q=4, labels=['Low', 'Medium', 'High', 'VeryHigh'])
        
        # 从姓名中提取称谓
        all_data['Title'] = all_data['Name'].str.extract(' ([A-Za-z]+)\\.', expand=False)
        all_data['Title'] = all_data['Title'].replace(['Lady', 'Countess','Capt', 'Col',\\
                                                      'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
        all_data['Title'] = all_data['Title'].replace('Mlle', 'Miss')
        all_data['Title'] = all_data['Title'].replace('Ms', 'Miss')
        all_data['Title'] = all_data['Title'].replace('Mme', 'Mrs')
        
        # 船舱等级
        all_data['HasCabin'] = (~all_data['Cabin'].isnull()).astype(int)
        
        # 3. 编码分类特征
        print("   编码分类特征...")
        
        # 性别编码
        le_sex = LabelEncoder()
        all_data['Sex_encoded'] = le_sex.fit_transform(all_data['Sex'])
        
        # 登船港口编码
        le_embarked = LabelEncoder()
        all_data['Embarked_encoded'] = le_embarked.fit_transform(all_data['Embarked'])
        
        # 称谓编码
        le_title = LabelEncoder()
        all_data['Title_encoded'] = le_title.fit_transform(all_data['Title'])
        
        # 年龄组编码
        le_age_group = LabelEncoder()
        all_data['AgeGroup_encoded'] = le_age_group.fit_transform(all_data['AgeGroup'])
        
        # 票价组编码
        le_fare_group = LabelEncoder()
        all_data['FareGroup_encoded'] = le_fare_group.fit_transform(all_data['FareGroup'])
        
        # 4. 选择最终特征
        feature_columns = [
            'Pclass', 'Sex_encoded', 'Age', 'SibSp', 'Parch', 'Fare',
            'Embarked_encoded', 'FamilySize', 'IsAlone', 'HasCabin',
            'Title_encoded', 'AgeGroup_encoded', 'FareGroup_encoded'
        ]
        
        # 分离训练集和测试集
        train_size = len(self.train_data)
        train_processed = all_data[:train_size]
        test_processed = all_data[train_size:]
        
        # 准备特征和标签
        X = train_processed[feature_columns]
        y = train_processed['Survived']
        X_submission = test_processed[feature_columns]
        
        self.feature_names = feature_columns
        
        print(f"✅ 特征工程完成!")
        print(f"   - 最终特征数量: {len(feature_columns)}")
        print(f"   - 特征列表: {feature_columns}")
        
        return X, y, X_submission
    
    def prepare_data(self):
        """准备训练数据"""
        print("\\n📊 准备训练数据...")
        
        # 特征工程
        X, y, X_submission = self.feature_engineering()
        
        # 划分训练集和验证集
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # 特征标准化
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        print(f"✅ 数据准备完成!")
        print(f"   - 训练集: {len(self.X_train)} 样本")
        print(f"   - 验证集: {len(self.X_test)} 样本")
        print(f"   - 特征数量: {self.X_train.shape[1]}")
        
        return self.X_train_scaled, self.X_test_scaled, self.y_train, self.y_test
    
    def train_models(self):
        """训练多种模型"""
        print("\\n🤖 训练多种模型...")
        
        # 定义模型
        models = {
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
            'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100),
            'Gradient Boosting': GradientBoostingClassifier(random_state=42, n_estimators=100),
            'SVM': SVC(random_state=42, probability=True)
        }
        
        # 训练和评估每个模型
        results = {}
        
        for name, model in models.items():
            print(f"   训练 {name}...")
            
            # 使用交叉验证评估
            cv_scores = cross_val_score(model, self.X_train_scaled, self.y_train, cv=5, scoring='accuracy')
            
            # 训练模型
            model.fit(self.X_train_scaled, self.y_train)
            
            # 验证集预测
            y_pred = model.predict(self.X_test_scaled)
            y_pred_proba = model.predict_proba(self.X_test_scaled)[:, 1]
            
            # 计算指标
            accuracy = accuracy_score(self.y_test, y_pred)
            auc = roc_auc_score(self.y_test, y_pred_proba)
            
            results[name] = {
                'model': model,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'val_accuracy': accuracy,
                'val_auc': auc,
                'y_pred': y_pred,
                'y_pred_proba': y_pred_proba
            }
            
            print(f"     CV准确率: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
            print(f"     验证准确率: {accuracy:.4f}")
            print(f"     验证AUC: {auc:.4f}")
        
        self.models = results
        
        # 选择最佳模型
        best_model_name = max(results.keys(), key=lambda x: results[x]['val_auc'])
        self.best_model = results[best_model_name]['model']
        
        print(f"\\n🏆 最佳模型: {best_model_name}")
        print(f"   验证AUC: {results[best_model_name]['val_auc']:.4f}")
        
        return results
    
    def evaluate_models(self):
        """评估模型性能"""
        print("\\n📊 模型性能评估...")
        
        if not self.models:
            print("❌ 请先训练模型")
            return
        
        # 创建评估可视化
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. 模型性能对比
        model_names = list(self.models.keys())
        cv_scores = [self.models[name]['cv_mean'] for name in model_names]
        val_scores = [self.models[name]['val_accuracy'] for name in model_names]
        auc_scores = [self.models[name]['val_auc'] for name in model_names]
        
        x = np.arange(len(model_names))
        width = 0.25
        
        axes[0, 0].bar(x - width, cv_scores, width, label='CV准确率', alpha=0.8)
        axes[0, 0].bar(x, val_scores, width, label='验证准确率', alpha=0.8)
        axes[0, 0].bar(x + width, auc_scores, width, label='验证AUC', alpha=0.8)
        
        axes[0, 0].set_xlabel('模型')
        axes[0, 0].set_ylabel('分数')
        axes[0, 0].set_title('模型性能对比', fontweight='bold')
        axes[0, 0].set_xticks(x)
        axes[0, 0].set_xticklabels(model_names, rotation=45)
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. ROC曲线
        for name in model_names:
            y_pred_proba = self.models[name]['y_pred_proba']
            fpr, tpr, _ = roc_curve(self.y_test, y_pred_proba)
            auc = self.models[name]['val_auc']
            axes[0, 1].plot(fpr, tpr, label=f'{name} (AUC = {auc:.3f})')
        
        axes[0, 1].plot([0, 1], [0, 1], 'k--', alpha=0.5)
        axes[0, 1].set_xlabel('假正率')
        axes[0, 1].set_ylabel('真正率')
        axes[0, 1].set_title('ROC曲线', fontweight='bold')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. 最佳模型的混淆矩阵
        best_model_name = max(self.models.keys(), key=lambda x: self.models[x]['val_auc'])
        best_y_pred = self.models[best_model_name]['y_pred']
        cm = confusion_matrix(self.y_test, best_y_pred)
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1, 0])
        axes[1, 0].set_title(f'{best_model_name} 混淆矩阵', fontweight='bold')
        axes[1, 0].set_xlabel('预测标签')
        axes[1, 0].set_ylabel('真实标签')
        
        # 4. 特征重要性（如果模型支持）
        if hasattr(self.best_model, 'feature_importances_'):
            importances = self.best_model.feature_importances_
            feature_importance = pd.DataFrame({
                'feature': self.feature_names,
                'importance': importances
            }).sort_values('importance', ascending=True)
            
            axes[1, 1].barh(range(len(feature_importance)), feature_importance['importance'])
            axes[1, 1].set_yticks(range(len(feature_importance)))
            axes[1, 1].set_yticklabels(feature_importance['feature'])
            axes[1, 1].set_xlabel('重要性')
            axes[1, 1].set_title('特征重要性', fontweight='bold')
            axes[1, 1].grid(True, alpha=0.3)
        else:
            axes[1, 1].text(0.5, 0.5, '该模型不支持\\n特征重要性分析', 
                           ha='center', va='center', transform=axes[1, 1].transAxes)
            axes[1, 1].set_title('特征重要性', fontweight='bold')
        
        plt.tight_layout()
        plt.show()
        
        # 打印详细评估报告
        print(f"\\n📋 最佳模型 ({best_model_name}) 详细报告:")
        print(classification_report(self.y_test, best_y_pred, 
                                  target_names=['未生还', '生还']))
    
    def hyperparameter_tuning(self):
        """超参数调优"""
        print("\\n⚙️ 超参数调优...")
        
        # 为最佳模型进行超参数调优
        best_model_name = max(self.models.keys(), key=lambda x: self.models[x]['val_auc'])
        
        if 'Random Forest' in best_model_name:
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [3, 5, 7, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
            base_model = RandomForestClassifier(random_state=42)
            
        elif 'Gradient Boosting' in best_model_name:
            param_grid = {
                'n_estimators': [100, 200],
                'learning_rate': [0.05, 0.1, 0.15],
                'max_depth': [3, 4, 5]
            }
            base_model = GradientBoostingClassifier(random_state=42)
            
        else:
            print(f"   {best_model_name} 使用默认参数")
            return self.best_model
        
        print(f"   为 {best_model_name} 进行网格搜索...")
        
        # 网格搜索
        grid_search = GridSearchCV(
            base_model, param_grid, cv=5, scoring='roc_auc', 
            n_jobs=-1, verbose=1
        )
        
        grid_search.fit(self.X_train_scaled, self.y_train)
        
        # 更新最佳模型
        self.best_model = grid_search.best_estimator_
        
        print(f"✅ 超参数调优完成!")
        print(f"   最佳参数: {grid_search.best_params_}")
        print(f"   最佳CV分数: {grid_search.best_score_:.4f}")
        
        return self.best_model
    
    def generate_predictions(self, X_submission):
        """生成提交预测"""
        print("\\n🔮 生成提交预测...")
        
        if self.best_model is None:
            print("❌ 请先训练模型")
            return None
        
        # 标准化测试数据
        X_submission_scaled = self.scaler.transform(X_submission)
        
        # 预测
        predictions = self.best_model.predict(X_submission_scaled)
        probabilities = self.best_model.predict_proba(X_submission_scaled)[:, 1]
        
        print(f"✅ 预测完成!")
        print(f"   预测生还率: {predictions.mean():.3f}")
        
        return predictions, probabilities
    
    def create_project_report(self):
        """创建项目报告"""
        print("\\n📝 生成项目报告...")
        
        best_model_name = max(self.models.keys(), key=lambda x: self.models[x]['val_auc'])
        best_result = self.models[best_model_name]
        
        report = f"""
# 泰坦尼克号生还预测项目报告

## 项目概述
- **目标**: 预测泰坦尼克号乘客生还情况
- **数据集**: 泰坦尼克号乘客信息
- **问题类型**: 二分类问题

## 数据分析结果
- **训练集规模**: {len(self.train_data)} 条记录
- **测试集规模**: {len(self.test_data)} 条记录
- **特征数量**: {len(self.feature_names)}
- **总体生还率**: {self.train_data['Survived'].mean():.3f}

## 关键发现
1. **性别影响**: 女性生还率显著高于男性
2. **社会地位**: 高等级船票乘客生还率更高
3. **年龄因素**: 儿童生还率相对较高
4. **家庭规模**: 中等规模家庭生还率最佳

## 模型性能
- **最佳模型**: {best_model_name}
- **交叉验证准确率**: {best_result['cv_mean']:.4f} (±{best_result['cv_std']:.4f})
- **验证集准确率**: {best_result['val_accuracy']:.4f}
- **验证集AUC**: {best_result['val_auc']:.4f}

## 特征工程
- 处理缺失值（年龄、登船港口等）
- 创建新特征（家庭规模、是否独自一人等）
- 从姓名提取称谓信息
- 对分类变量进行编码

## 模型对比
"""
        
        for name, result in self.models.items():
            report += f"- **{name}**: CV={result['cv_mean']:.4f}, Val={result['val_accuracy']:.4f}, AUC={result['val_auc']:.4f}\\n"
        
        report += f"""
## 结论与建议
1. {best_model_name} 在所有测试模型中表现最佳
2. 性别是最重要的预测因子
3. 社会经济地位对生还率有显著影响
4. 模型可用于类似历史事件分析

## 技术栈
- **数据处理**: Pandas, NumPy
- **可视化**: Matplotlib, Seaborn  
- **机器学习**: Scikit-learn
- **模型**: {best_model_name}
"""
        
        print("✅ 项目报告生成完成!")
        return report
    
    def run_complete_project(self):
        """运行完整项目"""
        print("🚢 开始泰坦尼克号生还预测完整项目...")
        print("=" * 60)
        
        # 1. 加载数据
        self.load_data()
        
        # 2. 探索性数据分析
        self.exploratory_data_analysis()
        
        # 3. 准备数据
        self.prepare_data()
        
        # 4. 训练多种模型
        self.train_models()
        
        # 5. 评估模型
        self.evaluate_models()
        
        # 6. 超参数调优
        self.hyperparameter_tuning()
        
        # 7. 重新评估调优后的模型
        print("\\n🔄 重新评估调优后的模型...")
        y_pred_final = self.best_model.predict(self.X_test_scaled)
        y_pred_proba_final = self.best_model.predict_proba(self.X_test_scaled)[:, 1]
        
        final_accuracy = accuracy_score(self.y_test, y_pred_final)
        final_auc = roc_auc_score(self.y_test, y_pred_proba_final)
        
        print(f"   最终准确率: {final_accuracy:.4f}")
        print(f"   最终AUC: {final_auc:.4f}")
        
        # 8. 生成项目报告
        report = self.create_project_report()
        
        print("\\n🎯 项目总结:")
        print(f"  • 成功完成端到端机器学习项目")
        print(f"  • 最佳模型准确率: {final_accuracy:.4f}")
        print(f"  • 模型AUC分数: {final_auc:.4f}")
        print(f"  • 特征工程创建 {len(self.feature_names)} 个特征")
        print("\\n🎉 泰坦尼克号生还预测项目完成!")
        
        return self, report

def main():
    """主函数"""
    print("🚢 第14章：完整AI项目实战")
    print("=" * 60)
    
    # 运行完整项目
    predictor, report = TitanicSurvivalPredictor().run_complete_project()
    
    print("\\n🎓 学习总结:")
    print("  • 掌握了完整的机器学习项目流程")
    print("  • 学会了数据探索和特征工程")
    print("  • 比较了多种机器学习算法")
    print("  • 进行了模型调优和性能评估")
    print("  • 具备了端到端项目开发能力")
    
    return predictor, report

if __name__ == "__main__":
    predictor, report = main()

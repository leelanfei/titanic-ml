# Titanic Survival Prediction - Complete Analysis Code
# 泰坦尼克号生存预测 - 完整分析代码
# Kaggle Score: 78.947%

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ========== Part 1: Load the data ==========
# 从 Kaggle 下载数据到 data/ 目录
# train.csv: https://www.kaggle.com/competitions/titanic/data
# test.csv: https://www.kaggle.com/competitions/titanic/data

train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')

print("=" * 50)
print("数据概览")
print("=" * 50)
print(f"训练集大小: {train.shape}")
print(f"测试集大小: {test.shape}")

# ========== Part 2: Exploratory Data Analysis (EDA) ==========
print("\n" + "=" * 50)
print("数据探索")
print("=" * 50)

# 2.1 查看缺失值
print("\n训练集缺失值:")
print(train.isnull().sum())

print("\n测试集缺失值:")
print(test.isnull().sum())

# 2.2 性别与生存率
print("\n性别生存率:")
print(f"女性: {train[train['Sex']=='female']['Survived'].mean():.2%}")
print(f"男性: {train[train['Sex']=='male']['Survived'].mean():.2%}")

# 2.3 舱位等级与生存率
print("\n舱位等级生存率:")
print(train.groupby('Pclass')['Survived'].mean())

# ========== Part 3: Data Cleaning ==========
print("\n" + "=" * 50)
print("数据清洗")
print("=" * 50)

# 关键：先用训练集计算统计量，再用训练集的统计量填充测试集
age_median = train['Age'].median()
fare_median = train['Fare'].median()

print(f"训练集 Age 中位数: {age_median}")
print(f"训练集 Fare 中位数: {fare_median}")

def clean_data(df, is_train=True):
    """
    数据清洗函数
    注意：用训练集的统计量填充测试集
    """
    df = df.copy()
    
    # 填充缺失值
    df['Age'] = df['Age'].fillna(age_median)
    df['Fare'] = df['Fare'].fillna(fare_median)
    
    if is_train:
        df['Embarked'] = df['Embarked'].fillna('S')  # 训练集：众数
    # 测试集的 Embarked 不需要填充（根据之前的检查）
    
    # 性别编码
    df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
    
    return df

train = clean_data(train, is_train=True)
test = clean_data(test, is_train=False)

print("✅ 缺失值填充完成")

# ========== Part 4: Feature Engineering ==========
print("\n" + "=" * 50)
print("特征工程")
print("=" * 50)

def feature_engineering(df):
    """
    特征工程函数
    创建新特征，增强模型预测能力
    """
    df = df.copy()
    
    # 4.1 从姓名提取称谓 (Title)
    df['Title'] = df['Name'].str.extract(r' ([A-Za-z]+)\.')
    
    # 称谓映射：Rare 类别合并稀有称谓
    df['Title'] = df['Title'].replace(
        ['Lady', 'Countess', 'Capt', 'Col', 'Don', 'Dr', 
         'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'],
        'Rare'
    )
    
    # Mlle/Mme/Ms 映射
    df['Title'] = df['Title'].replace(['Mlle', 'Mme', 'Ms'], 'Miss')
    
    # 数值编码
    title_map = {'Mr': 0, 'Miss': 1, 'Mrs': 2, 'Master': 3, 'Rare': 4}
    df['Title'] = df['Title'].map(title_map)
    df['Title'] = df['Title'].fillna(4)  # 未匹配的填充为 Rare
    
    # 4.2 家庭规模
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    
    # 4.3 是否独自出行
    df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
    
    # 4.4 年龄分段
    df['AgeBand'] = pd.cut(df['Age'], 
                           bins=[0, 12, 18, 35, 60, 100], 
                           labels=[0, 1, 2, 3, 4]).astype(int)
    
    # 4.5 登船港口 one-hot 编码
    df['Embarked_C'] = (df['Embarked'] == 'C').astype(int)
    df['Embarked_Q'] = (df['Embarked'] == 'Q').astype(int)
    df['Embarked_S'] = (df['Embarked'] == 'S').astype(int)
    
    return df

train = feature_engineering(train)
test = feature_engineering(test)

print("新增特征:", ['Title', 'FamilySize', 'IsAlone', 'AgeBand', 
                   'Embarked_C', 'Embarked_Q', 'Embarked_S'])

# ========== Part 5: Feature Selection ==========
print("\n" + "=" * 50)
print("特征选择")
print("=" * 50)

# 选择用于建模的特征
features = [
    'Pclass',      # 舱位等级
    'Sex',         # 性别
    'Age',         # 年龄
    'SibSp',       # 兄弟姐妹/配偶数量
    'Parch',       # 父母/子女数量
    'Fare',        # 票价
    'Title',       # 称谓
    'FamilySize',  # 家庭规模
    'IsAlone',     # 是否独自出行
    'AgeBand',     # 年龄段
    'Embarked_C',  # 登船港口-C
    'Embarked_Q',  # 登船港口-Q
    'Embarked_S',  # 登船港口-S
]

X = train[features]
y = train['Survived']

X_test = test[features]

print(f"特征数量: {len(features)}")
print(f"训练样本: {len(X)}, 测试样本: {len(X_test)}")

# ========== Part 6: Model Training ==========
print("\n" + "=" * 50)
print("模型训练")
print("=" * 50)

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold

# 使用随机森林分类器
model = RandomForestClassifier(
    n_estimators=100,  # 树的数量
    random_state=42     # 随机种子，保证可复现
)

# StratifiedKFold 分层交叉验证
# 确保每折的正负样本比例一致
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# 交叉验证评估
scores = cross_val_score(model, X, y, cv=skf)

print("各折交叉验证分数:")
for i, score in enumerate(scores, 1):
    print(f"  Fold {i}: {score:.4f}")

print(f"\n平均准确率: {scores.mean():.4f} ± {scores.std():.4f}")

# 训练最终模型
model.fit(X, y)
print("\n✅ 模型训练完成")

# ========== Part 7: Feature Importance ==========
print("\n" + "=" * 50)
print("特征重要性")
print("=" * 50)

importance = pd.DataFrame({
    'feature': features,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print(importance.to_string(index=False))

# 绘制特征重要性图
plt.figure(figsize=(10, 6))
plt.barh(importance['feature'], importance['importance'])
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.title('Feature Importance - Random Forest')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('images/feature_importance.png', dpi=150)
print("\n✅ 特征重要性图已保存到 images/feature_importance.png")

# ========== Part 8: Prediction & Submission ==========
print("\n" + "=" * 50)
print("预测与提交")
print("=" * 50)

# 生成预测
predictions = model.predict(X_test)

# 创建提交文件
submission = pd.DataFrame({
    'PassengerId': test['PassengerId'],
    'Survived': predictions.astype(int)
})

# 保存提交文件
submission.to_csv('submission.csv', index=False)
print(f"✅ 提交文件已保存到 submission.csv")
print(f"   预测生还人数: {predictions.sum()}")
print(f"   预测遇难人数: {len(predictions) - predictions.sum()}")

# ========== Part 9: Summary ==========
print("\n" + "=" * 50)
print("项目总结")
print("=" * 50)
print(f"""
 最终成绩: 78.947%

 关键成功因素:
   1. 使用训练集统计量填充测试集 (避免数据泄露)
   2. 从姓名提取称谓 (Title 是强特征)
   3. 家庭规模特征 (FamilySize, IsAlone)
   4. 年龄分段 (AgeBand)
   5. StratifiedKFold 交叉验证

 踩过的坑:
   1. 集成学习反而降低分数 (树模型太相似)
   2. 过度的特征工程可能适得其反
""")

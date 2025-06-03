import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, recall_score, f1_score, confusion_matrix, roc_curve
import matplotlib.pyplot as plt
import matplotlib

# 设置matplotlib中文字体，防止中文乱码
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False

# 1. 数据读取与预处理
file_path = '1_default of credit card clients.xls'
data = pd.read_excel(file_path, header=1)
data = data.rename(columns={'default payment next month': 'default'})
data = data.dropna()
data = data[(data['AGE'] >= 18) & (data['AGE'] <= 100)]
data['EDUCATION'] = data['EDUCATION'].map({1: 1, 2: 2, 3: 3, 4: 4, 5: 4, 6: 4, 0: 4})
data['MARRIAGE'] = data['MARRIAGE'].map({1: 1, 2: 2, 3: 3, 0: 3})

# 特征分组
base_features = ['SEX', 'EDUCATION', 'MARRIAGE', 'AGE']
dynamic_features = [f'PAY_{i}' for i in [0,2,3,4,5,6]] + \
                   [f'BILL_AMT{i}' for i in range(1,7)] + \
                   [f'PAY_AMT{i}' for i in range(1,7)]
X_base = data[base_features]
X_dynamic = data[dynamic_features]
y = data['default']

# 划分训练集和测试集
X_base_train, X_base_test, X_dyn_train, X_dyn_test, y_train, y_test = train_test_split(
    X_base, X_dynamic, y, test_size=0.2, random_state=42, stratify=y)

# 标准化
scaler_base = StandardScaler()
X_base_train_scaled = scaler_base.fit_transform(X_base_train)
X_base_test_scaled = scaler_base.transform(X_base_test)
scaler_dyn = StandardScaler()
X_dyn_train_scaled = scaler_dyn.fit_transform(X_dyn_train)
X_dyn_test_scaled = scaler_dyn.transform(X_dyn_test)

# 2. 逻辑回归（基础特征）
lr = LogisticRegression(max_iter=1000)
lr.fit(X_base_train_scaled, y_train)
base_prob = lr.predict_proba(X_base_test_scaled)[:,1]

# 3. 随机森林（动态特征）
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_dyn_train_scaled, y_train)
dyn_prob = rf.predict_proba(X_dyn_test_scaled)[:,1]

# 4. 融合概率
final_prob = 0.5 * base_prob + 0.5 * dyn_prob
final_pred = (final_prob >= 0.5).astype(int)

# 5. 评估
auc = roc_auc_score(y_test, final_prob)
acc = accuracy_score(y_test, final_pred)
recall = recall_score(y_test, final_pred)
f1 = f1_score(y_test, final_pred)
cm = confusion_matrix(y_test, final_pred)
print('融合模型评估结果：')
print(f'AUC: {auc:.4f}')
print(f'准确率: {acc:.4f}')
print(f'召回率: {recall:.4f}')
print(f'F1分数: {f1:.4f}')
print('混淆矩阵：')
print(cm)
print('-'*30)

def ks_score(y_true, y_prob):
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    return max(tpr - fpr)
print(f'融合模型KS值: {ks_score(y_test, final_prob):.4f}')

# 6. 信用评分体系
def prob_to_score(prob, A=600, B=20/np.log(2)):
    odds = prob / (1 - prob)
    score = A - B * np.log(odds)
    return score

scores = prob_to_score(final_prob)

def score_to_grade(score):
    if score >= 750:
        return 'A'
    elif score >= 700:
        return 'B'
    elif score >= 650:
        return 'C'
    elif score >= 600:
        return 'D'
    else:
        return 'E'

grades = [score_to_grade(s) for s in scores]
score_df = pd.DataFrame({'score': scores, 'grade': grades, 'true': y_test.values})

print('融合模型信用等级分布:')
print(score_df['grade'].value_counts())

plt.figure(figsize=(8,5))
plt.hist(scores, bins=30, edgecolor='k')
plt.title('信用分数分布（融合模型）')
plt.xlabel('分数')
plt.ylabel('人数')
plt.show() 
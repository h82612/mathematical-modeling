import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score, recall_score, f1_score, confusion_matrix, roc_curve
from imblearn.over_sampling import SMOTE
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

X = data.drop(['ID', 'default'], axis=1)
y = data['default']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

# 使用SMOTE处理样本不平衡
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

# 特征选择（基于随机森林）
from sklearn.ensemble import RandomForestClassifier
rf_selector = RandomForestClassifier(n_estimators=100, random_state=42)
rf_selector.fit(X_train, y_train)
importances = rf_selector.feature_importances_
indices = np.argsort(importances)[::-1]
top_n = 20
X_train_sel = X_train[:, indices[:top_n]]
X_test_sel = X_test[:, indices[:top_n]]

# 逻辑回归模型参数调优
param_grid_lr = {
    'C': [0.01, 0.1, 1, 10, 100],
    'class_weight': [None, 'balanced'],
    'solver': ['liblinear', 'saga']
}
grid_lr = GridSearchCV(LogisticRegression(max_iter=1000), param_grid_lr, cv=5, scoring='roc_auc')
grid_lr.fit(X_train_sel, y_train)
best_lr = grid_lr.best_estimator_
y_pred_lr = best_lr.predict(X_test_sel)
y_prob_lr = best_lr.predict_proba(X_test_sel)[:, 1]

def evaluate(y_true, y_pred, y_prob, model_name):
    auc = roc_auc_score(y_true, y_prob)
    acc = accuracy_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    print(f"{model_name}评估结果：")
    print(f"AUC: {auc:.4f}")
    print(f"准确率: {acc:.4f}")
    print(f"召回率: {recall:.4f}")
    print(f"F1分数: {f1:.4f}")
    print("混淆矩阵：")
    print(confusion_matrix(y_true, y_pred))
    print("-" * 30)
    return auc

def ks_score(y_true, y_prob):
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    return max(tpr - fpr)

evaluate(y_test, y_pred_lr, y_prob_lr, "逻辑回归")
print(f"逻辑回归KS值: {ks_score(y_test, y_prob_lr):.4f}")

def prob_to_score(prob, A=600, B=20/np.log(2)):
    odds = prob / (1 - prob)
    score = A - B * np.log(odds)
    return score

scores = prob_to_score(y_prob_lr)

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

print('逻辑回归信用等级分布:')
print(score_df['grade'].value_counts())

plt.figure(figsize=(8,5))
plt.hist(scores, bins=30, edgecolor='k')
plt.title('信用分数分布（逻辑回归）')
plt.xlabel('分数')
plt.ylabel('人数')
plt.show() 
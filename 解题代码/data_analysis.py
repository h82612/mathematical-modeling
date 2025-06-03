import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 1. 数据清洗与预处理
def load_and_clean_data():
    # 读取数据
    df = pd.read_excel('1_default of credit card clients.xls', skiprows=1)
    # 重命名列名，下一期是否违约
    df = df.rename({'default payment next month':'DEFAULT'},axis=1)
    
    # 检查并处理缺失值
    print('\n数据缺失值情况：')
    print(df.isnull().sum())
    
    # 处理异常值
    # 检查年龄的合理范围
    df = df[(df['AGE'] >= 18) & (df['AGE'] <= 100)]
    
    # 处理教育和婚姻状况中的未知值
    df['EDUCATION'] = df['EDUCATION'].map({1:1, 2:2, 3:3, 4:4, 5:4, 6:4, 0:4})
    df['MARRIAGE'] = df['MARRIAGE'].map({1:1, 2:2, 3:3, 0:3})
    
    return df

# 2. 探索性数据分析
def exploratory_analysis(df):
    # 基本统计描述
    print('\n数据基本统计描述：')
    print(df.describe())
    
    # 个人特征分析
    plt.figure(figsize=(15, 5))
    
    # 年龄分布
    plt.subplot(131)
    sns.histplot(data=df, x='AGE', hue='DEFAULT', multiple="stack")
    plt.title('年龄分布与违约关系')
    
    # 教育程度分布
    plt.subplot(132)
    edu_default = pd.crosstab(df['EDUCATION'], df['DEFAULT'])
    edu_default.plot(kind='bar', stacked=True)
    plt.title('教育程度与违约关系')
    plt.xticks(range(4), ['研究生', '大学', '高中', '其他'])
    
    # 婚姻状况分布
    plt.subplot(133)
    marriage_default = pd.crosstab(df['MARRIAGE'], df['DEFAULT'])
    marriage_default.plot(kind='bar', stacked=True)
    plt.title('婚姻状况与违约关系')
    plt.xticks(range(3), ['已婚', '单身', '其他'])
    
    plt.tight_layout()
    plt.savefig('personal_features.png')
    
    # 财务状况分析
    plt.figure(figsize=(15, 5))
    
    # 信用额度分布
    plt.subplot(131)
    sns.boxplot(data=df, y='LIMIT_BAL', x='DEFAULT')
    plt.title('信用额度与违约关系')
    
    # 账单金额分析
    plt.subplot(132)
    df['AVG_BILL_AMT'] = df[[f'BILL_AMT{i}' for i in range(1, 7)]].mean(axis=1)
    sns.boxplot(data=df, y='AVG_BILL_AMT', x='DEFAULT')
    plt.title('平均账单金额与违约关系')
    
    # 还款金额分析
    plt.subplot(133)
    df['AVG_PAY_AMT'] = df[[f'PAY_AMT{i}' for i in range(1, 7)]].mean(axis=1)
    sns.boxplot(data=df, y='AVG_PAY_AMT', x='DEFAULT')
    plt.title('平均还款金额与违约关系')
    
    plt.tight_layout()
    plt.savefig('financial_features.png')
    
    return df

# 3. 统计分析
def statistical_analysis(df):
    # 计算相关性
    numeric_cols = ['LIMIT_BAL', 'AGE', 'AVG_BILL_AMT', 'AVG_PAY_AMT'] + \
                   [f'PAY_{i}' for i in (0, 2, 3, 4, 5, 6)] + \
                   [f'BILL_AMT{i}' for i in range(1, 7)] + \
                   [f'PAY_AMT{i}' for i in range(1, 7)]
    
    correlation_matrix = df[numeric_cols + ['DEFAULT']].corr()
    
    # 绘制相关性热力图
    plt.figure(figsize=(12, 8))
    sns.heatmap(correlation_matrix[['DEFAULT']].sort_values(by='DEFAULT', ascending=False),
                annot=True, cmap='coolwarm', center=0)
    plt.title('特征与违约的相关性分析')
    plt.tight_layout()
    plt.savefig('correlation_analysis.png')
    
    # 卡方检验 - 分类变量
    categorical_vars = ['SEX', 'EDUCATION', 'MARRIAGE']
    print('\n分类变量与违约的卡方检验结果：')
    for var in categorical_vars:
        contingency_table = pd.crosstab(df[var], df['DEFAULT'])
        chi2, p_value = stats.chi2_contingency(contingency_table)[:2]
        print(f'{var}: chi2 = {chi2:.2f}, p-value = {p_value:.4f}')
    
    # 方差分析 - 连续变量
    continuous_vars = ['LIMIT_BAL', 'AGE', 'AVG_BILL_AMT', 'AVG_PAY_AMT']
    print('\n连续变量与违约的方差分析结果：')
    for var in continuous_vars:
        f_stat, p_value = stats.f_oneway(df[df['DEFAULT']==0][var],
                                        df[df['DEFAULT']==1][var])
        print(f'{var}: F = {f_stat:.2f}, p-value = {p_value:.4f}')

def main():
    # 1. 数据清洗与预处理
    df = load_and_clean_data()
    
    # 2. 探索性数据分析
    df = exploratory_analysis(df)
    
    # 3. 统计分析
    statistical_analysis(df)

if __name__ == '__main__':
    main()
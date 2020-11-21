import numpy as np
import seaborn as sns
import pandas as pd

from scipy import stats, integrate
import matplotlib.pyplot as plt
import seaborn as sns
# 频数直方图

# 生成100个成标准正态分布的随机数
x = np.random.normal(size=100)

# sns.distplot画频数直方图
# kde=True，进行核密度估计
sns.distplot(x,kde=True)
plt.show()


# 散点图
# 一般用散点图展示两个变量间的关系
mean, cov = [0, 1], [(1, .5), (.5, 1)]
# 二维多正态分数的数据
data = np.random.multivariate_normal(mean, cov, 200)
df = pd.DataFrame(data, columns=["x", "y"]) #将array结构转换为dataframe结构，添加列名
sns.jointplot(x='x', y='y', data=df, size=7)  # x,y参数也可以用x=df['x'], y=df['y'] 的形式传入
# sns.jointplot画双变量关系图，
# data传入dataframe，x，y设置两个变量数据，
# size设置图的大小
plt.show()

# 热度图
uniform_data = np.random.rand(3, 3)
# print (uniform_data)
heatmap = sns.heatmap(uniform_data, annot=True)
# 通过颜色的变化反映数字大小
plt.show()
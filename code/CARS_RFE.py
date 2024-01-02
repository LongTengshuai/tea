import pandas as pd
import numpy as np


# 导入RFE方法和线性回归基模型

from sklearn.feature_selection import RFE

from sklearn.linear_model import LinearRegression
from sklearn.cross_decomposition import PLSRegression

# 自变量特征
excelFile = pd.read_excel("F:\华南农业大学龙老师组\茶叶\茶叶茶多酚\实验项目\数据\dtest\可见+近红外\波段\CARS\特征波段融合\叶绿素a123xiu_testrong.xlsx", header=None)#正11_转置
excelFile = np.array(excelFile)
# print(excelFile)
featrues = excelFile[:, 1:]
category = excelFile[:, 0]

rfe = RFE(estimator=PLSRegression(copy=True, max_iter=1000, n_components=4, scale=True,
    tol=1e-06), n_features_to_select=30) # 定义递归的函数类型，这里采用的是LinearRegression函数，定义变量的数量，这里是11个。sklearn中还有其他函数可选

# fit 方法训练选择特征属性

sFeature = rfe.fit_transform(featrues, category) #这里选择t变量，即因变量。举例前面自变量为x, 这里的t即代表y，y = kx.

print(rfe.get_support())  # 查看满足条件的属性
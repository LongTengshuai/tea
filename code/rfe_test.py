"""
使用RFE进行特征选择：RFE是常见的特征选择方法，也叫递归特征消除。它的工作原理是递归删除特征，
并在剩余的特征上构建模型。它使用模型准确率来判断哪些特征（或特征组合）对预测结果贡献较大。
"""
import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

excelFile = pd.read_excel("F:\华南农业大学龙老师组\茶叶\茶叶茶多酚\实验项目\数据\dtest\可见+近红外\波段\CARS\特征波段融合\叶绿素a123xiu_rongCARS.xlsx", header=None)#正11_转置
excelFile = np.array(excelFile)
# print(excelFile)
featrues = excelFile[:, 1:]
category = excelFile[:, 0]
#testdataset = datasets.load_iris()
model = LogisticRegression()
rfe = RFE(model, 3)
rfe = rfe.fit(featrues, category)
print(rfe.support_)
print(rfe.ranking_)
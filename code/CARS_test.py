# 导入 pandas 读取数据
import pandas as pd
import numpy as np

import CARS

# 读取数据
excelFile = pd.read_excel("F:\华南农业大学龙老师组\茶叶\茶叶茶多酚\实验项目\数据\dtest\可见+近红外\波段\CARS\特征波段融合\类胡萝卜素123xiu_testrong.xlsx", header=None)#正11_转置
#excelFile = np.array(excelFile)

# m * n
print("数据矩阵 data.shape：",excelFile.shape)

# 50个样本， 600个 波段 第一列是 桃子糖度值 需要分离开
X = excelFile.values[:,1:]
# 等同操作
#X = data.drop(['Brix'], axis=1)

y = excelFile.values[:,0]
# 等同操作
# y = data.loc[:,'Brix'].values

print(f"X.shape:{X.shape}, y.shape:{y.shape}")
#建模筛选
lis = CARS.CARS_Cloud(X,y)
print("获取波段数：",len(lis))
print(lis)
#导出波段

X_=X[:,lis]
#print(X_)
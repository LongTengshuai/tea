import numpy as np

import pandas as pd

from sklearn.linear_model import Lasso

inputfile = 'F:\华南农业大学龙老师组\茶叶\茶叶茶多酚\实验项目\数据\dtest\可见+近红外\波段\CARS\特征波段融合\叶绿素a123xiu_rongCARS.csv' #输入的数据文件

data = pd.read_csv(inputfile) #读取数据

lasso = Lasso(1000)  #调用Lasso()函数，设置λ的值为1000

lasso.fit(data.iloc[:,:-1],data.iloc[:,-1])

print('相关系数为：',np.round(lasso.coef_,5))  #输出结果，保留五位小数

## 计算相关系数非零的个数

print('相关系数非零个数为：',np.sum(lasso.coef_ != 0))

mask = lasso.coef_ != 0  #返回一个相关系数是否为零的布尔数组

print('相关系数是否为零：',mask)

outputfile = 'F:\华南农业大学龙老师组\茶叶\茶叶茶多酚\实验项目\数据\dtest\可见+近红外\波段\CARS\特征波段融合\CARS_LASSO\叶绿素a.csv'  #输出的数据文件

new_reg_data = data.iloc[:, mask]  #返回相关系数非零的数据

new_reg_data.to_csv(outputfile)  #存储数据
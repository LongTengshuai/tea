import numpy as np
import pandas as pd
import xlsxwriter # 负责写excel
import math


excelFile = pd.read_excel("F:\华南农业大学龙老师组\茶叶\茶叶茶多酚\实验项目\数据\预处理数据\方案二\叶绿素a1234ipls_a_test.xlsx", header=None)#正11_转置
excelFile = np.array(excelFile)
# print(excelFile)
x = excelFile[:, 1:18]
y = excelFile[:, 19:36]
f=x/y
#保存预处理数据
filename =xlsxwriter.Workbook('F:\华南农业大学龙老师组\茶叶\茶叶茶多酚\实验项目\数据\预处理数据\方案二\叶绿素a1234ipls_b_test.xlsx') #创建工作簿
Worksheet= filename.add_worksheet('sheet1') #创建sheet
[h,l]=f.shape #h为行数，l为列数
for i in range (h):
    for j in range (l):
        Worksheet.write(i,j,f[i,j])
filename.close()
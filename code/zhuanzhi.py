import xlwt  # 负责写excel
import xlrd
import numpy as np
import pandas as pd
filename =xlwt.Workbook() #创建工作簿
sheet1 = filename.add_sheet(u'sheet1',cell_overwrite_ok=True) #创建sheet
#readbook_2 = xlrd.open_workbook('F:\华南农业大学龙老师组\茶叶\实验\反面_test_xiu.xlsx')
readbook_2 = xlrd.open_workbook('F:\华南农业大学龙老师组\茶叶\茶叶茶多酚\实验项目\数据\dtest\可见+近红外\光谱.xlsx')
sheet_2 = readbook_2.sheet_by_index(0)
Y = [sheet_2.cell_value(i, j) for i in range(0, 142) for j in range(0,365)]
Y1 = np.array(Y)
Y2 = Y1.reshape(142,365)
Y3 = np.transpose(Y2)
x= Y3[:,0:142] #20:170
[h,l]=x.shape #h为行数，l为列数
for i in range (h):
    for j in range (l):
        sheet1.write(i,j,x[i,j])
#filename.save('F:\华南农业大学龙老师组\茶叶\实验\(转置)反面_test_xiu.xls')
filename.save('F:\华南农业大学龙老师组\茶叶\茶叶茶多酚\实验项目\数据\dtest\可见+近红外\(转置)光谱.xls')

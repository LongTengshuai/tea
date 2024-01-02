import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split,cross_val_score,cross_validate # 交叉验证所需的函数
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import KFold,LeaveOneOut,LeavePOut,ShuffleSplit # 交叉验证所需的子集划分方法
from sklearn.model_selection import StratifiedKFold,StratifiedShuffleSplit # 分层分割
from sklearn.model_selection import GroupKFold,LeaveOneGroupOut,LeavePGroupsOut,GroupShuffleSplit # 分组分割
from sklearn.model_selection import TimeSeriesSplit # 时间序列分割
from sklearn import metrics
from sklearn.metrics import recall_score  # 模型度量
from sklearn.model_selection import cross_val_score#交叉验证
from sklearn.model_selection import LeaveOneOut#留一法完全交叉验证的方法
from scipy.signal import savgol_filter#SG平滑
import xlrd  #读取EXCEL
import matplotlib.pyplot  as plt#画图
from SNV import *
from plotHyperspectral import *
from MSC import *
from sklearn.decomposition import PCA, FactorAnalysis, TruncatedSVD, FastICA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV, RandomizedSearchCV, \
    learning_curve
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error as MSE, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.cross_decomposition import PLSRegression
readbook_2 = xlrd.open_workbook('F:\华南农业大学龙老师组\茶叶\实验\正面_test_xiu.xlsx')
#sheet_1 = readbook_1.sheet_by_index(0)
##sheet_2 = readbook_2.sheet_by_index(0)
sheet_2 = readbook_2.sheet_by_index(0)
#X=[sheet_1.cell_value(i, j) for i in range(0, sheet_1.nrows) for j in range(0, 225)]
#X1 = np.array(X)
#X2 = x1.reshape(sheet_1.nrows,225)
#xx = x2[2,0:176]

Y = [sheet_2.cell_value(i, j) for i in range(0, 225) for j in range(1,217)]
Y1 = np.array(Y)
Y2 = Y1.reshape(225,216)
Y3 = np.transpose(Y2)
#my_font = font_manager.FontProperties(fname=r"C:/Windows/Fonts/simsun.ttc", size=12)

x= Y3[:,0:225] #20:170
#y = Y3[:,176]
#x1 = preprocess_MSC(x)
#x1 = preprocess_SNV(x)
x1= savgol_filter(Y3, 5, 3, mode='nearest')
#x1=savgol_filter(x, 5, 3, mode= 'nearest')#SG
#X3 = savgol_filter(X, 5, 3, mode='nearest')
x_date=sheet_2.col_values(0,0)
#print(x_date)
for i in range(0,216):
    y_date=x1[i,:]
    plt.plot(x_date,y_date)
print('y_date',len(y_date))
print('x_date',len(x_date))
plt.xlabel("波段(nm)",fontsize=18)#设置横坐标，以及字体大小
plt.ylabel("反射率",fontsize=18)#设置纵坐标，以及字体大小
plt.title("茶叶光谱数据_sg",fontsize=18)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.rcParams['font.sans-serif']=['SimHei'] #显示中文标签
plt.rcParams['axes.unicode_minus']=False
plt.show()


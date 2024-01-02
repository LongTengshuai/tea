import numpy as np
import pandas as pd
import CARS
import xlsxwriter # 负责写excel
import math
from lssvr import LSSVR
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
from D1 import *
from D2 import *
from sklearn.decomposition import PCA, FactorAnalysis, TruncatedSVD, FastICA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV, RandomizedSearchCV, \
    learning_curve
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error as MSE, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.cross_decomposition import PLSRegression
from sklearn.kernel_ridge import KernelRidge

# 读取数据
excelFile = pd.read_excel("F:\华南农业大学龙老师组\茶叶\茶叶茶多酚\实验项目\数据\dtest\可见+近红外\数据\波段\红外CARS\氨基酸34_rong.xlsx", header=None)#正11_转置
#excelFile = np.array(excelFile)

# m * n
print("数据矩阵 data.shape：",excelFile.shape)

# 50个样本， 600个 波段 第一列是 桃子糖度值 需要分离开
X = excelFile.values[:,1:]
# 等同操作
#X = data.drop(['Brix'], axis=1)

category = excelFile.values[:,0]
# 等同操作
# y = data.loc[:,'Brix'].values

print(f"X.shape:{X.shape}, y.shape:{category.shape}")
#建模筛选
lis = CARS.CARS_Cloud(X,category)
print("获取波段数：",len(lis))
print(lis)
#导出波段
featrues=X[:,lis]
#处理方法
# 使用SNV预处理
#featrue = preprocess_SNV(featrues)
# 使用MSC预处理
#featrue= preprocess_MSC(featrues)

# SG平滑
#featrue = savgol_filter(featrues, 5, 3, mode='nearest')
# plotHyperspectralImage(wavelength=wavelength, featrues=featrue, title="SG")
#D1一阶差分
#featrue=D1(featrues)
#D2二阶差分
#featrue=D2(featrues)

#原始数据
featrue=featrues
#
# 进行数据降维PCA
#fqCategoryPCA = PCA(n_components=19,random_state=8)
#fqCategoryPCA = TruncatedSVD(n_components=28, random_state=8)
#fqCategoryPCA = FastICA(n_components=24, random_state=8)#21
for i in range(15,25):
    fqCategoryPCA = FastICA(n_components=i,random_state=0)#38,17
    #fqCategoryPCA = FactorAnalysis(n_components=26, random_state=0)#28,6#26,6#31,12#29,12
    featruesPCA = fqCategoryPCA.fit_transform(featrue)
    #print(featruesPCA.shape)
    # # # # 划分数据集
    #X_train, X_test, y_train, y_test = train_test_split(featruesPCA, category, train_size=0.80, test_size=0.20,
    #random_state=2)
    X_train,X_test,y_train,y_test=train_test_split(featruesPCA,category,test_size=0.2,random_state=6)


    #偏最小支持向量机
    #model = LSSVR(kernel='rbf', gamma=1.8)

    #最小二乘法
    #model = LinearRegression(normalize=True)

    #核岭回归
    model = KernelRidge(alpha=0.01, kernel='rbf', gamma=0.1)

    #偏最小二乘法
    #model=PLSRegression(copy=True, max_iter=1000, n_components=3, scale=True,
         #tol=1e-06)
    model.fit(X_train ,y_train)
    y_train_predict = model.predict(X_train)
    y_test_predict = model.predict(X_test)

    ##
    #
    print("训练集决定系数为:", r2_score(y_train, y_train_predict))
    print("训练集均方根误差为:", math.sqrt(MSE(y_train, y_train_predict)))
    print("测试集决定系数为:", r2_score(y_test, y_test_predict))
    print("测试集均方根误差为:", math.sqrt(MSE(y_test, y_test_predict)))
    #
    fig,ax=plt.subplots()
    #ax.scatter(x,y,c='r')
    plt.xlabel("Reference chlorophyll",fontsize=18)#设置横坐标，以及字体大小
    plt.ylabel("Predicted chlorophyll",fontsize=18)#设置纵坐标，以及字体大小
    ax.scatter(y_train, y_train_predict, c='red')
    plt.scatter(y_test, y_test_predict)
    plt.plot(y_train, y_train, c="black")
    plt.title("茶多酚D2_ipls_plsr",fontsize=18)
    #n = np.arange(len(y_test))
    #for i, txt in enumerate(n):
    #    ax.annotate(txt, (y_test[i], y_test_predict[i]))
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.rcParams['font.sans-serif']=['SimHei'] #显示中文标签
    plt.rcParams['axes.unicode_minus']=False
    #plt.show()

import numpy as np
import pandas as pd
import xlsxwriter # 负责写excel
import math
from openpyxl import Workbook
from lssvr import LSSVR
from sklearn.svm import SVR
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
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel
import sklearn.gaussian_process.kernels as k
from sklearn.kernel_ridge import KernelRidge

excelFile = pd.read_excel("F:\华南农业大学龙老师组\茶叶\茶叶茶多酚\实验项目\数据\dtest\可见+近红外\数据\波段\CARS\类胡萝卜素123xiu_test.xlsx", header=None)#正11_转置
excelFile = np.array(excelFile)
# print(excelFile)
featrues = excelFile[:, 1:]
category = excelFile[:, 0]




# 使用SNV预处理
#featrue = preprocess_SNV(featrues)
# 使用MSC预处理
featrue= preprocess_MSC(featrues)

# SG平滑
#featrue = savgol_filter(featrues, 5, 3, mode='nearest')
# plotHyperspectralImage(wavelength=wavelength, featrues=featrue, title="SG")
#D1一阶差分
#featrue=D1(featrues)
#D2二阶差分
#featrue=D2(featrues)

#保存预处理数据
#filename =xlsxwriter.Workbook('F:\华南农业大学龙老师组\茶叶\茶叶茶多酚\实验项目\数据\dtest\可见+近红外\波段\红外CARS\氨基酸34_SNV.xlsx') #创建工作簿
#Worksheet= filename.add_worksheet('sheet1') #创建sheet
#[h,l]=featrue.shape #h为行数，l为列数
#for i in range (h):
#    for j in range (l):
#        Worksheet.write(i,j,featrue[i,j])
#filename.close()

#原始数据
#featrue=featrues
#
# 进行数据降维PCA
#for i in range(16,25):
#fqCategoryPCA = PCA(n_components=26,random_state=0)
#fqCategoryPCA = TruncatedSVD(n_components=28, random_state=8)
#fqCategoryPCA = FastICA(n_components=24, random_state=8)#21
for i in range(17,26):
    fqCategoryPCA = FastICA(n_components=i,random_state=0)#38,17
    #fqCategoryPCA = FactorAnalysis(n_components=28, random_state=0)#28,6#26,6#31,12#29
    featruesPCA = fqCategoryPCA.fit_transform(featrue)
    #print(featruesPCA.shape)
    # # # # 划分数据集
    #X_train, X_test, y_train, y_test = train_test_split(featruesPCA, category, train_size=0.80, test_size=0.20,
    #random_state=2)
    X_train,X_test,y_train,y_test=train_test_split(featruesPCA,category,test_size=0.2,random_state=2)
    #...........
    #X_train=featrue
    #y_train=category
    #print(y_train)
    #
    ##fqCategoryPCA = FastICA(n_components=38,random_state=17)
    ##featruesPCA = fqCategoryPCA.fit_transform(X_train)
    # # # 调用test_RandomForestRegressor_max_depth
    # # test_RandomForestRegressor_max_depth(X_train, X_test, y_train, y_test)
    #
    #XGBOOST
    #model = XGBRegressor(random_state=8, n_jobs=4, min_child_weight=12, subsample=0.8, learning_rate=0.30266)
    ##model.fit(X_train, y_train)
    ##y_train_predict = model.predict(X_train)
    ##y_test_predict = model.predict(X_test)
    #偏最小支持向量机
    #model = LSSVR(kernel='rbf', gamma=3)

    #支持向量机
    #model=SVR(kernel='rbf', C=1e2, gamma=1.5)#高斯

    #GPR(高斯过程回归)
    #kernel = C(constant_value=0) * RBF(length_scale=1, length_scale_bounds=(1e-1,1e2))
    #model= GaussianProcessRegressor(kernel=kernel)

    #核岭回归

    model= KernelRidge(alpha=0.01, kernel='rbf', gamma=0.3)

    #最小二乘法
    #model = LinearRegression(normalize=True)
    #偏最小二乘法
    #model=PLSRegression(copy=True, max_iter=1000, n_components=2, scale=True,
    #tol=1e-06)
    model.fit(X_train ,y_train)
    y_train_predict = model.predict(X_train)
    y_test_predict = model.predict(X_test)
    #.........................
    #model.fit(X_train,y_train )
    #y_train_predict = model.predict(X_train)



    #print('The accuracy of the Logistic Regression is: {0}'.format(metrics.accuracy_score(y_train_predict,y_test)))
    #scores = cross_val_score(model, X_train, y_train, cv = 11,scoring = 'r2')
    #print( scores.mean())
    # ===================================直接调用交叉验证评估模型==========================
    #scores = cross_val_score(model, featruesPCA, category, cv=4)  #cv为迭代次数。
    #print(scores)  # 打印输出每次迭代的度量值（准确度）
    #print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))  # 获取置信区间。（也就是均值和方差）
    #print(clf.score(X_test, y_test))
    # ===================================直接调用交叉验证评估模型(可以对交叉验证方式进行指定，如验证次数，训练集测试集划分比例等)==========================
    #for i in range(21):
    #cv = ShuffleSplit(n_splits=5, test_size=.1, random_state=3)
    #scores=cross_val_score(model, featruesPCA, category,  cv=cv)
    #print(scores)

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
    #yn=np.array([y_test,y_test_predict])
    #yn=np.array([y_train,y_train_predict])
    #YN= np.transpose(yn)
    #print(YN)
    #filename =xlsxwriter.Workbook('D:\F盘文件\硕士毕业\硕博士论文\毕业论文图\最优结果\测试数据_LS_SVR.xlsx') #创建工作簿
    #Worksheet= filename.add_worksheet('sheet1') #创建sheet
    #[h,l]=YN.shape #h为行数，l为列数
    #for i in range (h):
    #        for j in range (l):
    #                Worksheet.write(i,j,YN[i,j])
    #filename.close()

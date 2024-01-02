import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
import xlrd  #读取EXCEL
import matplotlib.pyplot  as plt#画图

# 实现多元散射
def preprocess_MSC(data):
    # 样本数量
    n = data.shape[0]#数据集总行数
    k = np.zeros(n)
    b = np.zeros(n)

    M = np.mean(data, axis=0)#axis=0，那么输出矩阵是1行，求每一列的平均（按照每一行去求平均）；
    # axis=1，输出矩阵是1列，求每一行的平均（按照每一列去求平均）。还可以这么理解，axis是几，那就表明哪一维度被压缩成1。
    from sklearn.linear_model import LinearRegression

    for i in range(n):
        y = data[i, :]
        y = y.reshape(-1, 1)
        M = M.reshape(-1, 1)
        model = LinearRegression()
        model.fit(M, y)
        k[i] = model.coef_
        b[i] = model.intercept_

    spec_msc = np.zeros_like(data)
    for i in range(n):
        bb = np.repeat(b[i], data.shape[1])
        kk = np.repeat(k[i], data.shape[1])
        temp = (data[i, :] - bb) / kk
        spec_msc[i, :] = temp

    return spec_msc
#SNV
def preprocess_SNV(data):
    # 计算每一条光谱的均值
    mean = np.mean(data, axis=1, dtype=np.float64)
    # 计算每一条光谱的标准差
    std = np.std(data, axis=1, dtype=np.float64)
    for i in range(data.shape[0]):
        data[i, :] = (data[i, :] - mean[i]) / std[i]
    return data

# 一阶导
def D1(data):
    n, p = data.shape
    Di = np.ones((n, p - 1))
    for i in range(n):
        Di[i] = np.diff(data[i])
    return Di
#二阶导
def D2(data):
    n, p = data.shape
    Di = np.ones((n, p - 2))
    for i in range(n):
        Di[i] = np.diff(np.diff(data[i]))
    return Di
#一阶差分
def C1(sdata):
    """
    一阶差分
    """
    temp1 = pd.DataFrame(sdata)
    temp2 = temp1.diff(axis=1)
    temp3 = temp2.values
    return np.delete(temp3, 0, axis=1)




readbook_2 = pd.read_excel('F:\华南农业大学龙老师组\茶叶\茶叶茶多酚\实验项目\数据\dtest\可见+近红外\红外.xlsx')
#sheet_2 = readbook_2.sheet_by_index(0)
#Y = [sheet_2.cell_value(i, j) for i in range(0, 176) for j in range(1,142)]
#Y1 =np.array(Y)
#Y2 = Y1.reshape(176,141)
excelFile=np.array(readbook_2)
Y= excelFile[:, 1:]
X= excelFile[:, 0]
#Y3 = np.transpose(Y2)
#my_font = font_manager.FontProperties(fname=r"C:/Windows/Fonts/simsun.ttc", size=12)

#x= Y3[:,0:225] #20:170
#y = Y3[:,176]
#x1 = preprocess_MSC(x)
#x1 = preprocess_SNV(x)
y_date=Y
#x_date=sheet_2.col_values(0,0)
x_date=X
#print(x_date)
#for i in range(0,141):
#    y_date=x1[i,:]
plt.plot(x_date,y_date)
print('y_date',len(y_date))
print('x_date',len(x_date))
plt.xlabel("波段(nm)",fontsize=18)#设置横坐标，以及字体大小
plt.ylabel("反射率",fontsize=18)#设置纵坐标，以及字体大小
plt.title("茶叶光谱",fontsize=18)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.rcParams['font.sans-serif']=['SimHei'] #显示中文标签
plt.rcParams['axes.unicode_minus']=False
plt.show()
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=None, stratify=y)
#df = pd.read_excel('F:\华南农业大学龙老师组\茶叶\实验\正面_test.xlsx')
#print(preprocess_MSC(df))
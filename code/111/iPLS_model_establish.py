# 导入第三方库
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utility_spectrum import spxy, ipls
import scipy.io as sio
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import mean_squared_error

# 1.数据获取
mat = sio.loadmat('ndfndf.mat')
data = mat['ndfndf']
x, y = data[:, :1050], data[:, 1050]
print(x.shape, y.shape)

# 2.样本集划分
x_train, x_test, y_train, y_test = spxy(x, y, test_size=0.33)
print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)


intervals = 20
ipls(intervals, x_train, x_test, y_train, y_test)




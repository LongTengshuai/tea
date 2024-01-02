import numpy as np


# 实现多元散射
def preprocess_MSC(data):
    # 样本数量
    n = data.shape[0]
    k = np.zeros(n)
    b = np.zeros(n)

    M = np.mean(data, axis=0)
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

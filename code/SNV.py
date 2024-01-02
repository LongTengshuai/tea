import numpy as np


# SNV处理
def preprocess_SNV(data):
    # 计算每一条光谱的均值
    mean = np.mean(data, axis=1, dtype=np.float64)
    # 计算每一条光谱的标准差
    std = np.std(data, axis=1, dtype=np.float64)
    for i in range(data.shape[0]):
        data[i, :] = (data[i, :] - mean[i]) / std[i]
    return data

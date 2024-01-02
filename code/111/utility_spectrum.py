import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import *
from sklearn.model_selection import learning_curve
from sklearn.cross_decomposition import PLSRegression


def PlotSpectrum(spec):
    """
    :param spec: shape (n_samples, n_features)
    :return: plt
    """

    plt.figure(figsize=(5.2, 3.1), dpi=300)
    x = np.arange(400, 400 + 2 * spec.shape[1], 2)
    for i in range(spec.shape[0]):
        plt.plot(x, spec[i, :], linewidth=0.6)

    fonts = 8
    plt.xlim(350, 2550)
    # plt.ylim(0, 1)
    plt.xlabel('Wavelength (nm)', fontsize=fonts)
    plt.ylabel('absorbance (AU)', fontsize=fonts)
    plt.yticks(fontsize=fonts)
    plt.xticks(fontsize=fonts)
    plt.tight_layout(pad=0.3)
    plt.grid(True)
    return plt


def ks(x, y, test_size=0.2):
    """

    :param x: shape (n_samples, n_features)
    :param y: shape (n_sample, )
    :param test_size: the ratio of test_size (float)
    :return: spec_train: (n_samples, n_features)
             spec_test: (n_samples, n_features)
             target_train: (n_sample, )
             target_test: (n_sample, )
    """
    M = x.shape[0]             # 样本数量
    N = round((1-test_size) * M)
    samples = np.arange(M)     # 产生样本编号

    D = np.zeros((M, M))        # 初始化

    for i in range((M-1)):
        xa = x[i, :]
        for j in range((i+1), M):
            xb = x[j, :]
            D[i, j] = np.linalg.norm(xa-xb)  # 计算x欧式距离（返回上三角矩阵，对角线为0）

    maxD = np.max(D, axis=0)              # 求得每列元素的最大值
    index_row = np.argmax(D, axis=0)      # 求得每列元素最大值的行位置
    index_column = np.argmax(maxD)        # 求得最大元素的列位置

    m = np.zeros(N)
    m[0] = np.array(index_row[index_column])
    m[1] = np.array(index_column)
    m = m.astype(int)                     # m作为索引必须为int类型

    dminmax = np.zeros(N)
    dminmax[1] = D[m[0], m[1]]   # 存储最大最小值

    for i in range(2, N):  # 遍历待入选的样本
        pool = np.delete(samples, m[:i])  # 待入选样本编号
        dmin = np.zeros((M-i))         # 初始化待入选与已选样本的最小距离
        for j in range((M-i)):         # 已入选i个，在待入选的（M-i）个样本中操作
            indexa = pool[j]           # 待入选样本中的第j个样品
            d = np.zeros(i)            # 初始化第j个待选样本与已选样本之间的距离向量
            for k in range(i):         # 遍历入选样本
                indexb = m[k]          # 已入选的第k个样本
                if indexa < indexb:    # 因为D矩阵是上三角矩阵，d取D矩阵的上三角值（条件判断）
                    d[k] = D[indexa, indexb]
                else:
                    d[k] = D[indexb, indexa]
            dmin[j] = np.min(d)     # 第j个样本与已入选的样本的最小值
        dminmax[i] = np.max(dmin)   # 在最小值中找到最大值
        index = np.argmax(dmin)     # 在最小值中找到最大值的位置
        m[i] = pool[index]          # 进入入选样本集

    m_complement = np.delete(np.arange(x.shape[0]), m)    # 预测集样本编号

    spec_train = x[m, :]
    target_train = y[m]
    spec_test = x[m_complement, :]
    target_test = y[m_complement]
    return spec_train, spec_test, target_train, target_test


def spxy(x, y, test_size=0.2):
    """

    :param x: shape (n_samples, n_features)
    :param y: shape (n_sample, )
    :param test_size: the ratio of test_size
    :return: spec_train :(n_samples, n_features)
             spec_test: (n_samples, n_features)
             target_train: (n_sample, )
             target_test: (n_sample, )
    """
    x_backup = x
    y_backup = y   # 备份
    M = x.shape[0]            # 样本数量
    N = round((1-test_size) * M)
    samples = np.arange(M)    # 产生样本编号

    # 对Y进行标准化处理
    y = (y - np.mean(y))/np.std(y)

    # 初始化
    D = np.zeros((M, M))
    Dy = np.zeros((M, M))

    # 计算欧式距离
    for i in range(M-1):
        xa = x[i, :]
        ya = y[i]
        for j in range((i+1), M):
            xb = x[j, :]
            yb = y[j]
            D[i, j] = np.linalg.norm(xa-xb)      # 计算x欧式距离（返回上三角矩阵，对角线为0）
            Dy[i, j] = np.linalg.norm(ya - yb)   # 计算y欧式距离（返回上三角矩阵，对角线为0）

    Dmax = np.max(D)        # 寻找最大值
    Dymax = np.max(Dy)
    D = D/Dmax + Dy/Dymax   # 联合XY

    maxD = D.max(axis=0)               # 返回数组，每一个元素为D中每列元素最大值
    index_row = D.argmax(axis=0)       # 返回数组，每一个元素为D中每列元素最大值的位置
    index_column = maxD.argmax()       # 返回标量，D中最大的元素所在的列

    m = np.zeros(N)
    m[0] = index_row[index_column]
    m[1] = index_column
    m = m.astype(int)          # m作为索引必须为int类型

    dminmax = np.zeros(N)
    dminmax[1] = D[m[0], m[1]]  # 存储最大最小值

    for i in range(2, N):  # 遍历待入选的样本
        pool = np.delete(samples, m[:i])  # 待入选样本编号
        dmin = np.zeros(M-i)  # 初始化待入选与已选样本的最小距离
        for j in range(M-i):  # 已入选i个，在待入选的（M-i）个样本中操作
            indexa = pool[j]  # 带入选样本中的第j个样品
            d = np.zeros(i)   # 初始化第j个待选样本与已选样本之间的距离向量
            for k in range(i):  # 遍历入选样本
                indexb = m[k]   # 已入选的第k个样本
                if indexa < indexb:  # 因为D矩阵是上三角矩阵，d取D矩阵的上三角值（条件判断）
                    d[k] = D[indexa, indexb]
                else:
                    d[k] = D[indexb, indexa]
            dmin[j] = np.min(d)     # 第j个样本与已入选的样本的最小值
        dminmax[i] = np.max(dmin)   # 在最小值中找到最大值
        index = np.argmax(dmin)     # 在最小值中找到最大值的位置
        m[i] = pool[index]          # 进入入选样本集

    m_complement = np.delete(np.arange(x.shape[0]), m)     # 测试集样本编号

    spec_train = x[m, :]
    target_train = y_backup[m]
    spec_test = x[m_complement, :]
    target_test = y_backup[m_complement]

    return spec_train, spec_test, target_train, target_test


def mean_centralization(sdata):
    """
    均值中心化
    """
    temp1 = np.mean(sdata, axis=0)
    temp2 = np.tile(temp1, sdata.shape[0]).reshape((sdata.shape[0], sdata.shape[1]))
    return sdata - temp2


def standardlize(sdata):
    """
    标准化
    """
    from sklearn import preprocessing
    return preprocessing.scale(sdata)


def snv(sdata):
    """
    标准正态变量变换
    """
    temp1 = np.mean(sdata, axis=1)
    temp2 = np.tile(temp1, sdata.shape[1]).reshape((sdata.shape[0], sdata.shape[1]))
    temp3 = np.std(sdata, axis=1)
    temp4 = np.tile(temp3, sdata.shape[1]).reshape((sdata.shape[0], sdata.shape[1]))
    return (sdata - temp2)/temp4


def D1(sdata):
    """
    一阶差分
    """
    temp1 = pd.DataFrame(sdata)
    temp2 = temp1.diff(axis=1)
    temp3 = temp2.values
    return np.delete(temp3, 0, axis=1)


def D2(sdata):
    """
    二阶差分
    """
    temp2 = (pd.DataFrame(sdata)).diff(axis=1)
    temp3 = np.delete(temp2.values, 0, axis=1)
    temp4 = (pd.DataFrame(temp3)).diff(axis=1)
    spec_D2 = np.delete(temp4.values, 0, axis=1)
    return spec_D2


def msc(sdata):
    n = sdata.shape[0]  # 样本数量
    k = np.zeros(sdata.shape[0])
    b = np.zeros(sdata.shape[0])

    M = np.mean(sdata, axis=0)

    from sklearn.linear_model import LinearRegression
    for i in range(n):
        y = sdata[i, :]
        y = y.reshape(-1, 1)
        M = M.reshape(-1, 1)
        model = LinearRegression()
        model.fit(M, y)
        k[i] = model.coef_
        b[i] = model.intercept_

    spec_msc = np.zeros_like(sdata)
    for i in range(n):
        bb = np.repeat(b[i], sdata.shape[1])
        kk = np.repeat(k[i], sdata.shape[1])
        temp = (sdata[i, :] - bb)/kk
        spec_msc[i, :] = temp
    return spec_msc


def splitspectrum(interval_num, x_train, x_test):
    """
    :param interval_num:  int (common values are 10, 20, 30 or 40)
    :param x_train:  shape (n_samples, n_features)
    :param x_test:  shape (n_samples, n_features)
    :return: x_train_block:intervals splitting for training sets（dict）
            x_test_black： intervals splitting for test sets （dict）
    """
    feature_num = x_train.shape[1]

    x_train_block = {}
    x_test_black = {}
    remaining = feature_num % interval_num  # 用于检查是否能等分
    # （一）特征数量能够等分的情况
    if not remaining:
        interval_size = feature_num / interval_num  # 子区间波点数量
        for i in range(1, interval_num+1):
            # （1）取对应子区间的光谱数据
            feature_start, feature_end = int((i-1) * interval_size), int(i * interval_size)
            x_train_block[str(i)] = x_train[:, feature_start:feature_end]
            x_test_black[str(i)] = x_test[:, feature_start:feature_end]

    # （二）特征数量不能等分的情况(将多余波点等分到后面的几个区间里)
    else:
        separation = interval_num - remaining  # 前几个区间
        intervalsize1 = feature_num // interval_num
        intervalsize2 = feature_num // interval_num + 1

        # （2）前几个子区间(以separation为界)
        for i in range(1, separation+1):
            feature_start, feature_end = int((i-1) * intervalsize1), int(i * intervalsize1)
            x_train_block[str(i)] = x_train[:, feature_start:feature_end]
            x_test_black[str(i)] = x_test[:, feature_start:feature_end]

        # （3）后几个子区间(以separation为界)
        for i in range(separation+1, interval_num+1):
            feature_s = int((i - separation-1) * intervalsize2) + feature_end
            feature_e = int((i - separation) * intervalsize2) + feature_end
            x_train_block[str(i)] = x_train[:, feature_s:feature_e]
            x_test_black[str(i)] = x_test[:, feature_s:feature_e]

    return x_train_block, x_test_black


def ipls(intervals, x_train, x_test, y_train, y_test):
    """

    :param intervals: 区间数量
    :param x_train: shape (n_samples, n_features)
    :param x_test: shape (n_samples, n_features)
    :param y_train: shape (n_samples, )
    :param y_test: shape (n_samples, )
    :return:
    """
    x_train_block, x_test_black = splitspectrum(intervals, x_train, x_test)

    mse = []
    for i in range(1, intervals + 1):
        print("当前区间:", i)
        x_train_interval, x_test_interval = x_train_block[str(i)], x_test_black[str(i)]

        current_fn = x_train_interval.shape[1]
        if current_fn >= 100:
            ncom_upper = 100
        elif current_fn >= 50:
            ncom_upper = current_fn - 10
        else:
            ncom_upper = current_fn - 5
        ncomp = np.arange(5, ncom_upper)

        error = []
        for nc in ncomp:
            print("迭代当前主成分数量:", nc)
            pls = PLSRegression(n_components=nc,
                                scale=True,
                                max_iter=500,
                                tol=1e-06,
                                copy=True)
            pls.fit(x_train_interval, y_train.reshape(-1, 1))
            y_test_pred = pls.predict(x_test_interval)
            mse_temp = mean_squared_error(y_test, y_test_pred.ravel())
            error.append(mse_temp)
        mse.append(np.min(error))

    print(mse)
    plt.figure(figsize=(5.5, 4), dpi=300)
    plt.bar(np.arange(1, intervals + 1), mse, width=0.5, color='bgrk', linewidth=0.4)
    plt.xlabel("intervals")
    plt.ylabel("mse")
    plt.show()


def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
    """
    Generate a simple plot of the test and training learning curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
          - None, to use the default 3-fold cross-validation,
          - integer, to specify the number of folds.
          - :term:`CV splitter`,
          - An iterable yielding (train, test) splits as arrays of indices.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : int or None, optional (default=None)
        Number of jobs to run in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    train_sizes : array-like, shape (n_ticks,), dtype float or int
        Relative or absolute numbers of training examples that will be used to
        generate the learning curve. If the dtype is float, it is regarded as a
        fraction of the maximum size of the training set (that is determined
        by the selected validation method), i.e. it has to be within (0, 1].
        Otherwise it is interpreted as absolute sizes of the training sets.
        Note that for classification the number of samples usually have to
        be big enough to contain at least one sample from each class.
        (default: np.linspace(0.1, 1.0, 5))
    """
    plt.figure(figsize=(6, 4.8), dpi=300)
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt


def PlotRelativeCurves(y_true, y_pre):
    """

    :param y_true:  shape (n_samples, )
    :param y_pre:   shape (n_samples, )
    :return: plt
    """

    axis_start = (np.array([y_true.min(), y_pre.min()])).min()
    axis_end = (np.array([y_true.max(), y_pre.max()])).max()

    plt.figure(figsize=(5.5, 4), dpi=300)
    plt.scatter(y_true, y_pre, color='black', s=7)
    plt.plot([axis_start, axis_end], [axis_start, axis_end], color='blue', linewidth=1.5)

    fonts = 8
    # plt.xlim(350, 2550)
    # plt.ylim(0, 1)
    plt.xlabel('true target', fontsize=fonts)
    plt.ylabel('predicted target', fontsize=fonts)
    plt.yticks(fontsize=fonts)
    plt.xticks(fontsize=fonts)


    plt.tight_layout(pad=0.5)
    plt.grid()

    return plt


def PlotResidual(y_true, y_pre):
    """

    :param y_true: shape (n_samples, )
    :param y_pre: shape (n_samples, )
    :return: plt
    """

    residual = np.abs(y_true - y_pre)
    plt.figure(figsize=(4.6, 2.8), dpi=300)
    markerline, stemlines, baseline = plt.stem(np.arange(1, 33), residual)
    plt.setp(markerline, markersize=3, color='red')
    plt.setp(stemlines, linewidth=1, color='black')
    plt.setp(baseline, color='white', linewidth=1)


    fonts = 8
    # plt.xlim(-30, 1030)
    plt.ylim(0, 1.5)
    plt.xlabel("Samples", fontsize=fonts)
    plt.ylabel("Residual (%)", fontsize=fonts)
    plt.yticks(fontsize=fonts)
    plt.xticks(fontsize=fonts)

    # plt.legend()
    plt.tight_layout(pad=0.3)
    plt.grid(True)
    return plt


def evaluating(y_true, y_pre, samplesets):
    """

    :param y_true: (n_samples, )
    :param y_pre: (n_samples, )
    :samplesets: string
    :return: None
    """
    evs_ = explained_variance_score(y_true, y_pre)
    mae_ = mean_absolute_error(y_true, y_pre)
    mse_ = mean_squared_error(y_true, y_pre)
    r2_ = r2_score(y_true, y_pre)
    rmse_ = np.sqrt(mse_)
    rpd_ = np.std(y_true)/rmse_

    print("*"*100)
    print(samplesets + ' 解释方差得分  平均绝对误差  决定系数  均方误差  均方根误差  相对分析误差')
    print('结果     %6.4f       %6.4f    %6.4f   %6.4f   %6.4f        %6.4f' % (evs_, mae_, r2_, mse_, rmse_, rpd_))
    print("*"*100)


def ccm_plot(ccm):
    """

    :param ccm: correlation coefficient values  (n_samples, )
    :return: plt
    """
    var = len(ccm)-1
    base = 400
    plt.figure(figsize=(5, 3.6), dpi=300)
    plt.plot(np.arange(base, base + 2 * var, 2), ccm[:var], color='r', linewidth=0.8)

    fontsize = 8
    plt.xlabel('attribute', fontsize=fontsize)
    plt.ylabel('coefficient value', fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)

    plt.legend(['correlation coefficient'], loc='upper left', fontsize=fontsize)
    plt.tight_layout(pad=0.3)
    plt.grid()
    return plt


def featureWavebandLabel(spec, index_label):
    """

    :param spec: the spectra to be showed (n_samples, n_features)
    :param index_label: (n_labels, )
    :return: plt
    """
    base = 400
    label_wavelengths = base + 2*index_label
    mean_spec = np.mean(spec, axis=0)
    plt.figure(figsize=(5.2, 3.1), dpi=300)
    x = np.arange(base, base + 2 * spec.shape[1], 2)
    for i in range(spec.shape[0]):
        plt.plot(x, spec[i, :], linewidth=0.6, color='blue')

    for j in range(len(index_label)):
        xx = label_wavelengths[j]
        yy = mean_spec[index_label[j]]
        plt.plot([xx, xx], [yy+0.07, yy-0.07], linewidth=0.6, color='red')
    fonts = 8
    plt.xlim(350, 2550)
    plt.ylim(0, 1)
    plt.xlabel('Wavelength (nm)', fontsize=fonts)
    plt.ylabel('absorbance (AU)', fontsize=fonts)
    plt.yticks(fontsize=fonts)
    plt.xticks(fontsize=fonts)

    # plt.legend()
    plt.tight_layout(pad=0.3)
    plt.grid(True)
    return plt


# file = pd.read_excel('NDFNDF.xlsx')
# datas = file.values
# spec = datas[:, 1:1051]
# target = datas[:, -1]
#
# fig, ax = plt.subplots(figsize=(5.2, 3.1), dpi=300)
# x = np.arange(400, 400 + 2*spec.shape[1], 2)
# for i in range(spec.shape[0]):
#     ax.plot(x, spec[i, :], linewidth=0.6)
# fonts = 8
#
# ax.set_xlim(350, 2550)
# ax.set_ylim(0, 1)
# ax.set_xlabel('Wavelength (nm)', fontsize=fonts)
# ax.set_ylabel('absorbance (AU)', fontsize=fonts)
# xticklabels = np.arange(500, 3000, 500)
# yticklabels = np.arange(0, 1.2, 0.2)
# ax.set_xticks(xticklabels)
# ax.set_yticks(yticklabels)
# ax.set_xticklabels(labels=[str(round(xx, 1)) for xx in xticklabels], fontdict={'fontsize': fonts})
# ax.set_yticklabels(labels=[str(round(xx, 1)) for xx in yticklabels], fontdict={'fontsize': fonts})
# ax.grid(True)
#
# fig.tight_layout(pad=0.3)
# fig.savefig("./2.tiff")
# plt.show()


import matplotlib.pyplot as plt


# 绘制光谱图像
def plotHyperspectralImage(wavelength, featrues, title):
    for i in range(featrues.shape[1]):
        plt.plot(wavelength, featrues[i], linewidth=0.5)

    # 解决绘图中文不显示
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    plt.title(title, fontsize=14)
    plt.xlabel("Wavelength", fontsize=14)
    plt.ylabel("Reflectance", fontsize=14)
    plt.show()

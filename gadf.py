import numpy as np
import pandas as pd
from scipy import signal
from pyts.image import GramianAngularField
import matplotlib.pyplot as plt
import os

import matplotlib
matplotlib.use("TkAgg")

#数据集与标签的读取
x = pd.read_csv('spectra10.csv',header=None).values

# 标准正态化
def stand(data):
    x = data
    row_means = np.mean(x, axis=1)  # 每行的平均值
    row_stds = np.std(x, axis=1)  # 每行的标准差
    data_snv = (x - row_means[:, np.newaxis]) / row_stds[:, np.newaxis]
    return data_snv

# 数据标准化，平滑滤波
x_stand = stand(x)
x_stand_sg = signal.savgol_filter(x_stand, 13, 3)


# 生成角差场，角和场, n生成图像的大小
n = 128
gd = GramianAngularField(image_size=n, method='d')
gs = GramianAngularField(image_size=n, method='s')
gd_image = gd.fit_transform(x_stand_sg)
gs_image = gs.fit_transform(x_stand_sg)

# 显示第一个图像
plt.imshow(gd_image[0], cmap='viridis')
plt.axis('off')
plt.savefig('gd0',bbox_inches = 'tight',pad_inches=0)

plt.imshow(gs_image[0], cmap='viridis')
plt.axis('off')
plt.savefig('gs0',bbox_inches = 'tight',pad_inches=0)

# 创建文件夹保存图片
gd_name = "gd_images"
gs_name = "gs_images"
os.makedirs(gd_name, exist_ok=True)
os.makedirs(gs_name, exist_ok=True)
for i in range(len(gd_image)):
    plt.imsave(os.path.join(gd_name,f'image_{i+1}.png'), gd_image[i], cmap='viridis')
for i in range(len(gs_image)):
    plt.imsave(os.path.join(gs_name,f'image_{i+1}.png'), gs_image[i], cmap='viridis')

print("图像已保存。")
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')
import numpy as np
x = ['K100', 'K200', 'K300', 'K400']

UCF101 = [81.9, 87.1, 87.6, 89.6]
Diving48 = [42.3, 41.4, 42.2, 42.7]
HMDB51 = [54.9, 58.2, 62.4, 63.4]
RareAct = [53.3, 59.4, 58.5, 60.0]
SSV2 = [32.1, 33.0, 34.3, 34.8]

fig = plt.figure(figsize=(6,6), dpi=200)
plt.title('Source Domain Categories \n to Target Domain Accuracy  ')  # 折线图标题

plt.gca().set_aspect(0.055)
plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示汉字
plt.xlabel('Source Domain Category')  # x轴标题
plt.ylabel('Target Domain Accuracy(%)')  # y轴标题
x=[1,2,3,4]
plt.xticks([1,2,3,4], ['K100', 'K200', 'K300', 'K400'])

plt.plot(x, UCF101, marker='o', markerfacecolor='black' , color='green', markersize=3)  # 绘制折线图，添加数据点，设置点的大小
plt.plot(x, Diving48, marker='o', markerfacecolor='black' , color='blue', markersize=3)
plt.plot(x, HMDB51, marker='o', markerfacecolor='black',  color='red', markersize=3)
plt.plot(x, RareAct, marker='o', markerfacecolor='black', color='slategray', markersize=3)
plt.plot(x, SSV2, marker='o', markerfacecolor='black', color='orange', markersize=3)
'''
for a, b in zip(x, y_trx):
    plt.text(a, b, b, ha='center', va='bottom', fontsize=10)  # 设置数据标签位置及大小
for a, b in zip(x, y_strm):
    plt.text(a, b, b, ha='center', va='bottom', fontsize=10)
    '''
'''
for a, b in zip(x, UCF101):
    plt.text(a, b, b, ha='center', va='bottom', fontsize=10)
'''

plt.legend(['UCF101', 'Diving48', 'HMDB51', 'RareAct', 'SSV2'],loc='upper left')  # 设置折线名称

plt.grid(linestyle='-.')
#plt.gcf().set_facecolor(np.ones(3)* 240 / 255)   # 生成画布的大小
#plt.savefig('/home/guofei/few_shot_accuracy-ssv2.jpg', dpi=300)
plt.savefig('./sdNumber2tdAccuracy.jpg', dpi=300)
plt.show()  # 显示折线图

#plt.xlim(a, b)
import matplotlib.pyplot as plt
import numpy as np
x = ['K100', 'K200', 'K300', 'K400']

UCF101 = [81.9, 89.8, 93.54, 95.18]
Diving48 = [42.3, 43.4, 49.8, 53.9]
HMDB51 = [54.9, 59.3, 64.5, 69.0]
RareAct = [54.2, 65.2, 63.9, 70.3]
SSV2 = [32.1, 31.2, 38.83, 39.3]

fig = plt.figure(figsize=(6,6), dpi=300)
plt.title('Source Domain Number to Target Domain few-shot accuracy  ')  # 折线图标题


plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示汉字
plt.xlabel('Source Domain Class Number')  # x轴标题
plt.ylabel('Target Domain Accuracy(%)')  # y轴标题
plt.plot(x, UCF101, marker='o', markerfacecolor='white' , color='green', markersize=3)  # 绘制折线图，添加数据点，设置点的大小
plt.plot(x, Diving48, marker='o', markerfacecolor='white' , color='blue', markersize=3)
plt.plot(x, HMDB51, marker='o', markerfacecolor='white',  color='red', markersize=3)
plt.plot(x, RareAct, marker='o', markerfacecolor='white', color='slategray', markersize=3)
plt.plot(x, SSV2, marker='o', markerfacecolor='white', color='orange', markersize=3)
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

plt.legend(['UCF101', 'Diving48', 'TSA-HMDB51', 'RareAct', 'SSV2'])  # 设置折线名称

plt.grid(linestyle='-.')
#plt.gcf().set_facecolor(np.ones(3)* 240 / 255)   # 生成画布的大小
#plt.savefig('/home/guofei/few_shot_accuracy-ssv2.jpg', dpi=300)
plt.savefig('./sdNumber2tdAccuracy.jpg', dpi=300)
plt.show()  # 显示折线图

#plt.xlim(a, b)
# coding:gbk

import matplotlib.pyplot as plt

# 模型和参数数据
models = ['LSTM', 'GRU', 'Ref.[15]', 'Ref.[21]',
          'Ref.[22]', 'Ref.[26]', 'Ref.[27]', 'Ref.[28]', 'Ref.[29]', 'NTPGR']
gflpos = [8.30, 12.70, 5.50, 2.43, 2.89, 1.26, 1.07, 1.33, 0.96, 0.78]

# 虚构的精度数据（0到1之间的值）
accuracy_data = [83.15, 81.30, 91.13, 87.22, 89.74, 94.12, 95.32, 93.87, 96.51, 97.56]

# 使用'o'表示圆形点
plt.scatter(gflpos[0], accuracy_data[0], label=models[0], marker='o', s=100)
plt.scatter(gflpos[1], accuracy_data[1], label=models[1], marker='*', s=100)
plt.scatter(gflpos[2], accuracy_data[2], label=models[2], marker='v', s=100)
plt.scatter(gflpos[3], accuracy_data[3], label=models[3], marker='8', s=100)
plt.scatter(gflpos[4], accuracy_data[4], label=models[4], marker='D', s=100)
plt.scatter(gflpos[5], accuracy_data[5], label=models[5], marker='H', s=100)
plt.scatter(gflpos[6], accuracy_data[6], label=models[6], marker='d', s=100)
plt.scatter(gflpos[7], accuracy_data[7], label=models[7], marker='h', s=100)
plt.scatter(gflpos[8], accuracy_data[8], label=models[8], marker='s', s=100)
plt.scatter(gflpos[9], accuracy_data[9], label=models[9], marker='P', s=100)

# 添加标签和标题
plt.xlabel('ART/s', fontsize=14)
plt.ylabel('Accuracy/%', fontsize=14)
plt.title('Model Accuracy And ART', fontsize=14)

# 设置X轴和Y轴的刻度值
plt.yticks([80.0, 82, 84, 86, 88, 90, 92, 94, 96, 98])  # 设置Y轴刻度为指定的值

# 添加图例，指定位置为右下角
plt.legend(loc='upper right')

# 启用网格
plt.grid(True)

# 保存图表为图片文件（可选择的格式包括PNG、JPEG、SVG等）
plt.savefig('model_accuracy_plot.jpg', dpi=400)  # 文件名为'model_accuracy_plot.png'，dpi参数设置图像分辨率

# 显示图表
plt.show()


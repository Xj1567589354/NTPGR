# coding:gbk

import matplotlib.pyplot as plt

# 模型和参数数据
models = ['HRNet-W32', 'MobileNetV2', 'ShuffleNetV2', 'Small HRNet',
          'Lite-HRNet-30', 'Dite-HRNet-30', 'EPFFNet']
gflpos = [7.10, 1.5, 1.3, 0.5, 0.3, 0.3, 5.2]

# 虚构的精度数据（0到1之间的值）
accuracy_data = [73.4, 64.6, 59.9, 55.2, 67.2, 68.3, 75.1]

# 使用'o'表示圆形点
plt.scatter(gflpos[0], accuracy_data[0], label=models[0], marker='o', s=100)
plt.scatter(gflpos[1], accuracy_data[1], label=models[1], marker='*', s=100)
plt.scatter(gflpos[2], accuracy_data[2], label=models[2], marker='v', s=100)
plt.scatter(gflpos[3], accuracy_data[3], label=models[3], marker='8', s=100)
plt.scatter(gflpos[4], accuracy_data[4], label=models[4], marker='D', s=100)
plt.scatter(gflpos[5], accuracy_data[5], label=models[5], marker='H', s=100)
plt.scatter(gflpos[6], accuracy_data[6], label=models[6], marker='d', s=100)

# 添加标签和标题
plt.xlabel('#Param/M', fontsize=14)
plt.ylabel('mAP/%', fontsize=14)
plt.title('Model mAP vs GFLOPs on COCO', fontsize=14)

# 设置X轴和Y轴的刻度值
plt.yticks([52.5, 55, 57.5, 60, 62.5, 65, 67.5, 70, 72.5, 75, 77.5])  # 设置Y轴刻度为指定的值

plt.tick_params(axis='both', direction='in')

# 添加图例，指定位置为右下角
plt.legend(loc='lower right')

# 启用网格
plt.grid(True)

# 保存图表为图片文件（可选择的格式包括PNG、JPEG、SVG等）
plt.savefig('model_map_plot.wmf', dpi=400)  # 文件名为'model_accuracy_plot.png'，dpi参数设置图像分辨率

# 显示图表
plt.show()


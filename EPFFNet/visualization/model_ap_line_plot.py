import matplotlib.pyplot as plt

# 模型和参数数据
models = ['HRNet-DStage', 'HRNet-TStage', 'HRNet']
parameters = [1.2, 7.8, 28.5]  # 假设这是模型的参数数量

colors = [(0.9804, 0.7804, 0.5804), (0.9608, 0.5569, 0.5843),
          (0.6706, 0.5882, 0.8745)]

# 虚构的精度数据（0到1之间的值）
accuracy_data = [43.6, 72.0, 73.4]

# 创建柱状图
bars = plt.bar(parameters, accuracy_data, width=3, align='center')

# 创建折线图
plt.plot(parameters, accuracy_data, marker='o', linestyle='-', color='black', label='Model Accuracy')

# 添加标签和标题，并设置字体大小
plt.xlabel('#Params', fontsize=14)
plt.ylabel('mAP(COCO Val)', fontsize=14)
plt.title('Model Parameters And Accuracy', fontsize=14)

# 设置X轴和Y轴的刻度值
plt.xticks([1.0, 10.0, 20.0, 30.0])  # 设置Y轴刻度为指定的值

for bar, color in zip(bars, colors):
    bar.set_color(color)

# 设置y轴范围和标签
plt.ylim(40, 80)
plt.yticks([i for i in {40, 50, 60, 70, 80}])

# 启用网格
plt.grid(True)

# 添加数据标签
plt.annotate(models[0], (parameters[0], accuracy_data[0]), textcoords="offset points", xytext=(40, 0), ha='center', fontsize=12)
plt.annotate(models[1], (parameters[1], accuracy_data[1]), textcoords="offset points", xytext=(0, 5), ha='center', fontsize=12)
plt.annotate(models[2], (parameters[2], accuracy_data[2]), textcoords="offset points", xytext=(0, 5), ha='center', fontsize=12)

# 保存图表为图片文件（可选择的格式包括PNG、JPEG、SVG等）
plt.savefig('model_ap_line_plot.jpg', dpi=400)  # 文件名为'model_accuracy_line_plot.png'，dpi参数设置图像分辨率

# 显示图表
plt.show()

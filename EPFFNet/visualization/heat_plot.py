import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# 1. 定义元素和手势
elements = ['Vehicles', 'Light', 'Pedestrian', 'Trees', 'Railings']
gestures = ['Stop', 'Move Straight', 'Turn Left', 'Turn Right', 'Lane Change', 'Slow Down', 'Pull Over']

# 2. 指定准确率的范围
lower_bound = 0.75  # 最小值
upper_bound = 0.85  # 最大值

# 3. 构建准确率矩阵（指定范围内的随机数据）
accuracy_matrix = np.random.uniform(lower_bound, upper_bound, (len(gestures), len(elements)))

# 3. 绘制热力图
plt.figure(figsize=(12, 8))
ax = sns.heatmap(accuracy_matrix, annot=True, fmt=".2f", cmap='YlGnBu',
                 xticklabels=elements, yticklabels=gestures)

plt.yticks(fontsize=14, rotation=45)  # 将 y 轴标签倾斜 30 度
plt.xticks(fontsize=14)  # 设置 x 轴标签的字体大小

# 4. 添加标题和标签
plt.title('Impact Of Complex Factors On Accuracy', fontsize=18)
# plt.savefig('heat_plot_16.png', dpi=400)

plt.show()

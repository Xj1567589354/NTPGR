import matplotlib.pyplot as plt

# 数据
categories = ['Stop', 'Move Straght', 'Turn Left', 'Turn Left to Stay', 'Turn Right', 'Lane Change', 'Slow Down', 'Pull Over']
accuracy = [98.78, 97.56, 97.56, 95.22, 97.59, 98.78, 97.22, 95.22]
# 为每个类别指定不同的颜色
colors = ['#92A5D1', '#C5DFF4', '#AEB2D1', '#D9B9D4', '#7C9895', '#C9DCC4', '#DAA87C', '#F4EEAC']

# 创建横向条状图
plt.figure(figsize=(10, 6))
plt.barh(categories, accuracy, color=colors)
plt.xlabel('Accuracy/%', fontsize=14)
plt.title('Accuracy Of Different Gestures', fontsize=14)

# 旋转类别标签
plt.xticks(fontsize=14)
plt.yticks(rotation=45, fontsize=14)  # 将类别标签设置为斜着排列

plt.xlim(85, 100)

# 显示图表
plt.tight_layout()  # 确保布局不重叠

# 启用网格
plt.grid(True)

plt.savefig('hand_accuracy.png', dpi=400)
# 显示图表
plt.show()
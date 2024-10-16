import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)  # 设置随机种子

# 手势数据的准确率范围（每个手势模拟10个随机数据）
hand_signal_data = {
    'Stop': np.random.uniform(96.56, 98.76, 10),
    'Move Straght': np.random.uniform(95.22, 97.56, 10),
    'Turn Left': np.random.uniform(95.22, 97.56, 10),
    'Turn Left to Stay': np.random.uniform(95.06, 97.76, 10),
    'Turn Right': np.random.uniform(95.56, 97.59, 10),
    'Lane Change': np.random.uniform(96.56, 98.76, 10),
    'Slow Down': np.random.uniform(97.73, 99.36, 10),
    'Pull Over': np.random.uniform(97.50, 100.00, 10),
}

# 设置颜色
colors = ['#92A5D1', '#C5DFF4', '#E8BE74', '#D9B9D4', '#7C9895', '#C9DCC4', '#DAA87C', '#F4EEAC']

# 将数据转换为列表形式
data_to_plot = [hand_signal_data[key] for key in hand_signal_data.keys()]

# 绘制箱线图
fig, ax = plt.subplots(figsize=(8, 6))
box_width = 0.6  # 设置箱子的宽度

# 为每种手势设置不同颜色
for i, (key, values) in enumerate(hand_signal_data.items()):
    box = ax.boxplot(values,
                     positions=[i],  # 确保每个箱线图在不同的位置
                     widths=box_width,  # 设置箱子的宽度
                     patch_artist=True,
                     boxprops=dict(facecolor=colors[i], color='black'),
                     flierprops=dict(marker='o', color='red', markersize=5),
                     medianprops=dict(color='#8B0000'),
                     whiskerprops=dict(color='black'))

# 设置图表标题和标签
ax.set_title('Accuracy Of Different Gestures', fontsize=14)
ax.set_ylabel('Accuracy/%', fontsize=14)
ax.set_xticklabels(hand_signal_data.keys(), rotation=45, ha='right', fontsize=14)

plt.yticks([95, 96, 97, 98, 99, 100])  # 设置Y轴刻度为指定的值

# # 启用网格
# plt.grid(True)

# 显示箱线图
plt.tight_layout()
# plt.savefig('hand_accuracy_box.png', dpi=400)
plt.show()

import matplotlib.pyplot as plt

# 数据
objects = ['Model B', 'SE', 'ECA', 'MECA', 'CA', 'SCA']
values = [73.8, 74.15, 74.57, 75, 75.81, 76.15]
# colors = [(85, 31, 51), (203, 187, 193), (189, 65, 70), (228, 183, 188), (236, 198, 140), (275, 228, 200)]
colors = [(0.34, 0.1215, 0.2), (0.796, 0.734, 0.7568), (0.741, 0.2548, 0.2745), (0.8941, 0.7176, 0.7373),
          (0.9255, 0.7765, 0.5490), (0.9608, 0.8941, 0.7843)]

values_add = ['73.8', '+0.35', '+0.765', '+1.2', '+0.81', '+1.15']

# 创建柱状图
bars = plt.bar(objects, values, width=0.55, align='center')

# 在每个柱体上方标注数字
for i, value in enumerate(values):
    plt.text(i, value + 0.1, str(values_add[i]), ha='center', va='bottom')

for bar, color in zip(bars, colors):
    bar.set_color(color)

# # 设置每个柱体的边框为黑色
# for bar in bars:
#     bar.set_edgecolor((0.2, 0.2, 0.2))

# 设置y轴范围和标签
plt.ylim(73, 76.5)
plt.yticks([i for i in {73.5, 74, 74.5, 75, 75.5, 76, 76.5}])

# 添加标题和标签
plt.title('COCO Human Pose Estimation', fontsize=14)

# 移除右边和顶部的边框线
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)

# 保存图表为图片文件（可选择的格式包括PNG、JPEG、SVG等）
plt.savefig('bar_plot.jpg', dpi=400)

# 显示图形
plt.show()
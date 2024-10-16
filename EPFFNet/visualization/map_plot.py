# coding:gbk

import matplotlib.pyplot as plt

# ģ�ͺͲ�������
models = ['HRNet-W32', 'MobileNetV2', 'ShuffleNetV2', 'Small HRNet',
          'Lite-HRNet-30', 'Dite-HRNet-30', 'EPFFNet']
gflpos = [7.10, 1.5, 1.3, 0.5, 0.3, 0.3, 5.2]

# �鹹�ľ������ݣ�0��1֮���ֵ��
accuracy_data = [73.4, 64.6, 59.9, 55.2, 67.2, 68.3, 75.1]

# ʹ��'o'��ʾԲ�ε�
plt.scatter(gflpos[0], accuracy_data[0], label=models[0], marker='o', s=100)
plt.scatter(gflpos[1], accuracy_data[1], label=models[1], marker='*', s=100)
plt.scatter(gflpos[2], accuracy_data[2], label=models[2], marker='v', s=100)
plt.scatter(gflpos[3], accuracy_data[3], label=models[3], marker='8', s=100)
plt.scatter(gflpos[4], accuracy_data[4], label=models[4], marker='D', s=100)
plt.scatter(gflpos[5], accuracy_data[5], label=models[5], marker='H', s=100)
plt.scatter(gflpos[6], accuracy_data[6], label=models[6], marker='d', s=100)

# ��ӱ�ǩ�ͱ���
plt.xlabel('#Param/M', fontsize=14)
plt.ylabel('mAP/%', fontsize=14)
plt.title('Model mAP vs GFLOPs on COCO', fontsize=14)

# ����X���Y��Ŀ̶�ֵ
plt.yticks([52.5, 55, 57.5, 60, 62.5, 65, 67.5, 70, 72.5, 75, 77.5])  # ����Y��̶�Ϊָ����ֵ

plt.tick_params(axis='both', direction='in')

# ���ͼ����ָ��λ��Ϊ���½�
plt.legend(loc='lower right')

# ��������
plt.grid(True)

# ����ͼ��ΪͼƬ�ļ�����ѡ��ĸ�ʽ����PNG��JPEG��SVG�ȣ�
plt.savefig('model_map_plot.wmf', dpi=400)  # �ļ���Ϊ'model_accuracy_plot.png'��dpi��������ͼ��ֱ���

# ��ʾͼ��
plt.show()


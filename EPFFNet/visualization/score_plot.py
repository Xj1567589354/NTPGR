# coding:gbk

import matplotlib.pyplot as plt

# ģ�ͺͲ�������
models = ['LSTM', 'GRU', 'Ref.[15]', 'Ref.[21]',
          'Ref.[22]', 'Ref.[26]', 'Ref.[27]', 'Ref.[28]', 'Ref.[29]', 'NTPGR']
gflpos = [8.30, 12.70, 5.50, 2.43, 2.89, 1.26, 1.07, 1.33, 0.96, 0.78]

# �鹹�ľ������ݣ�0��1֮���ֵ��
accuracy_data = [83.15, 81.30, 91.13, 87.22, 89.74, 94.12, 95.32, 93.87, 96.51, 97.56]

# ʹ��'o'��ʾԲ�ε�
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

# ��ӱ�ǩ�ͱ���
plt.xlabel('ART/s', fontsize=14)
plt.ylabel('Accuracy/%', fontsize=14)
plt.title('Model Accuracy And ART', fontsize=14)

# ����X���Y��Ŀ̶�ֵ
plt.yticks([80.0, 82, 84, 86, 88, 90, 92, 94, 96, 98])  # ����Y��̶�Ϊָ����ֵ

# ���ͼ����ָ��λ��Ϊ���½�
plt.legend(loc='upper right')

# ��������
plt.grid(True)

# ����ͼ��ΪͼƬ�ļ�����ѡ��ĸ�ʽ����PNG��JPEG��SVG�ȣ�
plt.savefig('model_accuracy_plot.jpg', dpi=400)  # �ļ���Ϊ'model_accuracy_plot.png'��dpi��������ͼ��ֱ���

# ��ʾͼ��
plt.show()


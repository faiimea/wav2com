import numpy as np
import os
from scipy.io.wavfile import read
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from generate_array_feature import mald_feature, get_filelist

# ...[省略其他代码和import语句]...

input_path = '/home/fazhong/Github/czx2/example/data4'
features = []
labels = []
for root, dirs, files in os.walk(input_path):
    for file in files:
        file_path = os.path.join(root, file)
        rate, wavdata = read(file_path)
        
        features.append(list(mald_feature(rate, wavdata)))
        if 'feng' in file:
            labels.append(0)
        elif 'meng' in file:
            labels.append(1)
        elif 'jc' in file:
            labels.append(2)
        else:
            labels.append(3)
        
features = np.array(features)
labels = np.array(labels)

# 创建t-SNE模型
tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)

# 执行t-SNE降维
tsne_results = tsne.fit_transform(features)

# 绘制t-SNE结果并保存图像，不包含点的标识，并增加图例
def plot_with_labels(low_dim_embs, labels, filename='/home/fazhong/Github/czx/tsne_name1.pdf'):  # 修改为保存为PDF
    assert low_dim_embs.shape[0] >= len(labels), "More labels than embeddings"
    plt.figure(figsize=(14, 14))  # in inches
    label_names = ['User1', 'User2', 'User3', 'User4']  # 标签名称
    colors = ['blue', 'green', 'red', 'purple']  # 各标签对应颜色
    for i, color in enumerate(colors):
        idx = labels == i
        plt.scatter(low_dim_embs[idx, 0], low_dim_embs[idx, 1], c=color, label=label_names[i])
    
    # 增强坐标轴标签的可读性
    plt.xlabel('Component 1', fontsize=24)
    plt.ylabel('Component 2', fontsize=24)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    #plt.grid(True)  # 添加网格线
    
    plt.legend(prop={'size': 25}, loc=1)  # 添加图例
    plt.savefig(filename, format='pdf')  # 指定保存为PDF
    plt.close()  # 关闭图形，避免在Notebook中显示
    print(f"t-SNE plot with labels saved as {filename}")

# 调用plot_with_labels函数来绘制散点图并保存，不包含点的标识，并增加图例
plot_with_labels(tsne_results, labels)
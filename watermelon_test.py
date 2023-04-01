import numpy as np
from decisiontree import DecisionTree, right_test
import matplotlib.pyplot as plt
from PIL import Image
import os


if __name__ == '__main__':
    # 读取数据集
    data = np.loadtxt('./data/watermelon.data',
                      dtype=str, delimiter=',',
                      usecols=(0, 1, 2, 3, 4, 5, 6, 7),
                      encoding='utf-8')
    # 读取标签
    labels = np.loadtxt('./data/watermelon.data',
                        dtype=str, delimiter=',',
                        usecols=8, encoding='utf-8')

    # 数据集属性名称
    ft = ['色泽', '根蒂', '敲声', '纹理', '脐部', '触感', '密度', '含糖率']

    # 数据集前6个属性为离散属性，后两个属性为连续属性
    ft_at = ["d", "d", "d", "d", "d", "d", "c", "c"]

    # 采样70%的数据位训练数据集，30%的数据位测试数据集
    np.random.seed(1)
    sample = np.random.rand(labels.size)
    train_data = data[sample < 0.7]
    train_labels = labels[sample < 0.7]
    test_data = data[sample >= 0.7]
    test_labels = labels[sample >= 0.7]

    # 生成决策树
    dt = DecisionTree(ft, ft_at)

    # 训练决策树
    dt.fit(train_data, train_labels)

    # 生成并显示决策树图片
    dt.tree.grap_dot('./dot/watermelon_test.dot')
    os.system("dot ./dot/watermelon_test.dot -T png -o ./figure/watermelon_test.png")
    img = Image.open("./figure/watermelon_test.png")
    img.show()

    # 查看决策树深度与预测正确率的关系
    right_list = []
    # 决策树深度为2~10
    depth_list = list(range(1, 10))
    for d in depth_list:
        dt = DecisionTree(ft, ft_at, max_depth=d)
        dt.fit(train_data, train_labels)
        right_list.append(right_test(dt.tree, test_data, test_labels))

    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.xlabel('决策树深度')
    plt.ylabel('正确率')
    plt.plot(depth_list, right_list)
    plt.show()

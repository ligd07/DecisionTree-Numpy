import numpy as np
from decisiontree import DecisionTree, right_test
import matplotlib.pyplot as plt
from PIL import Image
import os


if __name__ == '__main__':
    # 读取数据集
    data = np.loadtxt('./data/iris.data',
                      delimiter=',',
                      usecols=(0, 1),
                      encoding='utf-8')
    # 读取标签
    labels = np.loadtxt('./data/iris.data',
                        dtype=str, delimiter=',',
                        usecols=4, encoding='utf-8')

    # 数据集属性名称
    ft = ['花萼长度', '花萼宽度']

    # 数据集所有属性均为连续属性
    ft_at = ["c", "c"]

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
    dt.tree.grap_dot('./dot/iris_test1.dot')
    os.system("dot ./dot/iris_test1.dot -T png -o ./figure/ris_test1.png")
    img = Image.open("./figure/ris_test1.png")
    img.show()

    # 生成网络坐标
    test_x, test_y = np.meshgrid(np.arange(data[:, 0].min(), data[:, 0].max(), 0.01),
                                 np.arange(data[:, 1].min(), data[:, 1].max(), 0.01))
    test_xy = np.array([test_x.ravel(), test_y.ravel()]).T

    test_z = []
    # 计算所有坐标处的值
    for i in range(test_xy.shape[0]):
        z = dt.tree.predict(test_xy[i])
        test_z.append(z)

    test_z = np.array(test_z)

    # 绘制坐标网络
    plt.figure(1)
    plt.scatter(test_xy[test_z == 'Iris-setosa', 0],
                test_xy[test_z == 'Iris-setosa', 1],
                c='y')
    plt.scatter(test_xy[test_z == 'Iris-versicolor', 0],
                test_xy[test_z == 'Iris-versicolor', 1],
                c='m')
    plt.scatter(test_xy[test_z == 'Iris-virginica', 0],
                test_xy[test_z == 'Iris-virginica', 1],
                c='c')

    # 绘制训练数据集
    plt.scatter(train_data[train_labels == 'Iris-setosa', 0],
                train_data[train_labels == 'Iris-setosa', 1],
                c='r')
    plt.scatter(train_data[train_labels == 'Iris-versicolor', 0],
                train_data[train_labels == 'Iris-versicolor', 1],
                c='g')
    plt.scatter(train_data[train_labels == 'Iris-virginica', 0],
                train_data[train_labels == 'Iris-virginica', 1],
                c='b')

    # 绘制测试数据集
    plt.scatter(test_data[test_labels == 'Iris-setosa', 0],
                test_data[test_labels == 'Iris-setosa', 1],
                c='r', marker='*')
    plt.scatter(test_data[test_labels == 'Iris-versicolor', 0],
                test_data[test_labels == 'Iris-versicolor', 1],
                c='g', marker='*')
    plt.scatter(test_data[test_labels == 'Iris-virginica', 0],
                test_data[test_labels == 'Iris-virginica', 1],
                c='b', marker='*')

    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.title('测试正确率'+str(round(right_test(dt.tree, test_data, test_labels), 3)*100)+'%')
    plt.xlabel('花萼长度')
    plt.ylabel('花萼宽度')

    # 查看决策树深度与预测正确率的关系
    right_list = []
    # 决策树深度为2~10
    depth_list = list(range(1, 10))
    for d in depth_list:
        dt = DecisionTree(ft, ft_at, max_depth=d, criterion='gini')
        dt.fit(train_data, train_labels)
        right_list.append(right_test(dt.tree, test_data, test_labels))

    plt.figure(2)
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.xlabel('决策树深度')
    plt.ylabel('正确率')
    plt.plot(depth_list, right_list)
    plt.show()


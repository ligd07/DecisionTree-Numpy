import numpy as np
from tree import Tree


def entropy(y):
    """
        信息熵计算函数

    Parameters
    ----------
    y:numpy.ndarray
        输入数据，1维数组

    Returns
    -------
    ent:float
        输入数据的信息熵
    """
    class_list = np.unique(y, return_counts=True)
    class_count = class_list[0].size

    ent = 0
    for k in range(class_count):
        pk = class_list[1][k] / len(y)
        ent += -pk * np.log(pk)
    return ent


def gini(y):
    """
        基尼系数计算函数

    Parameters
    ----------
    y:numpy.ndarray
        输入数据，1维数组

    Returns
    -------
    g:float
        输入数据的基尼系数
    """
    class_list = np.unique(y, return_counts=True)
    class_count = class_list[0].size

    g = 1
    for k in range(class_count):
        pk = class_list[1][k] / len(y)
        g -= pk ** 2
    return g


def get_most_class(y):
    """
        给出输入数据中数量最多的数据

    Parameters
    ----------
    y:numpy.ndarray
        输入数据，1维数组

    Returns
    -------
    float|str
        输入数据中数量最多的数据

    """
    class_list = np.unique(y, return_counts=True)
    max_labels = class_list[1].argmax()
    return class_list[0][max_labels]


def right_test(tree, x, y):
    """
        使用输入的决策树tree判断给定数据x的种类，并与给定标签y对比，返回判断的正确率

    Parameters
    ----------
    tree:Tree
        决策树
    x:numpy.ndarray
        数据
    y:numpy.ndarray
        数据标签
    Returns
    -------
    right_rate:float
        预测正确率
    """
    right_num = 0
    for i, j in zip(x, y):
        if tree.predict(i) == j:
            right_num += 1
    right_rate = right_num / y.size

    return right_rate


class DecisionTree:
    def __init__(self, feature_list, feature_attributes, criterion="entropy", max_depth=5):
        """
            初始化决策树

        Parameters
        ----------
        feature_list:list
            数据集属性名称列表
        feature_attributes:list
            数据集属性连续性列表，连续为'c'，离散为'd'
        criterion:str
            entropy：使用信息熵
            gini：基尼系数
        max_depth:int
            决策树最大深度，决策树深度从0开始
        """
        self.max_depth = max_depth
        self.feature_attributes = feature_attributes
        self.feature = feature_list
        if criterion == 'entropy':
            self.information_criterion = entropy
        elif criterion == 'gini':
            self.information_criterion = gini

        self.tree = None

    def fit(self, x, y):
        """
            决策树训练函数

        Parameters
        ----------
        x:numpy.ndarray
            数据集
        y:numpy.ndarray
            数据标签
        """
        self.tree = self.create_tree(x, y, 0)

    def div_feature(self, x, y, feature, t=0):
        """
            根据属性分割数据集
        Parameters
        ----------
        x:numpy.ndarray
            数据集
        y:numpy.ndarray
            数据标签
        feature:int
            分割的属性索引，当属性连续时，按照t值将数据分为两份，当属性离散时，则属性有几类便将数据集分为几类
        t:float
            当属性连续时，按照t值将数据分为两份，当属性为离散值时该值不起作用

        Returns
        -------
        div_x,div_y,div_z:(list,list,list)
            div_x:数据集分类结果。

            div_y:标签分类结果。

            div_z:分类依据。若按照连续属性分类则该列表存储对应数据分类是否小于等于t值，小于等于则为True，否则为False；
            若按照离散属性分类，则该列表存储对应分类的属性。
        """
        div_x = []
        div_y = []
        div_z = []
        if self.feature_attributes[feature] == "c":
            tmp = x[:, feature].astype(float)
            div_x.append(x[tmp <= t])
            div_x.append(x[tmp > t])
            div_y.append(y[tmp <= t])
            div_y.append(y[tmp > t])
            div_z.append(True)
            div_z.append(False)
        else:
            for i in np.unique(x[:, feature]):
                div_x.append(x[x[:, feature] == i])
                div_y.append(y[x[:, feature] == i])
                div_z.append(i)
        return div_x, div_y, div_z

    def information_gain(self, x, y, feature, t=0):
        """
            信息增益

        Parameters
        ----------
        x:numpy.ndarray
            数据集
        y:numpy.ndarray
            数据标签
        feature:int
            分割的属性索引，当属性连续时，按照t值将数据分为两份后计算信息增益，
            当属性离散时，则属性有几类便将数据集分为几类后计算信息增益
        t:float
            当属性连续时，按照t值将数据分为两份，当属性为离散值时该值不起作用

        Returns
        -------
        float
            信息增益
        """
        ent = self.information_criterion(y)
        div_x = self.div_feature(x, y, feature, t)[1]
        gain = 0
        for lab in div_x:
            gain += lab.shape[0] / x.shape[0] * self.information_criterion(lab)

        return ent - gain

    def get_best_feature(self, x, y):
        """
            获取最佳信息增益的属性索引
        Parameters
        ----------
        x:numpy.ndarray
            数据集
        y:numpy.ndarray
            数据标签

        Returns
        -------
        best_a, best_t:(int, float)
            beat_a:最佳属性索引。
            best_t:当最佳属性为连续值时的最佳分割值，
                    当最佳属性为离散值时返回None。
        """
        best_a = 0
        best_t = 0
        best_gain = 0

        for a in range(x.shape[1]):
            if self.feature_attributes[a] == "c":
                data_tmp = np.sort(x[:, a].astype(float), axis=0)
                for j in range(x.shape[0] - 1):
                    t = (data_tmp[j] + data_tmp[j + 1]) / 2
                    gain = self.information_gain(x, y, a, t)
                    if gain >= best_gain:
                        best_gain = gain
                        best_a = a
                        best_t = t
            else:
                gain = self.information_gain(x, y, a)
                if gain >= best_gain:
                    best_gain = gain
                    best_a = a
                    best_t = None

        return best_a, best_t

    def create_tree(self, x, y, d):
        """
            构建决策树

        Parameters
        ----------
        x:numpy.ndarray
            数据集
        y:numpy.ndarray
            数据标签
        d:int
            决策树深度

        Returns
        -------
        Tree
            决策树
        """
        if np.unique(y).size <= 1:
            return Tree({'类别': y[0],
                         '数量': np.unique(y, return_counts=True),
                         '熵': 0,
                         'depth': d})

        if d >= self.max_depth:
            return Tree({'类别': get_most_class(y),
                         '数量': np.unique(y, return_counts=True),
                         '熵': self.information_criterion(y),
                         'depth': d})

        best_feature = self.get_best_feature(x, y)

        leaf_data, leaf_labels, div_z = self.div_feature(x, y, *best_feature)

        if True not in [i.size == 0 for i in leaf_labels]:
            tree = Tree({'特征': (best_feature[0], self.feature[best_feature[0]]),
                         '判断': best_feature[1],
                         '数量': np.unique(y, return_counts=True),
                         '类别': get_most_class(y),
                         '熵': self.information_criterion(y),
                         'depth': d,
                         'div': div_z,
                         'leaf': []})
            for i, j in zip(leaf_data, leaf_labels):
                tree_tmp = self.create_tree(i, j, d + 1)
                tree['leaf'].append(tree_tmp)
        else:
            tree = Tree({'类别': get_most_class(y),
                         '数量': np.unique(y, return_counts=True),
                         '熵': self.information_criterion(y),
                         'depth': d})

        return tree

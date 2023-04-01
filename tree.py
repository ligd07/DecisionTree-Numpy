class Tree(dict):
    n = 0

    def __init__(self, tree):
        """
            依据输入的字典形式的决策树生成Tree

        Parameters
        ----------
        tree:dict
            字典形式的决策树
        """
        super(Tree, self).__init__(tree)

    def predict(self, x):
        """
            依据输入数据预测种类

        Parameters
        ----------
        x:numpy.ndarray
            属性数据，1维数组
        Returns
        -------
        label:str
            预测的种类
        """
        if 'leaf' in self:
            if self['判断'] is None:
                if x[self['特征'][0]] in self['div']:
                    i = self['div'].index(x[self['特征'][0]])
                    label = self['leaf'][i].predict(x)
                else:
                    label = self['类别']
            else:
                if x[self['特征'][0]].astype(float) <= self['判断']:
                    label = self['leaf'][0].predict(x)
                else:
                    label = self['leaf'][1].predict(x)
        else:
            label = self['类别']
        return label

    def grap_dot(self, path="test.dot"):
        """
            生成.dot文件，以便graphviz生成图像

        Parameters
        ----------
        path:str
            dot文件路径和文件名
        """
        Tree.n = 0
        s = self.grap()
        with open(path, "a", encoding='utf-8') as f:
            f.truncate(0)
            f.write('digraph demo\n{\nnode[color = "blue"]\n')
            f.write(s)
            f.write('}')

    def grap(self):
        """
            以字符串形式返回决策树的节点和连接方式，以便graphviz生成图像

        Returns
        -------
        s:str
            字符串形式决策树节点和连接方式
        """
        if 'leaf' in self:
            s = str(Tree.n) + '[' + 'label=\"'
            for num in range(self['数量'][0].size):
                s += self['数量'][0][num] + ':' + str(round(self['数量'][1][num], 3)) + '\\n'

            s += '熵:' + str(round(self['熵'], 3)) + '\\n'

            if self['判断'] is None:
                s += str(self['特征'][1]) + '\\n'
            else:
                s += str(self['特征'][1]) + '<=' + str(
                    round(self['判断'], 3)) + '\\n'

            s += '\",fontname="FangSong",shape="box"]' + '\n'
            dad = str(Tree.n)
            Tree.n += 1
            for i in range(len(self['leaf'])):
                s += dad + '->' + str(Tree.n) + '[label="' + str(self['div'][i]) + '",fontname="FangSong"]' + '\n' + \
                     self['leaf'][i].grap()
        else:
            s = str(Tree.n) + '[' + 'label=\"'
            for num in range(self['数量'][0].size):
                s += self['数量'][0][num] + ':' + str(round(self['数量'][1][num], 3)) + '\\n'
            s += '类别:' + str(self['类别']) + \
                 '\\n熵:' + str(round(self['熵'], 3)) + \
                 '\",fontname="FangSong",shape="box"]' + '\n'
            Tree.n += 1

        return s

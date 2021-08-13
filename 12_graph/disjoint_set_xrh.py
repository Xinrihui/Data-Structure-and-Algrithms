# -*- coding: UTF-8 -*-
from collections import *


class DisjointSet:
    """

    by XRH 
    date: 2020-09-26 

    实现 并查集( 不相交集合的森林 )

    功能：
    1. 判断两个元素 是否在一个集合中
    2. 对两个元素 所在的不同的集合 进行合并
    3. 输出 元素 所在的集合
    4. 输出 无向图中的 所有 连通分量
    5. 输出 集合 的个数

    """

    def __init__(self, edge_list=None):

        self.__hash_key_node = {}
        self.__hash_node_key = {}

        self.parent = [None]  # 记录每一个节点的 root , 第 0 位空出, 标号 从1开始

        self.rank = [None]  # 记录 每一个节点的 秩

        if edge_list is not None:

            for edge in edge_list:
                self.add(edge)

    def add_node(self, node_key):
        """
        独立的 节点 加入 并查集

        :param node_key: 
        :return: 
        """

        if node_key not in self.__hash_key_node:
            index = len(self.parent)  # 标号 从1开始

            self.__hash_key_node[node_key] = index
            self.__hash_node_key[index] = node_key

            self.parent.append(index)

            self.rank.append(1)

    def add(self, edge):
        """
        以边的 形式 将元素加入 并查集

        若 图中没有环路, 则边加入 成功, 返回 True
        若 图中有环路, 则边加入失败, 返回 False

        :param edge: (node_a,node_b)
        :return: 
        """

        node_a_key = edge[0]
        node_b_key = edge[1]

        flag = True

        if node_a_key == node_b_key:  # 边的左右两点 相等

            if node_a_key not in self.__hash_key_node:
                index = len(self.parent)  # 标号 从1开始

                self.__hash_key_node[node_a_key] = index
                self.__hash_node_key[index] = node_a_key

                self.parent.append(index)

                self.rank.append(1)

                print("graph exist circle ,edge({}, {})".format(node_a_key, node_b_key))

                flag = False

        else:
            if node_a_key not in self.__hash_key_node:
                index = len(self.parent)  # 标号 从1开始

                self.__hash_key_node[node_a_key] = index
                self.__hash_node_key[index] = node_a_key

                self.parent.append(index)

                self.rank.append(1)

            if node_b_key not in self.__hash_key_node:
                index = len(self.parent)

                self.__hash_key_node[node_b_key] = index
                self.__hash_node_key[index] = node_b_key

                self.parent.append(index)

                self.rank.append(1)

            node_a = self.__hash_key_node[node_a_key]
            node_b = self.__hash_key_node[node_b_key]

            if self.__find(node_a) != self.__find(node_b):  # 根节点不同 说明不在一个 集合中

                self.__union(node_a, node_b)  # 合并两个集合

                flag = True

            else:

                print("graph exist circle ,edge({}, {})".format(node_a_key, node_b_key))

                flag = False

        return flag

    def countSetsNum(self):
        """
        返回 并查集中集合的数目

        :return: 
        """

        root_set = set()

        for node in range(1, len(self.parent)):
            root_set.add(self.__find(node))

        return len(root_set)

    def __find(self, node):
        """
        找到 node 的根节点 即 node 所属的集合, 并进行路径压缩

        :param node: 
        :return: 
        """

        if self.parent[node] != node:
            root = self.__find(self.parent[node])

            self.parent[node] = root

        root = self.parent[node]

        return root

    def __union(self, node1, node2):

        """
        找出 node1 和 node2 对应的集合, 并将 它们按秩合并
        ( 把 高度较小的树 合并到高度较大的树中 )

        :param node1: 
        :param node2: 
        :return: 
        """

        root1 = self.__find(node1)
        root2 = self.__find(node2)

        if self.rank[root1] < self.rank[root2]:  # 把 高度较小的树 合并到高度较大的树中, 树的高度不变

            self.parent[root1] = root2


        elif self.rank[root1] > self.rank[root2]:

            self.parent[root2] = root1


        else:  # 两个 集合树的 高度相同, 则 随意合并到其中一个树中，并且 树的高度 +1

            self.parent[root2] = root1

            self.rank[root1] += 1

    def compareTwoKey(self, key1, key2):
        """
        判断 两个键 是否在 一个 集合中

        1.根据 键找到对应的 Node 
        2. 找出 Node 对应的集合 
        3. 比较两者 是否在一个集合中

        :param key: 
        :return: 
        """
        node1 = self.__hash_key_node[key1]
        node2 = self.__hash_key_node[key2]

        return self.__find(node1) == self.__find(node2)

    def getSet(self, key):
        """
        输出 key 所在的集合 的所有的元素

        :param key: 
        :return: 
        """

        res = None

        if key in self.__hash_key_node:

            node = self.__hash_key_node[key]

            root = self.__find(node)

            node_list = []

            for i in range(1, len(self.parent)):

                p = self.parent[i]

                if p == root:
                    node_list.append(i)

            res = [self.__hash_node_key[node] for node in node_list]

        return res

    def getConnectedComponent(self):
        """
        输出 无向图 中所有的 连通分量

        :return: 
        """

        connected_component_node = defaultdict(set)

        for i in range(1, len(self.parent)):

            p = self.__find(i)

            connected_component_node[p].add(i)

        connected_component = {}

        for root_node, set_nodes in connected_component_node.items():
            key = self.__hash_node_key[root_node]

            connected_component[key] = [self.__hash_node_key[node] for node in set_nodes]

        return connected_component

if __name__ == '__main__':


    edge_list=[('a','b'),('b','c'),('b','d'),('e','f'),('g','e'),('h','i'),('j','j')]


    disjoint_set = DisjointSet(edge_list)

    # print(disjoint_set.parent)
    # print(disjoint_set.rank)

    print(disjoint_set.countSetsNum())

    print(disjoint_set.compareTwoKey('a','d'))

    print(disjoint_set.getSet('d'))
    print(disjoint_set.getSet('f'))
    print(disjoint_set.getSet('z'))

    print(disjoint_set.getConnectedComponent())



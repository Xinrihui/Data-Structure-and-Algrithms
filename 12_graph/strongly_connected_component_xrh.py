#!/usr/bin/python
# -*- coding: UTF-8 -*-

from collections import *


class solution():
    """
    Kosaraju 实现 求解 有向图的强连通分量
    
    """

    def dfs_timestamp(self, G):
        """
        非连通图的 DFS

        (1) 每一个节点都要 作为 DFS 的入口
        (2) 返回 每一个点的 发现时间 和 完成时间

        :param G: 
        :return: 
        """

        self.discover_time = dict()  # 顶点的发现时间
        self.finish_time = dict()  # 顶点的 完成时间

        self.visited=set() # 被发现过的节点


        nodes_list = G.keys()

        t = 0

        for node in nodes_list:

            if node not in self.visited:

                t = self.__dfs_timestamp_process(G, node, t)

        return self.discover_time, self.finish_time

    def __dfs_timestamp_process(self, G, current, t=0):
        """
        基于递归 的 DFS
        给节点加上了时间戳 

        :param G: 图
        :param current: 当前顶点
        :param t: 时间点 
        :return: 
        """

        t += 1  # Set discover time
        self.discover_time[current] = t

        self.visited.add(current)

        for node in G[current]:  # Explore neighbors

            if node not in self.visited:

                t = self.__dfs_timestamp_process(G, node, t)  # t是一个变量，带入函数的时候值被复制过来，因此在其他递归函数的修改将失效

        t += 1  # Set finish time
        self.finish_time[current] = t

        return t



    def dfs_timestamp_output_scc(self, GT, order):
        """
        
        DFS 返回 整个 非连通的有向图 G 的所有的 连通分量
        
        :param GT: 
        :param order: 可以 指定 DFS 的入口的顺序
        :return: 
        """

        self.discover_time = dict()  # 顶点的发现时间
        self.finish_time = dict()  # 顶点的 完成时间

        self.visited = set()  # 被发现过的顶点

        nodes_list = order

        t = 0

        res_scc=[]

        S_prev=set()

        for node in nodes_list:

            if node not in self.visited:

                t = self.__dfs_timestamp_process(GT, node, t)
                res_scc.append( self.visited-S_prev)

                S_prev=self.visited.copy()

        return res_scc



    def tr(self,G):  # Transpose (rev. edges of) G
        """
        求图 G 的转置
        
        :param G: 
        :return: 
        """
        GT = {}

        for u in G:
            GT[u] = set()  # Get all the nodes in there

        for u in G:
            for v in G[u]:
                GT[v].add(u)  # Add all reverse edges

        return GT

    def scc(self,G):
        """
        有向图 G 的强连通 分量
        
        :param G: 
        :return: 
        """

        #1. DFS 得到 所有节点的 开始和 结束时间
        discover_time, finish_time = self.dfs_timestamp(G)

        # print(finish_time)

        # 2. 节点的 结束时间 逆序
        order= sorted(finish_time.items(),key=lambda x:x[1],reverse=True)

        order=[ele[0] for ele in order]

        # print(order)

        # 3. 图G 的转置
        GT=self.tr(G)


        res_scc=self.dfs_timestamp_output_scc(GT,order)

        return res_scc


class solution3:
    """
    DFS 求解 无向图的 连通分量
    
    """


    def walk(self, G, s, S=set()):  # Walk the graph from node s
        """
        非递归 DFS 遍历图
        :param G: 
        :param s: 
        :param S: 
        :return: 
        """

        # P 用于存放已探索的节点
        # Q 是用来存放待探索节点

        P, Q = dict(), set()  # P: 记录 已经访问过的节点的 前驱节点 Predecessors  Q:"to do" queue

        P[s] = None  # s has no predecessor

        Q.add(s)  # We plan on starting with s
        while Q:  # Still nodes to visit
            u = Q.pop()  # Pick one, arbitrarily
            # for v in G[u].difference(P.keys(), S): #Key        # New nodes?
            for v in G[u] - (set(P.keys()) | S):
                Q.add(v)  # We plan to visit them!
                P[v] = u  # Remember where we came from
        # print P
        return P

    def components(self,G):  # The connected components
        comp = []
        seen = set()  # Nodes we've already seen
        for u in G.keys():
            # print'u', u
            # Try every starting point
            if u in seen: continue  # Seen? Ignore it
            C = self.walk(G, u)  # Traverse component
            seen.update(C.keys())  # Add keys of C to seen
            # print seen
            comp.append(C)  # Collect the components
        return comp


class solution2:
    """
    Tarjan  实现 求解 强连通分量

    """
    pass



if __name__ == '__main__':
    sol = solution()

    G = {
        0: set([1, 3]),
        1: set([0, 2, 4]),
        2: set([1, 5]),
        3: set([0, 4]),
        4: set([1, 3, 5, 6]),
        5: set([2, 4, 7]),
        6: set([4, 7]),
        7: set([5, 6])
    }


    # print(sol.dfs(G, 0, 2))


    G = {
        'u': set(['v', 'x']),
        'v': set(['y']),
        'x': set(['v']),
        'y': set(['x']),
        'w': set(['y','z']),
        'z':set(['z'])
        }

    # print(sol.dfs_timestamp(G))

    G = {
        'c': set(['d', 'g']),
        'd': set(['c', 'h']),
        'f': set(['g']),
        'g': set(['f', 'h']),
        'a': set(['b']),
        'b': set(['e','c','f']),
        'e': set(['a','f']),
        'h':set(['h'])

        }

    # print(sol.scc(G))


    G = {
        'a': set(['b']),
        'b': set(['c']),
        'c':set()
        }

    # print(sol.scc(G))

    sol2=solution2()
    G = {
        0: {2:1},
        1: {1:1,2:1},
        2: {1:1},
        }

    # print(sol2.dfs_find_cricle(G))

    G = {
        0: set([1, 2]),
        1: set([0, 2]),
        2: set([0, 1]),
        3: set([4, 5]),
        4: set([3, 5]),
        5: set([3, 4])
        }

    sol3 = solution3()
    # print (sol3.components(G))















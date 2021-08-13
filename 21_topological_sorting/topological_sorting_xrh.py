#!/usr/bin/python
# -*- coding: UTF-8 -*-

from collections import *

class solution():


    def Kahn(self,graph):

        # 1.统计所有节点的入度
        in_degree=defaultdict(int)
        for source in graph: # 所有 节点的入度 初始化为 0
            in_degree[source]=0

        for source in graph:
            for target in graph[source]:
                in_degree[target]+=1

        # print(in_degree)

        # 2. 找出 入度为0 的节点放入 队列queue 中
        queue=deque()
        for node in in_degree:
            if in_degree[node]==0:
                queue.append(node)

        # 3. 输出 入度为0 的节点到结果集中，并将 相关联节点的入度 -1
        res=[]
        while len(queue)>0:

            current=queue.popleft()
            res.append(current)

            for node in graph[current]:
                in_degree[node]-=1

                if in_degree[node]==0:  # 入度为0 的节点加入 queue
                    queue.append(node)

        # 4. 判断是否存在环路
        flag=False
        if len(res)<len(graph): # 拓扑排序的顶点个数 小于 图的顶点个数，说明存在环路
            flag=True

        return flag,res


    def by_dfs(self, graph):

        """
        利用 深度优先 遍历 实现 拓扑排序：
        
        1. DFS中顶点的 出栈顺序 即 为逆拓扑序。
        2.递归处理 每一个顶点，把每一个顶点作为入口，做 DFS 
    
        :param graph: 

        :return: 
                
        结果错误（未解决）：
        ['v4', 'v3', 'v2', 'v5', 'v1']
        正确结果：
        ['v4', 'v3', 'v2', 'v1', 'v5']
        
        """
        res = []
        visited = set()

        for s in graph: # graph 中的每一个节点都可为 DFS 的入口

            if s not in visited: # 若节点 被访问过则忽略
                stack = []
                stack.append(s)
                visited.add(s)  # 初始化，起点 s 需标记为 已被访问

                while len(stack) > 0:

                    current = stack.pop()
                    print(current)  # 访问节点
                    res.append(current) #TODO: 要先递归 后访问，即 不能 先访问 v1,而应该 从 v1 继续向深处遍历 直到 v5，然后访问 v5；
                                        #TODO: 联想 二叉树的 前序遍历 和 中序遍历的 区别

                    for node in graph[current]:  # 遍历 current节点 周围的子节点

                        if node not in visited:  # 已经访问过的节点 无需再加入 queue
                            stack.append(node)
                            visited.add(node)  # 加入 stack 就算被访问过


        return res[::-1]

    def by_dfs_recursive(self,graph):

        """
        利用 深度优先 遍历 实现 拓扑排序：

        1.递归处理 每一个顶点，把每一个顶点作为入口，做 DFS 
        2. 先递归 后 访问，访问结果列表的逆序 即为 拓扑排序
        
        :param graph: 

        :return: 
        """
        is_visit =set()
        li = []

        def dfs(graph, start_node):

            for end_node in graph[start_node]:
                if end_node not in is_visit:

                    is_visit.add(end_node)
                    dfs(graph, end_node)

            print(start_node) # 先递归 后访问
            li.append(start_node)

        for start_node in graph:
            if start_node not in is_visit:
                is_visit.add(start_node)
                dfs(graph, start_node)

        li.reverse()
        return li

    def by_dfs_recursive_v1(self, graph):
        """
        利用 深度优先 遍历 实现 拓扑排序：
        
        1.通过邻接表构造逆邻接表。邻接表中，边 s->t 表示 s 先于 t 执行，也就是 t 要依赖 s。
        在逆邻接表中，边 s->t 表示 s 依赖于 t，s 后于 t 执行。
        
        2.递归处理每个顶点。对于顶点 vertex 来说，
        我们先输出 指向它的所有顶点，也就是说，先把它依赖的所有的顶点输出了，然后再输出自己。
        
        :param graph: 
        :return: 
        """

        inverse_graph=defaultdict(set)

        # 1.通过邻接表构造逆邻接表
        for node in graph:
            inverse_graph[node]=set()

        for start_node in graph:
            for end_node in graph[start_node]:
                if start_node not in inverse_graph[end_node]:
                    inverse_graph[end_node].add(start_node)

        # print(inverse_graph)

        self.result=[]
        self.visited=set()

        # 2.递归处理每个顶点
        for node in inverse_graph:

            if node not in self.visited:
                self.visited.add(node)
                self.__dfs(inverse_graph,node)

        return self.result

    def __dfs(self,graph,s):
        """
        图的深度优先遍历 
        :param graph: 
        :param s: 
        :return: 
        """
        for node in graph[s]:

            if node not in self.visited:
                self.visited.add(node)
                self.__dfs(graph,node) # 先把它依赖的所有的顶点递归了,
        self.result.append(s)  # 然后再访问自己.



if __name__ == '__main__':

    sol = solution()

    graph = {
        'v1': set(['v5']),
        'v2': set(['v1']),
        'v3': set(['v1', 'v5']),
        'v4': set(['v2', 'v5']),
        'v5': set([]),
    }

    # print(sol.Kahn(graph))

    print(sol.by_dfs(graph))
    print(sol.by_dfs_recursive(graph))

    # print(sol.by_dfs_recursive_v1(graph))


    graph_with_circle = {
        'v1': set(['v5']),
        'v2': set(['v1']),
        'v3': set(['v1', 'v5']),
        'v4': set(['v2', 'v5']),
        'v5': set(['v1']),
    }
    print(sol.Kahn(graph_with_circle))






















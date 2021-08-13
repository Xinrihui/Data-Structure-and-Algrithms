# -*- coding: UTF-8 -*-
from collections import *

from  priority_queue_xrh import *

from  disjoint_set_xrh import *

class solutions:

    def Prim(self,graph):
        """
        最小生成树 的 Prim 算法
        
        利用 优先队列
        :param graph: 
        :return: 最小生成树 的边的 list: [('1', 1, '3'), ('3', 4, '6'), ('6', 2, '4'), ('3', 5, '2'), ('2', 3, '5')]
                ( 边开始节点, 边长 , 边的结束节点 )
        """

        visited=set() # S 集合

        Not_visited = Priority_Queue( key_func=lambda x: x[0], compare_func=lambda x: x[1]) # V-S 集合

        for node in graph:

            Not_visited.push((node,float('inf'),None)) # (node, distance, S_node)  S集合中的节点 S_node 到 V-S 集合 中的节点node 的 最短的距离为 distance

        # 初始化, 设置node1 的距离为 最小，为了让 node1 首先被弹出
        Not_visited.update(('1', 0, None))

        res_edge=[] # 记录 最小生成树 的边, ( 边开始节点, 边长 , 边的结束节点 )

        while len(Not_visited)>0:

            current=Not_visited.pop()  # (初始化)弹出 Node1:  (node1, 0, None) current= ('1', 0, None)
                                       #
                                       # 弹出 与 S 集合的距离 最短的 node
                                       #    1. Node3: (node3 ,1 ,node1 ) current= ('3', 1, '1')

            current_node= current[0]

            visited.add(current[0])

            res_edge.append((current[2] ,current[1],current_node)) # 记录 最短的边 ( edge_node1, length , edge_node2 ) ( 边开始节点, 边长 , 边的结束节点 )

            # 因为 current_node 加入S集合，使得 S 集合 与 V-S 集合中的节点 的 最短边会发生变化  ，所以 优先队列中节点的值 需要更新, 同时记录 最短的边
            for edge in graph[current_node]:

                if edge[0] not in visited: # 排除掉S 集合中的节点( V-S集合中的节点)

                    target_node=edge[0] # current_node -> target_node
                    distance=edge[1]

                    if distance < Not_visited.get_byKey(target_node)[1] : # 距离比原来短了
                        # 更新 优先队列中 节点的值
                        Not_visited.update( (target_node, distance ,current_node) ) # (target_node, distance, S_node )

            # print(Not_visited.hash_table)

        return  res_edge[1:] # 指向Node1 的边为空，因此忽略


    def Kruskal(self, graph):
        """
        最小生成树 的 Kruskal 算法

        :param graph: 
        :return: 最小生成树 的边的 list: [('1', 1, '3'), ('3', 4, '6'), ('6', 2, '4'), ('3', 5, '2'), ('2', 3, '5')]
                ( 边开始节点, 边长 , 边的结束节点 )
        """

        edges=[]

        visited=set()

        #找到所有的边 并对边进行去重（graph 为无向图，所以 node1-> node2 和 node2 -> node1 等价）

        for start_node,edge_list in graph.items():

            for end_node,edge_length in edge_list:

                if end_node not in visited:

                    edges.append((start_node,edge_length,end_node))


            visited.add(start_node)

        # 按照 边的长度 正序排序
        edges= sorted(edges,key=lambda ele : ele[1])

        # print(edges) #[('1', 1, '3'), ('4', 2, '6'), ('2', 3, '5'), ('3', 4, '6'), ('1', 5, '4'), ('2', 5, '3'), ('3', 5, '4'), ('1', 6, '2'), ('3', 6, '5'), ('5', 6, '6')]

        # 初始化 find
        find={node:node  for node in graph}

        def update_find_Recursion(find,current, find_max, find_min):

            find[current] = find_max

            for end_node,edge_length in graph[current]:

                if find[end_node] == find_min:
                    update_find_Recursion(find,end_node, find_max, find_min)

        res_edge = []  # 记录 最小生成树 的边 ( 边开始节点, 边长 , 边的结束节点 )

        for edge in edges:

            node_s=edge[0]
            node_e=edge[2]

            if find[node_s]!=find[node_e]: # 说明 边的两个端点 不在一个 联通分量中

                res_edge.append(edge) # edge 选入 最小生成树的边

                # 连通分量的标记：
                # 把 这个边的 开始节点 node_s 所在的联通分量  和 结束节点 node_e 所在的联通分量 都统一到一个 联通分量中

                if find[node_s]>find[node_e]:
                    node_max=node_s
                    node_min=node_e

                else:
                    node_max=node_e
                    node_min=node_s

                find_max=find[node_max]
                find_min=find[node_min]

                update_find_Recursion(find,node_min,find_max,find_min)


        return res_edge


    def Kruskal_v2(self, graph):
        """
        最小生成树 的 Kruskal 算法

        利用 并查集 判断 边上的 两个点 是否在一个 连通分量中
        
        :param graph: 
        :return: 最小生成树 的边的 list: [('1', 1, '3'), ('3', 4, '6'), ('6', 2, '4'), ('3', 5, '2'), ('2', 3, '5')]
                ( 边开始节点, 边长 , 边的结束节点 )
        """

        edges=[]

        visited=set()

        #找到所有的边 并对边进行去重（graph 为无向图，所以 node1-> node2 和 node2 -> node1 等价）

        for start_node,edge_list in graph.items():

            for end_node,edge_length in edge_list:

                if end_node not in visited:

                    edges.append((start_node,edge_length,end_node))


            visited.add(start_node)

        # 按照 边的长度 正序排序
        edges= sorted(edges,key=lambda ele : ele[1])

        # print(edges) #[('1', 1, '3'), ('4', 2, '6'), ('2', 3, '5'), ('3', 4, '6'), ('1', 5, '4'), ('2', 5, '3'), ('3', 5, '4'), ('1', 6, '2'), ('3', 6, '5'), ('5', 6, '6')]

        # 初始化 并查集
        disjoin_set= DisjointSet()

        res_edge = []  # 记录 最小生成树 的边 ( 边开始节点, 边长 , 边的结束节点 )

        for edge in edges: # edge=('1', 1, '3')

            node_s=edge[0]
            node_e=edge[2]

            if disjoin_set.add((node_s,node_e))==True: # 说明 边的两个端点 不在一个 联通分量中

                res_edge.append(edge) # edge 选入 最小生成树的边


        return res_edge

if __name__ == '__main__':

    sol = solutions()
    graph = {
        '1': [('2', 6), ('3', 1),('4', 5)],  #(2, 6) 1节点到2节点的 距离为6 ; (3, 1) 1节点到3节点的 距离为1
        '2': [('1', 6),('3',5) ,('5', 3)],
        '3': [('1', 1),('2', 5),('4', 5),('5', 6),('6',4)],
        '4': [('1', 5), ('3', 5) , ('6' ,2)],
        '5': [('2', 3),('3', 6),('6' ,6)],
        '6': [('5',6),('3',4),('4',2)]
    }

    # print(sol.Prim(graph))
    print(sol.Kruskal(graph))

    print(sol.Kruskal_v2(graph))
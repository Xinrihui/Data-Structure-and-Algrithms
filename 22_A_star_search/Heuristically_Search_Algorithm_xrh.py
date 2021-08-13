# -*- coding: UTF-8 -*-
from collections import *

import heapq

from  priority_queue_xrh import *


class solutions:

    def __hManhattan(self,v1,v2):
        """
        计算 两个点的 曼哈顿距离
        :param v1: (x1,y1)
        :param v2: (x2,y2)
        :return: 
        """
        return abs(v1[0]-v2[0])+abs(v1[1]-v2[1])

    def dijkstra_v1(self, graph, start_node, end_node):
        """
        利用 优先队列 实现 单源最短路径算法 dijkstra
        
        1.将优先队列的实现打包成一个类 Priority_Queue , 实现了 对堆中元素的 value 更新后，会自动调整堆
        2.第一次 只将 起始节点 装入优先队列中
        3.每次 从优先队列中弹出 距离最小的节点，对此节点做松弛操作，若它的一跳范围内有节点 距离变小了，则把它们装入 优先队列 
        
        详见 《数据机构与算法之美》 -> 最短路径

        :param graph: 
        :param start_node: 起点
        :param end_node: 终点
        :return: 
        """

        pre_node = {node: start_node for node in graph}  # 记录每个顶点的前驱顶点
        pre_node[start_node] = None  # 起点 没有前驱节点

        distance = {node: float('inf') for node in graph}
        distance[start_node] = 0  # 我们把起始顶点的 dist 值初始化为 0

        heap = Priority_Queue([(start_node, 0)], key_func=lambda x: x[0], compare_func=lambda x: x[1])  #

        while len(heap) > 0:

            current = heap.pop()  # 弹出 堆中最小的元素
            print(current)

            if current[0] == end_node:  # 起点 到 终点的最短路径产生了
                break

            left_node = current[0]  # 当前节点 即 边的左端点
            curr_dis = current[1]  # 起点 到 当前节点的最短距离

            for edge in graph[left_node]:  # 遍历当前 节点 出度的边

                right_node = edge[0]  # 边 的右端点
                new_dis = curr_dis + edge[1]

                if new_dis < distance[right_node]:
                    distance[right_node] = new_dis  # 更新 起点 到right_node 的距离
                    pre_node[right_node] = left_node  # 更新 right_node 的前驱节点

                    if heap.has_Key(right_node):
                        heap.update_byKey(right_node, new_dis)

                    else:
                        heap.push((right_node, new_dis))

        path = []
        current = end_node

        while current != None:
            path.append(current)
            current = pre_node[current]

        return path[::-1]

    def A_star_search_v1(self, graph,position, start_node, end_node):
        """
        利用 优先队列 实现 单源最短路径算法 A*

        :param graph: 
        :param position:  所有点的坐标位置
        :param start_node: 起点
        :param end_node: 终点
        :return: 
        """

        pre_node = {node: start_node for node in graph}  # 记录每个顶点的前驱顶点
        pre_node[start_node] = None  # 起点 没有前驱节点

        distance = {node: float('inf') for node in graph}
        distance[start_node] = 0  # 我们把起始顶点的 dist 值初始化为 0

        heap = Priority_Queue([(start_node, 0)], key_func=lambda x: x[0], compare_func=lambda x: x[1])  #

        flag=0
        while len(heap) > 0 and flag==0:

            current = heap.pop()  # 弹出 堆中最小的元素
            print(current)


            left_node = current[0]  # 当前节点 即 边的左端点
            curr_dis = current[1]  # 起点 到 当前节点的最短距离

            for edge in graph[left_node]:  # 遍历当前 节点 出度的边

                right_node = edge[0]  # 边 的右端点

                new_dis=curr_dis + edge[1] + self.__hManhattan(position[left_node],position[right_node])     # 计算估价函数

                if new_dis < distance[right_node]:
                    distance[right_node] = new_dis  # 更新 起点 到right_node 的距离
                    pre_node[right_node] = left_node  # 更新 right_node 的前驱节点

                    if heap.has_Key(right_node):
                        heap.update_byKey(right_node, new_dis)

                    else:
                        heap.push((right_node, new_dis))

                if right_node == end_node:#只要到达t就可以结束while了
                    flag=1 # 退出 标志置为 1
                    break


        path = []
        current = end_node

        while current != None:
            path.append(current)
            current = pre_node[current]

        return path[::-1]

    def A_star_search(self, graph, position,start_node, end_node):
        """
        单源最短路径算法 A*
        
        基于 dijkstra() 改造 
        
        :param graph: 
        :param position:  所有点的坐标位置
        :param start_node: 起点
        :param end_node: 终点
        :return: 
        """

        pre_node = {node: start_node for node in graph}  # 记录每个顶点的前驱顶点
        pre_node[start_node] = None  # 起点 没有前驱节点

        distance = {node: float('inf') for node in graph}  # 起始点到 所有顶点的距离
        distance[start_node] = 0  # 我们把起始顶点的 dist 值初始化为 0

        Not_visited = Priority_Queue(key_func=lambda x: x[0], compare_func=lambda x: x[1])  # V-S 集合

        for node in graph:
            Not_visited.push((node, float('inf')))

        Not_visited.update((start_node, 0))  # 起点 的距离为0 肯定先弹出

        flag=0 # 算法退出条件

        while len(Not_visited) > 0 and flag==0:

            current = Not_visited.pop()  # 弹出 堆中最小的元素
            # print(current)

            if current[0] == end_node:  # 早停：起点 到 终点的最短路径产生了
                break

            left_node = current[0]  # 当前节点 即 边的左端点
            curr_dis = current[1]  # 起点 到 当前节点的最短距离

            for edge in graph[left_node]:  # 遍历当前 节点 出度的边

                if Not_visited.has_Key(edge[0]):

                    right_node = edge[0]  # 边 的右端点
                    new_dis = curr_dis + self.__hManhattan(position[right_node],position[left_node])

                    if new_dis < distance[right_node]:
                        distance[right_node] = new_dis  # 更新 起点 到right_node 的距离
                        pre_node[right_node] = left_node  # 更新 right_node 的前驱节点

                        Not_visited.update((right_node, new_dis))

                        if right_node == end_node: # 摸到终点了 ，算法终止
                            flag=1
                            break

        path = []
        current = end_node

        while current != None:
            path.append(current)
            current = pre_node[current]

        return  distance[end_node],path[::-1]


if __name__ == '__main__':

    sol = solutions()

    graph = {
        0: [(1,20),(4,60),(5,60),(6,60)], # 0节点到1节点的 距离为20
        1: [(0,20),(2,20)],
        2: [(1,20),(3,10)],
        3: [(2,10),(12,40),(13,30)],
        4: [(0,60),(8,50),(12,40)],
        5: [(0,60),(8,70),(9,80),(10,50)],
        6: [(0,60),(7,70),(13,50)],
        7: [(6,70),(11,50)],
        8: [(4,50),(5,70),(9,50)],
        9: [(5,80),(8,50),(10,60)],
        10:[(5,50),(9,60),(11,60)],
        11:[(7,50),(10,60)],
        12:[(3,40),(4,40)],
        13:[(3,30),(6,50)]
    }

    position={
        0:(320,630), # 图中顶点的坐标
        1:(300,630),
        2:(280,625),
        3:(270,630),
        4:(320,700),
        5:(360,620),
        6:(320,590),
        7:(370,580),
        8:(350,730),
        9:(390,690),
        10:(400,620),
        11:(400,560),
        12:(270,670),
        13:(270,240)
    }

    # print(sol.dijkstra_v1_1(graph, 0, 10))

    print(sol.A_star_search(graph,position, 0, 10))















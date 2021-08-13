# -*- coding: UTF-8 -*-
from collections import *

import heapq

from  priority_queue_xrh import *

class ComapreHeap(object):

    def __init__(self, initial=None, key=lambda x: x):
        self.key = key

        self.length=len(initial)

        if initial:
            self._data = [(key(item), item) for item in initial]
            heapq.heapify(self._data)
        else:
            self._data = []


    def __len__(self):

        return  self.length

    def push(self, item):
        self.length+=1
        heapq.heappush(self._data, (self.key(item), item))

    def pop(self):
        self.length-=1
        return heapq.heappop(self._data)[1]

    def heapify(self):

        heapq.heapify(self._data)



class solutions:

    def dijkstra_deparated(self,graph,start_node,end_node):
        """
        利用小顶堆 实现 单源最短路径算法 dijkstra
        
        :param graph: 
        :param start_node: 起点
        :param end_node: 终点
        :return: 
        """

        pre_node={ node:start_node for node in graph } #记录每个顶点的前驱顶点
        pre_node[start_node]=None # 起点 没有前驱节点

        distance = { node:float('inf') for node in graph }
        distance[start_node]=0  #我们把起始顶点的 dist 值初始化为 0

        heap = ComapreHeap([(start_node,0)], key=lambda x: x[1]) # 小顶堆，堆顶元素为最小

        while len(heap)>0:

            current=heap.pop()  # 弹出 堆中最小的元素
            print(current)
            left_node=current[0]  #当前节点 即 边的左端点
            curr_dis=current[1] #起点 到 当前节点的最短距离

            for edge in graph[left_node]: # 遍历当前 节点 出度的边

                right_node=edge[0] # 边 的右端点
                new_dis=curr_dis+ edge[1]

                if new_dis < distance[right_node]:

                    distance[right_node]=new_dis #更新 起点 到right_node 的距离
                    pre_node[right_node]=left_node # 更新 right_node 的前驱节点

                    heap.push((right_node,new_dis)) # TODO：堆中 会出现重复的节点。此处应修改为 更新 堆中相应节点的 距离，可利用 优先队列实现

        path=[]
        current=end_node

        while current!=None:

            path.append(current)
            current=pre_node[current]

        return path[::-1]

    def dijkstra_deparated_2(self, graph, start_node, end_node):
        """
        利用 优先队列 实现 单源最短路径算法 dijkstra，解决了 堆中出现重复的节点问题

        :param graph: 
        :param start_node: 起点
        :param end_node: 终点
        :return: 
        """

        pre_node = {node: start_node for node in graph}  # 记录每个顶点的前驱顶点
        pre_node[start_node] = None  # 起点 没有前驱节点

        distance = {node: float('inf') for node in graph}
        distance[start_node] = 0  # 我们把起始顶点的 dist 值初始化为 0

        hash_table = {} # 建立 节点 到 (节点，距离) 的索引

        list_0=[start_node,0]
        hash_table[start_node]=list_0 # list 为可变对象，list_0 是一个地址

        heap = ComapreHeap([list_0], key=lambda x: x[1])  # 小顶堆，堆顶元素为最小

        while len(heap) > 0:

            current = heap.pop()  # 弹出 堆中最小的元素
            print(current)

            if current[0]==end_node: #起点 到 终点的最短路径产生了
                break

            left_node = current[0]  # 当前节点 即 边的左端点
            curr_dis = current[1]  # 起点 到 当前节点的最短距离

            for edge in graph[left_node]:  # 遍历当前 节点 出度的边

                right_node = edge[0]  # 边 的右端点
                new_dis = curr_dis + edge[1]

                if new_dis < distance[right_node]:
                    distance[right_node] = new_dis  # 更新 起点 到right_node 的距离
                    pre_node[right_node] = left_node  # 更新 right_node 的前驱节点

                    if right_node in hash_table:
                        hash_table[right_node][1]=new_dis # hash_table[right_node] 返回的是 list 的地址
                        heap.heapify()

                    else:
                        list_x = [right_node, new_dis]
                        hash_table [right_node] = list_x  # list 为可变对象，list_x 是一个地址
                        heap.push(list_x)

        path = []
        current = end_node

        while current != None:
            path.append(current)
            current = pre_node[current]

        return path[::-1]


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

        heap = Priority_Queue([(start_node, 0)],key_func=lambda x: x[0] , compare_func=lambda x: x[1])

        while len(heap) > 0:

            current = heap.pop()  # 弹出 堆中最小的元素
            print(current)

            if current[0]==end_node: #起点 到 终点的最短路径产生了
                break

            left_node = current[0]  # 当前节点 即 边的左端点
            curr_dis = current[1]  # 起点 到 当前节点的最短距离

            for edge in graph[left_node]:  # 遍历当前 节点 出度的边

                right_node = edge[0]  # 边 的右端点
                new_dis = curr_dis + edge[1]

                if new_dis < distance[right_node]:
                    distance[right_node] = new_dis  # 更新 起点 到right_node 的距离
                    pre_node[right_node] = left_node  # 更新 right_node 的前驱节点

                    if  heap.has_Key(right_node): #
                        heap.update_byKey(right_node,new_dis)

                    else:
                        heap.push((right_node,new_dis))

        path = []
        current = end_node

        while current != None:
            path.append(current)
            current = pre_node[current]

        return path[::-1]

    def dijkstra(self, graph, start_node, end_node):
        """
        单源最短路径算法 dijkstra
        
         1.V-S 集合 的所有节点 全部加入 优先队列中（距离设置为 +无穷），起点的 距离设置为 0 
         2.每次从 V-S 集合中弹出 距离最小的 节点X，对X 做松弛操作， 只有还在 V-S 的集合中的节点 才能松弛  
         
         详见《高级算法与分析》 -> 8.贪心算法(2)
        
        :param graph: 
        :param start_node: 起点
        :param end_node: 终点
        :return: 
        """

        pre_node = {node: start_node for node in graph}  # 记录每个顶点的前驱顶点
        pre_node[start_node] = None  # 起点 没有前驱节点

        distance = {node: float('inf') for node in graph} # 起始点到 所有顶点的距离
        distance[start_node] = 0  # 我们把起始顶点的 dist 值初始化为 0

        Not_visited = Priority_Queue( key_func=lambda x: x[0], compare_func=lambda x: x[1])  # V-S 集合

        for node in graph:

            Not_visited.push(( node,float('inf') ))

        Not_visited.update((start_node,0)) # 起点 的距离为0 肯定先弹出

        while len(Not_visited) > 0:

            current = Not_visited.pop()  # 弹出 堆中最小的元素
            # print(current)

            if current[0] == end_node:  # 早停：起点 到 终点的最短路径产生了
                break

            left_node = current[0]  # 当前节点 即 边的左端点
            curr_dis = current[1]  # 起点 到 当前节点的最短距离

            for edge in graph[left_node]:  # 遍历当前 节点 出度的边

                if Not_visited.has_Key(edge[0]):

                    right_node = edge[0]  # 边 的右端点
                    new_dis = curr_dis + edge[1]

                    if new_dis < distance[right_node]:

                        distance[right_node] = new_dis  # 更新 起点 到right_node 的距离
                        pre_node[right_node] = left_node  # 更新 right_node 的前驱节点

                        Not_visited.update((right_node, new_dis))


        path = []
        current = end_node

        while current != None:
            path.append(current)
            current = pre_node[current]

        return distance,path[::-1]


if __name__ == '__main__':

    sol = solutions()
    graph = { # 图例见  《数据结构与算法之美》-> 高级算法 -最短路径
        0: [(1,10),(4,15)], # 0节点到1节点的 距离为10 ; 0节点到4节点的 距离为15
        1: [(2,15),(3,2)],
        2: [(5,5)],
        3: [(2,1),(5,12)],
        4: [(5,10)],
        5: []
    }

    # print(sol.dijkstra_v1(graph, 0, 4))
    # print(sol.dijkstra_v1(graph,0,2))

    # print(sol.dijkstra(graph, 0, 5))
    # print(sol.dijkstra(graph,0,2))

    graph_with_circle = {
        0: [(1,10),(4,20)], # 0节点到1节点的 距离为10 ; 0节点到4节点的 距离为 20
        1: [(2,15),(3,2)],
        2: [(5,5)],
        3: [(2,1),(5,12)],
        4: [(5,10)],
        5: [(4,1)] # 出现环路 4->5 5->4 ; 5节点到4节点的 距离为 1
    }

    # print(sol.dijkstra_v1(graph_with_circle, 0, 4))

    print(sol.dijkstra(graph_with_circle, 0, 4))



































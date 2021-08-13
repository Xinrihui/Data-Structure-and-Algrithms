#!/usr/bin/python
# -*- coding: UTF-8 -*-

from collections import *

class solution():
    """
    
    by XRH 
    date: 2020-02-13 
    
    广度优先和 深度优先 遍历图 (基础)
    
    """


    def get_path_v1(self,prev,s,t):
        """
        prev :用来记录搜索路径。当我们从顶点 s 开始，广度优先搜索到顶点 t 后，prev 数组中存储的就是搜索的路径。不过，这个路径是反向存储的。prev[w]存储的是，顶点 w 是从哪个前驱顶点遍历过来的。比如，我们通过顶点 2 的邻接表访问到顶点 3，那 prev[3]就等于 2
        递归访问  prev 得到 路径
        :param prev: 前置数组 
        :param s: 起点
        :param t: 终点
        :return: 
        """

        path=[]

        current=t

        while current!=-1:

            path.append(current)

            if current==s:
                break

            current=prev[current]

        else:
            return []

        return path[::-1]

    def get_path_v2 (self,prev,s,t):
        """
        在 get_path_v1 基础上， 用栈实现 path[::-1]
        
        :param prev: 
        :param s: 0
        :param t: 6
        :return: 
        """

        stack=[]
        current=t

        path=[]

        while current!=-1:

            stack.append(current)

            if current==s:
                break

            current=prev[current]

        while len(stack)>0: # stack=[6,4,3,0]

            path.append(stack.pop())

        return path


    def bfs(self,graph,s,t):
        """
        广度优先搜索图
        （1）在无向图中 从 起点s 开始 搜索到 终点t，并记录搜索路径
        （2）通过前置数组 prev 记录路径，数组中的 元素 对应每一个节点的 前驱节点
         
         ref: https://time.geekbang.org/column/article/70891
         
        :param graph:    
        :param s:  
        :param t: 
        :return: 
        """
        queue=deque()
        queue.append(s)

        visited=set()
        visited.add(s) # 初始化，起点 s 需标记为 已被访问

        prev=[-1]*len(graph)

        while len(queue)>0:

            N=len(queue)

            for i in range(N):

                current=queue.popleft()
                print(current) # 访问节点

                for node in graph[current]: # 遍历 current节点 周围的子节点

                    if node == t:
                        prev[node]=current  # 记录 node 的 父亲节点为 current

                        return True,self.get_path_v1(prev,s,t)

                    if node not in visited: # 已经访问过的节点 无需再加入 queue
                        queue.append(node)
                        visited.add(node) # 加入 queue 就代表 被访问，避免 由于环路的存在而 导致 queue 存在重复元素；
                                         # eg. 在访问1节点时，会把4节点加入队列，访问3节点时，也会把4节点加入 队列，这样队列中就存在着重复的 4 节点
                        prev[node] = current

        return False,[]

    def find_xD_friends(self,graph,s,X=2):
        """
        给一个用户，如何找出这个用户的所有 X 度（其中包含一度、二度...X度）好友关系
        
        1.使用 广度优先搜索
        
        :param graph: 
        :param s: 
        :return: 
        """

        queue=deque()
        queue.append(s)

        visited=set()
        visited.add(s)  # 初始化，起点 s 需标记为 已被访问

        level=0 # 度
        all_level_friends=[] # 所有度的 好友

        while len(queue)>0:

            N=len(queue)

            level_friends=[]

            for i in range(N):

                current=queue.popleft()
                print(current) # 访问节点

                level_friends.append(current) # 加入 X度好友 集合

                for node in graph[current]: # 遍历 current节点 周围的子节点

                    if node not in visited: # 已经访问过的节点 无需再加入 queue
                        queue.append(node)
                        visited.add(node)


            print('level:',level,' friends:',level_friends)
            all_level_friends.append(level_friends)

            if level == X:
                break

            level+=1


        return all_level_friends[1:]



    def dfs(self,graph,s,t):

        """
        深度优先搜索
        起点为 s ，终点为 t 
        
        1.用栈 实现 (非递归)
        2.返回搜索的路径，并不是 s 到 t 的最短路径
        
        :param graph: 
        :param s: 
        :param t: 
        :return: 
        """

        stack=[]
        stack.append(s)
        visited=set()
        visited.add(s) # 初始化，起点 s 需标记为 已被访问

        prev = [-1] * len(graph)

        while len(stack) > 0:

            current=stack.pop()
            print(current)  # 访问节点

            for node in graph[current]:  # 遍历 current节点 周围的子节点

                if node == t:
                    prev[node] = current  # 记录 node 的 父亲节点为 current

                    return True, self.get_path_v1(prev, s, t)

                if node not in visited:  # 已经访问过的节点 无需再加入 queue
                    stack.append(node)
                    visited.add(node) # 加入 stack 就算被访问过

                    prev[node] = current

        return False,[]


    def dfs_v1(self,graph,s,t):
        """
        深度优先搜索
        起点为 s ，终点为 t 

        1.递归实现
        2.返回搜索的路径

        :param graph: 
        :param s: 
        :param t: 
        :return: 
        """
        self.graph=graph
        self.end=t

        p=s
        prev_path=[p]

        self.visited = set()
        self.visited.add(s)

        path=self._dfs_recursive(p,prev_path)

        return path

    def _dfs_recursive(self,current,prev_path):

        # print(prev_path)
        # print(current)

        if current==self.end: # 递归结束条件
            return prev_path

        for node in self.graph[current]:  # 遍历 current节点 周围的子节点

            if node not in self.visited:  # 已经访问过的节点 无需再加入 queue

                self.visited.add(node)  # 加入 stack 就算被访问过

                path=self._dfs_recursive(node,prev_path+[node])

                if path !=None:

                    return path

    def find_xD_friends_v1(self,graph,s,X=2):
        """
        给一个用户，如何找出这个用户的所有 X 度（其中包含一度、二度...X度）好友关系
        
        1.使用 深度优先搜索
        
        :param graph: 
        :param s: 
        :return: 
        """
        self.graph=graph
        self.X=X

        p=s
        level=0

        self.visited = set()
        self.visited.add(s)

        self.all_level_friends ={}  # 所有度的 好友

        for i in range(1,self.X+1):
            self.all_level_friends[i]=set() #初始化 all_level_friends

        self._dfs_recursive_v1(p,level)

        return self.all_level_friends

    def _dfs_recursive_v1(self,current,level):


        if level==self.X+1: # 递归结束条件
            return

        if level!=0:
            self.all_level_friends[level].add(current)

        for node in self.graph[current]:  # 遍历 current节点 周围的子节点

            if node not in self.visited:  # 已经访问过的节点 无需再加入 queue

                self.visited.add(node)  # 加入 stack 就算被访问过

                self._dfs_recursive_v1(node,level+1)


class DFS():
    """
    by XRH 
    date: 2020-09-29 
    
    深度优先 遍历图 (高级)
    
    有向图 无向图 均适用
    
    1.深度优先遍历 并 输出节点的 发现时间 和 完成时间
    
    2. 输出 DFS 搜索树 (树边)
    
    3. 找到 图中的 前向边 反向边 和 交叉边
    
    4. 判断 图中是否存在环路
    
    5. 输出 图中的 所有的环路 
       
       (1) 有向图 
       (2) 无向图
        

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

        # self.visited.add(current) # 节点被发现

        for node in G[current]:  # Explore neighbors

            if node not in self.visited:

                self.visited.add(node)

                t = self.__dfs_timestamp_process(G, node, t)  # t是一个变量，带入函数的时候值被复制过来，因此在其他递归函数的修改将失效

        t += 1  # Set finish time
        self.finish_time[current] = t

        return t


    def dfs_timestamp_parent(self, G):
        """
        非连通图的 DFS

        (1) 每一个节点都要 作为 DFS 的入口
        (2) 返回 每一个点的 开始时间 和 结束时间
        (3) 返回 每一个点的 父亲节点
        (4) 记录 节点的颜色 

        :param G: 
        :return: 
        """

        self.discover_time = dict()  # 顶点的发现时间
        self.finish_time = dict()  # 顶点的 完成时间

        self.colour = dict()  # 顶点的 颜色

        self.parent=dict() # 记录节点的 父亲

        for node in G.keys():
            self.parent[node] = None
            self.colour[node] = 'white' #初始化 都是白色节点 代表 节点未被发现

        nodes_list = G.keys()

        t = 0

        for node in nodes_list:

            if self.colour[node] == 'white':

                t = self.__dfs_timestamp_parent_process(G, node, t)

        return self.discover_time, self.finish_time , self.parent


    def __dfs_timestamp_parent_process(self, G, current, t=0):
        """
        基于递归 的 DFS
        给节点加上了时间戳

        :param G: 图
        :param current: 当前顶点
        :param t: 时间点 
        :return: 
        """

        self.colour[current] = 'gray' # 节点被发现 置为 灰色

        t += 1  # Set discover time
        self.discover_time[current] = t


        for node in G[current]:  # Explore neighbors

            if self.colour[node] == 'white':
                self.parent[node] = current
                t = self.__dfs_timestamp_parent_process(G, node, t)

        t += 1  # Set finish time

        self.finish_time[current] = t

        self.colour[current] = 'black' # 从current节点 出发的边都已被探索, 置为黑色

        return t


    def dfs_tree(self, G, directed=True):
        """
        非连通图的 DFS 

        (1) 每一个节点都要 作为 DFS 的入口
        (2) 返回  从每一个节点 进入图 并进行遍历产生的 深度优先树 
        (3) 对图中所有的边 进行分类, 包括 树边 前向边 后向边 和 交叉边
        (4) 是否有 反向边 可以判断 图中是否存在环路
        
        :param G: 
        :param directed:  有向图 标记位, 默认为 有向图
        :return: 
        """

        self.discover_time = dict()  # 顶点的发现时间
        self.finish_time = dict()  # 顶点的 完成时间

        self.colour = dict()  # 顶点的 颜色

        self.parent=dict() # 记录节点的 父亲

        self.forest= dict() # 深度优先树 组成的森林

        self.tree_edges=[] #树边
        self.backward_edges=[] # 后向边

        # self.forward_cross_edges=[] # 前向边或交叉边

        self.forward_edges=[] # 前向边
        self.cross_edges = []  # 交叉边


        for node in G.keys():

            self.parent[node] = None
            self.colour[node] = 'white' #初始化 都是白色节点 代表 节点未被发现

        nodes_list = G.keys()

        t = 0

        for node in nodes_list:

            if self.colour[node] == 'white':

                root=node
                self.forest[root] = defaultdict(set)  # 以 root 作为根节点的 DFS树 组成了 森林

                if directed == True: # 有向图

                    t = self.__dfs_tree_process(G, node, root,t)

                else:
                    self.visited_edge=set()

                    t = self.__dfs_tree_undirected_process(G, node, root, t)


        print("discover_time:{}".format(self.discover_time))
        print("finish_time:{}".format(self.finish_time))

        if len(self.backward_edges)>0:
            print("circle nums >= {}".format(len(self.backward_edges)) ) # 输出 图中的环的个数

        print("tree_edges:{}".format(self.tree_edges))
        print("backward_edges:{}".format(self.backward_edges))
        print("forward_edges:{}".format(self.forward_edges))
        print("cross_edges:{}".format(self.cross_edges))

        return  self.forest,self.backward_edges


    def __dfs_tree_process(self, G, current, root,t=0):
        """
        基于递归 的 DFS 
        
        适用于 有向图
        
        :param G: 图
        :param current: 当前顶点
        :param root: 深度优先 搜索树的 根节点 
        :param t: 时间点 
        
        :return: 
        """

        self.colour[current] = 'gray' # 节点被发现 置为 灰色

        t += 1  # Set discover time
        self.discover_time[current] = t


        for node in G[current]:  # Explore neighbors

            if self.colour[node] == 'white':

                self.parent[node] = current

                self.forest[root][current].add(node) # current -> node 是以root 为根节点的 DFS树的边

                self.tree_edges.append((current,node)) # 树边

                t = self.__dfs_tree_process(G, node,root, t)

            elif self.colour[node] == 'gray':

                self.backward_edges.append((current,node))

            elif self.colour[node] == 'black':

                # self.forward_cross_edges.append((current, node))

                if self.discover_time[current] < self.discover_time[node]:

                    self.forward_edges.append((current, node))

                else:

                    self.cross_edges.append((current, node))


        t += 1  # Set finish time

        self.finish_time[current] = t

        self.colour[current] = 'black' # 从current节点 出发的边都已被探索, 置为黑色

        return t

    def __dfs_tree_undirected_process(self, G, current, root, t=0):
        """
        基于递归 的 DFS
        
        适用于 无向图
        
        保证每条边 只访问一次
        
        :param G: 图
        :param current: 当前顶点
        :param root: 深度优先 搜索树的 根节点 
        :param t: 时间点 

        :return: 
        """

        self.colour[current] = 'gray'  # 节点被发现 置为 灰色

        t += 1  # Set discover time
        self.discover_time[current] = t

        for node in G[current]:  # 访问边

            # 重复的边 不要访问
            if (node,current) not in self.visited_edge: # 当前的边为 current -> node 则 上一次访问过的边为 node->current

                if self.colour[node] == 'white':

                    self.parent[node] = current

                    self.forest[root][current].add(node)  # current -> node 是以root 为根节点的 DFS树的边

                    self.tree_edges.append((current, node))  # 树边

                    self.visited_edge.add((current,node))

                    t = self.__dfs_tree_undirected_process(G, node, root, t) # TODO: 记得改 函数名

                elif self.colour[node] == 'gray':

                    self.backward_edges.append((current, node))

                    self.visited_edge.add((current, node))

                elif self.colour[node] == 'black':

                    # self.forward_cross_edges.append((current, node))

                    if self.discover_time[current] < self.discover_time[node]:

                        self.forward_edges.append((current, node))

                    else:

                        self.cross_edges.append((current, node))

                    self.visited_edge.add((current, node))


        t += 1  # Set finish time

        self.finish_time[current] = t

        self.colour[current] = 'black'  # 从current节点 出发的边都已被探索, 置为黑色

        return t


    def get_circles_DirectedGraph(self,G):
        """
        输出 有向图中 所有的环
        
        :param G: 
        :return: 
        """

        # 1. 找出 图中的 反向边
        _,backward_edges=self.dfs_tree(G)

        res_circles=[]

        for reverse_edge in backward_edges:

            u=reverse_edge[0]
            v=reverse_edge[1]

            # 2. 依据 反向边 (u,v) 找出 所有的 v->u 的路径
            path_list=self.find_all_path(G,v,u)

            # 3.  v->u 的路径 拼接上 (u,v) 组成 最终的环路

            circle_paths=[]
            for path in path_list:

                circle_paths.append(path+[v])

            res_circles.append(circle_paths)


        return res_circles


    def find_all_path(self,G,start_node,end_node):
        """
        回溯法 
        
        找出 图中 start_node -> end_node 的所有路径
        
        :param start_node: 
        :param end_node: 
        :return: 
        """

        self.path_list=[]

        self.end_node=end_node

        path=[start_node]
        current_node=start_node

        self.__find_all_path_process(G,current_node,path)

        return self.path_list

    def __find_all_path_process(self,G,current_node,path):

        if current_node == self.end_node:

            # print(path)

            self.path_list.append(path)

            return

        else:

            for node in G[current_node]:

                if node not in path:

                    self.__find_all_path_process(G,node,path+[node])


    def get_circles_UndirectedGraph(self, G):
        """
        输出 无向图中 所有的环

        :param G: 
        :return: 
        """

        # 1. 得到 图中的 DFS搜索树 和 反向边
        forest,backward_edges=self.dfs_tree(G,directed=False)

        print(forest)

        res_circles=[]

        for reverse_edge in backward_edges:

            u=reverse_edge[0]
            v=reverse_edge[1]

            # 2. 依据 反向边 (u,v)  在 DFS搜索树 中找到 一条 v -> u 的路径
            path=self.find_path(forest,v,u)

            # 3.  v->u 的路径 拼接上 (u,v) 组成 最终的环路

            res_circles.append(path+[v])


        return res_circles

    def find_path(self,forest,v,u):
        """
        从 DFS搜索树 组成的 森林中 找到 一条 v->u 的路径
        
        :param forest: 
        :param v: 
        :param u: 
        :return: 
        """
        for root in forest.keys():

            if v in forest[root].keys(): # 判断 v 节点在 那一颗 DFS 树中

                DFS_tree=forest[root]

                current_node=v
                end_node=u

                path=[current_node]

                res_path=self.__find_path_process(DFS_tree,current_node,end_node,path)

                return res_path


    def __find_path_process(self,G,current_node,end_node,path):

        if current_node == end_node:

            # print(path)

            return path

        else:

            for node in G[current_node]:

                if node not in path:

                    res_path=self.__find_path_process(G,node,end_node,path+[node])

                    if res_path!=None:

                        return res_path



if __name__ == '__main__':

    sol=solution()

    G = {
        1: set([0, 2, 4]),
        0: set([1, 3]),
        2: set([1,5]),
        3: set([0,4]),
        4: set([1,3,5,6]),
        5: set([2,4,7]),
        6: set([4,7]),
        7: set([5,6])
        }

    # print(sol.bfs(G,0,6))

    # print(sol.bfs(G, 0, 2))

    # print(sol.dfs(G, 0, 2))

    # print(sol.dfs_v1(G, 0, 2))

    # print(sol.dfs_v1(G, 0, 3))

    # print(sol.find_xD_friends(G,0,2))

    # print(sol.find_xD_friends_v1(G,0,2))


    dfs=DFS()

    G = {
        'u': set(['v', 'x']),
        'v': set(['y']),
        'x': set(['v']),
        'y': set(['x']),
        'w': set(['y','z']),
        'z':set(['z'])
        }

    # print(dfs.dfs_timestamp(G))

    # print(dfs.dfs_timestamp_parent(G))

    # print(dfs.dfs_tree(G))

    G = {
        's': set(['w', 'z']),
        'z': set(['y','w']),
        'y': set(['x']),
        'x': set(['z']),
        'w':set(['x']),
        't': set(['v','u']),
        'v': set(['s', 'w']),
        'u': set(['t']),

    }

    # print(dfs.dfs_tree(G))

    """
output[0]:

discover_time:{'s': 1, 'z': 2, 'y': 3, 'x': 4, 'w': 7, 't': 11, 'v': 12, 'u': 14}
finish_time:{'x': 5, 'y': 6, 'w': 8, 'z': 9, 's': 10, 'v': 13, 'u': 15, 't': 16}

circle nums >= 2
backward_edges:[('x', 'z'), ('u', 't')]
forward_edges:[('s', 'w')]
cross_edges:[('w', 'x'), ('v', 's'), ('v', 'w')]
{'s': defaultdict(<class 'set'>, {'s': {'z'}, 'z': {'y', 'w'}, 'y': {'x'}}), 't': defaultdict(<class 'set'>, {'t': {'v', 'u'}})}
    
    """

    # print(dfs.get_circles_DirectedGraph(G))

    """
    [
    [['z', 'w', 'x', 'z'], ['z', 'y', 'x', 'z']],
    [['t', 'u', 't']]
    ]    
    """

    undirected_G={ # 枚举出所有的 顶点

        '0': set(['1']),
        '1': set(['0','2','3']),
        '2': set(['1','5','4']),
        '3': set(['1','4']),
        '4':set(['2','3']),
        '5': set(['2']),
        '6': set(['7']),
        '7': set(['6'])

    }

    print(dfs.get_circles_UndirectedGraph(undirected_G))















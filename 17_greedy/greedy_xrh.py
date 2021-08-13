#!/usr/bin/python
# -*- coding: UTF-8 -*-

from numpy import *
import heapq

class solutions:

    def childs_with_sugers(self,childs,sugers):
        """
        分糖给小朋友，一个小朋友只能拿一块糖，糖不能分割
        :param childs: 
        :param sugers: 
        :return: 
        """
        childs=sorted(childs)
        sugers=sorted(sugers)

        res=[]
        # if len(childs) > len(sugers) :
        j=0
        for i,child in enumerate(childs):

           while j < len(sugers):
               if sugers[j] >= child:
                    res.append([child,sugers[j]])
                    j+=1
                    break
               j=j+1
           else: # 正常结束 while 循环 则运行下面代码
               print('suger is not enough') # 糖已经分配完了，有的小朋友没有糖吃
               break # 跳出for 循环

        return res

    def regions_overlap(self,regions,L):
        """
        区间覆盖问题 
        
        给定一个长度为 m的区间，再给出 n条线段的起点和终点（注意这里是闭区间），
        求最少使用多少条线段可以将整个区间完全覆盖。
        
        ref: https://www.cnblogs.com/acgoto/p/9824723.html
        :param regions: [ [2,6],[1,4],[3,6],[3,7],[6,8],[2,4],[3,5] ]
        :param L:  8 
        :return: [ [1,4] ,[3,7],[6,8] ]
        """
        regions=sorted(regions, key=lambda d: d[0]) # 按照区间的 左端点进行排序

        # print(regions)

        right_most=1
        res=[]

        while right_most<L:

            left_small=list(filter(lambda x: x[0] <= right_most, regions)) # 过滤出左端点 小于 right_most 的区间

            right_max=max(left_small,key=lambda x:x[1]) # 选这些区间 中 右端点最大的一个

            res.append(right_max)

            right_most=right_max[1] #更新 已覆盖线段的 右端点

        return res

    def max_regions_not_intersect(self, regions, L):
        """
        最多 不相交区间（活动选择问题）
        
        假设我们有 n 个区间，区间的起始端点和结束端点分别是[l1, r1]，[l2, r2]，[l3, r3]，……，[ln, rn]。我们从这 n 个区间中选出一部分区间，
        这部分区间满足两两不相交（端点相交的情况不算相交），最多能选出多少个区间呢？
        
        ref: https://time.geekbang.org/column/article/73188
        :param regions: [[6,8],[2,4],[3,5],[1,5],[5,9],[8,10]]
        :param L: 10
        :return: [[2,4],[6,8],[8,10]]
        """
        regions = sorted(regions, key=lambda d: d[0])  # 按照区间的 左端点进行排序
        # print(regions)

        right_most=0
        res=[]

        while right_most<L:

            left_small=list(filter(lambda x: x[0] >= right_most, regions)) # 过滤出左端点 大于 right_most 的区间 ，这样能避免重合

            right_max=min(left_small,key=lambda x:x[1]) # 选这些区间 中 右端点最小的一个，这样 能留出更多的剩余空间

            res.append(right_max)

            right_most=right_max[1] #更新 已覆盖线段的 右端点

        return res

    def activity_selection(self, regions):

        """
        活动选择问题

        假设我们有 n 个活动，活动的 开始端点和结束时间 分别是[l1, r1]，[l2, r2]，[l3, r3]，……，[ln, rn]
        我们从这 n 个 活动中选出一部分活动，
        这部分活动的时间不冲突，即 满足 两两不相交（端点相交的情况不算相交），最多能选出多少个活动呢？
        
        贪心策略： 结束时间 早的 活动 优先 
        
        :param regions: [[6,8],[2,4],[3,5],[1,5],[5,9],[8,10]]
        
        :return: [[2,4],[6,8],[8,10]]
        """
        regions = sorted(regions, key=lambda d: d[1])  # 按照 结束时间 对活动 进行排序

        selected=[] #被选中的活动

        next_earliest_start=0 # 保证与上一个活动不冲突的 情况下，下一个活动的最早开始时间

        for region in regions:

            start=region[0]
            end=region[1]

            if start >= next_earliest_start: # 活动的开始时间 满足 要求

                selected.append(region)
                next_earliest_start=end


        return selected

    def minimum_delay_scheduling(self, duration, deadline):
        """
        最小延迟 调度 问题
        
        n 项任务，每一项任务 消耗的时间为 duration，每一项任务的 截止时间为 deadline
        任务超过截止时间才完成会产生延迟
        
        求 所有任务中  发生最大延迟的任务 所产生的延迟 达到最小的 调度策略，在这一策略下的 最大延迟
        
        贪心策略： 截止时间 早的任务优先 
        
        :param duration: [5,8,4,10,3]
        :param deadline: [10,12,15,11,20]
        :return: 
        """

        N=len(duration) # 任务个数

        deadline=array(deadline)

        jobs_deadline= [ (i,deadline[i]) for i in range(N)] # (任务标号, 任务截止时间)

        jobs_deadline = sorted(jobs_deadline, key=lambda ele: ele[1])

        jobs_delay = zeros(N, dtype=int)  # 记录 各个任务的 延迟时间

        current_time=0 # 记录 当前的时间点

        for job in jobs_deadline:

            job_NO=job[0]
            job_deadline=job[1]

            current_time+=duration[job_NO]

            if current_time >  job_deadline: # 时间超过了 deadline

                jobs_delay[job_NO]= current_time-job_deadline


        print('jobs_delay:',jobs_delay)
        max_delay=max(jobs_delay)

        return max_delay


    def bulk_bag_problem(self, weights,values, capacity):
        """
        散装背包问题 (背包问题变形)
        
        与普通背包问题 的关键区别：每件物品均可分解
        
        一个可以容纳 100kg 物品的背包， 有 5 种豆子，每种豆子的 总量 和 总价值都各不相同。
        为了让背包中所装物品的总价值最大，我们如何选择在背包中装哪些豆子？每种豆子又该装多少呢
        
        采用贪心策略：尽可能多放 单位重量价值 最大的 物品
        
        :param weights: [100,30,60,20,50]
        :param values: [100,90,120,80,75]
        :param capacity: 100
        :return: 
        """
        N=len(weights) # 物品种类

        weights=array(weights)
        values=array(values)

        unit_weight_value= values/weights

        unit_weight_value= [ (i,unit_weight_value[i]) for i in range(N)] # (标号,物品的 单位重量的价值)

        unit_weight_value= sorted(unit_weight_value,key=lambda ele:ele[1],reverse=True)
        #按照 物品单位重量的价值 逆序排序 [(3, 4.0), (1, 3.0), (2, 2.0), (4, 1.5), (0, 1.0)]

        bag_weight=capacity
        bag_value=0

        bag_items= zeros(N, dtype=float) # 背包中 每一样物品的 重量

        for item in unit_weight_value:

            if bag_weight <=0:
                break

            item_NO=item[0]

            if bag_weight >= weights[item_NO]: # 背包 容量 足以装下全部的 item_NO 物品

                bag_weight -= weights[item_NO] #
                bag_value += values[item_NO]

                bag_items[item_NO]=weights[item_NO]

            else: # 背包 容量 不足

                bag_items[item_NO]=bag_weight # 剩下的背包容量 全部装 item_NO 物品
                bag_weight = 0 #
                bag_value += bag_weight*item[1]


        return bag_value,bag_items

    def optimal_loading(self, weights, capacity):
        """
        
        最优装载问题 (背包问题变形)
        
        即 物品的价值都为 1 的 01 背包问题
        
        n 个集装箱 重量为 weights, 船的载重能力限制为 capacity, 每一个集装箱 都小于 capacity, 
        如何 选择 使得 可以装载更多 的集装箱
        
        采用贪心策略：轻的物品优先装入
        
        :param weights: [100,30,60,20,50]
        :param capacity: 100 
        :return: 
        """

        N=len(weights) # 物品种类

        weights=array(weights)

        items_weight= [ (i,weights[i]) for i in range(N)] # (标号,物品的重量)

        items_weight = sorted(items_weight, key=lambda ele: ele[1])

        bag_weight=capacity

        bag_items= [] # 记录 放入背包的物品

        for item in items_weight:

            if bag_weight <=0:
                break

            item_NO=item[0]

            if bag_weight >= item[1] : # 背包 容量 能装下 item_NO 物品

                bag_weight -= weights[item_NO]
                bag_items.append(item_NO)


        return bag_items





class ComapreHeap(object):
    def __init__(self, initial=None, key=lambda x: x):
        self.key = key
        if initial:
            self._data = [(key(item), item) for item in initial]
            heapq.heapify(self._data)
        else:
            self._data = []

    def push(self, item):
        heapq.heappush(self._data, (self.key(item), item))

    def pop(self):
        return heapq.heappop(self._data)[1]

class TreeNode(object):
    def __init__(self,key=None,value=None):
        self.key=key
        self.value=value
        self.left=None
        self.right=None

class huffman_tree:
    """
    霍夫曼 前缀编码
    
    使用 {0,1} 按照字符出现的 频率，并根据贪心策略 生成二叉编码树对字符集进行前缀编码
    
     ref: 
    （1）《算法导论》
    （2）https://time.geekbang.org/column/article/73188
    
    """

    def __init__(self, char_list):

        self.huffman_encode_tree=self.__encode(char_list)

        self.__decode_all()

    def encode(self, string):
        """
        将字符串 编码为 以 {0,1} 表示的 字节流 
        :param string: 'acbf'
        :return: '01001011100'
        """
        bytes=[]

        for char in string:
            bytes.append(self.char_dict[char])

        return ''.join(bytes)

    def decode(self,bytes):
        """
        解码一段 以 {0,1} 表示的 字节流 
        :param bytes: '01001011100'
        :return: 'acbf'
        """
        root = self.huffman_encode_tree
        p=root

        res_string=[]

        for byte in bytes:

            if byte=='0': # 走左子树
                p=p.left

            else: # 走右子树
                p=p.right

            #TODO : 解码失败的处理

            if p.left==None and p.right==None: # 走到叶子节点了
                res_string.append(p.key)
                p=root

        return ''.join(res_string)

    def __decode_all(self):
        """
        返回 霍夫曼 编码树 上 char_list 中所有字符 和其 对应的 编码
        :return: char_dict= {'a': '0', 'c': '100', 'b': '101', 'f': '1100', 'e': '1101', 'd': '111'} 
        """
        root=self.huffman_encode_tree
        self.char_dict={}

        self.__tree_pre_order(root,[])

        for char,char_bytes in self.char_dict.items():

            self.char_dict[char]=''.join(str(e) for e in char_bytes)


        return self.char_dict

    def __tree_pre_order(self,root,pre_list):

        if root.left== None and root.right == None: # 说明到达叶子节点
            self.char_dict[root.key]=pre_list

        else:
           if root.left !=None:
               self.__tree_pre_order(root.left,pre_list+[0])
           if root.right!=None:
               self.__tree_pre_order(root.right, pre_list + [1])


    def __encode(self,char_list):
        """
        
        将 char_list 中的字符，生成一颗 huffman 编码树 
        
        :param char_list: [('a',45),('b',13),('c',12),('d',16),('e',9),('f',5)]
        :return: 
        """

        leaf_nodes= [TreeNode(ele[0] ,ele[1]) for ele in char_list ]

        heap_nodes=ComapreHeap(leaf_nodes,key=lambda x:x.value)

        N=len(char_list)

        root_node=None

        for i in range(N-1): # N 为叶节点个数，要执行N-1次的 叶节点的合并操作

            root_node=TreeNode()

            left_node=heap_nodes.pop()
            right_node=heap_nodes.pop()

            root_node.key='s'+str(i) # 非叶子节点的 Key
            root_node.value=left_node.value+ right_node.value

            root_node.left=left_node
            root_node.right=right_node

            # print('root:',root_node.value)
            # print('root.left:', root_node.left.value)
            # print('root.right:', root_node.right.value)

            heap_nodes.push(root_node)


        return root_node



if __name__ == '__main__':

    sol = solutions()
    childs=[3,4,5,6,7,8] # 小孩0-小孩5 想要的糖果的 重量
    sugers=[1,2,3,4,5] # 现有的各个糖果的 重量

    # print(sol.childs_with_sugers(childs,sugers))

    regions=[ [2,6],[1,4],[3,6],[3,7],[6,8],[2,4],[3,5] ]
    L=8
    # print(sol.regions_overlap(regions,L))

    regions=[[6,8],[2,4],[3,5],[1,5],[5,9],[8,10]]
    L=10
    # print(sol.max_regions_not_intersect(regions, L))
    # print(sol.activity_selection(regions))

    duration= [5, 8, 4, 10, 3]
    deadline= [10, 12, 15, 11, 20]
    # print(sol.minimum_delay_scheduling(duration,deadline))


    weights= [100, 30, 60, 20, 50]
    values= [100, 90, 120, 80, 75]
    capacity= 100
    # print(sol.bulk_bag_problem(weights,values,capacity))

    # print(sol.optimal_loading(weights,capacity))

    char_list=[('a',45),('b',13),('c',12),('d',16),('e',9),('f',5)]
    huffman_tree=huffman_tree(char_list)
    print('char_dict:',huffman_tree.char_dict)

    bytes= '01001011100'
    print(huffman_tree.decode(bytes))
    print(huffman_tree.encode('acbf'))





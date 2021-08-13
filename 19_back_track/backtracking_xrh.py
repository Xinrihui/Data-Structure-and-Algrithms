#!/usr/bin/python
# -*- coding: UTF-8 -*-

import itertools
from numpy import *

def test_CombinationAndPermutation():
    """
    利用 itertools 实现 排列 和 组合
    ref https://www.cnblogs.com/xiao-apple36/p/10861830.html
    :return: 
    """

    #1.组合
    for i in itertools.combinations('ABC', 1):
        print(i)

    for i in itertools.combinations('ABC', 2):
        print(i)
    # 输出 AB AC BC

    for i in itertools.combinations('ABC', 3):
        print(i)
    # 输出  ABC


    #2.排列
    for i in itertools.permutations('ABC', 2):
        print(''.join(i), end=",")
    # 输出 BC BD CB CD DB DC
    print('\r')


    #3. 笛卡尔积
    a = (1, 2)
    b = ('A', 'B', 'C')
    c = itertools.product(a, b)
    for i in c:
        print(i)

    for i in itertools.product('ABC', repeat=2): # a='ABC' b='ABC' a与b 做笛卡尔积
        print(''.join(i), end=",")
    print('\n')

class CombinationAndPermutation:
    """
    纯手工 实现 排列 与 组合 
    ref: https://docs.python.org/zh-cn/3.7/library/itertools.html
    
    :return: 
    """
    # 1. 笛卡尔积
    @staticmethod
    def product(*args, repeat=1):
        """
        笛卡尔积
        eg.
         product('ABCD', 'xy') --> Ax Ay Bx By Cx Cy Dx Dy
         product(range(2), repeat=3) --> 000 001 010 011 100 101 110 111        
        :param args: 
        :param repeat: 
        :return: 
        """

        pools = [tuple(pool) for pool in args] * repeat # [('A', 'B', 'C'), ('A', 'B', 'C')]
        result = [[]]

        # for pool in pools:
        #     result = [x + [y] for x in result for y in pool]

        for pool in pools:
            res=[]
            for y in pool:
                for x in result:
                    res.append(x + [y])
            # print(res) #  [['A'], ['B'], ['C']] ;
                         #  [['A', 'A'], ['B', 'A'], ['C', 'A'], ['A', 'B'], ['B', 'B'], ['C', 'B'], ['A', 'C'], ['B', 'C'], ['C', 'C']]
            result=res


        for prod in result:
            yield tuple(prod)

    # for ele in product('ABC', repeat=2):
    #     print(ele)

    # 2. 排列
    @staticmethod
    def permutations(iterable, r=None):
        """
        排列
        permutations() 可被改写为 product() 的子序列，
        只要将含有重复元素（来自输入中同一位置的）的项排除。
        :param iterable: 
        :param r: 
        :return: 
        """
        pool = tuple(iterable)
        n = len(pool)
        r = n if r is None else r
        for indices in product(range(n), repeat=r):

            if len(set(indices)) == r: # len(set( ('A','A')))=1
                yield tuple(pool[i] for i in indices)

    # for ele in permutations('ABC', 3):
    #     print(ele)


    # 3. 组合
    @staticmethod
    def combinations(iterable, r):
        """
        组合
        
        combinations() 被改写为 permutations() 过滤后的子序列，（相对于元素在输入中的位置）元素不是有序的。
        :param iterable: 
        :param r: 
        :return: 
        """
        pool = tuple(iterable)
        n = len(pool)
        for indices in CombinationAndPermutation.permutations(range(n), r):

            if sorted(indices) == list(indices):
                yield tuple(pool[i] for i in indices)

    @staticmethod
    def combinations_v2(iterable, r):
        """
        组合
        
        eg.
        combinations('ABCD', 2) --> AB AC AD BC BD CD
        combinations(range(4), 3) --> 012 013 023 123
        
        :param iterable: 
        :param r: 
        :return: 
        """

        pool = tuple(iterable) # ('A', 'B', 'C')
        n = len(pool) # n=3
        if r > n: # r=2
            return
        indices = list(range(r)) # indices=[0,1]

        yield tuple(pool[i] for i in indices) # (pool[0],pol[1])

        while True:
            for i in reversed(range(r)): # i=1
                                         # i=0 ;

                                         # i=1
                if indices[i] != i + n - r: # i + n - r: 1+3-2 =2 ;
                                            #            0+3-2=1
                    break
            else:
                return
            indices[i] += 1 # indices=[0,2] ;
                            # indices=[1,2] ;
                            #
            for j in range(i + 1, r):
                indices[j] = indices[j - 1] + 1
            yield tuple(pool[i] for i in indices) # (pool[0],pol[2]) ;
                                                  # (pool[1],pol[2])

    # for ele in combinations('ABC',2):
    #     print(ele) #  ('A', 'B')  ('A', 'C') ('B', 'C')



from functools import reduce

def test_reduce():
    """
   reduce 函数的使用
   reduce把一个函数作用在一个序列[x1, x2, x3, ...]上，这个函数必须接收两个参数，reduce把结果继续和序列的下一个元素做累积计算，其效果就是：
   reduce(f, [x1, x2, x3, x4]) = f(f(f(x1, x2), x3), x4)
   
   ref: https://www.liaoxuefeng.com/wiki/1016959663602400/1017329367486080
    :return: 
    """

    def add(x, y):
        return x + y

    print(reduce(add, [1, 3, 5, 7, 9]))

    def fn(x, y):
        return x * 10 + y

    print(reduce(fn, [1, 3, 5, 7, 9])) #把序列[1, 3, 5, 7, 9]变换成整数 13579


    DIGITS = {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9}

    def char2num(s):
        return DIGITS[s]

    def str2int(s):
        return reduce(lambda x, y: x * 10 + y, map(char2num, s))

    print(str2int('123'))


class solution_N_quene:
    """
    by XRH 
    date: 2020-08-23 
    
    国际象棋中的 N 皇后 问题
    
    """

    def N_quene(self,N):

        self.N=N
        self.states= zeros((N,N),dtype=int) # N X N 的棋盘

        self.res=[]

        prev_list=[]
        level=0

        #  初始情况 level=0 , 第一次放皇后(放在 第0行), 不需要考虑 和之前的皇后冲突
        for j in range(self.N):

            self.states[level][j]=1
            self._process(level+1,prev_list+[j]) # 一行一行 往下放皇后
            self.states[level][j] = 0

        return self.res

    def _check_legality(self,level,current):
        """
        检查 当前皇后的摆放位置 是否 和之前的皇后 冲突
        :param level: 
        :param current: 
        :return: 
        """

        # 1.向上检查 同一列
        i=level-1
        while i>=0:

            if self.states[i][current]==1: # 发现冲突

                return False

            i-=1

        # 2. 向上检查 45度的 左斜线
        i = level - 1
        j=current -1
        while i >= 0 and j >= 0:

            if self.states[i][j]==1: # 发现冲突

                return False

            i-=1
            j-=1

        # 3. 向上检查 45度的 右斜线
        i = level - 1
        j=current + 1
        while i >= 0 and j < self.N:

            if self.states[i][j]==1: # 发现冲突

                return False

            i-=1
            j+=1

        return True

    def _process(self,level,prev_list):

        if level==self.N: # 递归结束条件:
                          # 遍历到搜索树的叶子节点,找到可行解
                          # 搜索树的最后 一行 self.N-1 摆上皇后, level+1 = self.N

            # print(prev_list)

            self.res.append(prev_list)

        elif level< self.N:

            for j in range(self.N):

                if self._check_legality(level,j):

                    self.states[level][j] = 1 # 皇后摆在 棋盘上的 [level][j] 上
                    self._process(level+1,prev_list+[j])
                    self.states[level][j] = 0 # 把皇后 从棋盘上的 [level][j] 拿下来



class solution_zero_one_bag_weight:

    """
    01 背包问题 (只有 背包重量 , 没有背包价值)
    
    """

    def zero_one_bag_weight_iteration(self,weights, capacity):
        """
        迭代的方法解 01 背包问题
        
        :param weights: 
        :param capacity: 
        :return: 
        """

        L = len(weights)

        max_bag_weight = 0

        res_bag = ()

        for l in range(1, L + 1):
            # 遍历 背包的 中的物品件数 l=1, l=2, l=3

            for bag in itertools.combinations(weights, l):  # 背包中的物品件数 为l 时，列举所有可能的物品的组合

                bag_weight = sum(map(lambda x: x[1], bag))  # 背包中 所有物品的重量求和
                # print('bag:', bag, ' weight: ', bag_weight)

                if bag_weight > max_bag_weight and bag_weight <= capacity:
                    max_bag_weight = bag_weight
                    res_bag = bag

        return max_bag_weight, res_bag

    def zero_one_bag_weight(self,weights,capacity):
        self.weights=weights
        self.capacity=capacity

        self.max_bag_weight=0
        self.res_bag=()

        current_bag=[]
        self.__process(0,current_bag) #current_bag 表示当前已经装进去的物品；i表示考察到哪个物品了；

        return self.max_bag_weight,self.res_bag

    def __process(self,i,current_bag):

        if i <= len(self.weights):

            bag_weight = sum(map(lambda x: x[1], current_bag))

            if bag_weight > self.capacity: #搜索剪枝: 当发现已经选择的物品的重量超过 Wkg 之后，我们就停止继续探测剩下的物品
                return

            print('bag:', current_bag, ' weight: ', bag_weight)

            if bag_weight >self.max_bag_weight:
                self.max_bag_weight=bag_weight
                self.res_bag = current_bag

            self.__process(i+1,current_bag) # 第i 个物品不放入背包

            if i < len(self.weights):
                self.__process(i + 1, current_bag+[self.weights[i]]) # 第i 个物品 放入背包


class solution_zero_one_bag_value:
    """
    by XRH 
    date: 2020-08-23 
    
    01 背包问题 , 物品带价值, 求背包在满足重量的条件下, 背包总价值最大的物品放置策略

    """

    def zero_one_bag_value(self, weights, values, capacity):
        """
        01 背包问题的 回溯求解
        
        建立搜索空间树, 搜索树的节点信息为: 是否放入第 i 个 物品 (物品id )
        
        搜索树叶子节点 代表找到一个 可行解
        
        物品的数量 N , 算法的时间复杂度 O(2^N)
        
        :param weights: 
        :param values: 
        :param capacity: 
        :return: 
        """

        self.N=len(weights) # N 个物品

        self.weights=weights
        self.values=values
        self.capacity=capacity

        item_NO=0

        bag_weight=0
        bag_value=0

        prev_items=[]

        self.max_value=float('-inf')
        self.max_value_bag_item=[]

        self._process(item_NO,bag_weight,bag_value,prev_items) # 考虑 第0个物品 是否装入背包

        return self.max_value_bag_item


    def _process(self, item_NO, bag_weight, bag_value, prev_items):

        if item_NO == self.N:  # 递归结束条件: 最后一个 物品已考察完

            if bag_value > self.max_value:  # 找到了 更大价值的背包

                self.max_value = bag_value
                self.max_value_bag_item = prev_items

        if item_NO < self.N:

            # case1. 装入 item_NO 物品
            if bag_weight + self.weights[item_NO] <= self.capacity:  # 背包的重量不能超(搜索剪枝)

                self._process(item_NO + 1, bag_weight + self.weights[item_NO],
                              bag_value + self.values[item_NO], prev_items + [item_NO])

            # case2. 不装入 item_NO 物品
            self._process(item_NO + 1, bag_weight,
                          bag_value, prev_items)


    def zero_one_bag_value_cache(self, weights, values, capacity):
        """
        01 背包问题的 回溯求解 (带备忘录)
        
        1. 在朴素的搜索空间树的节点加上: 背包的重量 和 背包的价值 的状态 , 并利用这两个状态对搜索空间进行裁剪
        2. 使用备忘录 记录 搜索树的节点信息 (物品id, 背包重量, 背包价值) , 若遇到重复的子树, 直接找备忘录即可
        
        3. 把搜索树的节点信息 单独抽取出来 作为 动态规划中的 子问题的状态矩阵, 并写出 递推方程 即可得 动态规划 解法
              
        详见 《数据结构与算法之美》 -> 基础算法-动态规划
        
        :param weights: 
        :param values: 
        :param capacity: 
        :return: 
        """
        self.N = len(weights)  # N 个物品

        self.weights = weights
        self.values = values
        self.capacity = capacity

        self.states = zeros((self.N, capacity + 1), dtype=int)

        item_NO = 0

        bag_weight = 0
        bag_value = 0

        prev_items = []

        self.max_value = float('-inf')
        self.max_value_bag_item = []

        self._process_cache(item_NO, bag_weight, bag_value, prev_items)  # 考虑 第0个物品 是否装入背包

        return self.max_value_bag_item


    def _process_cache(self,item_NO,bag_weight,bag_value,prev_items):

        # print('item:{}, bag_weight:{}, bag_value:{}'.format(item_NO, bag_weight, bag_value))

        if item_NO == self.N: #递归结束条件: 最后一个 物品已考察完

            if bag_value> self.max_value: # 找到了 更大价值的背包

                self.max_value=bag_value
                self.max_value_bag_item=prev_items

        if item_NO < self.N:

            # case1. 装入 item_NO 物品
            after_bag_weight=bag_weight+self.weights[item_NO]
            after_bag_value=bag_value+self.values[item_NO]

            if after_bag_weight<=self.capacity: # 背包的重量不能超 背包的容量(搜索剪枝)

                if after_bag_value> self.states[item_NO][after_bag_weight]: # 装入后背包的价值 比 备忘录中记录的 考虑同一个 物品, 在相同的背包重量下的价值大 (搜索剪枝)

                    self.states[item_NO][after_bag_weight]=after_bag_value

                    self._process_cache(item_NO+1, # 考察 item_NO+1 号物品
                                  after_bag_weight,
                                  after_bag_value,prev_items+[item_NO]) #

            # case2. 不装入 item_NO 物品
            self._process_cache(item_NO+1,bag_weight,
                              bag_value,prev_items)


class soulution_bag_problem:
    """
    by XRH 
    date: 2020-08-23 
    
    背包问题
    
    与 01 背包问题的区别在于, 在背包问题中 每一个物品可以取多个
    
    """

    def bag_problem_BranchBound(self, weights, values, capacity):
        """
        分支限界法 解背包问题
        
        适用于组合优化问题
        
        :param weights: 
        :param values: 
        :param capacity: 
        :return: 
        """

        self.N=len(weights)
        self.capacity = capacity

        items=[ (values[i],weights[i])  for i in range(self.N)] # (values[i],weights[i])

        self.items_sorted= sorted(items, key= lambda ele: (ele[0]/ele[1]) ,reverse=True) # 单位重量的价值排序

        self.item_pre_value= list(map(lambda x:x[0]/x[1],self.items_sorted))

        # print( self.items_sorted)
        # print(self.item_pre_value)

        self.max_bound = float('-inf')
        self.max_value_bag_item = []

        # self.max_value_bag_items = zeros(self.N,dtype=int)

        item_NO = 0
        bag_weight = 0
        bag_value = 0

        prev_items = [] # [(物品id, 物品个数),...]

        root_bound=float('inf')

        self._process(item_NO, bag_weight, bag_value, prev_items,root_bound)  # 考虑 第0个物品 是否装入背包


        return self.max_value_bag_item


    def _process(self, item_NO, bag_weight, bag_value, prev_items,root_bound):

        if item_NO == self.N:  # 递归结束条件: 最后一个 物品已考察完( 找到新的可行解 )

            if bag_value > self.max_bound:  # 找到更大价值的 背包
                                            # 若新的 可行解的优化函数值大于 当前的界, 则把 当前界 更新为该可行解的值

                self.max_bound = bag_value
                self.max_value_bag_item = prev_items

        if item_NO < self.N:

            # 装入 item_NO 物品

            max_num= ( self.capacity // self.items_sorted[item_NO][1]) # item_NO 物品 最多装入的个数

            for i in range(0,max_num+1):

                current_bag_weight=bag_weight + i*self.items_sorted[item_NO][1]

                if current_bag_weight <= self.capacity: # 背包重量不能超

                    current_bag_value = bag_value + i * self.items_sorted[item_NO][0]
                    current_prev_items= prev_items+[(item_NO,i)]

                    current_root_bound= current_bag_value+ (self.capacity-current_bag_weight)*self.item_pre_value[item_NO]

                    if root_bound > self.max_bound: # 分支定界: 以A为根节点的 子树下所有节点的值 必然小于根节点A; 若根节点A的代价函数值 比当前界小, 则此分支可以丢弃

                        self._process(item_NO + 1, current_bag_weight,
                                  current_bag_value, current_prev_items,current_root_bound)






class solution_pattern_match:
    """
    正则表达式 匹配
    """

    def match(self,pattern,text):

        self.pattern=pattern # 正则表达式
        self.text=text # 待匹配的字符串

        self.pattern_len=len(pattern)
        self.text_len=len(text)

        pattern_j=0
        text_i=0

        self.match_flag=False

        self.__process(text_i,pattern_j)

        return self.match_flag

    def __process(self,text_i,pattern_j):

        if self.match_flag == True: #如果已经匹配了，就不要继续递归了
            return

        if text_i==self.text_len:
            if pattern_j == self.pattern_len:
                # pattern 和 text 都到了末尾，说明模式匹配成功
                self.match_flag = True
                return

        if text_i<self.text_len and pattern_j< self.pattern_len: #保证数组不越界

            if self.pattern[pattern_j]=='*' :
                for index in range(text_i,self.text_len+1):  # 为了让指针 指向 text的末尾 ，self.text_len+1
                    # '*' 可以匹配任意个数的字符
                    #递归 检查 从text 的当前指针指向的位置 到 text 的末尾 与 pattern_j+1 的匹配
                    self.__process(index, pattern_j+1)

            elif self.pattern[pattern_j]=='?' :
                self.__process(text_i, pattern_j + 1) # 匹配0个字符
                self.__process(text_i+1, pattern_j + 1) #匹配1个字符

            else: # self.pattern[pattern_j] 为普通字符
                if self.text[text_i]==self.pattern[pattern_j]:
                    self.__process(text_i + 1, pattern_j + 1)

class solution_Traveling_Salesman_Problem:

    def tsp_recursive(self,city_distance,start_point):
        """
        
        递归 + 剪枝 解 旅行推销员问题
        
        ref: https://www.cnblogs.com/dddyyy/p/10084673.html
        
        :param city_distance: 城市的距离矩阵
        :param start_point:  起点城市
        :return: 
        """

        self.city_distance=city_distance
        self.city_names=list(range(len(city_distance)))

        self.start_point=start_point

        self.min_cost=float('inf') # 取一个足够大的数 作为初始的 最小路径代价
        self.min_cost_path=[]

        self.path_length=len(city_distance)+1 # path_length=5

        stage = 0  # stage0：把起始的点 放入 当前路径中
        current_path=[start_point]
        current_cost=0

        self.__process(stage+1,current_path,current_cost)

        return self.min_cost,self.min_cost_path

    def __process(self,stage,current_path,current_cost):

        if stage < self.path_length-1: # stage1-stage3

            if current_cost >= self.min_cost:
                return

            # print(current_path)
            for next_city in set(self.city_names)-set(current_path): # 每个城市只会访问一次，从没走过的城市中选一个访问

                current_city=current_path[-1]
                cost= current_cost + self.city_distance[current_city][next_city] # 路径代价为：当前的路径代价 + 现在所在城市到下一个城市的距离
                self.__process(stage + 1, current_path+[next_city],cost)

        elif stage==self.path_length-1: #stage4: 最后要回到 起点

            cost = current_cost + self.city_distance[current_path[-1]][self.start_point]
            current_path=current_path+[self.start_point] # 把 起始节点 加到路径的末尾
            if cost < self.min_cost: #
                self.min_cost=cost
                self.min_cost_path=current_path

    def tsp_dp(self, city_distance,start_city=0):
        """
        动态规划 解 旅行商问题
        
        ref: https://blog.csdn.net/qq_39559641/article/details/101209534
        :param city_distance: 
        :param start_city:  
        :return: 
        """
        N=len(city_distance) # 城市的数目 5

        # M= pow(2,N-1) # N-1=4
        # states = zeros((N, M), dtype=int)

        states_dist={} #记录 到达状态的最短距离

        states_prev_node={} # 记录 到达该状态的 最佳的上一个状态

        city_set= set(list(range(N)))

        mid_city_set=city_set-set([start_city]) #{1,2,3,4}
        # print(mid_city_set)

        for V_length in range(0,N-1): # 0,1,2,3

            for mid_start in  mid_city_set:

                for V in itertools.combinations(mid_city_set-set([mid_start]), V_length):# 取组合

                    print(mid_start ,V) # mid_start=4, V=()
                                        # mid_start=3, V=(4,)
                                        # mid_start=2, V=(3, 4)

                    if len(V)==0:
                        states_dist[(mid_start, V)] = city_distance[mid_start][start_city]
                        states_prev_node[(mid_start, V)]=None
                    else:
                        min_dist=float('inf')
                        min_node=None

                        for city in V:

                            # v=set(V)-set([city]) # TypeError: unhashable type: 'set'
                            v=tuple( sorted(set(V)-set([city])) ) # TODO: sorted( set(V)-set([city]) )

                            dist=city_distance[mid_start][city]+states_dist[(city,v)]
                            if dist<min_dist:
                                min_dist=dist
                                min_node=(city, v)

                        states_dist[(mid_start,V)]=min_dist
                        states_prev_node[(mid_start,V)]=min_node

        # 求原问题的解 d(start_city=0,mid_city_set={1, 2, 3, 4})
        print(start_city ,tuple(mid_city_set) )
        min_dist = float('inf')
        min_node = None
        for city in mid_city_set:

            v = tuple(sorted(set(mid_city_set) - set([city])))

            dist = city_distance[start_city][city] + states_dist[(city, v)]
            if dist < min_dist:
                min_dist = dist
                min_node = (city, v)

        states_dist[(start_city ,tuple(mid_city_set) )] = min_dist
        states_prev_node[(start_city ,tuple(mid_city_set))] = min_node

        print(states_dist)
        print(states_prev_node)

        #反向推出 最短路径 经过的节点
        path=[start_city]
        prev_node=(start_city ,tuple(mid_city_set))
        for i in range(N-1):

            node=states_prev_node[prev_node]
            path.append(node[0])
            prev_node=node

        path.append(start_city)

        return min_dist,path


    def tsp_BranchBound(self,city_distance,start_point):
        """
        分支 定界法 求 旅行推销员问题
        
        :param city_distance: 
        :param start_point: 
        :return: 
        """
        self.city_distance = city_distance
        self.citys = list(range(len(city_distance)))

        self.start_point = start_point

        self.min_cost = float('inf')  # 取一个足够大的数 作为初始的 最小路径代价
        self.min_cost_path = []

        self.path_length = len(city_distance) + 1  # path_length=5

        stage = 0  # stage0：把起始的点 放入 当前路径中
        current_path = [start_point]
        current_cost = 0

        self.__process_BranchBound(stage + 1, current_path, current_cost) # 考察下一个顶点

        return self.min_cost, self.min_cost_path


    def __cost_func(self, current_path, current_cost):
        """
        代价函数

        在搜索树中, 以当前路径的最后一个节点 作为根节点的子树;
        求出 在该子树中, 我们能找到的最优的可行解 对应的 代价函数的 下界

        :param current_node: 
        :param current_group: 
        :return: 
        """
        end_node=current_path[-1]

        end_node_min_edge=min( [ l for l in self.city_distance[end_node] if l!=0] )

        lower_bound=0

        for node in set(self.citys) - set(current_path):

            node_min_edge=min( [ l for l in self.city_distance[node] if l!=0] )
            lower_bound+=node_min_edge

        return current_cost+ end_node_min_edge+lower_bound


    def __process_BranchBound(self, stage, current_path, current_cost):
        """
        
        :param stage: 
        :param current_path: 当前路径
        :param current_cost:  当前路径的长度
        :return: 
        """

        if stage < self.path_length - 1:  # stage1-stage3

            if self.__cost_func( current_path, current_cost) >= self.min_cost: # 分支定界: 对于极小化问题, 代价函数的的下界 比 搜索树当前的界大, 直接回溯
                return

            # print(current_path)

            for next_city in set(self.citys) - set(current_path):  # 每个城市只会访问一次，从没走过的城市中选一个访问

                current_city = current_path[-1]
                cost = current_cost + self.city_distance[current_city][next_city]  # 路径代价为：当前的路径代价 + 现在所在城市到下一个城市的距离

                self.__process_BranchBound(stage + 1, current_path + [next_city], cost)

        elif stage == self.path_length - 1:  # stage4: 最后要回到 起点

            cost = current_cost + self.city_distance[current_path[-1]][self.start_point]
            current_path = current_path + [self.start_point]  # 把 起始节点 加到路径的末尾

            if cost < self.min_cost:
                self.min_cost = cost
                self.min_cost_path = current_path







class solution_loading_problem:

        """

        装载问题  

        n 个集装箱 重量为 weights, 两艘船, 它们的载重能力分别为 c1 和 c2 
        
        如何 选择合理的 装载方案 , 能把这  n 个集装箱 都装上船

        原问题 => 第一艘船 在不超过 载重限制的情况下, 尽量装满, 若剩下的物品重量小于 第二艘船的载重(可以全部装入第二艘船),则此方案可行   
        
        
        """

        def loading_problem_recursive(self, weights , capacity , capacity2 ):
            """
            递归法 解 装载问题 
            
            :param weights: 
            :param capacity: 第一艘船的 载重限制
            :param capacity2: 第二艘船的 载重限制
            :return: 
            """

            self.N = len(weights)  # N 个集装箱

            self.weights = weights
            self.capacity = capacity


            self.max_weight = float('-inf')
            self.max_weight_bag_item = []

            item_NO = 0
            bag_weight = 0
            prev_items = []

            # 1. 找出 第一艘船 在不超过 载重限制的 最大装载重量的方案

            self._process(item_NO, bag_weight, prev_items)  # 考虑 第0个集装箱 是否 装船

            # 2. 剩余的 集装箱, 第一艘船 是否能装完
            item_set=set(self.max_weight_bag_item)

            print('第一艘船 装入的集装箱：',item_set)
            rest_item_set= set(list(range(self.N)))-item_set

            print('第二艘船 装入的集装箱：', rest_item_set)

            bag2_weight=0
            for item in rest_item_set:

                bag2_weight+=weights[item]

            flag=False

            if bag2_weight<=capacity2: flag=True

            return flag

        def _process(self, item_NO, bag_weight, prev_items):

            if item_NO == self.N:  # 递归结束条件: 最后一个 集装箱已考察完

                if bag_weight > self.max_weight:  # 找到了 总装载重量更高的 装载方案

                    self.max_weight = bag_weight
                    self.max_weight_bag_item = prev_items

            if item_NO < self.N:

                # case1. 装入 item_NO 集装箱
                if bag_weight + self.weights[item_NO] <= self.capacity:  # 船的载重限制 不能超(搜索剪枝)

                    self._process(item_NO + 1, bag_weight + self.weights[item_NO],
                                prev_items + [item_NO])

                # case2. 不装入 item_NO 集装箱
                self._process(item_NO + 1, bag_weight,
                               prev_items)

        def loading_problem_iteration(self, weights, capacity, capacity2):
            """
            递归法 解 装载问题 

            :param weights: 
            :param capacity: 第一艘船的 载重限制
            :param capacity2: 第二艘船的 载重限制
            :return: 
            """
            N = len(weights)  # N 个集装箱

            weights= sorted(weights,reverse=True)

            max_weight = float('-inf')
            max_weight_bag_item = []

            item_NO = 0
            bag_weight = 0
            prev_items = []

            while True:

                while item_NO <= N-1:

                    after_bag_weight= bag_weight + weights[item_NO]

                    # case1. 装入 item_NO 集装箱
                    if after_bag_weight <= capacity:

                        bag_weight= after_bag_weight
                        prev_items.append(item_NO)


                    item_NO+=1

                if item_NO==N: # 最后一个集装箱 已经考察完

                    if bag_weight>max_weight:

                        max_weight=bag_weight
                        max_weight_bag_item=array(prev_items) # 复制 prev_items

                # 开始回溯

                if len(prev_items)==0: # 没有一个 集装箱被装入, 说明所有分支 都已经回溯到了
                    break

                root=prev_items[-1]

                bag_weight-=weights[root] # root 集装箱 之前走的是装入分支, 现在选择不装入分支
                item_NO = root + 1
                prev_items.pop(-1) # 不装入 root 集装箱, 需要从 prev_items 中删除


            return max_weight_bag_item


class solution_Graph_Coloring_Problem():

    def mGCP(self,mapGraph,m):
        """
        递归 解图的 m 着色 问题
        
        
        详见 《数据结构与算法之美》 -> 基础算法-回溯法
        
        :param mapGraph: 
        :param m: 
        :return: 
        """

        self.mapGraph=mapGraph
        self.colors= list(range(1,m+1))
        self.nodeNum= len(mapGraph) # nodeNum=5

        self.res_nodes_color=[] # 节点的着色策略

        #node0: 初始节点 可以选择 所有颜色中的任意一种颜色上色
        node=0
        current_nodes_color=[]
        for color in self.colors:

            self._process_find_all( node+1 ,current_nodes_color+[color])


        return len(self.res_nodes_color),self.res_nodes_color

    def _process_find_all(self,node,current_nodes_color):
        """
        递归 找到所有的 可行解 
        
        :param node: 当前待上色的节点
        :param current_nodes_color:  当前已经上过色的节点 所上的颜色, 
                            eg. [1,2,3] 0号节点上颜色 1, 1号节点上颜色 2 , 2号节点上颜色 3   
        :return: 
        """

        if node < self.nodeNum:  # node1-node4

            node=node # 当前待上色的节点
            adjacent_nodes=self.mapGraph[node] #待上色节点的 相邻节点

            adjacent_nodes=adjacent_nodes[:len(current_nodes_color)] #相邻节点中 截取 已经上过色的节点 (current_colors 中按照节点的顺序 记录各个节点所上的颜色)

            adjacent_nodes_colors=[ current_nodes_color[index]
                                    for index,node in enumerate(adjacent_nodes) if node==1] #相邻节点中 已经上过色的节点的颜色 集合

            available_colors=set(self.colors)-set(adjacent_nodes_colors) #当前待上色的节点的 可选颜色

            for color in available_colors:
                self._process_find_all(node + 1, current_nodes_color + [color])


        elif node== self.nodeNum: # 说明 所有节点都已经成功上色

            self.res_nodes_color.append(current_nodes_color)


    def mGCP_v1(self, mapGraph, m):
        """
        递归 解图的 m 着色 问题

        递归找出一个可行解, 其他的解 利用对称性 直接生成

        :param mapGraph: 
        :param m: 
        :return: 
        """

        self.mapGraph = mapGraph
        self.colors = list(range(1, m + 1))
        self.nodeNum = len(mapGraph)  # nodeNum=5

        self.flag = False

        self.nodes_color = []  # 节点的着色策略

        # 1. 找到一个 可行解

        # node0: 初始节点 可以选择 所有颜色中的任意一种颜色上色
        node = 0
        current_nodes_color = []
        for color in self.colors:
            self._process_find_one(node + 1, current_nodes_color + [color])

        # 2. 生成其他的 可行解
        # 种子 self.nodes_color= [1, 2, 3, 4, 1]

        self.res_nodes_color=[]

        for permutation in itertools.permutations(self.colors, m): # 颜色的 排列

            dict={ i+1:ele  for i,ele in enumerate(permutation)}

            new_nodes_color= list(map( lambda ele: dict[ele] ,self.nodes_color))

            self.res_nodes_color.append(new_nodes_color)


        return len(self.res_nodes_color), self.res_nodes_color

    def _process_find_one(self, node, current_nodes_color):
        """
        递归 一个可行解 即可退出 

        :param node: 当前待上色的节点
        :param current_nodes_color:  当前已经上过色的节点 所上的颜色, 
                            eg. [1,2,3] 0号节点上颜色 1, 1号节点上颜色 2 , 2号节点上颜色 3   
        :return: 
        """

        if self.flag:
            return

        if node < self.nodeNum:  # node1-node4

            node = node  # 当前待上色的节点
            adjacent_nodes = self.mapGraph[node]  # 待上色节点的 相邻节点

            adjacent_nodes = adjacent_nodes[
                             :len(current_nodes_color)]  # 相邻节点中 截取 已经上过色的节点 (current_colors 中按照节点的顺序 记录各个节点所上的颜色)

            adjacent_nodes_colors = [current_nodes_color[index]
                                     for index, node in enumerate(adjacent_nodes) if node == 1]  # 相邻节点中 已经上过色的节点的颜色 集合

            available_colors = set(self.colors) - set(adjacent_nodes_colors)  # 当前待上色的节点的 可选颜色

            for color in available_colors:
                self._process_find_one(node + 1, current_nodes_color + [color])


        elif node == self.nodeNum:  #  说明 所有节点都已经成功上色

            self.flag=True
            self.nodes_color=current_nodes_color


class solution_biggest_group:
    """
    最大团问题
    
    
    """

    def biggest_group_BranchBound(self,graph):
        """
        分支定界法 求图的最大团
        
        :param graph: 
        :return: 
        """

        self.graph = graph
        self.nodeNum = len(graph)  # nodeNum=5

        self.max_bound = float('-inf')
        self.res_biggest_group = []  # 图的最大团

        # node0: 第一个节点
        current_node = 0
        current_group = []

        self._process(current_node+1, current_group + [current_node]) # 第一个节点 加入 团

        self._process(current_node+1, current_group ) # 第一个节点 不入 团

        return len(self.res_biggest_group), self.res_biggest_group


    def __check_connected(self,current_node,current_group):
        """
        判断 node 和团中的点是否 有边相连
        :param current_node: 
        :param current_group: 
        :return: 
        """
        flag=True

        for node in current_group:

            if self.graph[current_node][node]==0:
                flag=False
                break

        return flag

    def __cost_func(self,current_node,current_group):
        """
        代价函数
        
        目前的团 可能扩展为 极大团的 顶点数的上界
        
        :param current_node: 
        :param current_group: 
        :return: 
        """
        C=len(current_group)  # 目前团的顶点数

        return C + self.nodeNum-current_node


    def _process(self, current_node, current_group):
        """
        递归 找到 最优解 

        :param current_node: 当前 节点
        :param current_group:  当前的团
        
        
        :return: 
        """
        print(current_node, current_group)

        if current_node < self.nodeNum:  # node1-node4

            if self.__cost_func(current_node, current_group) >= self.max_bound:  # 分支定界 , 若代价函数的值大于界，则有继续搜索此分支的需要
                                                                                 # 代价函数 表示了 此子树 可能出现的 最优可行解的 上界

                # case1: current_node 节点 纳入最大团
                if self.__check_connected(current_node,current_group):# 判断 current_node 和团中的点是否 有边相连


                        self._process(current_node + 1, current_group + [current_node])



                # case1 和 case2 是根节点的两个分支, 都要走到
                # case2: current_node 节点 不纳入最大团
                self._process(current_node + 1, current_group )


        elif current_node == self.nodeNum:  # 找到可行解

            if len(current_group) > self.max_bound: # 可行解比当前界好，更新界

                self.max_bound=len(current_group)

                self.res_biggest_group= array(current_group)


class solution_Circle_permutation:
    """
    圆排列 问题
    
    
    """

    def circle_permutation_BranchBound(self,R_list):
        """
        回溯 + 分支定界法 求解 圆排列 问题
        
        :param R_list:  圆的半径 列表
        :return: 
        """
        self.R_list = R_list

        self.stageNum = len(R_list) # self.stageNum=6
        self.n= len(R_list)

        self.circles = list(range(len(R_list)))


        self.min_length = float('inf')  # 取一个足够大的数 作为初始的 最小 圆的排列长度
        self.min_length_permutation = []


        stage = 0  # stage0

        current_x=0 #  排列中 第 0个圆的初始坐标为0

        current_length = 0 # 当前 圆排列的长度
        current_permutation=[] # 当前 已经放入 排列的圆

        self.__process_BranchBound(stage , current_x, current_permutation, current_length)


        return self.min_length, self.min_length_permutation

    def __cost_func(self, k,x_prev,circle_next,current_permutation,available_circles):
        """
        计算代价函数

        在搜索树中, 以当前路径的最后一个节点 作为根节点的子树;
        求出 在该子树中, 我们能找到的最优的可行解 对应的 代价函数的 下界

        :param k:
        
        :param x_prev: 上一个圆 的坐标  x(k-1) k-1 个圆的 坐标
        
        :param circle_next: 加入排列的下一个圆的标号 
        
        :param current_permutation: 当前的排列 
        :param available_circles: 未加入 排列的 圆的标号
        
        :return x : k 个圆的坐标
        :return l : 当前圆排列的长度
        :return L : 代价函数
        
        """

        if k==0:

            r_first =self.R_list[circle_next] # r(0)
            r_prev=0  #  第一次  r(k-1)=0

        else:

            circle_prev = current_permutation[-1]  # 排列中的 最后一个圆的标号

            circle_first = current_permutation[0]  # 排列中的 第一个圆的标号

            r_prev = self.R_list[circle_prev]  # r(k-1)

            r_first = self.R_list[circle_first]  # r(0)


        r = self.R_list[circle_next]  # r(k)

        d=2*sqrt(r*r_prev) # d(k)

        x=x_prev+d # x(k)  k 个圆的坐标

        l= x+r+r_first # l(k)

        min_r= min([self.R_list[circle]  for circle in available_circles]) # 未被选择到的圆中, 半径最小的圆的半径

        L= x+(self.n-(k+1))*(2*min_r)+min_r+r_first # L(k)


        return float("%0.2f"%x), float("%0.2f"%l),float("%0.2f"%L) # 精度控制, 保留2位小数

    def __process_BranchBound(self, stage, current_x,current_permutation, current_length):
        """

        :param stage: 
        :param current_x: 圆的坐标
        :param current_permutation: 当前 圆排列
        :param current_length:  当前 圆排列的长度
        :return: 
        """

        if stage < self.stageNum :  # stage0-stage5

            available_circles=set(self.circles) - set(current_permutation)

            for circle_next in available_circles:

                x,l,L=self.__cost_func(stage,current_x, circle_next, current_permutation,
                                       available_circles)

                if L >= self.min_length:  # 分支定界: 对于极小化问题, 代价函数的的下界 比 搜索树当前的界大, 直接回溯
                    continue

                self.__process_BranchBound(stage+1, x , current_permutation+[circle_next], l)


        elif stage == self.stageNum:  # stage6 找到一个 可行解

            if current_length < self.min_length:
                self.min_length = current_length
                self.min_length_permutation = current_permutation


class solution_Continuous_postage:
    """
    连续邮资 问题
    
    n 种邮票, 一共 m 张 , 每一种的邮票的面值 <=100, 
    第一种邮票的面值 为 1 , 
    求 可以贴出的 连续邮资的最大的值为多少
    
    
    """

    def continuous_postage(self,n,m,max_value):
        """
        回溯 + 动态规划
        
        :param n: 邮票种类
        :param m: 邮票总的张数
        :param max_value: 邮票最大面值
        :return: 
        """
        self.n= n
        self.m = m

        self.column_num=max_value*self.m+1

        self.states= zeros((self.n,self.column_num),dtype=int) # 选择前i 种邮票 凑到邮资j 使用的最少邮票个数

        self.max_continuous_postage = float('-inf')  # 最大连续邮资
        self.max_continuous_postage_stamps = [] # 达到 最大连续邮资 选择的邮票的 面值

        # 1. 初始条件, 只能选 第一种邮票
        stamp_list=[1] # 第一种邮票的面值 为 1

        stage=0
        self.states[0,:]= list(range(self.column_num))
        r= self.m

        self.__process(stage+1,stamp_list,r)

        return self.max_continuous_postage,self.max_continuous_postage_stamps

    def __cal_continuous_postage(self,stage,next_stamp,states=None):
        """
        利用动态规划求解 在当前邮票面值列表下, 能凑成的连续邮资的上界 
        
        思路与  硬币选择问题 (背包问题变形) 类似
        
        :param stage: 
        :param next_stamp: 选取的下一个邮票的面值
        :param stamp_list: 当前(已经选取的)邮票面值列表
        :return: 
        """
        if states==None:
            states=self.states

        r=0

        for j in range(0,self.column_num): # 邮资总额 j


            min_num=float('inf') # 凑到 邮资总额 j 需要的最少邮票张数
            min_num_t=0 #  凑到 邮资总额 j 需要使用的 面值 为 next_stamp 的邮票的 张数

            for t in range(1,self.m+1):

                value= j-t*next_stamp  # 使用 t 张面值为 next_stamp 的邮票后,剩余的价值

                if value>=0:

                    current_num= t+ states[stage-1,value]

                    if current_num < min_num:

                        min_num=current_num
                        min_num_t=t


            # 在 "选择面值为 next_stamp 的邮票" 和 "不选择面值为 next_stamp 的邮票" 中选邮票数目最少的

            states[stage,j]= min(min_num,self.states[stage-1,j])


            # if self.states[stage,j] > self.m: #凑到 邮资总额 j 需要的最少邮票张数 大于 总邮票个数 m , 说明断点 r 出现
            #     r=j-1
            #     break

        # 找断点
        for j in range(0, self.column_num):

            if states[stage,j] > self.m: #凑到 邮资总额 j 需要的最少邮票张数 大于 总邮票个数 m , 说明断点 r 出现

                r=j-1
                break

        return r,states


    def __process(self,stage,stamp_list,r):
        """
        
        :param stage: 
        :param stamp_list: 当前已经选取的邮票面值列表
        :param r: 当前的连续邮资的上界 
        :return: 
        """

        if stage < self.n :  # stage0-stage3

            print('stage:{} ,x:{} ,r:{}'.format(stage,stamp_list, r))

            next_stamp_lower_bound=stamp_list[-1]+1 # 可以选择的邮票的下界
            next_stamp_upper_bound=r+1 # 可以选择的邮票的上界

            print('lower_bound:{} upper_bound:{}'.format(next_stamp_lower_bound,next_stamp_upper_bound))


            for next_stamp in range(next_stamp_lower_bound,next_stamp_upper_bound+1):

                # print('next_stamp:{} '.format(next_stamp))

                # print('before stage:{}'.format(stage))
                # print(self.states)

                r,_=self.__cal_continuous_postage(stage,next_stamp)


                self.__process(stage+1, stamp_list+[next_stamp], r)

                self.states[stage:, :] = zeros((self.n-stage,self.column_num),dtype=int) # 对 self.states 复位

                # print('after stage:{}'.format(stage))
                # print(self.states)

        elif stage == self.n:  # stage4 找到一个 可行解

            print("--------------------------")
            print('stage:{} ,x:{} ,r:{}'.format(stage, stamp_list, r))
            print("--------------------------")

            if r>self.max_continuous_postage:

                self.max_continuous_postage=r
                self.max_continuous_postage_stamps=array(stamp_list)




class Test:

    def test_all(self):

        sol = solution_N_quene()
        # print(sol.N_quene(4))


        sol = solution_zero_one_bag_weight()
        items_info = [('a', 2), ('b', 2), ('c', 4), ('d', 6), ('e', 3)]
        capacity = 9
        # print(sol.zero_one_bag_weight_iteration(items_info, capacity))
        # print(sol.zero_one_bag_weight(items_info, capacity))

        sol = solution_zero_one_bag_value()

        values = [12, 11, 9, 8]
        weights = [8, 6, 4, 3]
        capacity = 13

        # print(sol.zero_one_bag_value(weights,values,capacity))

        weights = [2, 2, 4]
        values = [3, 4, 8]
        capacity = 9
        # print(sol.zero_one_bag_value_cache(weights, values, capacity))


        sol = soulution_bag_problem()
        weights = [2, 3, 4, 7]
        values = [1, 3, 5, 9]
        capacity = 10
        # print(sol.bag_problem_BranchBound(weights, values, capacity))

        sol=solution_loading_problem()
        weights = [90, 80, 40, 30, 20, 12, 10]
        # weights = [40, 30, 20, 12, 10, 80, 90]
        capacity1=152
        capacity2=130
        # print(sol.loading_problem_recursive(weights,capacity1,capacity2))
        # print(sol.loading_problem_iteration(weights,capacity1,capacity2))

        sol2 = solution_pattern_match()
        pattern = 'a*d'
        text = 'abcd'
        # print(sol2.match(pattern,text))
        # print(sol2.match('a*d?f', 'abcdef'))
        # print(sol2.match('', 'ab'))
        # print(sol2.match('a*', 'ab'))
        # print(sol2.match('a?', 'ab'))
        # print(sol2.match('*', 'ab'))
        # print(sol2.match('?', 'a'))
        # print(sol2.match('**', 'ab'))
        # print(sol2.match('??', 'ab'))


        city_distance = [
            [0, 4, 3, 1],
            [4, 0, 1, 2],
            [3, 1, 0, 5],
            [1, 2, 5, 0],
        ]
        sol3 = solution_Traveling_Salesman_Problem()
        # print(sol3.tsp_recursive(city_distance,0))


        city_distance2 = [
            [0, 3, float('inf'), 8, 9],
            [3, 0, 3, 10, 5],
            [float('inf'), 3, 0, 4, 3],
            [8, 10, 4, 0, 20],
            [9, 5, 3, 20, 0],
        ]
        # print(sol3.tsp_dp(city_distance2, 0))
        # print(sol3.tsp_dp(city_distance2, 1))
        # print(sol3.tsp_dp(city_distance2, 2))
        # print(sol3.tsp_dp(city_distance2, 3))
        # print(sol3.tsp_dp(city_distance2, 4))

        city_distance3 = [
            [0, 5, 9, 4],
            [5, 0, 13, 2],
            [9, 13, 0, 7],
            [4, 2, 7, 0],
        ]

        # print(sol3.tsp_BranchBound(city_distance3, 0))

        mapGraph = [
            [0, 1, 1, 1, 0],
            [1, 0, 1, 1, 1],
            [1, 1, 0, 1, 0],
            [1, 1, 1, 0, 1],
            [0, 1, 0, 1, 0]
        ]
        sol4 = solution_Graph_Coloring_Problem()

        # print(sol4.mGCP(mapGraph, 4))
        # print(sol4.mGCP_v1(mapGraph, 4))

        # print(sol4.mGCP(mapGraph, 3))

        Graph = [
            [0, 1, 1, 1, 1],
            [1, 0, 0, 1, 0],
            [1, 0, 0, 1, 1],
            [1, 1, 1, 0, 1],
            [1, 0, 1, 1, 0]
        ]

        sol5= solution_biggest_group()

        # print(sol5.biggest_group_BranchBound(Graph))


        R=[1,1,2,2,3,5]
        sol6= solution_Circle_permutation()
        # print(sol6.circle_permutation_BranchBound(R))

        sol7=solution_Continuous_postage()
        n=4
        m=3

        print(sol7.continuous_postage(n,m,10))


if __name__ == '__main__':

    sol=Test()
    sol.test_all()







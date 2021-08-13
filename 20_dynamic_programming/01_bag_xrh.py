#!/usr/bin/python
# -*- coding: UTF-8 -*-

from numpy import *

class solution_zero_one_bag_weight:

    def run_recursive_cache(self,weights,capacity):
        """
        回溯法 解 01背包问题 , 并缓存子问题的解
        ref: https://time.geekbang.org/column/article/74788 
        :param weights: 
        :param capacity: 
        :return: 
        """
        self.weights=weights
        self.capacity=capacity

        self.max_bag_weight=0
        # self.res_bag=()

        self.cache = zeros((len(weights)+1 , capacity+1), dtype=bool) # 默认为 False

        i=0
        current_bag_weight=0
        self.__f( i, current_bag_weight)

        return self.max_bag_weight

    def __f(self, i , current_bag_weight):

        if i <= len(self.weights): # i= 0,1,2,3

            if current_bag_weight > self.capacity: #搜索剪枝: 当发现已经选择的物品的重量超过 Wkg 之后，我们就停止继续探测剩下的物品
                return

            if self.cache[i][current_bag_weight] == True: # 判断值相等 用 '=='
                print('(',i,',',current_bag_weight,')','has cached')
                return
            else:
                self.cache[i][current_bag_weight] = True

            print(i, current_bag_weight) # 树的先序列遍历


            if current_bag_weight > self.max_bag_weight:
                self.max_bag_weight =current_bag_weight

            self.__f(i+1,current_bag_weight) # f(1,0)

            if i < len(self.weights):
                self.__f(i+1, current_bag_weight + self.weights[i] ) #f(1,2)

    def run_dynamic_programming(self,weights,capacity):
        """
        动态规划法 解 01背包问题
        ref: https://time.geekbang.org/column/article/74788 
        :param weights: 
        :param capacity: 
        :return: 
        """
        states = zeros((len(weights), capacity + 1), dtype=int)

        row_num=shape(states)[0]
        col_num=shape(states)[1]

        for i in range(row_num):

            states[i][0]=1 # 第i个物品 不放入背包

            if  weights[i] < capacity :
                states[i][weights[i]]=1 # 第i个物品放入背包


            for j in range(col_num):
                if states[i-1][j]==1:
                    states[i][j] = 1 # 第i个物品 不放入背包

                    if j+weights[i] < capacity + 1:
                        states[i][j+weights[i]] = 1 # 第i个物品放入背包

        print(states)
        res_index=0
        for i  in  range(col_num-1,-1,-1): # 从后往前遍历列，找到第一个为1的，即为 最大物品重量和
            if states[-1][i]==1:
                res_index=i
                break

        return res_index

    def run_dynamic_programming_v2(self,weights,capacity):
        """
        动态规划法v2 解 01背包问题：
        （1）降低 空间复杂度， O(n*w) -> O(2w)
        
        ref: https://time.geekbang.org/column/article/74788 
        :param weights: 
        :param capacity: 
        :return: 
        """
        states = zeros((capacity + 1), dtype=int)

        row_num=len(weights)
        col_num=capacity + 1

        for i in range(row_num):

            next_states = zeros((capacity + 1), dtype=int)
            next_states[0]=1 # 第i个物品 不放入背包

            if  weights[i] < capacity :
                next_states[weights[i]]=1 # 背包中 只有 第i个物品

            for j in range(col_num):
                if states[j]==1:
                    next_states[j] = 1 # 第i个物品 不放入背包

                    if j+weights[i] < capacity + 1:
                        next_states[j+weights[i]] = 1 # 第i个物品放入背包

            states=next_states

        print(states)
        res_index=0
        for i  in  range(col_num-1,-1,-1):
            if states[i]==1:
                res_index=i
                break

        return res_index

    def run_dynamic_programming_v3(self, weights, capacity):
        """
        动态规划法v3 解 01背包问题：
        （1） 进一步降低 空间复杂度: O(2w) -> O(w)，
         (2) 降低时间复杂度: 对 states 的列的遍历 做 剪枝
         (3) 当重复使用一个 states 时，对 states 的列的遍历 j 需要从大到小来处理。如果我们按照 j 从小到大处理的话，会出现 for 循环重复计算的问题。

        ref: https://time.geekbang.org/column/article/74788 
        :param weights: 
        :param capacity: 
        :return: 
        """
        states = zeros((capacity + 1), dtype=int)

        row_num = len(weights)
        col_num = capacity + 1

        # 对状态0 特殊处理：
        states[0] = 1
        if weights[0] < capacity + 1:
            states[weights[0]] = 1  # 第0个物品放入背包

        for i in range(1,row_num): # 从状态1开始
            col_end=capacity-(weights[i]) # 剪枝: 大于col_end 的j 不用考虑，因为 j + weights[i] 必然大于 背包的容量
            for j in range(col_end,-1,-1):
                if states[j] == 1:
                    states[j + weights[i]] = 1  # 第i个物品放入背包


        print(states)

        res_index = 0
        for i in range(col_num - 1, -1, -1):
            if states[i] == 1:
                res_index = i
                break

        return res_index

import sys
class  solution_double11advance:

    def run_dynamic_programming_v2(self, prices, condition):
        """
        双十一 购物节有 “满 200 元减 50 元 ” 的促销。假设女朋友的购物车中有 n 个（n>100）想买的商品，她希望从里面选几个，
        在凑够满减条件的前提下，让选出来的商品价格总和最大程度地接近满减条件（200 元），这样就可以极大限度地“薅羊毛”。
        请给出选择商品的方案。
        
        空间复杂度： O(2*w) w为 conditionx2
        
        :param prices: 各个商品的价格
        :param condition: 满减的条件
        :return: 选择的商品方案
        """
        capacity = condition * 2

        states = zeros(capacity + 1, dtype=int)

        print('states memory_size:', sys.getsizeof(states))

        row_num = len(prices)
        col_num = condition + 1

        bag_states = []


        for i in range( row_num):  # 从第0个状态开始

            next_states = zeros((capacity + 1), dtype=int)

            next_states[0] = 1  # 第i个物品 不放入背包

            if prices[i] < capacity:
                next_states[prices[i]] = 1  # 背包中 只有 第i个物品

            for j in range(col_num):
                if states[j] == 1:
                    next_states[j] = 1  # 第i个物品 不放入背包

                    if j + prices[i] < capacity + 1:
                        next_states[j + prices[i]] = 1  # 第i个物品放入背包

            states = next_states

            bag_states.append([index for index in range(capacity + 1) if
                               states[index] != 0])  # 记录 各个状态i 下 背包的总价

        print('next_states memory_size:', sys.getsizeof(next_states))

        print(bag_states)  #
        print('bag_states memory_size:', sys.getsizeof(bag_states))

        min_price = 0
        for i in range(condition, capacity + 1):  # 满减问题中，我们要找 大于等于 200（满减条件）的值中最小的
            if states[i] == 1:
                min_price = i
                break

        # 根据背包中物品的总价，反推出 背包中的物品
        bag = []
        bag_price = min_price

        for i in range(row_num - 1, -1, -1):
            if i > 0:
                if bag_price - prices[i] in bag_states[i - 1] :
                    # case1: 第i个物品装入背包
                    # 1.bag_states[i-1] 上一个状态 背包的 总价的情况
                    # 3. (当前背包的重量 - 第i个物品的价格) 为 上一个状态的 背包的总价 则说明 第i 个物品被放入背包

                    bag.append(i)
                    bag_price = bag_price - prices[i]

                else:
                    # case2: 第i个物品未装入背包,背包的总价不变
                    pass

            elif i == 0:
                if bag_price == prices[i]:
                    bag.append(i)


        return min_price, bag

    def run_dynamic_programming_v3(self, prices, condition):
        """
        空间复杂度： O(w) 
        :param prices: 各个商品的价格
        :param condition: 满减的条件
        :return: 选择的商品方案
        """
        capacity = condition * 2

        states = zeros(capacity + 1, dtype=int)

        print('states memory_size:', sys.getsizeof(states))

        row_num = len(prices)
        col_num = condition + 1

        bag_states = []

        states[0] = 1  # 第0个 状态 特殊处理
        if prices[0] < capacity + 1:
            states[prices[0]] = 1  # 第0个物品放入背包

        for i in range(1, row_num):  # 从第1个状态开始

            col_end = capacity - (prices[i])  # 剪枝
            for j in range(col_end, -1, -1):
                if states[j] == 1:
                    states[j + prices[i]] = 1  # 第i个物品放入背包

            bag_states.append([index for index in range(capacity + 1) if
                               states[index] != 0])  # 记录 各个状态i 下 背包的总价的所有情况

        print(bag_states)  #
        print('bag_states memory_size:', sys.getsizeof(bag_states))

        min_price = 0
        for i in range(condition, capacity + 1):  # 满减问题中，我们要找 大于等于 200（满减条件）的值中最小的
            if states[i] == 1:
                min_price = i
                break

        # 根据背包中物品的总价，反推出 背包中的物品
        bag = []
        bag_price = min_price

        for i in range(row_num - 1, -1, -1):
            if i > 0:
                if bag_price - prices[i] in bag_states[i - 1] :
                    # case1: 第i个物品装入背包
                    # 1.bag_states[i-1] 上一个状态 背包的 总价的情况
                    # 3. (当前背包的重量 - 第i个物品的价格) 为 上一个状态的 背包的总价 则说明 第i 个物品被放入背包

                    bag.append(i)
                    bag_price = bag_price - prices[i]

                else:
                    # case2: 第i个物品未装入背包,背包的总价不变
                    pass

            elif i == 0:
                if bag_price == prices[i]:
                    bag.append(i)


        return min_price, bag

    def run_dynamic_programming_v4(self, prices, condition):
        """
        存储 动态规划 状态的数组 过于稀疏，尝试用其他数据结构  dict 替代 数组，进一步降低内存占用
        空间复杂度： 与 condition 的大小无关
        
        对比 bag_states 的内存空间：
         v3 :states memory_size: 3300 
         v4 :dict states memory_size: 1184
         
        :param prices: 各个商品的价格
        :param condition: 满减的条件
        :return: 选择的商品方案
        """
        capacity = condition * 2

        states = {}

        print('dict states(None) memory_size:', sys.getsizeof(states))

        print(' prices memory_size:', sys.getsizeof(prices))

        row_num = len(prices)
        col_num = condition + 1

        bag_states = []

        for i in range(row_num):  # 从第0个状态开始

            next_states = {}

            next_states[0] = 1  # 第i个物品 不放入背包

            if prices[i] < capacity:
                next_states[prices[i]] = 1  # 背包中 只有 第i个物品

            for j in range(col_num):
                if states.get(j)==1:
                    next_states[j] = 1  # 第i个物品 不放入背包

                    if j + prices[i] < capacity + 1:
                        next_states[j + prices[i]] = 1  # 第i个物品放入背包

            states = next_states

            bag_states.append(states.keys())  # 记录 各个状态i 下 背包的总价

        print('dict states memory_size:', sys.getsizeof(states))

        print(bag_states)  #
        print('bag_states memory_size:', sys.getsizeof(bag_states))

        min_price = 0
        for i in range(condition, capacity + 1):  # 满减问题中，我们要找 大于等于 200（满减条件）的值中最小的
            if states.get(i)==1:
                min_price = i
                break

        # 根据背包中物品的总价，反推出 背包中的物品
        bag = []
        bag_price = min_price

        for i in range(row_num - 1, -1, -1):
            if i > 0:
                if bag_price - prices[i] in bag_states[i - 1]:
                    # case1: 第i个物品装入背包
                    # 1.bag_states[i-1] 上一个状态 背包的 总价的情况
                    # 3. (当前背包的重量 - 第i个物品的价格) 为 上一个状态的 背包的总价 则说明 第i 个物品被放入背包

                    bag.append(i)
                    bag_price = bag_price - prices[i]

                else:
                    # case2: 第i个物品未装入背包,背包的总价不变
                    pass

            elif i == 0:
                if bag_price == prices[i]:
                    bag.append(i)

        return min_price, bag


class solution_zero_one_bag_value:

    def run_recursive_cache(self, weights, values, capacity):
        """
        回溯法（子问题缓存） 解  01背包问题 升级版（上价值） 
        ref: https://time.geekbang.org/column/article/74788 
        :param weights: 
        :param values:
        :param capacity: 
        :return: 
        """
        self.weights = weights
        self.values = values

        self.capacity = capacity

        self.max_bag_value = 0

        self.cache_value = zeros((len(weights) + 1, capacity + 1), dtype=int)

        i = 0
        current_bag_weight = 0
        current_bag_value=0

        self.__f(i, current_bag_weight,current_bag_value)

        return self.max_bag_value

    def __f(self, i, current_bag_weight,current_bag_value):

        if i <= len(self.weights):  # i= 0,1,2,3

            if current_bag_weight > self.capacity:  # 搜索剪枝: 当发现已经选择的物品的重量超过 Wkg 之后，我们就停止继续探测剩下的物品
                return

            if current_bag_value<self.cache_value[i][current_bag_weight] :
                print('(', i, ',', current_bag_weight,',', ')', 'has cached',', the value is ',self.cache_value[i][current_bag_weight])
                return
            else:
                self.cache_value[i][current_bag_weight] = current_bag_value

            print(i, current_bag_weight,current_bag_value)  # 树的先序列遍历

            if current_bag_value > self.max_bag_value:
                self.max_bag_value = current_bag_value

            self.__f(i + 1, current_bag_weight,current_bag_value)  # f(1,0,0)

            if i < len(self.weights):
                self.__f(i + 1, current_bag_weight + self.weights[i],current_bag_value+self.values[i])  # f(1,2,3)


    def run_dynamic_programming_v2(self, weights,values, capacity):
        """
        动态规划法v2 解 01背包问题（升级版） ：
        空间复杂度 ： O(2w) 
        ref: https://time.geekbang.org/column/article/74788 
        :param weights: 
        :param values: 
        :param capacity: 
        :return: 
        """
        states_value = zeros((capacity + 1), dtype=int)

        row_num = len(weights)
        col_num = capacity + 1

        bag_states=[]


        for i in range(row_num):

            next_states_value = zeros((capacity + 1), dtype=int) #当前背包中的物品价值之和


            if weights[i] <= capacity: next_states_value[weights[i]]=values[i] # 背包中 只有 第i 个物品

            # col_end=capacity-(weights[i])

            for j in range(col_num):

                if  states_value[j] > 0: #上一次 背包中的物品价值之和

                    if next_states_value[j] < states_value[j]: #把每一层中 (i, j) 重复的状态合并，只记录 物品价值 最大的那个状态
                        next_states_value[j]=states_value[j]  # 价值大才更新

                    if j + weights[i]<= capacity:
                        next_states_value[j + weights[i]] =states_value[j]+ values[i]

            states_value = next_states_value

            print(i,states_value)

            bag_states.append( [[0,0]] + [ [index,states_value[index]]  for index in range( capacity + 1 ) if states_value[index]!=0 ]) # 状态转移矩阵过于稀疏，只保留有值的记录


        print(bag_states) #   [col0 , col1]=[8,17] ( col1 为背包的重量 col2 为包内物品的价值)
        # [
        #   [[0, 0], [2, 3]],
        #   [[0, 0], [2, 6], [4, 9]],
        #   [[0, 0], [2, 6], [4, 9], [6, 14], [8, 17]]
        # ]

        #背包 最重的 那个状态 不一定是 价值最大的
        max_value=max(states_value)
        max_value_index=argmax(states_value)

        # 根据最大的背包价值，反推出 背包中的物品
        bag=[]
        bag_weight=max_value_index # 包中的物品价值最大时，背包的重量
        bag_value=max_value

        for i in range(row_num-1,-1,-1):
            if i>0:
                if  (bag_weight-weights[i] in  array(bag_states[i-1])[:,0] ) and (bag_value-values[i] in  array(bag_states[i-1])[:,1] ):
                # case1: 第i个物品装入背包
                # 1.bag_states[i-1] 上一个状态
                # 2. array(bag_states[i-1])[:,0] 上一个状态的 背包的重量; array(bag_states[i-1])[:,1] 上一个状态的 背包的价值
                # 3. (当前背包的重量 - 第i个物品的重量) 为 上一个状态的 背包的重量 同时 (当前背包的价值 - 第i个物品的价值) 为 上一个状态的 背包的价值，则说明 第i 个物品被放入背包

                    bag.append(i)
                    bag_weight=bag_weight-weights[i]
                    bag_value=bag_value-values[i]

                else:
                # case2: 第i个物品未装入背包,背包的重量和价值不变
                    pass

            elif i==0:
                if bag_weight==weights[i] and bag_value==values[i]:
                    bag.append(i)

        return max_value,bag



if __name__ == '__main__':

    weights=[2,2,4,6,3]
    # weights = [2,2,4]
    sol = solution_zero_one_bag_weight()
    # print(sol.run_recursive_cache(weights,8))
    # print(sol.run_recursive_cache(weights, 21))

    # print(sol.run_dynamic_programming_v3(weights,9))


    weights = [2, 2, 4]
    values=[3,6,8]

    sol2=solution_zero_one_bag_value()
    # print(sol2.run_recursive_cache(weights,values,5))
    # print(sol2.run_dynamic_programming_v2(weights, values, 5))
    print(sol2.run_dynamic_programming_v2(weights, values, 9))

    weights = [2, 2, 4, 6, 3]
    values=[3,4,8,9,6]

    # print(sol2.run_recursive_cache(weights, values, 17))
    # print(sol2.run_dynamic_programming_v2(weights, values, 17))

    prices=[10,20,400,100,50,60]
    sol3=solution_double11advance()

    # print(sol3.run_dynamic_programming_v3(prices,200))
    # print(sol3.run_dynamic_programming_v4(prices, 200))
    # print(sol3.run_dynamic_programming_v3(prices,400))
    # print(sol3.run_dynamic_programming_v4(prices, 400))


















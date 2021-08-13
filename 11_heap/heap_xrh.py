#!/usr/bin/python
# -*- coding: UTF-8 -*-

import timeit
import numpy as np

import random as rand

import math

class Heap:

    def __init__(self, input_list,flag='large'): # 默认为 构建大顶堆
        """
        flag=='large' 构建 大顶堆
        flag=='small' 构建 小顶堆
        :param input_list: 
        :param flag: 
        """

        self.heap_list=input_list
        self.heap_list.insert(0,None) # 0 号位置 空置

        self.length=len(self.heap_list) # 初始 状态 heap_list 的长度为1

        self.flag=flag

        for i in range(self.length//2,0,-1): #我们对下标从 n/2 开始到 1 的数据进行堆化,下标是 n/2+1 到 n 的节点是叶子节点，我们不需要堆化

            if self.flag=='large':
                self.__up_to_down_max_heapify(i) # 从子树的 根节点开始 往下堆化

            elif self.flag=='small':
                self.__up_to_down_min_heapify(i)  # 从子树的 根节点开始 往下堆化

    def __up_to_down_max_heapify(self,p):
        """
        从上往下 堆化，构建大顶堆
        :param p: 
        :return: 
        """
        while p<self.length:

            left_child=2*p
            right_child=2*p+1

            max_index=p

            # 找出 当前节点 左子节点 和 右子节点  中的最大节点
            if left_child<self.length and self.heap_list[left_child] > self.heap_list[max_index]:
                max_index=left_child

            if  right_child <self.length and self.heap_list[right_child] > self.heap_list[max_index]:
                max_index=right_child

            self.heap_list[max_index], self.heap_list[p] = self.heap_list[p], self.heap_list[max_index] # 当前节点与 最大节点进行交换

            if p==max_index: #最大节点 还是 当前节点，说明无需做堆的调整 ，退出循环
                break
            p=max_index

    def __up_to_down_min_heapify(self,p):
        """
        从上往下 堆化，构建小顶堆
        :param p: 
        :return: 
        """
        while p<self.length:

            left_child=2*p
            right_child=2*p+1

            min_index=p

            # 找出 当前节点 左子节点 和 右子节点  中的最小节点
            if left_child<self.length and self.heap_list[left_child] < self.heap_list[min_index]:
                min_index=left_child

            if  right_child <self.length and self.heap_list[right_child] < self.heap_list[min_index]:
                min_index=right_child

            self.heap_list[min_index], self.heap_list[p] = self.heap_list[p], self.heap_list[min_index] # 当前节点与 最大节点进行交换

            if p==min_index: #最小节点 还是 当前节点，说明无需做堆的调整 ，退出循环
                break
            p=min_index


    def __down_to_up_max_heapify(self,p):
        """
         从下往上 堆化，构建大顶堆
        :param p: 
        :return: 
        """
        while p >0 :
            parent=p//2
            min_index = p

            # 找出 子 父 节点中 的最小节点
            if parent >0 and self.heap_list[parent]<self.heap_list[p]:
                min_index = parent

            self.heap_list[p],self.heap_list[min_index]=self.heap_list[min_index],self.heap_list[p] # 当前节点和 最小节点 进行交换

            if p == min_index:  # 最小节点 还是 当前节点，说明无需做堆的调整 ，退出循环
                break

            p=min_index

    def __down_to_up_min_heapify(self,p):
        """
         从下往上 堆化，构建小顶堆
        :param p: 
        :return: 
        """
        while p >0 :
            parent=p//2
            max_index = p

            # 找出 子节点,父节点中 的最大节点
            if parent >0 and self.heap_list[parent]>self.heap_list[p]:
                max_index = parent

            self.heap_list[p],self.heap_list[max_index]=self.heap_list[max_index],self.heap_list[p] # 当前节点和 最小节点 进行交换

            if p == max_index:  # 最大节点 还是 当前节点，说明无需做堆的调整 ，退出循环
                break

            p=max_index

    def insert(self,val):
        """
        堆中插入一个元素
        :param val: 
        :return: 
        """
        self.heap_list.append(val) #把新插入的元素放到堆的最后
        self.length+=1

        if self.flag == 'large':
            self.__down_to_up_max_heapify(self.length - 1)

        elif self.flag == 'small':
            self.__down_to_up_min_heapify(self.length - 1)




    def pop(self):
        """
        弹出堆顶的元素
        :return: 
        """
        res=self.heap_list[1]

        self.heap_list[1]=self.heap_list[self.length-1]

        self.heap_list.pop()
        self.length-=1

        p=1

        if self.flag == 'large':
            self.__up_to_down_max_heapify(p)

        elif self.flag == 'small':
            self.__up_to_down_min_heapify(p)


        return res

    def sort(self):
        """
        实现 堆排序
        如果是大顶堆，为正序
        如果是小顶堆，为逆序
        :return: 
        """
        for i in range(self.length-1,0,-1):
            self.heap_list[i],self.heap_list[1]=self.heap_list[1],self.heap_list[i]
            self.length-=1

            if self.flag == 'large':
                self.__up_to_down_max_heapify(1)

            elif self.flag == 'small':
                self.__up_to_down_min_heapify(1)


        return self.heap_list[1:]

    def _draw_heap(self,data):
        """
        格式化打印 堆
        :param data:
        :return:
        """
        data=data[1:]
        length = len(data)

        if length == 0:
            return 'empty heap'

        ret = ''
        for i, n in enumerate(data):
            ret += str(n)
            # 每行最后一个换行
            if i == 2**int(math.log(i+1, 2)+1) - 2 or i == len(data) - 1:
                ret += '\n'
            else:
                ret += ', '

        return ret

    def __repr__(self):
        return self._draw_heap(self.heap_list)


class soultions:

    def topK_init(self,K):
        """
        动态数组的 最大的 K 个元素
        
        可以一直都维护一个 大小为 K 的小顶堆，当有数据被添加到集合中时，我们就拿它与堆顶的元素对比。
        如果比堆顶元素大，我们就把堆顶元素删除，并且将这个元素插入到堆中
        :param ele: 
        :param K: 
        :return: 
        """

        if K>0 and type(K)==int:
            self.Heap=Heap([],flag='small')

            self.capacity=K

            return 1
        else:
            return 0

    def topK_insert(self,ele):

        if self.Heap.length < self.capacity+1: # 堆还未 装满
            self.Heap.insert(ele)

        else:

            if ele > self.Heap.heap_list[1]: # ele 与 堆顶的元素 对比
                top_ele=self.Heap.pop()
                # print('pop: ',top_ele)
                self.Heap.insert(ele)

        return  self.Heap

    def median_init(self):
        """
        动态数据集合，求中位数
        
        维护两个堆，一个大顶堆，一个小顶堆。
        大顶堆中存储前半部分数据（小的数据），小顶堆中存储后半部分数据（大的数据）
        
        :return: 
        """

        self.max_Heap=Heap([],flag='large')
        self.min_Heap=Heap([],flag='small')

        self.N=(self.max_Heap.length-1)+(self.min_Heap.length-1)

    def median_insert(self,ele):
        """
        动态插入 新的元素
        :param ele: 
        :return: 
        """

        if self.N==0 or ele < self.max_Heap.heap_list[1]: #如果新加入的数据小于等于大顶堆的堆顶元素，就将这个新数据插入到大顶堆
            self.max_Heap.insert(ele)

        else:
            self.min_Heap.insert(ele)

        self.N = (self.max_Heap.length-1)+(self.min_Heap.length-1)

        if self.N % 2==0: # n 是偶数

            while self.max_Heap.length!=self.min_Heap.length:

                if self.max_Heap.length<self.min_Heap.length:
                    ele=self.min_Heap.pop()
                    self.max_Heap.insert(ele)

                elif self.max_Heap.length > self.min_Heap.length:
                    ele = self.max_Heap.pop()
                    self.min_Heap.insert(ele)

        else: #n 是奇数

            while self.max_Heap.length!=self.min_Heap.length+1:

                if self.max_Heap.length<self.min_Heap.length+1:
                    ele=self.min_Heap.pop()
                    self.max_Heap.insert(ele)

                elif self.max_Heap.length > self.min_Heap.length+1:
                    ele = self.max_Heap.pop()
                    self.min_Heap.insert(ele)

    def median_output(self):
        """
        输出 中位数 
        :return: 
        """

        if self.N>=1:
            return self.max_Heap.heap_list[1]




if __name__ == '__main__':

    ##-------- part1 堆和 堆排序 ------------##

    # max_heap= Heap([7,5,19,8,4,1,20,13,16],flag='large')
    # print(max_heap)
    #
    # max_heap.insert(22)
    # print(max_heap)
    #
    # print('pop: ',max_heap.pop())
    # print(max_heap)
    #
    # print(max_heap.sort())
    #
    # min_heap=Heap([7,5,19,8,4,1,20,13,16],flag='small')
    # print(min_heap.sort())

    ##-------------- part1 end -----------------##

    ##-------- part2 堆的应用 ------------##

    sol = soultions()
    l = np.random.randint(20, size=10)
    l1=list(l)
    print(l1)

    # 1. 求 topK

    # sol.topK_init(3)
    # for ele in l1:
    #     heap=sol.topK_insert(ele)
    #     print(heap)


    # 2. 求 中位数

    sol.median_init()

    for ele in l1:
        sol.median_insert(ele)

    print(sol.max_Heap)
    print(sol.min_Heap)
    print(sol.median_output())




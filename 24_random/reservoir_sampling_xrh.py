
#!/usr/bin/python
# -*- coding: UTF-8 -*-

import random

from deprecated import deprecated

class Solution:
    """
    蓄水池抽样

    它的使用场景是从不知道大小的集合里，只需要一次遍历，就能够等概率的提取 k个元素。

    Author: xrh
    Date: 2021-06-20

    """

    def __init__(self, nums):

        self.nums = nums

    def pick(self ) :
        """
        蓄水池抽样  K=1

        :return:
        """

        nums_it = iter(self.nums) # 使用迭代器节省内存; 每次调用, 迭代器需要初始化

        i = 1
        res = None

        for ele in nums_it: # 迭代器 一次只访问一个元素

            p = random.random() # 随机生成的一个实数，它在[0,1)范围内。

            if p < (1 / i):
                res = ele

            i += 1

        return res

    @deprecated(version='1.0', reason="You should use another function")
    def pick_K_deprecated(self , K) :
        """
        蓄水池抽样

        返回长度为 K 的结果列表

        :param K:
        :return:
        """

        nums_it = iter(self.nums) # 使用迭代器节省内存; 每次调用, 迭代器需要初始化

        i = 1
        res = []

        for ele in nums_it: # 迭代器 一次只访问一个元素, 模拟一个无限的流

            if i<=K:
                res.append(ele)

            else: # i>K

                p = random.random() # 随机生成的一个实数，它在[0,1)范围内。

                if p < (K / i):

                    idx = random.randint(0,K-1) # 随机生成的一个整数，它在[0,K-1] 范围内。
                    res[idx]=ele


            i += 1

        return res


    def pick_K(self , K) :
        """
        蓄水池抽样

        返回长度为 K 的结果列表

        优化:
        1. 由取两次随机数 改为只取一次

        :param K:
        :return:
        """

        nums_it = iter(self.nums) # 使用迭代器节省内存; 每次调用, 迭代器需要初始化

        i = 1
        res = []

        for ele in nums_it: # 迭代器 一次只访问一个元素, 模拟一个无限的流

            if i<=K:
                res.append(ele)

            else: # i>K

                idx = random.randint(0, i - 1)  # 随机生成的一个整数，它在[0,i-1] 范围内

                if idx < K-1 :
                    res[idx]=ele

            i += 1

        return res

if __name__ == '__main__':

    a = [1, 2, 3, 3, 3]
    sol = Solution(a)

    # for __ in range(10):
    #     print(sol.pick())

    for __ in range(10):
        print(sol.pick_K(2))














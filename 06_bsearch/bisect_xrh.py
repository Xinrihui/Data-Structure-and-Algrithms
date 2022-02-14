# !/usr/bin/python
# -*- coding: UTF-8 -*-

import timeit

from collections import *

import numpy as np


class bisect(object):
    """
    实现 python 自带的 bisect 模块
    
    by XRH 
    date: 2021-03-01
     
    """

    def bisect_key_left(self, nums, target):
        """

        (1) 若查找不到 target  则 bisect_key_left 与  bisect_key_right 都返回  右边的插入位置
            若能找到 target ，则  bisect_key_left 返回左边的插入位置 而 bisect_key_right 返回右边的插入位置

        (2) 若能找到 target , 且 nums 中有 重复元素(target) , 返回 第一个 target 出现的位置(即最左边的位置) 


        :param nums: 
        :param target: 
        :return: flag,idx
        """

        l = 0
        r = len(nums) - 1

        while l <= r:
            mid = l + (r - l) // 2

            if target == nums[mid]:

                if mid == 0 or nums[mid - 1] != target:
                    return True, mid

                else:
                    r = mid - 1


            elif target < nums[mid]:
                r = mid - 1

            else:
                l = mid + 1

        return False, l

    def bisect_key_right(self, nums, target):
        """
        (1) 若查找不到 target  则 bisect_key_left 与  bisect_key_right 都返回  右边的插入位置
            若能找到 target ，则  bisect_key_left 返回左边的插入位置 而 bisect_key_right 返回右边的插入位置

        (2) 若能找到 target ,且 nums 中有 重复元素(target) 返回 最后一个 出现的位置(最右边的位置) 的右边

        :param nums: 
        :param target: 
        :return: flag,idx
        """

        l = 0
        r = len(nums) - 1

        while l <= r:
            mid = l + (r - l) // 2

            if target == nums[mid]:

                if mid == len(nums) - 1 or nums[mid + 1] != target:
                    return True, mid + 1

                else:
                    l = mid + 1


            elif target < nums[mid]:
                r = mid - 1

            else:
                l = mid + 1

        return False, l


if __name__ == '__main__':

    bis = bisect()

    # IDE 测试 阶段：

    # 找不到 key
    # print(bis.bisect_key_left([], 1))  # 0
    # print(bis.bisect_key_left([3, 5, 7], 1))  # 0
    # print(bis.bisect_key_left([3, 5, 7], 4))  # 1
    # print(bis.bisect_key_left([3, 5, 7], 6))  # 2
    # print(bis.bisect_key_left([3, 5, 7], 8))  # 3

    # 找到 key
    # print(bis.bisect_key_left([3, 5, 7], 3))  # 0
    # print(bis.bisect_key_left([3, 5, 7], 5))  # 1

    # 找到 key 且key 重复
    print(bis.bisect_key_left([1, 3, 3, 3, 5, 7], 3))  # 1


    # # 找不到 key
    # print(bis.bisect_key_right([], 1))  # 0
    # print(bis.bisect_key_right([3, 5, 7], 1))  # 0
    # print(bis.bisect_key_right([3, 5, 7], 4))  # 1
    # print(bis.bisect_key_right([3, 5, 7], 6))  # 2
    # print(bis.bisect_key_right([3, 5, 7], 8))  # 3
    #
    # # 找到 key
    # print(bis.bisect_key_right([3, 5, 7], 3))  # 1
    # print(bis.bisect_key_right([3, 5, 7], 5))  # 2
    #
    # # 找到 key 且key 重复
    # print(bis.bisect_key_right([1, 3, 3, 3, 5, 7], 3))  # 4



















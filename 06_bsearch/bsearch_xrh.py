#!/usr/bin/python
# -*- coding: UTF-8 -*-

import timeit
import numpy as np

import sys
import random as rand



class solutions:

    def bsearch(self, nums,target):
        """
        二分查找 
        :param nums: 
        :param target: 
        :return: 
        """

        nums=sorted(nums)

        left=0
        right=len(nums)-1

        while left<=right:

            mid= left+(right-left)//2 #防止 left+right 过大溢出（其实 python 也不可能溢出）

            if nums[mid]==target:
                return mid # 返回下标

            elif target>nums[mid]:
                left=mid+1

            elif target<nums[mid]:
                right=mid-1

        return None

    def sqrt(self,a):
        """
        求一个数的平方根 ,要求精确到小数点后 6 位
        
        分治法
        :param a: 
        :return: 
        """
        left = 0
        right=float(a)

        while right-left >= 1.0e-6:

            mid= (right+left)/2.0

            if pow(mid,2)==a:
                return mid

            elif a>pow(mid,2):

                left = mid

            elif a<pow(mid,2):
                right = mid

        return left

    def besearch_first_equal(self,nums,target):
        """
        查找第一个值等于给定值的元素
        :param nums: [1,3,4,5,6,8,8,8,11,18]
        :param target: 8
        :return: 5
        """
        left=0
        right=len(nums)-1

        while left<=right:

            mid= left+(right-left)//2 #防止 left+right 过大溢出

            if nums[mid]==target:
                #往左边找
                if mid==0 or (mid-1>=0 and nums[mid-1]!=target):
                    return mid
                else:
                    while mid>=0 and nums[mid]==target:
                        mid-=1
                    return mid+1


            elif target>nums[mid]:
                left=mid+1

            elif target<nums[mid]:
                right=mid-1

        return None

    def besearch_last_equal(self,nums,target):
        """
        查找最后一个值等于给定值的元素
        :param nums: [1,3,4,5,6,8,8,8,11,18]
        :param target: 8
        :return: 7
        """
        left=0
        right=len(nums)-1

        while left<=right:

            mid= left+(right-left)//2 #防止 left+right 过大溢出

            if nums[mid]==target:
                #往右边找
                if mid== len(nums)-1 or (mid+1<=len(nums)-1 and nums[mid+1]!=target):
                    return mid
                else:
                    while mid<=len(nums)-1 and nums[mid]==target:
                        mid+=1
                    return mid-1


            elif target>nums[mid]:
                left=mid+1

            elif target<nums[mid]:
                right=mid-1

        return None


    def besearch_first_large(self,nums,target):
        """
        查找第一个大于等于给定值的元素
        :param nums: [1,3,4,5,6,8,8,8,11,18]
        :param target: 7
        :return: 5 nums[5]==8
        """
        left=0
        right=len(nums)-1

        while left<=right:

            mid= left+(right-left)//2 #防止 left+right 过大溢出

            if target>nums[mid]:
                left=mid+1


            elif target<=nums[mid]:
                # 往左边找
                if mid == 0 or (mid - 1 >= 0 and nums[mid - 1] < target):
                    return mid
                else:
                    while mid >= 0 and nums[mid]  >= target:
                        mid -= 1
                    return mid + 1

        return None

    def besearch_last_small(self,nums,target):
        """
        查找最后一个小于等于给定值的元素
        :param nums: [1,3,4,5,6,8,8,8,11,18]
        :param target: 7
        :return: 4 nums[4]==6
        """
        left=0
        right=len(nums)-1

        while left<=right:

            mid= left+(right-left)//2 #防止 left+right 过大溢出

            if target>=nums[mid]:
                # 往右边找
                if mid== len(nums)-1 or (mid+1<=len(nums)-1 and nums[mid+1] > target):
                    return mid
                else:
                    while mid <= len(nums) - 1 and nums[mid] <= target:
                        mid += 1
                    return mid - 1

            elif target<nums[mid]:
                right=mid+1

        return None


    def __bsearch(self, nums,target,left,right):
        """
        二分查找 
        :param nums: 
        :param target: 
        :return: 
        """


        while left<=right:

            mid= left+(right-left)//2 #防止 left+right 过大溢出（其实 python 也不可能溢出）

            if nums[mid]==target:
                return mid # 返回下标

            elif target>nums[mid]:
                left=mid+1

            elif target<nums[mid]:
                right=mid-1

        return None

    def besearch_loop_sorted(self,nums,target):
        """
        有序数组是一个循环有序数组，比如 4，5，6，1，2，3。
        实现一个求 “值等于给定值” 的二分查找算法
        
        :param nums: [4,5,6,1,2,3]
        :param target: 6
        :return: 2
        """

        divide_index=0 #分界下标

        i = 1
        while i<len(nums):

            if nums[i] < nums[i-1]:
                divide_index=i
                break
            i+=1

        nums1_left=0
        nums1_right=divide_index-1

        nums2_left=divide_index
        nums2_right=len(nums)-1

        res=None

        if target <= nums[nums1_right]:
            res=self.__bsearch(nums,target,nums1_left,nums1_right)

        else: # target>nums[nums1_right]
            res = self.__bsearch(nums, target, nums2_left, nums2_right)

        return res



if __name__ == '__main__':

    sol = solutions()

    ##-------- part1 二分查找 基础 ------------##
    # l=np.random.randint(int(1e7),size=int(1e7)) # 1000 万个整数
    # l1=list(l)
    # print('l1 list memory_size:', sys.getsizeof(l1),'B') # 90000112 B= 90MB
    #
    # l2=set(l)
    # print('l2 hash memory_size:', sys.getsizeof(l2),'B') #  268435680 B= 268MB
    #
    # start = timeit.default_timer()
    # print('by bsearch: ')
    # print(sol.bsearch(l1, 100)) # 在l1 中查找100
    # end = timeit.default_timer()
    # print('time: ', end-start ,'s')
    #
    # start = timeit.default_timer()
    # print('by hash search: ')
    # print( 100 in l2 )
    # end = timeit.default_timer()
    # print('time: ', end-start ,'s')


    print(pow(9,0.5),sol.sqrt(9))
    print(pow(4, 0.5), sol.sqrt(4))
    print(pow(10, 0.5), sol.sqrt(10))

    ##-------------- part1 end -----------------##


    ##-------- part2 二分查找 进阶 ------------##


    ##-------------- part2 end -----------------##

    # print(sol.besearch_first_equal( [1,3,4,5,6,8,8,8,11,18],8))
    # print(sol.besearch_last_equal([1, 3, 4, 5, 6, 8, 8, 8, 11, 18], 8))

    # print(sol.besearch_first_large([1, 3, 4, 5, 6, 8, 8, 8, 11, 18], 7))
    # print(sol.besearch_last_small([1, 3, 4, 5, 6, 8, 8, 8, 11, 18], 7))

    print(sol.besearch_loop_sorted([4,5,6,1,2,3],6))
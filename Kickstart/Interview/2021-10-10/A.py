#!/usr/bin/python
# -*- coding: UTF-8 -*-

import timeit
from collections import *
import numpy as np


class Solution1:

    # 满足要求的连续最长子数组
    def __solve_Q1(self, prices, budget):
        """
        1.设置 区间的左端点 和 右端点
        2.计算当前区间的和
          若区间和小于限制, 则右端点往右移动, 区间的和加上右端点的值
          若区间和大于限制, 则左端点往左移动, 区间的和减去左端点的值

        :param prices:
        :param budget:
        :return:
        """

        left = 0
        right = -1

        max_length = float('-inf')
        region_sum = 0

        while right < len(prices) and left < len(prices):  # 7

            print('left:{}, right:{}, region_sum:{}'.format(left, right, region_sum))

            if region_sum <= budget:

                while region_sum <= budget and right < len(prices):  # 随时检查数组越界

                    region_length = right - left + 1
                    max_length = max(region_length, max_length)

                    right += 1  # 注意检查数组越界
                    if right < len(prices):
                        region_sum += prices[right]

            else:

                while region_sum > budget and left < len(prices):
                    region_sum -= prices[left]
                    left += 1

        return max_length

    # 满足要求的连续最长子数组
    def solve_Q1(self, prices, budget):
        """
        1.设置 区间的左端点 和 右端点
        2.计算当前区间的和
          若区间和小于限制, 则右端点往右移动, 区间的和加上右端点的值
          若区间和大于限制, 则左端点往左移动, 区间的和减去左端点的值

        :param prices:
        :param budget:
        :return:
        """

        left = 0
        right = 0

        max_length = 0
        region_sum = 0

        while right < len(prices)+1:  # 8+1=9

            print('left:{}, right:{}, region_sum:{}'.format(left, right, region_sum))

            if region_sum <= budget:

                while region_sum <= budget and right < len(prices)+1:  # 随时检查数组越界

                    region_length = right - left
                    max_length = max(region_length, max_length)

                    right += 1  # 注意检查数组越界
                    if right < len(prices)+1:
                        region_sum += prices[right-1]


            else:

                while region_sum > budget and left < len(prices):
                    region_sum -= prices[left]
                    left += 1

        return max_length




class Solution2:

    def find_qualify_square(self, prices, start, end, diagonal_region_sum, budget):
        """
        找到 一个合格的正方形区域, 区域内的正方形的和小于 budget

        :param prices:
        :param start:
        :param end:
        :param basic_region_sum:
        :param budget:
        :return:
        """

        right = end
        left = start
        region_sum = diagonal_region_sum

        # 向右边走
        while right < len(prices[0]):

            if region_sum <= budget:
                return True

            right += 1
            if right < len(prices[0]):
                region_sum = region_sum + np.sum(prices[start:end+1, right]) - np.sum(prices[start:end+1, left])
                left += 1

        up = start
        down = end
        region_sum = diagonal_region_sum

        # 往下走
        while down < len(prices):

            if region_sum <= budget:
                return True

            down += 1
            if down < len(prices):
                try:
                    region_sum = region_sum + np.sum(prices[down, start:end+1]) - np.sum(prices[up, start:end+1])
                except Exception as err:
                    print(err)  # debug 时 , 在此处打断点

                up += 1

        return False


    def solve_Q2(self, prices, budget):
        """
         满足要求的最大正方形的边长

        :param prices:
        :param budget:
        :return:
        """

        prices = np.array(prices)

        start = 0
        end = 0

        max_length = 0  # 所求最大区域的边长

        diagonal_region_sum = prices[0][0]  # 主对角线上的正方形区域

        diagonal_len = len(prices)  # 整个正方形的对角线的长度

        while start <= end and end < diagonal_len:  # 3

            print('start:{}, end:{}, diagonal_region_sum:{}'.format(start, end, diagonal_region_sum))

            if self.find_qualify_square(prices, start, end, diagonal_region_sum, budget) is True:  # 找到了


                region_length = end - start + 1
                max_length = max(region_length, max_length)
                # expand region
                end += 1  # 注意检查数组越界
                if end < diagonal_len:

                    # 在原区域的右下角增加一行和一列
                    diagonal_region_sum += np.sum(prices[end, start:end+1]) + np.sum(prices[start:end+1, end]) - prices[end, end]


            else: # 找不到
            # move region

                # 在原区域的左上角减去一行一列
                diagonal_region_sum -= np.sum(prices[start, start:end + 1]) + np.sum(prices[start:end + 1, start]) - prices[start, start]
                start += 1

                end += 1 # 注意检查数组越界
                if end < diagonal_len:
                    #  在原区域的右下角增加一行和一列
                    diagonal_region_sum += np.sum(prices[end, start:end + 1]) + np.sum(prices[start:end + 1, end]) - prices[end, end]


        return max_length




class Test:

    def test_solve_Q1(self):
        sol = Solution1()

        # Case1
        prices = [3, 2, 3, 1, 4, 1, 1, 3]
        budget = 7
        assert sol.solve_Q1(prices, budget) == 4
        print('test pass')

        # Case2
        prices = [1, 8, 2, 2]
        budget = 7
        assert sol.solve_Q1(prices, budget) == 2
        print('test pass')

        # Case3
        prices = [1, 8, 2, 2, 7]
        budget = 7
        assert sol.solve_Q1(prices, budget) == 2
        print('test pass')

        # Case4
        prices = [8]
        budget = 7
        assert sol.solve_Q1(prices, budget) == 0
        print('test pass')

        # Case5
        prices = [7]
        budget = 7
        assert sol.solve_Q1(prices, budget) == 1
        print('test pass')

    def test_solve_Q2(self):
        sol = Solution2()

        # Case1
        prices = [[7, 8, 9],
                  [3, 1, 2],
                  [3, 2, 1]]

        budget = 7
        assert sol.solve_Q2(prices, budget) == 2

        # Case2
        prices = [[7, 8, 8, 1, 1],
                 [3, 6, 1, 1, 1],
                 [3, 2, 1, 1, 1],
                 [0, 0, 1, 1, 1],
                 [0, 0, 0, 0, 9]]

        budget = 9
        assert sol.solve_Q2(prices, budget) == 3

        # Case3
        prices = [[7]]

        budget = 7
        assert sol.solve_Q2(prices, budget) == 1

        # Case4
        prices = [[8]]

        budget = 7
        assert sol.solve_Q2(prices, budget) == 0


if __name__ == '__main__':
    # IDE 测试 阶段：

    test = Test()

    # test.test_solve_Q1()

    test.test_solve_Q2()


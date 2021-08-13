
#!/usr/bin/python
# -*- coding: UTF-8 -*-

import random

from deprecated import deprecated

class Solution:
    """

    Knuth （Fisher-Yates）洗牌算法

    Author: xrh
    Date: 2021-06-20

    """

    def __init__(self, nums ):
        self.nums_bak = [ele for ele in nums]

        self.nums = nums

    def reset(self) :
        """
        Resets the array to its original configuration and return it.
        """
        self.nums = [ele for ele in self.nums_bak]

        return self.nums

    def shuffle(self) :
        """
        Returns a random shuffling of the array.

        """
        nums = self.nums

        L = len(nums)

        for i in range(L - 1, 0, -1):

            # try:

            #  nums= [1, 2, 3, 4, 5]
            #  i=4  ridx=4 自己和自己互换可行

            ridx = random.randint(0, i)  # 随机生成的一个整数，它在[0,i] 范围内 ;
                                         # 必须包含i 以保证 每个元素出现在数组的任何位置的概率一样

            # except Exception as err:
            #     print(err)  # debug 时 , 在此处打断点

            nums[ridx], nums[i] = nums[i], nums[ridx]

        return nums

if __name__ == '__main__':

    a = [1, 2, 3, 4, 5]
    sol = Solution(a)

    for __ in range(10):
      print(sol.shuffle())


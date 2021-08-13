#!/usr/bin/python
# -*- coding: UTF-8 -*-


from collections import *

from numpy import *


class KMP:
    def get_next(self, pattern):
        """

        next 数组 表示 模式串的子串中 前缀和后缀相匹配的最长的长度

        时间复杂度： O(m) m 为 模式串的长度
        空间复杂度： O(m)

        eg.  pattern = 'a b c d a b c a'
                index:  0 1 2 3 4 5 6 7
             next   =   0 0 0 0 1 2 3 1

         next[4]=1 ： pattern 的子串 pattern[0:4+1]= 'a b c d a'  前缀为 'a' 后缀为 'a' ，前缀==后缀 
         next[5]=2 ： pattern 的子串 pattern[0:5+1]= 'a b c d a b'  前缀为 'ab' 后缀为 'ab' ，前缀==后缀 
         next[6]=3 ： pattern 的子串 pattern[0:6+1]= 'a b c d a b c'  前缀为 'abc' 后缀为 'abc' ，前缀==后缀 


        ref: https://www.bilibili.com/video/BV1Ys411d7yh?from=search&seid=767115274745432726
        :param pattern: 
        :return: 
        """
        m = len(pattern)
        next = [0] * m

        j = 0
        for i in range(1, m):

            if pattern[i] == pattern[j]:
                next[i] = j + 1
                j += 1

            else:  # pattern[i]!=pattern[j]

                # M1:
                while pattern[i] != pattern[j]:  # j 要 向前移动

                    if j == 0:  # pattern[i]!=pattern[j] and j==0
                        next[i] = 0
                        break

                    j = next[j - 1]

                else:  # pattern[i]==pattern[j] 正常退出 while ; j==0 or j!=0
                    next[i] = j + 1

                    # M2 :
                    # while pattern[i] != pattern[j] and j!=0: # j==0 或者 pattern[i] == pattern[j] 跳出循环
                    #     j = next[j - 1]
                    #
                    # if pattern[i]==pattern[j] :
                    #     next[i] = j + 1
                    #
                    # else:
                    #     next[i]=0 #   pattern[i]!=pattern[j] => j==0

        return next

    def match(self, main_string, pattern):
        """

        发现不匹配的字符后，我尽可能的多移动一点

        匹配的时间复杂度： O(n)  n为 主串的长度
        求 next 数组的时间复杂度：O(m) m 为 模式串长度
        整个 kmp 的复杂度 ： O(m+n)

        ref: https://www.bilibili.com/video/BV1Ys411d7yh?from=search&seid=767115274745432726
        :param main_string: 主串
        :param pattern:  模式串
        :return: (匹配的开始位置，匹配的结束位置)
        """
        n = len(main_string)
        m = len(pattern)

        i = 0  # 主串指针
        j = 0  # 模式串指针

        next = self.get_next(pattern)

        # print('next: ',next)

        while i < n and j < m:

            if main_string[i] == pattern[j]:
                i += 1
                j += 1

            else:
                if j != 0:
                    j = next[j - 1]

                else:  # main_string[i]!=pattern[j]  and j==0
                    i += 1

        if j == m:
            return i - m, i
        else:
            return None, None

    def find_all(self,main_string, pattern):
        """
        多次匹配的 字符串查找算法：
        （1）bf rk bm kmp 都是单次匹配的 字符串查找算法，即在 main_string 中找 pattern ，找到一次就退出；
        （2）循环调用  kmp 把 main_string 中 出现所有 pattern 都找出来
        :return: 
        """
        # right=-1
        left=-1

        res=[]

        while True:

            # offset = right + 1

            offset = left + 1

            left,right=self.match(main_string[offset:],pattern) # 从 已匹配 的 左端点的 后一位 开始重新搜索 main_string

            if right==None: # 找不到 ，退出循环
                break

            left=left+offset
            right=right+offset

            res.append((left,right))

        return res


if __name__ == '__main__':

    kmp=KMP()

    # print(kmp.match('abxabcabcaby','abcaby'))

    print(kmp.match('abxabcabcaby', 'ab'))

    # print(kmp.match('ababababca','abababca'))

    print(kmp.find_all('ababaeabac', 'ab'))
    # print(kmp.find_all('ababaeabac', 'aba'))


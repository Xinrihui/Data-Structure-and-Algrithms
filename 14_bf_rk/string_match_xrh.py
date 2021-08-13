#!/usr/bin/python
# -*- coding: UTF-8 -*-

from collections import *

from numpy import *

class solutions:

    def bf(self,main_string, pattern):
        """
        字符串匹配，bf暴搜
        :param main: 主串
        :param pattern: 模式串
        :return:
        """
        if len(main_string)==0 or len(pattern)==0 or len(main_string)<len(pattern):
            return None

        start=0
        length=len(pattern) # length= 3
        end=start+length-1  # start=0 end=2

        while end< len(main_string):

            if main_string[start:end+1]== pattern:
                return start,end

            else:
                start+=1
                end = start + length - 1

        return None

    def bf_2D(self,main_string, pattern):
        """
        二维空间的 字符串匹配
        :param main: 主串 
        :param pattern: 模式串
        :return:
        """
        main_string=array(main_string)
        pattern=array(pattern)

        main_row_num=main_string.shape[0]
        main_col_num=main_string.shape[1]

        pattern_row_num=pattern.shape[0]
        pattern_col_num=pattern.shape[1]


        pattern_1D=pattern.flatten() # 把 pattern 拍成一维向量

        for i in range(0,main_row_num-pattern_row_num+1):

            for j in range(0,main_col_num-pattern_col_num+1):

                if ''.join((main_string[i:i+pattern_row_num,j:j+pattern_col_num]).flatten())== ''.join(pattern_1D): # ''.join() ：把 list 转换为字符串

                    return (i,i+pattern_row_num),(j,j+pattern_col_num)

        return  None


    def simple_hash(self, s, start, end):
        """
        计算子串的哈希值
        每个字符取 acs-ii 码后求和
        :param s:
        :param start:
        :param end:
        :return:
        """

        res=0
        for c in s[start:end+1]:
            res+=ord(c)

        return res

    def rk(self,main_string, pattern):
        """
        RK 算法，利用 hash 算法计算 pattern 和 main_string 的 hash 值进行比较
        :param main_string: 
        :param pattern: 
        :return: 
        """

        if len(main_string)==0 or len(pattern)==0 or len(main_string)<len(pattern):
            return None

        length=len(pattern) # length= 3
        pattern_hash=self.simple_hash(pattern,0,length-1)

        start=0
        end = start + length - 1  # start=1 end=3
        prev_main_hash=0

        while end< len(main_string):

            if start==0: # 初始条件
                main_hash= self.simple_hash(main_string,0,length-1)

            else:  # start=1 start=2 ....
                main_hash=prev_main_hash - ord(main_string[start-1])+ ord(main_string[end])

            if main_hash== pattern_hash:

                if main_string[start:end+1]== pattern:# 可能出现 hash 冲突
                    return start,end

            prev_main_hash=main_hash
            start+=1
            end = start + length - 1

        return None



class solutions2:

    def generate_bc(self,pattern):
        """
        生成坏字符哈希表
        :param pattern:
        :param m:
        :param bc:
        :return:
        """

        def gene_default():
            return -1

        bc=defaultdict(gene_default)

        for i in range(len(pattern)):
            # bc[ord(pattern[i])] = i
            bc[ pattern[i] ] = i

        return bc

    def generate_gs(self,pattern):
        """
        好后缀预处理
        :param pattern:
        :param suffix:
        :param prefix:
        :return:
        """
        m=len(pattern)

        suffix=(-1)*ones(m+1,dtype=int)
        prefix=zeros(m+1,dtype= bool) # [False,False,....]

        for i in range(m-1): # 遍历 从 0 到 i 的模式串的子串 （i 可以是 0 到 m-2）与 整个模式串，求 公共后缀子串
            k=1
            for j in range(i,-1,-1):
                if pattern[j]== pattern[m-k]:
                    suffix[k]=j # 公共后缀子串的长度是 k , j 表示 公共后缀子串 的起始下标
                    # print(suffix)
                    if j==0: #公共后缀子串也是模式串的前缀子串
                        prefix[k]=True
                        # print(prefix)
                    k+=1
                else:
                    break

        return suffix,prefix

    def move_by_gs(self,j, m, suffix, prefix):
        """
        通过好后缀计算移动值
        需要处理三种情况：
        1. 整个好后缀在 pattern仍能找到
        2. 好后缀里存在 *后缀子串* 能和 pattern的 *前缀* 匹配
        3. 其他
        :param j:
        :param m:
        :param suffix:
        :param prefix:
        :return:
        """

        k=m-j # 好后缀的 长度 ; 好后缀 pattern[j:length]

        if suffix[k] !=-1: # 好后缀 在 pattern 中能找到第二个
            y=j-suffix[k]+1

        else: # 好后缀 在 pattern 中 找不到了
            y=m
            for r in range(j+1,m): # 好后缀的 后缀
                k1= m-r # 后缀的长度
                if prefix[k1]==True:
                    y=r

        return y


    def bm(self,main_string, pattern):

        bc=self.generate_bc(pattern)

        suffix,prefix=self.generate_gs(pattern)

        length=len(pattern) # length= 3

        start=0
        end = start + length - 1  # start=1 end=3

        while end< len(main_string):

            for i in range(length-1,-1,-1):

                if main_string[start+i]!=pattern[i]: # 找到了坏字符 pattern[i] ，同时 好后缀为 pattern[i+1:]

                    #1. 使用 坏字符(bc) 规则 计算后移位数
                    si=i

                    xi=bc[pattern[i]]
                    x=si-xi

                    # 2. 使用 好后缀 (gs) 规则 计算后移位数
                    y=1
                    if i < length-1: # 如果有好后缀的话 ; i=length-1 说明 pattern 末尾的第一个字符即为 坏字符
                        y = self.move_by_gs(i, length, suffix, prefix)

                    step=max(x,y)

                    break

            else: # 正常执行完 for 循环 说明 pattern 与  main_string 匹配上了
                return start, end


            start += step
            end = start + length - 1

            # print(step, main_string[start:end+1])

        return None


class solutions3:


    def get_next(self,pattern):
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

        j=0
        for i in range(1,m):

            if pattern[i]==pattern[j]:
                next[i]=j+1
                j+=1

            else: # pattern[i]!=pattern[j]

                # M1:
                while pattern[i]!=pattern[j] : # j 要 向前移动

                    if j==0: # pattern[i]!=pattern[j] and j==0
                        next[i] = 0
                        break

                    j=next[j-1]

                else: # pattern[i]==pattern[j] 正常退出 while ; j==0 or j!=0
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

    def kmp(self,main_string, pattern):
        """
        
        发现不匹配的字符后，我尽可能的多移动一点
        
        匹配的时间复杂度： O(n)  n为 主串的长度
        求 next 数组的时间复杂度：O(m) m 为 模式串长度
        整个 kmp 的复杂度 ： O(m+n)
        
        ref: https://www.bilibili.com/video/BV1Ys411d7yh?from=search&seid=767115274745432726
        :param main_string: 
        :param pattern: 
        :return: 
        """
        n=len(main_string)
        m=len(pattern)

        i=0 # 主串指针
        j=0 # 模式串指针

        next=self.get_next(pattern)

        # print('next: ',next)

        while i<n and j<m:

            if main_string[i]==pattern[j]:
                i+=1
                j+=1

            else:
                if j!=0:
                    j=next[j-1]

                else: # main_string[i]!=pattern[j]  and j==0
                    i+=1

        if j== m:
            return i-m,i
        else:
            return None,None

    def get_next_depreatured(self, pattern):
            """
            next数组生成
            （暴力解法）
            找到 pattern 所有前缀 子串中，最长可匹配 前缀子串
            eg.pattern='ababacd'

            前缀子串1  'abab' ， 它的 最长可匹配 前缀子串为 'ab' （'abab' 的前缀子串 为 'ab'，后缀子串为 'ab'，匹配成功）

            :param pattern:
            :return:
            """
            m = len(pattern)
            next = (-1) * ones(m, dtype=int)  # next[0]=-1 ; next[-1]=-1

            for k in range(1, m - 1):  # 遍历 pattern 的所有前缀子串  pattern[0:k+1]

                longest_prefix = -1  # -1代表没有 最长可匹配前缀子串
                j = k
                i = 0
                while i <= m - 1 and j >= 1:  # pattern[:i+1] 为 前缀 ；pattern[j:k+1] 为 后缀

                    if pattern[:i + 1] == pattern[j:k + 1]:  # 前缀 和 后缀 相匹配
                        longest_prefix = i  # 更新 最长可匹配前缀子串 的 结尾字符下标

                    i += 1
                    j -= 1

                next[k] = longest_prefix  # 记录 结尾字符下标为k 的前缀子串 的最长可匹配前缀子串 的结尾下标

            return next

    def get_next_v1(self, pattern):

        """
        next数组生成
        (利用 动态规划 )
        注意：
        理解的难点在于next[i]根据next[0], next[1]…… next[i-1]的求解
        next[i]的值依赖于前面的next数组的值，求解思路：
        1. 首先取出前一个最长的匹配的前缀子串，其下标就是next[i-1]
        2. 对比下一个字符，如果匹配，直接赋值next[i]为next[i-1]+1，因为i-1的时候已经是最长
        *3. 如果不匹配，需要递归去找次长的匹配的前缀子串，这里难理解的就是递归地方式，next[i-1]
            是i-1的最长匹配前缀子串的下标结尾，则 *next[next[i-1]]* 是其次长匹配前缀子串的下标
            结尾
        *4. 递归的出口，就是在次长前缀子串的下一个字符和当前匹配 或 遇到-1，遇到-1则说明没找到任
            何匹配的前缀子串，这时需要找pattern的第一个字符对比

        ps: next[m-1]的数值其实没有任何意义，求解时可以不理。网上也有将next数组往右平移的做法。
        
        ref: https://time.geekbang.org/column/article/71845
        
        :param pattern:
        :return:
        """
        m = len(pattern)
        next = [-1] * m

        next[0] = -1

        # for i in range(1, m):
        for i in range(1, m - 1):
            j = next[i - 1]  # 取i-1时匹配到的最长前缀子串
            while j != -1 and pattern[j + 1] != pattern[i]:
                j = next[j]  # 次长的前缀子串的下标，即是next[next[i-1]]

            # 根据上面跳出while的条件，当j=-1时，需要比较pattern[0]和当前字符
            # 如果j!=-1，则pattern[j+1]和pattern[i]一定是相等的
            if pattern[j + 1] == pattern[i]:  # 如果接下来的字符也是匹配的，那i的最长前缀子串下标是next[i-1]+1
                j += 1
            next[i] = j

        return next


    def kmp_v1(self,main_string, pattern):
        """
         kmp 字符串匹配 算法 实现
         
         ref: https://time.geekbang.org/column/article/71845
        :param main_string: 
        :param pattern: 
        :return: 
        """

        length=len(pattern)

        start=0
        end = start + length - 1

        next=self.get_next_v1(pattern)

        # print('next: ',next)

        while end< len(main_string):

            step=1
            for j in range(length):

                if j>0 and main_string[start + j] != pattern[j]: # 找到坏字符，说明前面的是 好前缀 ;
                                            #  j=0 and main_string[start + j] != pattern[j]  说明 首字符就是 坏字符，也就没有 好前缀
                    k=next[j-1]+1
                    step= j-k

                    break
            else:  # 正常执行完 for 循环 说明 pattern 与  main_string 匹配上了
                return start, end

            start += step
            end = start + length - 1

            # print(step, main_string[start:end+1])

        return None,None

    def get_next_v2(self, pattern):
        """
        
        求next数组的过程完全可以看成字符串匹配的过程，即以 模式字符串 为 主字符串，以 模式字符串的 前缀 为 目标字符串，
        一旦字符串匹配成功，那么当前的next值就是匹配成功的字符串的长度。
        
        ref: https://www.zhihu.com/question/21923021/answer/281346746
        :param pattern: 
        :return: 
        """

        m = len(pattern)
        next = [-1] * m

        next[0] = -1

        i=0
        j=-1

        while i<m-1:

            if j==-1 or pattern[i]==pattern[j]:
                i += 1
                j += 1
                next[i]=j
            else:
                j=next[j]

        return next

    def kmp_v2(self, main_string, pattern):
        """
         kmp 实现 思路2 

         ref: https://www.zhihu.com/question/21923021/answer/281346746
         
        :param main_string: 
        :param pattern: 
        :return: 
        """

        i=0
        j=0

        next=self.get_next_v2(pattern)
        print('next: ', next)

        while i<len(main_string) and j<len(pattern):

            if j==-1 or main_string[i]==pattern[j]:
                i+=1
                j+=1
            else:
                j=next[j]

        if j==len(pattern):
            return i-j
        else:
            return None,None



    def find_all(self,main_string, pattern):
        """
        多次匹配的 字符串查找算法：
        （1）bf rk bm kmp 都是单次匹配的 字符串查找算法，即在 main_string 中找 pattern ，找到一次就退出；
        （2）循环调用  kmp 把 main_string 中所有的 pattern 都找出来
        :return: 
        """
        # right=-1
        left=-1

        res=[]

        while True:

            # offset = right + 1

            offset = left + 1

            left,right=self.kmp(main_string[offset:],pattern) # 从 已匹配 的 左端点的 后一位 开始重新搜索 main_string

            if right==None: # 找不到 ，退出循环
                break

            left=left+offset
            right=right+offset

            res.append((left,right))

        return res



if __name__ == '__main__':

    ##-------- part1 bf 和 rk 算法 ------------##
    sol=solutions()

    main_string='baddef'
    pattern='add'

    # print(sol.bf(main_string,pattern))
    # print(sol.rk(main_string, pattern))

    # print(sol.bf('ab', 'a'))
    # print(sol.bf('abc', 'abc'))
    # print(sol.bf('ab', ''))
    #
    # print(sol.rk('ab', 'a'))
    # print(sol.rk('abc', 'abc'))
    # print(sol.rk('ab', ''))


    # m_str = 'a'*10000
    # p_str = 'a'*200+'b'
    #
    # start = timeit.default_timer()
    # print('by bf: ')
    # print(sol.bf(m_str,p_str))
    # end = timeit.default_timer()
    # print('time: ', end-start ,'s')
    #
    # start = timeit.default_timer()
    # print('by rk: ')
    # print(sol.rk(m_str,p_str))
    # end = timeit.default_timer()
    # print('time: ', end-start ,'s')

    main_string = [['d','a','b','c'], ['e','f','a','d' ], ['c','c' ,'a','f' ], ['d','e' , 'f','c' ]]
    pattern=[['c','a'],['e','f']]

    # print(sol.bf_2D(main_string,pattern))

    ##-------------- part1 end -----------------##

    ##-------------- part2 BM 算法 --------------##
    sol2=solutions2()

    # print(sol2.generate_gs('cabcab'))

    m_str = 'dfasdeeeetewtweyyyhtruuueyytewtweyyhtrhrth'
    p_str = 'eyytewtweyy'

    # print(sol2.bm('abcacabdc','abd')) # 测试 坏字符原则

    # print(sol2.bm('abcacabcbcbacabc', 'abbcabc')) # 测试 好后缀 原则
    # print(sol2.bm('abcacabcbcbacabc', 'cbacabc'))

    # print(sol2.bm(m_str,p_str))

    ##-------------- part2 end -----------------##


    ##-------------- part3 KMP 算法 --------------##

    sol3=solutions3()
    # print(sol3.kmp('ababaeabac', 'ababacd'))
    m_str = "aabbbbaaabbababbabbbabaaabb"
    p_str = "abbabbbabaa"

    # print(sol3.get_next('ababacd'))
    # print(sol3.get_next_v2('ababacd'))

    # print(sol3.get_next('abababca'))
    # print(sol3.get_next_v1('abababca'))

    # print(sol3.get_next(p_str)) #[-1, -1, -1, 0, 1, 2, -1, 0, 1, 0,-1]

    # print(sol3.kmp_v1('ababaeabac','ababacd'))
    # print(sol3.kmp_v1(m_str, p_str))

    # print(sol3.kmp_v2('ababababca','abababca'))


    # print(sol3.get_next_v1('abcdabca'))

    # print(sol3.get_next('abcdabca'))
    # print(sol3.get_next('aabaabaaa'))

    # print(sol3.kmp('abxabcabcaby','abcaby'))

    print(sol3.kmp('abxabcabcaby', 'ab'))

    # print(sol3.kmp('ababababca','abababca'))

    # print(sol3.find_all('ababaeabac', 'ab'))
    # print(sol3.find_all('ababaeabac', 'aba'))

    ##-------------- part3 end -----------------##








    


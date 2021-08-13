#!/usr/bin/python
# -*- coding: UTF-8 -*-

import timeit

from numpy import *

import math

import random as rand

class solutions:

    def inversion_regions(self,nums):
        """
        排序算法 中，我们用有序度来表示一组数据的有序程度，用逆序度表示一组数据的无序程度。
        假设我们有 n 个数据，我们期望数据从小到大排列，那完全有序的数据的有序度就是 n(n-1)/2，逆序度等于 0；相反，倒序排列的数据的有序度就是 0，逆序度是 n(n-1)/2。
        
        M1:暴力枚举
        :param nums: [2,4,3,1,5,6]
        :return: [(2,1) ,(4,3) ,(4,1) ,(3,1)]
        """
        res=[]
        for i in range(len(nums)):
            left=nums[i]

            for j in range(i,len(nums)):
                right=nums[j]
                if left>right:
                    res.append((left,right))
        return len(res),res

    def inversion_regions_v2(self, nums):
        """
        M2: 分治法
        套用分治的思想来求数组 A 的逆序对个数。我们可以将数组分成前后两半 A1 和 A2，
        分别计算 A1 和 A2 的逆序对个数 K1 和 K2，然后再计算 A1 与 A2 之间的逆序对个数 K3。
        那数组 A 的逆序对个数就等于 K1+K2+K3。
        
        利用归并排序的思想：
        归并排序中有一个非常关键的操作，就是将两个有序的小数组，合并成一个有序的数组。
        每次合并操作，我们都计算逆序对个数，把这些计算出来的逆序对个数求和，就是这个数组的逆序对个数了。
        :param nums: 
        :return: 
        """
        N=len(nums)

        self.nums=nums

        self.res_region=[]

        self.inversion_num=0

        self.__merge_sort(0,N-1) # N-1 划重点！

        return self.inversion_num,self.nums

    def __merge_sort(self,left,right):

        if right>left: # 划重点！递归注意递归退出条件！

            #1.分解和递归：左半边有序 和 右半边有序
            mid = (right + left) // 2 # (right + left) 划重点！

            self.__merge_sort(left,mid)
            self.__merge_sort(mid+1,right) # mid+1 划重点！ left=0 right=1 mid=0 , self.__process(1 ,1)
            # self.__process(mid , right)  # left=0 right=1 mid=0 会永远陷入 self.__process(0 ,1) 的循环

            #2. 合并：把左半边有序和右半边有序合起来 ，整个序列有序
            self.__merge(left,mid,right)

    def __merge(self,left, mid , right):

        merge_cache=[]

        i = left
        j = mid+1

        while i <=mid and j<=right:

            if self.nums[i] > self.nums[j]:
                merge_cache.append(self.nums[j])
                j+=1

            else:
                self.inversion_num += j - (mid + 1)  # self.nums[i] 肯定比 self.nums[mid + 1:j+1] 的所有元素 都大
                                                     # eg. nums=[2,3,4,|1,5,6]  i=0 j=4 mid+1=3
                                                     #
                merge_cache.append(self.nums[i])

                i+=1

        if i <=mid :
            self.inversion_num += (right - mid)*(mid-i+1) # eg. nums=[2,3,4,|-2,-1,0] , [2,3,4] 的所有元素 比 [-2,-1,0] 都大 ，所以逆序对 一共有 3*3=9 个
            merge_cache=merge_cache+self.nums[i:mid+1]

        if j<=right:

            merge_cache=merge_cache+self.nums[j:right+1]

        # print('left:',left,'mid:',mid,'right:',right )
        # print('left part:', self.nums[left:mid+1],'right part:',self.nums[mid+1:right+1])
        # print('merge_cache:',merge_cache)
        # print('inversion_num:', self.inversion_num)

        self.nums[left:right+1]=merge_cache

    def max_subarray(self,nums):
        """
        
        最大子数组问题 （股票问题）
        
        :param nums: [-2,1,-3,4,-1,2,1,-5,4]
        :return: 
        
        """

        if len(nums)==1:
            return (nums[0],nums)


        mid=len(nums)//2

        # 1.分解 和 递归：最大子数组 在 mid 的 左边 或者 右边
        left_max_val,left_list=self.max_subarray(nums[:mid]) # 左半部分 的 最大子数组的值，和子数组
        right_max_val,right_list=self.max_subarray(nums[mid:])

        # 2. 中间状况：最大子数组 横跨了mid 的 左边和 右边的元素；时间复杂度 : O(n)

        mid_max_val_left = float('-inf') #最大子数组 的左半部分 的值
        left_row=mid-1

        mid_max_val_right = float('-inf') #最大子数组 的右半部分 的值
        right_row=mid

        # 找到 最大子数组 的左半部分 最大的情况
        left_sum=0
        i=mid-1 # mid >=1
        while i>=0:  # 时间复杂度 : O(n/2)
            left_sum +=nums[i]

            if left_sum>mid_max_val_left:
                mid_max_val_left= left_sum
                left_row= i

            i=i-1

        # 找到 最大子数组 的右半部分 最大的情况
        right_sum=0
        j=mid  # mid >=1
        while j < len(nums):# 时间复杂度 : O(n/2)
            right_sum +=nums[j]

            if right_sum>mid_max_val_right:
                mid_max_val_right= right_sum
                right_row= j

            j=j+1

        # 最大子数组 的左半部分 和 右半部分 拼一起
        mid_max_val=mid_max_val_left+mid_max_val_right
        mid_list=nums[left_row:right_row+1]

        result_list=[(left_max_val,left_list), (right_max_val,right_list),(mid_max_val,mid_list)]

        #3.合并：取 左边 右边 和中间 里的最优解
        max_result=max(result_list,key=lambda x: x[0])

        return max_result



class solutions1:

    def Hanoi(self, source, target, pillars, N):
        """
        Hanoi 塔问题

        将 N 个 盘子从 源柱子 A 搬到 目的柱子 B

        :param source: 源柱子
        :param target: 目的柱子
        :param pillars: 柱子的集合（一共 3 根）
        :param N: 
        :return: 
        """
        if N == 1:
            print(source, '--->', target)  # 只有一个 盘子直接 搬

        else:
            tmp = list(pillars - set([source, target]))[0]  # 排除掉 源柱子 和 目的柱子 剩下的为 中间柱子

            self.Hanoi(source, tmp, pillars, N - 1)  # 将 最上面的 N-1 个盘子 搬到中间柱子（保持盘子从小到大的顺序）
            print(source, '--->', target)  # 把最下面 的盘子 搬到目的节点
            self.Hanoi(tmp, target, pillars, N - 1)  # 把中间柱子的盘子 搬到目的节点


    def quick_Pow(self,x, n):
        """
        基于分治法 的快速幂乘法
        
        x^n 其中 n 必为整数
        
        :param x: 
        :param n: 
        :return: 
        """

        flag = 0 # n 为负数的标记

        if n < 0:
            flag = 1 # n 为负数
            n = abs(n)

        if n == 1 and flag == 0:
            return x
        elif n == 1 and flag == 1:
            return 1 / x
        elif n == 0:
            return 1
        else:
            if n % 2 == 0:
                sub_problem = self.quick_Pow(x, n // 2)
                result = sub_problem * sub_problem
            else:
                sub_problem = self.quick_Pow(x, (n - 1) // 2)  # (n-1) 括号要加
                result = sub_problem * sub_problem * x

        return result if flag == 0 else 1.0 / result

    def quick_matrix_pow(self, x, N):
        """
        基于分治法 的快速 方阵(nxn)的 幂乘法

        x^N 其中 N 必为 正整数
        
        1.numpy 中 无此函数 ：
          power(x,n) , x**n 都是对矩阵中的 每一个元素( element-wise ) 求幂 
        
        :param x: 
        :param n: 
        :return: 
        """
        x=  array(x)

        if x.shape[0] != x.shape[1]:
            raise Exception('x:{}不是一个方阵，无法计算 矩阵的幂乘'.format(x))

        L=x.shape[0]

        if N == 0:
            return   ones(shape=(L,L))

        elif N==1:
            return x

        else: # N==2,3,4

            if N % 2 == 0:
                sub_problem = self.quick_matrix_pow(x, N // 2)
                result =   dot(sub_problem, sub_problem)
            else:
                sub_problem = self.quick_matrix_pow(x, (N - 1) // 2)  # (n-1) 括号要加
                result =    dot(  dot(sub_problem, sub_problem), x)

        return result

    def fibonacci_downToUp(self,n):
        """
        自底向上 求解 
        
        :param n: 
        :return: 
        """

        if n == 0:
            return 0
        elif n == 1:
            return 1
        else:
            Fibonacci = [0] * (n+1)

            Fibonacci[1] = 1

            for i in range(2, n+1):
                Fibonacci[i] = Fibonacci[i - 1] + Fibonacci[i - 2]

            return Fibonacci[-1]


    def fibonacci_qucik( self, n):
        """
        分治法 求解 斐波那契数列 
        
       i= 0 1 2 3 4 5 6  7 8 
          0,1,1,2,3,5,8,13,21,...
        
        :param n: 
        :return: 
        """

        if n ==0:
            return 0

        elif n ==1 or n==2:

            return 1

        else: # n=3,4,...

            a=  array([ [1,1],
                         [1,0] ])

            b=self.quick_matrix_pow(a,n-1)

            return b[0][0]

    def Int_Bit_multiplication(self,X,Y):
        """
        整数位乘问题
        
        X,Y 是 n 位二进制数 , 满足 n=2^k 
        
        1.取一个数的 高 k 位:  X >> k
        
        eg. X=8 k=2
        
        0b1000 >>2 = 0b10
        
        2.取一个数的 低 k 位: X & (2^k-1) 
            
        eg. X=8 k=2
                        
            0b1000
         &  0b0011  
         =  00
        
        0b0011= 2^2-1
             
        3.移位保留 原符号
        
        eg. X=-8 k=2
        
        -0b1000 >>2  = -0b10
        
        
        4.移位操作的 优先级最低 ，一定要加括号 
        
                  
        :param X: 10 进制表示的数
        :param Y: 10 进制表示的数
        :return: 
        """
        # X_bin=bin(X) # '0bxx ' 是一个字符串
        # Y_bin = bin(Y)

        n=max(X.bit_length(),Y.bit_length()) # 得到 二进制的位数

        if n<=1 :

            return X*Y

        else:

            A= X >> (n//2) # 取 X 的高(n//2) 位
            B=  X & (2**(n//2)-1)  # 取 X 的低(n//2) 位

            C= Y >> (n//2) # 取 Y 的高(n//2) 位
            D=  Y & (2**(n//2)-1)  # 取 Y 的低(n//2) 位


            AC=self.Int_Bit_multiplication(A,C)
            BD=self.Int_Bit_multiplication(B,D)

            AD_BC=self.Int_Bit_multiplication(A-B,D-C)+AC+BD

            return (AC<<n) + (AD_BC<<(n//2)) + BD # 移位运算符 优先级低，记得加括号

    def quick_dot(self,A, B):
        """
        
        利用分治法的 快速矩阵乘法 (Strassen 矩阵乘法)
        
        A B 为 n 阶矩阵, n=2^k
        
        :param A: 
        :param B: 
        :return: 
        """

        n = len(A)
        half = n // 2

        if n == 1:
            return A * B

        a = A[0:half, 0:half]
        b = A[0:half, half:]
        c = A[half:, 0:half]
        d = A[half:, half:]
        e = B[0:half, 0:half]
        f = B[0:half, half:]
        g = B[half:, 0:half]
        h = B[half:, half:]

        P1 = self.quick_dot(a, f - h)
        P2 = self.quick_dot(a + b, h)
        P3 = self.quick_dot(c + d, e)
        P4 = self.quick_dot(d, g - e)
        P5 = self.quick_dot(a + d, e + h)
        P6 = self.quick_dot(b - d, g + h)
        P7 = self.quick_dot(a - c, e + f)

        r = P5 + P4 - P2 + P6
        s = P1 + P2
        t = P3 + P4
        u = P5 + P1 - P3 - P7


        result = zeros((n, n), dtype=int16)
        result[0:half, 0:half] = r
        result[0:half, half:] = s
        result[half:, 0:half] = t
        result[half:, half:] = u

        return result



class solutions2:

    def min_dist_node_1D_naive(self,nodes):
        """
        最接近点对问题：
        给定 线段 上的 n个点，找其中的一对点，使得在n个点的所有点对中，该点对的距离最小。
        
        M1: 线段上的点进行排序，用一次线性扫描就可以找出最接近点对
        :param nodes: [1,4,5,7,9]
        :return: 
        """
        nodes=sorted(nodes)
        prev_node=nodes[0]

        min_dist=float('inf')
        min_dist_nodes=None
        for i in range(1,len(nodes)):
            current_node= nodes[i]

            if current_node-prev_node < min_dist:


                min_dist=current_node-prev_node
                min_dist_nodes=(prev_node,current_node)

            prev_node=current_node

        return min_dist,min_dist_nodes

    def min_dist_node_1D(self,nodes):
        """
        给定 线段 上的 n个点，找其中的一对点，使得在n个点的所有点对中，该点对的距离最小。
        
        M2 ： 分治法（要想快，用分治）
        ref: https://blog.csdn.net/liufeng_king/article/details/8484284
        :param nodes: [1,3,5,6,9]
        :return: 
        """
        if len(nodes)<=1: # 注意递归退出条件
            min_dist=float('inf')
            res_node=None

            return (min_dist,res_node)

        elif len(nodes)==2: # 注意考虑 递归到原子状况，即最小的子问题
            min_dist=abs(nodes[0]-nodes[1])
            res_node=(nodes[0],nodes[1])

            return (min_dist,res_node)


        min_dist = float('inf')
        res_node = None

        #1.分解：找到子问题 的切割方法，保证左右子问题规模近似
        m= (min(nodes)+max(nodes))/2 #找到切分点

        s1= list(filter(lambda t:t<=m, nodes)) # m的 左边的区间为 s1
        s2= list(filter(lambda t:t>m, nodes))  # m 的 右边区间为 s2
        # print(nodes_small)

        # 2. 递归：求解 s1 和 s2 中的 最接近点对
        min_dist_s1, s1_node  =self.min_dist_node_1D(s1)
        min_dist_s2, s2_node = self.min_dist_node_1D(s2)

        # 3. 中间情况： 横跨 s1 和 s2 的最接近点对
        # 4. 合并：三种情况（s1中的 最接近点对 ,s2中的 最接近点对，横跨 s1 和 s2 的最接近点对）中 找到最优解
        if min_dist_s1 < min_dist_s2:
            min_dist=min_dist_s1
            res_node=s1_node

        else:
            min_dist = min_dist_s2
            res_node = s2_node

        s1 = list(filter(lambda t: m-min_dist<t<=m , s1))
        s2 = list(filter(lambda t: m<=t<m+min_dist , s2))

        if len(s1)==1 and len(s2)==1:

            if abs(s1[0]-s2[0])< min_dist:

                min_dist=abs(s1[0]-s2[0])
                res_node=(s1[0],s2[0])

        return (min_dist,res_node)

    def min_dist_node_2D_naive(self, nodes):
        """
        给定 2D 平面 上的 n个点，找其中的一对点，使得在n个点的所有点对中，该点对的距离最小。
        
        M1: 枚举法 
        
        :param nodes: 
        :return: 
        """
        min_dist=float('inf')
        res_node=None

        for i in range(len(nodes)):
            left=nodes[i]

            for j in range(i+1,len(nodes)):
                right=nodes[j]

                # dist= math.sqrt((left[0]-right[0])**2+(left[1]-right[1])**2)
                dist = (left[0] - right[0]) ** 2 + (left[1] - right[1]) ** 2

                if dist<min_dist:
                    min_dist=dist
                    res_node=(left,right)

        return min_dist,res_node


    def __partion(self,nums, left, right):

        pivot = rand.randint(left, right)

        nums[right], nums[pivot] = nums[pivot], nums[right]

        storeIndex = left

        for i in range(left, right):
            if nums[i] < nums[right]:
                nums[i], nums[storeIndex] = nums[storeIndex], nums[i]

                storeIndex += 1

        nums[right], nums[storeIndex] = nums[storeIndex], nums[right]
        return storeIndex

    def quick_select(self,nums, left, right, smallest_i):
        """
        基于 快速排序的 快速 选择，即找到 数组中第 N小的元素，若 N=len(nums)/2 即找到中位数
        平均时间复杂度：O(n)
        eg1.
        # l=[5,4,3,6,9,0]
        # target=quick_select( l, 0, len(l)-1, 0)  # smallest_i=0 即最小的元素 ，输出 0
        :param nums: 
        :param left: 左端点
        :param right: 右端点 
        :param smallest_i: 第 N小的 元素 
        :return: 
        """

        if left < right:

            pivot_new = self.__partion(nums, left, right)  # pivot_new is the index
            # print ("index:", pivot_new, "ele:", nums[pivot_new])
            # print ("list: ", nums)

            if pivot_new == smallest_i:
                return nums[pivot_new]
            elif pivot_new < smallest_i:  # smallest_i is the last position of the sorted list
                return self.quick_select(nums, pivot_new + 1, right, smallest_i)  # right
            else:
                return self.quick_select(nums, left, pivot_new - 1, smallest_i)  # left

        else:
            return nums[left]

    def __dist(self,node1,node2):
        """
        计算 两点的 距离的 平方
        :param node1: 
        :param node2: 
        :return: 
        """
        return  (node1[0] - node2[0]) ** 2 + (node1[1] - node2[1]) ** 2

    def min_dist_node_2D(self, nodes):
        """
        给定 2D 平面 上的 n个点，找其中的一对点，使得在n个点的所有点对中，该点对的距离最小。
        
        M2 : 分治法
        :param nodes: 
        :return: 
        """
        if len(nodes)<=1: # 注意递归退出条件
            min_dist=float('inf')
            res_node=None

            return (min_dist,res_node)

        elif len(nodes)==2: # 注意考虑 递归到原子状况，即最小的子问题

            min_dist=self.__dist(nodes[0],nodes[1])
            res_node=(nodes[0],nodes[1])

            return (min_dist,res_node)

        nodes_x=list(map(lambda t:t[0],nodes))

        m=self.quick_select(nodes_x,0,len(nodes_x)-1, len(nodes_x)//2)
        # m= (min(nodes_x)+max(nodes_x))/2 #找到切分点


        s1= list(filter(lambda t:t[0]<=m, nodes)) # m的 左边的区间为 s1
        s2= list(filter(lambda t:t[0]>m, nodes))  # m 的 右边区间为 s2
        # print(nodes_small)

        min_dist_s1, s1_node  =self.min_dist_node_2D(s1)
        min_dist_s2, s2_node = self.min_dist_node_2D(s2)

        res_node=None

        if min_dist_s1 < min_dist_s2:
            min_dist=min_dist_s1
            res_node=s1_node

        else:
            min_dist = min_dist_s2
            res_node = s2_node

        d=min_dist

        # 用 P1和 P2 分别表示直线l的左边和右边的宽为 d 的2个垂直长条
        P1 = list(filter(lambda t: m - d < t[0] <= m, s1))
        P2 = list(filter(lambda t: m <= t[0] < m+d , s2))

        #将P1和P2中所有点按其y坐标排好序，则对P1中所有点 p，
        # 对排好序的点列作一次扫描，就可以找出所有最接近点对的候选者，对P1中每一点最多只要检查P2中排好序的 相继6个点，
        # 即 从P点的纵坐标yp 向上检查 P2中的三个点，向下检查 P2中的三个点。

        P_y_sorted=sorted(P1+P2,key=lambda t: t[1]) # 按照 纵坐标 对 P1,P2  进行排序
        P1=set(P1)
        P2=set(P2)

        for i,node in enumerate(P_y_sorted):

            if node in P1:
                # 向上找4个
                j=i
                up_times=0
                while j>=0:

                    if up_times>4:
                        break

                    if  P_y_sorted[j] in P2:
                       dist=self.__dist(node,P_y_sorted[j])
                       up_times+=1

                       if dist<min_dist:
                           min_dist = dist
                           res_node = (node,P_y_sorted[j])
                    j=j-1

                # 向下找4个
                j=i
                down_times=0
                while j < len(P_y_sorted):

                    if down_times>4:
                        break

                    if  P_y_sorted[j] in P2:
                       dist=self.__dist(node,P_y_sorted[j])
                       down_times+=1

                       if dist<min_dist:
                           min_dist = dist
                           res_node = (node,P_y_sorted[j])
                    j=j+1


        return (min_dist,res_node)



class test_chips:

    """
    by XRH 
    date: 2020-08-13 

    芯片测试问题

    从 n 片芯片中 挑出一片好的 芯片 ，使用最少的 测试次数 
    
    前提假设 ：
    n 片芯片中 好芯片比坏芯片 至少多一片
    
        
    功能：

    1. 生成 n 片待测试的 芯片，其中 好坏芯片个数 随机，但是保证 前提假设成立
    
    2. 测试台 同时测试 两个芯片 A 和 B , 它们 互相 说对方是否为 好芯片.
      若为 好芯片 则会说实话 , 若为 坏芯片 则随便说
        
    3. 朴素方法 检测 n 个 芯片, 找到一个 好芯片
    
    4. 利用分治策略 检测 n 个 芯片, 找到一个 好芯片
    
    """

    def generate_chips(self, n):
        """
        生成 n 片待测试的 芯片
        
        0 代表 坏芯片，1代表好芯片 
        
        :param n: 10
        :return: chips
        
        (index, flag) list  
        eg.
        [(1, 1), (9, 0), (7, 0), (6, 0), (8, 0), (4, 1), (2, 1), (5, 1), (3, 1), (0, 1)]
        
        """
        difference_num=1 # 好芯片 比 坏芯片多1个
        # difference_num=random.randint(1,n) # 好芯片 比 坏芯片多的 [1,n] 个

        unqualified_num= (n- difference_num)//2 # 坏芯片个数

        qualified_num = n- unqualified_num # 好芯片个数

        qualified_chips= [(i,1)  for i in range(qualified_num)] # 1 代表好芯片

        unqualified_chips= [(i,0)  for i in range(qualified_num,qualified_num+unqualified_num)] # 0 代表 坏芯片

        chips= qualified_chips+unqualified_chips

        random.shuffle(chips) # 打乱 顺序

        return chips


    def test_platform(self,A,B):
        """
        测试台 同时测试 两个芯片 A 和 B , 它们 互相 说对方是否为 好芯片
        
        :param A: (index, flag)  flag 是芯片实际好坏的标记
                  eg. (9, 0)
        :param B: 
        :return: 
        """
        A=A[1]
        B = B[1]

        res_flag=None

        if A==1 and B==1:  # A,B 都是好芯片

            res_flag=[1,1]

        elif A==1 and B==0: # A 为好 B 为坏

            res_flag = [0, random.randint(0,1)]  # A说B 是坏芯片, B随便乱说

        elif A == 0 and B == 1:  # A 为坏 B 为好

            res_flag = [ random.randint(0, 1),0 ]

        else: # A,B 都是 坏芯片

            res_flag = [random.randint(0, 1), random.randint(0, 1)]

        return res_flag


    def find_one_qualified_chip_naive(self,chips):
        """
        朴素方法 检测 n 个 芯片, 找到一个 好芯片
        
        时间复杂度：O(n^2)
        
        :param chips:(index, flag) list 
         eg. [(1, 1), (9, 0), (7, 0), (6, 0), (8, 0), (4, 1), (2, 1), (5, 1), (3, 1), (0, 1)]
        :return: 
        """

        n=len(chips)

        qualified_index=-2 # 好芯片的标号

        # flag=0 # 找到好芯片的标记

        for i in range(n): # 从 第0 个芯片开始


            A=chips[i] # 待检测的 芯片

            qualified_votes=0 # 说A 是 好芯片的票数
            unqualified_votes=0 # 说A 是 坏芯片的票数

            for j in range(i+1,n): # 其他所有的芯片来 检测 A 芯片

                B=chips[j]

                report=self.test_platform(A,B)

                if report[1]==1: # B 说 A 是好芯片
                    qualified_votes+=1
                else:# B 说 A 是坏芯片
                    unqualified_votes+=1

            if n%2 ==0: # n 为偶数

                if qualified_votes>= n//2: # 第 i 个芯片为好芯片

                    qualified_index=i

                    break

            else: # n 为奇数

                if qualified_votes >= (n-1) // 2:  # 第 i 个芯片为好芯片

                    qualified_index = i

                    break


        return chips[qualified_index]

    def find_one_qualified_chip(self, chips):


        self.qualified_chip = None

        self.__find_one_qualified_chip(chips)

        return self.qualified_chip


    def __find_one_qualified_chip(self, chips):
        """
        利用分治策略 检测 n 个 芯片, 找到一个 好芯片
            
        时间复杂度：
        
        T(n)=T(n/2)+ O(n)
        
        T(n)=O(n)
        
        :param chips: [1, 1, 0, 1, 1, 1, 1, 1]
        :return: 
        """

        n = len(chips)

        qualified_index = -2

        if n in (1,2):

            self.qualified_chip =chips[0]

            return

        elif n==3:

            A=chips[0]
            B=chips[1]

            report=self.test_platform(A,B)

            if report[0]==1 and  report[1]==1: # 只可能为2个都为好(因为 好芯片比坏芯片至少多一个 所以 好芯片个数大于2个)

                self.qualified_chip=chips[0]

            else: # chips[0] chips[1] 1好1坏，说明 chips[2] 必然为好芯片

                self.qualified_chip = chips[2]

            return


        if n%2 !=0: # n 为奇数

            # 轮空的芯片 单独测试一轮
            A = chips[0] # 被测试的芯片

            qualified_votes = 0  # 说A 是 好芯片的票数
            unqualified_votes = 0  # 说A 是 坏芯片的票数

            for j in range(1, n):  # 其他所有的芯片来 检测 A 芯片

                B = chips[j]

                report = self.test_platform(A, B)

                if report[1] == 1:  # B 说 A 是好芯片
                    qualified_votes += 1
                else:  # B 说 A 是坏芯片
                    unqualified_votes += 1


            # n 为奇数
            if qualified_votes >= (n - 1) // 2:  # 第 0 个芯片为好芯片

                qualified_index = 0

                self.qualified_chip=chips[qualified_index]


            else:  # 第 0 个芯片为 坏芯片

                self.find_one_qualified_chip(chips[1:]) # 把坏芯片扔掉


        else:  # n 为 偶数

            # n 个 芯片分为 n//2 组
            chips_group = []
            for i in range(0, len(chips), n//2):
                chips_group.append(chips[i:i + (n//2)])


            chips=[]

            for i in range(n//2):

                A=chips_group[0][i]
                B=chips_group[1][i]

                report=self.test_platform(A,B)

                if report[0]==1 and report[1]==1: # 一个小组内 两片都互相 说好, 留下其中一片

                    chips.append(A)

                # 其他 所有情况, 该组芯片全部丢弃

            self.find_one_qualified_chip(chips) # 分治法 关键：子问题 与 原问题 有相同的性质
                                                #              即满足 分组筛选过后的chips 中 好芯片的 个数 至少比 坏芯片多一个



if __name__ == '__main__':

    sol = solutions()
    # print(sol.inversion_regions([2, 4, 3, 1, 5, 6]))
    # print(sol.inversion_regions_v2( [2, 4, 3, 1, 5, 6]))

    # def inversion_regions 的大数据量 测试 ，统计运行时间
    # regions=  random.randint(100,size=10000)
    #
    # regions=list(regions)

    # start = timeit.default_timer()
    # print('by inversion_regions: ')
    # print(sol.inversion_regions(regions)[0])
    # end = timeit.default_timer()
    # print('time: ', end - start, 's')
    #
    # start = timeit.default_timer()
    # print('by inversion_regions_v2: ')
    # print(sol.inversion_regions_v2(regions)[0])
    # end = timeit.default_timer()
    # print('time: ', end-start ,'s')

#-------------------------------------------------------------#
    nums=[-2, 1, -3, 4, -1, 2, 1, -5, 4]
    # print(sol.max_subarray(nums))


#-------------------------------------------------------------#

    sol = solutions1()

    # print(sol.Hanoi('A','C',set(['A','B','C']),3))

    # print(sol.quick_Pow(2,10))

    a = [[1, 1],
         [1, 0]]

    # print(sol.quick_matrix_pow(a, 3))

    # print(sol.fibonacci_downToUp(4))

    # print(sol.fibonacci_qucik(4))

    # print(sol.Int_Bit_multiplication(8,8))

    A = arange(16).reshape(4, 4)
    B=ones( (4,4), dtype=int16 )

    # print (sol.quick_dot(A,B))
    # print (dot(A,B))

# -------------------------------------------------------------#


    sol = solutions2()
    # nodes_1D=10*(  random.rand(100))
    nodes_1D=rand.sample(range(0,100),10) # [0,1000]中 抽样 100个不重复的数

    # nodes_1D=[1,3,5,6,9]
    nodes_1D=list(nodes_1D)
    # print(nodes_1D)


    # print(sol.min_dist_node_1D_naive(nodes_1D))
    # print(sol.min_dist_node_1D(nodes_1D))

    x=rand.sample(range(0,100),10)
    y=rand.sample(range(0,100),10)
    nodes_2D=[(x[i],y[i]) for i in range(len(x))]

    # print(nodes_2D)
    # print(sol.min_dist_node_2D_naive(nodes_2D))
    # print(sol.min_dist_node_2D(nodes_2D))


    # min_dist_node_2D 的大数据量 测试 ，统计运行时间
    # x=rand.sample(range(0,10000),5000)
    # y=rand.sample(range(0,10000),5000)
    # nodes_2D=[(x[i],y[i]) for i in range(len(x))]
    #
    # start = timeit.default_timer()
    # print('by min_dist_node_2D_naive: ')
    # print(sol.min_dist_node_2D_naive(nodes_2D))
    # end = timeit.default_timer()
    # print('time: ', end - start, 's')
    #
    # start = timeit.default_timer()
    # print('by min_dist_node_2D: ')
    # print(sol.min_dist_node_2D(nodes_2D))
    # end = timeit.default_timer()
    # print('time: ', end-start ,'s')


# -------------------------------------------------------------#

    sol3=test_chips()

    chips=sol3.generate_chips(10)

    print(chips)

    # print(sol3.find_one_qualified_chip_naive(chips))

    print(sol3.find_one_qualified_chip(chips))











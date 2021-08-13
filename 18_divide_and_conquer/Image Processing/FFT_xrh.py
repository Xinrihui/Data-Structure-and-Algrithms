#!/usr/bin/python
# -*- coding: UTF-8 -*-

import timeit

import numpy as np

import math

import random as rand

class calculate_polynomial:
    """

    by XRH 
    date: 2020-08-19 

    计算多项式的 方法集合

    """

    def cal_poly_naive(self,a_list,x_list):
        """
        求解多项式的值 (朴素方法) 
        
        对于 x_list 的每一个 x 求 A(x)
        
        length(a)=n
        length(x_list)=n
                
        A(x)= a[0]+ a[1]x + a[2]x^2 + ... + a[n-1]x^(n-1)
        每一个 x 做 1+2+..+(n-1)= O(n^2) 次乘法
        
        时间复杂度: O(n^3)
        
        :param a_list: 
        :param x_list: 
        :return: 
        """

        res_A_list=[]

        for x in x_list:

            A=0
            for i,a in enumerate(a_list):

                A+= a*(x**i) # TODO: python 中整数本来是不用考虑溢出的问题

            res_A_list.append(A)

        return  res_A_list

    def cal_poly_v1(self,a_list,x_list):
        """
        求解多项式的值 ( 将递归转换为 自底向上迭代 )
        
        对于 x_list 中的每一个 x 求 多项式 A(x)
        
        length(a)=n
        length(x_list)=2n
        
        A(x)= a[0]+ a[1]x + a[2]x^2 + ... + a[n-1]x^(n-1)
                
        A(1,x)=a[n-1]
        A(2,x)=a[n-2] + xA(1,x)
        ...
        A(n,x)= a[0]+ xA(n-1,x)
        
        对每一个 x 做 n 次乘法, 计算 多项式的 时间复杂度为 O(n)
        
        总的时间复杂度: O(n^2)
        
        :param a_list: 
        :param x_list: 
        :return: 
        """
        res_A_list=[]

        n=len(a_list)

        for x in x_list:

            A=0

            for i in range(n-1,-1,-1):

                A = a_list[i]+x*A

            res_A_list.append(A)

        return  res_A_list

    def cal_poly_v2(self, a_list, x_list):
        """
        求解多项式的值 (分治法)

        对于 x_list 中的每一个 x 求 多项式 A(x)

        length(a)=n 假设 n 必为偶数 
        
        length(x_list)=2n

        A(x)= a[0]+ a[1]x + a[2]x^2 + ... + a[n-1]x^(n-1)

        A(x) = A_even(x^2) + xA_odd(x^2)  合并步骤只做 2 次乘法 时间复杂度 O(1)
          
        T(n)= 2T(n/2) + O(1) ,  T(n)=O(n)
        
        计算 1个x 的多项式的值 时间复杂度为 O(n)

        总的时间复杂度: O( n^2 )

        :param a_list: 
        :param x_list: 
        :return: 
        """

        res_A_list=[]


        a_list=np.array(a_list,dtype='int64')

        for x in x_list:

            A=self.__process_v2(a_list,x)
            res_A_list.append(A)

        return  res_A_list

    def __process_v2(self,a_list,x):
        """
        分治策略 , 利用递归实现
        计算每一个 多项式的值
        
        :param a_list: 
        :param x: 
        :return: 
        """

        n=len(a_list)

        if n==1: # 递归的结束条件, 递归到的原子状态(最小子问题)

            return a_list[0]

        elif n>1:

            # even = [i for i in range(n) if i % 2 == 0]  # 偶数序号 TODO: 时间复杂度 O(n) 拖累了算法的效率
            # odd = [i for i in range(n) if i % 2 != 0]  # 奇数序号

            a_list_even= a_list[0::2] #  偶次数 多项式的 系数
            a_list_odd = a_list[1::2] #  奇次数 多项式的 系数


            return self.__process_v2(a_list_even,x*x)+x*self.__process_v2(a_list_odd,x*x)



    def cal_poly_vector(self, a_list, x_list):
        """
        分治法 求解多项式的值

        利用向量化的方法 对  x_list 整体求 多项式向量 A(x_list)

        length(a)=n 假设 n 必为偶数 

        length(x_list)=2n

        A(x)= a[0]+ a[1]x + a[2]x^2 + ... + a[n-1]x^(n-1)

        A(x) = A_even(x^2) + xA_odd(x^2)  合并步骤只做 3n 次乘法
        
         T(n)= 2T(n/2) + O(n) ,  T(n)=O(nlogn)
        
        计算 x_vector 的多项式向量 的时间复杂度为 O(nlogn)
        
        总的时间复杂度:  O( nlogn )  
 
        :param a_list: 
        :param x_list: 
        :return: 
        """


        a_vector = np.array(a_list, dtype='int64') # 默认 int 为 int32, 当数值巨大时 可能会发生溢出
        x_vector = np.array(x_list, dtype='int64')

        res_A_vector = self.__process_vector(a_vector, x_vector)

        return res_A_vector

    def __process_vector(self, a_vector, x_vector):
        """
        分治策略
        计算 多项式 向量 

        :param a_list: 
        :param x: 
        :return: 
        """

        n = len(a_vector)

        if n == 1:  # 递归的结束条件, 递归到的原子状态(最小子问题)

            # return a_vector[0]*ones(len(x_vector),dtype=int) # 默认 int 为 int32
            return a_vector[0] * np.ones(len(x_vector), dtype='int64')

        elif n > 1:

            a_list_even= a_vector[0::2] #  偶次数 多项式的 系数
            a_list_odd = a_vector[1::2] #  奇次数 多项式的 系数

            return self.__process_vector(a_list_even, x_vector * x_vector) + x_vector * self.__process_vector(a_list_odd, x_vector * x_vector)

    def _W_Conjugation(self, N, B):
        """
        计算 旋转因子 (逆时针旋转)

        与顺时针旋转 的旋转因子 共轭(Conjugation)
        
        _W_Conjugation(4,0) # (1)
        _W_Conjugation(4,1) # 1j
        _W_Conjugation(4,2) # (-1)
        _W_Conjugation(4,3) # (-1j)
        _W_Conjugation(4,4) # (1)
        
        :param N: 
        :param B: 
        :return: 
        """

        mid = (2 * math.pi / N) * B

        real = math.cos(mid)

        image = math.sin(mid)

        return complex(float("%0.8f" % real), float("%0.8f" % image))  # 控制浮点数的精度, 保留8位小数

    def one_2nth_root(self,n):
        """
        求 1 的 2n 次方根, 一共 2n 个数
        :param n: 
        :return: [w0,w1,..w2n-1]
        """

        return [ self._W_Conjugation(2*n,i) for i in range(2*n)]

    def cal_poly_vector_one_2nth_root(self, a_list, x_list):
        """
        分治法 求解多项式的值 (限定 x_list 为 1 的 2n 次方根)

        利用向量化的方法 对  x_list 整体求 多项式向量 A(x_list)

        A(x)= a[0]+ a[1]x + a[2]x^2 + ... + a[n-1]x^(n-1)
        
        
        A(x) = A_even(x^2) + xA_odd(x^2)  x^2 可以预先计算好, 所以合并步骤只做 n 次乘法
        
          x^2 和 x 之间的关系为：
          1. w 是 1 的 2n 次方根, 则 w^2 是 1 的 n 次方根, 因此有：
            
            w0^2=w0 w1^2=w2 w2^2=w4....
            从 w0 开始 隔1个 取 1个 得到 
            x^2= [w0,w2,w4,..,w14,w0,w2,w4,..,w14]

        
        T(n)= 2T(n/2) + O(n) , T(n)=O(n)

        计算 x_vector 的多项式向量 的时间复杂度为 O(n)

        总的时间复杂度:  O( nlogn )  

        :param a_list: 
        :param x_list: 
        :return: 
        """

        a_vector = np.array(a_list, dtype='int64')  # 默认 int 为 int32, 当数值巨大时 可能会发生溢出
        x_vector = np.array(x_list, dtype='complex128')

        self.x_vector_square= np.concatenate ((x_vector[0::2],x_vector[0::2])) # x^2 和 x 之间的关系为：
                                                                               #  1. w 是 1 的 2n 次方根, 则 w^2 是 1 的 n 次方根
                                                                               #  2. 从 w0 开始 隔1个 取1个

        res_A_vector = self.__process_vector_one_2nth_root(a_vector, x_vector)

        return res_A_vector

    def __process_vector_one_2nth_root(self, a_vector, x_vector):
        """
        分治策略
        计算 多项式 向量 

        :param a_list: 
        :param x: 
        :return: 
        """

        n = len(a_vector)

        if n == 1:  # 递归的结束条件, 递归到的原子状态(最小子问题)

            return a_vector[0] * np.ones(len(x_vector), dtype='int64')

        elif n > 1:

            a_list_even = a_vector[0::2]  # 偶次数 多项式的 系数
            a_list_odd = a_vector[1::2]  # 奇次数 多项式的 系数


            return self.__process_vector_one_2nth_root(a_list_even, self.x_vector_square) + x_vector * self.__process_vector_one_2nth_root(
                a_list_odd, self.x_vector_square)


class OneDimension_FFT:
    """
    
    by XRH 
    date: 2020-08-19 
    
    一维 离散信号(序列) 的快速傅里叶变换
    
    """

    def base_2FFT(self,in1,in2,W):
        """
        蝶形运算符 (基 2FFT) 实现
        
        :param in1: 
        :param in2: 
        :param W: 
        :return: 
        """

        out1= in1 + W*in2
        out2= in1 - W*in2

        return out1,out2


    def _W(self,N,B):
        """
        计算 旋转因子(顺时针旋转)
        
       详见 《数字信号处理》-> 4. 快速傅里叶变换 FFT         
        
        _W(4,0) # (1-0j)
        _W(4,1) # -1j
        _W(4,2) # (-1-0j)
        _W(4,3) # (-0+1j)
        _W(4,4) # (1+0j)
        
        2.复数的计算
        ref: https://www.cnblogs.com/crawer-1/p/8242017.html
        
        :param N: 
        :param B: 
        :return: 
        """

        mid = -(2 * math.pi / N) * B

        real = math.cos(mid)

        image = math.sin(mid)

        return complex(float("%0.8f" % real), float("%0.8f" % image))  # 控制浮点数的精度, 保留8位小数

    def _reverse_range(self,N):
        """
        抽取输入中，决定 输入顺序的 倒序算法
        
        :param N: 4 
        :return: [0, 2, 1, 3]
        """
        res=[]

        bit_num=int(np.log2(N)) # 位数

        for i in range(N):

            ele=list(bin(i)[2:])

            if len(ele)< bit_num:

                ele= ['0']*(bit_num-len(ele)) + ele

            ele.reverse()

            ele_str=''.join(ele)

            ele= int(ele_str,2) # 将2进制表示的ele_str 转换为 int
            res.append(ele)

        return res


    def trans_nFFT(self,input_seq):
        """
         N 点 FFT
        利用 基 2FFT 将 N点 DFT 转换为 N/2 个2点DFT , 实现 (离散信号)序列 从时间域到频率域的投影
        
        时间复杂度: O( (n/2)log(n) )

        一共 log(n) 级 
        每一级中 有 (n/2) 个蝶形运算符, 每一个蝶形运算符 只做一次乘法
        
        详见 《数字信号处理》-> 4. 快速傅里叶变换 FFT
    
        :param input_seq: [1,0.5,0,0.5,1,1,0.5,0]
        :return: 
        """
        N = len(input_seq)

        input_seq=np.array(input_seq)

        #1. 抽取输入

        sample_index= self._reverse_range(N)

        sample_input= input_seq[sample_index]

        #2. 分级计算
        # N 点 基2 FFT 的运算流图 的级数为 M
        M=int(np.log2(N)) #  N=8 M=3

        num_2FFT=N//2  # 每一级有 N//2 个蝶形


        # 2.1 蝶形运算符的 运算次序
        cal_order = []

        for level in range(M):  # 级别

            cal_level_order = []

            i = 0
            j = 0
            num = 0
            flag = [0] * N

            while num < num_2FFT:  # 遍历 每一级的 N/2 个蝶形运算符

                # 第 num 个蝶形运算符

                while flag[i] == 1:  # 该位被占用, 往后找 直到找到一个空位作为 蝶形运算符的 输入1
                    i += 1

                flag[i] = 1  # 蝶形运算符的 输入1 的标号为 i
                j = i + (2 ** level)  # 蝶形运算符的 输入2 的标号为 j

                flag[j] = 1

                cal_level_order.append((i, j)) #

                num += 1

            cal_order.append(cal_level_order)

        # print(cal_order) #[[(0, 1), (2, 3), (4, 5), (6, 7)],
                        # [(0, 2), (1, 3), (4, 6), (5, 7)],
                        # [(0, 4), (1, 5), (2, 6), (3, 7)]]

        for level in range(M): # level= 0, 1 ,2

            output=[0]*N

            L = level + 1 #
            J = list(range(2 ** (L - 1)))

            J_list= J* ( num_2FFT // len(J))

            for i in range(num_2FFT): # 计算 第level 级下, 第i个蝶形

                in1_index= cal_order[level][i][0]
                in2_index= cal_order[level][i][1]

                output[in1_index],output[in2_index]=self.base_2FFT(sample_input[in1_index],sample_input[in2_index],
                                                                   self._W(N,J_list[i]*(2**(M-L) )) )

            sample_input=output

        return  sample_input



if __name__ == '__main__':


    sol = OneDimension_FFT()

    # print(sol._reverse_range(4))
    # print(sol.trans_nFFT([1,0.5,0,0.5,1,1,0.5,0]))


    sol1=calculate_polynomial()

    # x_list = list(range(16))
    # a_list= np.ones(8,dtype=int)

    # print(sol1.cal_poly_v1(a_list, x_list))
    # print(sol1.cal_poly_v2(a_list, x_list))
    # print(sol1.cal_poly_vector(a_list, x_list))


    # print(sol1.one_2nth_root(8))

    x_list = sol1.one_2nth_root(8)
    a_list = np.ones(8, dtype=int)

    print(sol1.cal_poly_vector_one_2nth_root(a_list, x_list))





 # ------------- 计算多项式 cal_poly() 的大数据量 测试 ，统计运行时间 ---------------#

 #    x_list=np.random.randint(100,size=10000) # len(x_list)=10000
 #    a_list=rand.sample(range(0,10),8) # 从[0,1,2..,9] 中抽取 8 个
 #
 #    start = timeit.default_timer()
 #    print('by cal_poly_naive: ')
 #    print(sol1.cal_poly_naive(a_list, x_list)) # TODO：数值过大导致溢出  RuntimeWarning: overflow encountered in long_scalars A+= a*(x**i)
 #    end = timeit.default_timer()
 #    print('time: ', end - start, 's')
 #
 #
 #    start = timeit.default_timer()
 #    print('by cal_poly_v1: ')
 #    print(sol1.cal_poly_v1(a_list, x_list)) # TODO：数值过大导致溢出 RuntimeWarning: overflow encountered in long_scalars A = a_list[i]+x*A
 #    end = timeit.default_timer()
 #    print('time: ', end - start, 's')
 #
 #    start = timeit.default_timer()
 #    print('by cal_poly_v2: ')
 #    print(sol1.cal_poly_v2(a_list, x_list)) #
 #    end = timeit.default_timer()
 #    print('time: ', end-start ,'s')
 #
 #    start = timeit.default_timer()
 #    print('by cal_poly_vector: ')
 #    print(sol1.cal_poly_vector(a_list, x_list)) #
 #    end = timeit.default_timer()
 #    print('time: ', end-start ,'s')

# -----------------------------------------------------#










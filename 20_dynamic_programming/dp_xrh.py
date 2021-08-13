#!/usr/bin/python
# -*- coding: UTF-8 -*-


from numpy import *

class solutions:

    def yh_triangle_dp(self,num):
        """
        对杨辉三角 进行改造，每个位置的数字可以随意填写，经过某个数字只能到达下面一层相邻的两个数字。
        假设你站在第一层（最高层），往下移动，我们把移动到最底层所经过的所有数字之和，定义为路径的长度。
        
        求从最高层移动到最底层的最短路径长度。
        by 动态规划 
        :param num: 
        :return: 
        """

        states=[num[0][0]]

        for i in range(1,len(num)):
            next_states=zeros(i+1, dtype=int)

            #第一个 和最后一个元素特殊处理
            next_states[0]=states[0]+num[i][0]
            next_states[-1] = states[-1] + num[i][-1]

            # 中间的元素 使用 状态转移方程
            for j in range(1,i):
                next_states[j]=min(states[j-1]+num[i][j],states[j]+num[i][j])

            states=next_states

        print(states)

        return min(states)

    def coins_select(self, coin_values, pay):
        """    
        假设我们有几种不同币值的硬币 v1，v2，……，vn（单位是元）。
        如果我们要支付 w 元，求最少需要多少个硬币。
        比如，我们有 3 种不同的硬币，1 元、3 元、5 元，我们要支付 9 元，最少需要 3 个硬币（3 个 3 元的硬币）。
        
        :param coin_values: [1,3,5]
        :param pay: 9
        :return: 3,[1,3,5]
        """
        col_num=pay+1

        states_num = zeros(col_num, dtype=int) # 记录达到 某价格 所需要的最少硬币的数量
        states_coin=zeros(col_num, dtype=int) # 记录达到 某价格 最后一次取的硬币的面值

        for i in  range(1,col_num):
            min_num=float('inf')

            num = float('inf')
            coin=None

            for value in coin_values:

                if i-value>=0:
                    num=states_num[i-value]+1

                if num<min_num:
                    min_num=num
                    coin=value

            states_num[i]=min_num
            states_coin[i]=coin

        # print(states_num)
        # print(states_coin)

        pay_num=states_num[-1] # 达到 最终状态 pay=9 所需要的 硬币的个数
        last_coin=states_coin[-1] # 达到 最终状态 最后一次取的硬币的 面值
        coin_list=[last_coin] # 把 最后一枚 加入到 硬币列表
        sum_value=pay

        for i in range(pay_num-1):
            sum_value=sum_value-last_coin
            last_coin= states_coin[sum_value]

            coin_list.append(last_coin)

        return pay_num,coin_list

    def matrix_multiplication(self,P):
        """
        矩阵 链乘 的 规划
        
        根据 矩阵 乘法的 结合律： ABC = A(BC) 
        确定矩阵乘法的 次序 使得 元素相乘的 总次数 最少
        
        eg.
        P = [10 ,100 , 5, 50]
        A1: 10x100  A2: 100x5  A3:5x50
        
        矩阵乘法的 元素 相乘的次数
        (A1A2)A3: 10x100x5 + 10x5x50=7500 
        A1(A2A3):  100x5x50 + 10x100x50 = 75000
        
        显然，第一种 乘法次序 元素相乘的总次数少
        
        :param P: 矩阵链 [30,35,15,5,10,20]
        :return: 
        """

        L=len(P)

        states = zeros((L,L), dtype=int) # 记录 子矩阵链 P[i:j+1] 的最少元素相乘的次数

        s= zeros((L,L), dtype=int) # 记录 每一个子矩阵链 达到最少元素相乘次数 的最后一次划分的切割位置 k

        # 1. 子问题的划分 和 计算顺序
        for r in range(1,L):

            for i in range(0, L - r):

                j = i + r

                # 2. 递推方程
                if r == 1:
                    states[i][j] = 0

                elif r == 2:
                    states[i][j] = P[i] * P[i + 1] * P[i + 2]

                elif r > 2:  # r==3,4...

                    min_cost = float('inf')
                    min_cost_k=-2

                    for k in range(i + 1, j):

                        cost= states[i][k]+states[k][j]+ P[i] * P[k] * P[j]

                        if cost<min_cost: # 记录 最少的 元素相乘次数

                            min_cost=cost
                            min_cost_k = k

                    states[i][j]= min_cost
                    s[i][j]=min_cost_k

        print(states)
        print(s)

        res_min_cost=states[0][L-1]

        #3.解的追踪

        # res_k_list=[]

        def track_k_Recursion(left,right):
            """
            递归 追踪 切分位置 k 
            
            :param left: 
            :param right: 
            :return: 
            """

            k=s[left][right]

            if k!=0:
                res_k_list.append(k)
                track_k_Recursion(left,k)
                track_k_Recursion(k,right)


        def track_k(left, right):
            """
            非递归（利用栈） 追踪 切分位置 k 

            :param left: 
            :param right: 
            :return: 
            """

            stack=[]

            res_k_list=[]

            stack.append((left,right))

            while len(stack)>0:

                (left, right)=stack.pop()

                k = s[left][right]

                if k != 0:
                    res_k_list.append(k)
                    stack.append((k, right))
                    stack.append((left, k))


            return res_k_list


        # track_k_Recursion(0,L-1)

        res_k_list=track_k(0, L - 1)


        return res_min_cost,res_k_list


    def investment_problem(self,f_matrix,m):
        """
        投资问题 (整数规划)
        
        投资总金额 m ，一共有 n 个项目，每一个 项目的 投资金额 和 对应的回报 数目 由 f_matrix 表示
        
        f_matrix 中，行为 投资金额 ，列 为 相应项目的 回报 即 
        f(x,i) 表示 对 i 项目 投资 x 元 得到的回报；
        
        尝试 找到一个投资效益最大的 方案
        
        
        f_matrix:
        x   f(x,1)  f(x,2)  f(x,3)  f(x,4)
        0    0      0       0       0
        1    11     0       2       20
        2    12     5       10      21
        3    13     10      30      22
        4    14     15      32      23
        5    15     20      40      24
        
        m=5
        
        :param f_matrix: 
        :param m: 投资总金额
        :return: 
        """
        f_matrix=array(f_matrix)

        n=len(f_matrix[0]) # 项目个数

        states = zeros((n, m+1), dtype=int)
        s = zeros((n, m + 1), dtype=int)

        # 1. 子问题的划分 和 计算顺序

        states[0,:]=f_matrix[:,0] # k=0 可选的 只有 0 号项目
        s[0,:]= list(range(m+1))

        for k in range(1,n): # 可选的 项目 0,1,..k

            for x in range(m+1): # 投资金额为 x

                C_max=float('-inf')
                t_max=0

                # 2. 递推方程
                for t in range(0,x+1):

                    C= f_matrix[t,k]+states[k-1][x-t]  # 给 项目k 投资 t 元 的 总收益

                    if C>C_max:
                        C_max=C
                        t_max=t

                states[k,x]=C_max
                s[k,x]=t_max


        print(states)
        print(s)

        # 3.解的追踪
        investment_strategy= zeros( n, dtype=int)

        balance=m # 结余的 钱

        for k in range(n-1,-1,-1):

            k_t= s[k,balance] # 给 第k 个项目 t 元钱

            balance=balance-k_t # 还剩下 的钱

            investment_strategy[k]=k_t


        return states[-1,-1],investment_strategy


    def bag_problem(self, weights,values, capacity):
        """
        普通背包问题
         
        每一个物品可以放多个 , 区别于 01背包问题
        
        :param weights: 
        :param values: 
        :param capacity: 
        :return: 
        """

        n=len(weights) # 物品的个数

        states = zeros((n, capacity+1), dtype=int) # 背包达到的最大价值
        s = (-1)*ones((n, capacity + 1), dtype=int) # 背包达到的最大价值 时 装入 物品的最大标号; -1 表示无法装入任何物品


        # 1. 子问题的划分 和 计算顺序

        # 初始化
        states[0, :] = [ (i//weights[0])*values[0] for i in range(capacity+1) ]  # 只能选 NO.0个物品(k=0)时, 背包重量分别为 0,1,..,capacity 时 它的最大总价值

        for i in range(capacity + 1): # 只能选 NO.0个物品(k=0)时, 背包重量分别为 0,1,..,capacity 并达到 最大总价值时 装入 物品的 最大标号
            if i>= weights[0] :
                s[0, i]=0


        for k in range(1,n): # 可选的 物品 0,1,..k

            for x in range(1,capacity+1): # 背包的重量 x

                #2. 递推方程
                if x-weights[k]<0: # 第k 个物品 背包放不下
                    states[k,x]=states[k-1,x]

                    s[k, x] = s[k - 1, x]

                else:
                    add_k=states[k,x-weights[k]]+values[k] #放入 第k 个物品
                    not_add_k= states[k-1,x] # 不放入 第k 个物品

                    if add_k>=not_add_k:
                        states[k, x]=add_k
                        s[k, x]=k
                    else:
                        states[k, x] = not_add_k
                        s[k, x] = s[k - 1, x]


        print(states)
        print(s)

        # 3.解的追踪

        bag_items = zeros(n, dtype=int) # 背包中 每一样物品的 个数

        bag_weight = capacity  # 背包中的 剩余空间

        k=n-1

        while bag_weight>0:

            item_id= s[k, bag_weight] # 背包达到最大价值时 物品的最大标号

            if item_id==-1: # 此重量下 放不了 任何物品
                break

            bag_items[item_id]+=1 # 背包中 放入 1个 item_id 号物品

            bag_weight=bag_weight-weights[item_id]


        return states[-1][-1],bag_items

    def coins_select_advanced(self, weights, values, amount):
        """
        硬币 选择 问题 (背包问题变形)

        n 种 硬币 重量 weights 价值 values
        (第一种 硬币 必然 value=1  ,这保证了 必然能 凑出要求的金额 amount )
        
        需要支付的 总金额为 amount , 找到一种 重量最轻的 付钱的方式, 将硬币放入背包
        
        :param weights: [1,1,1,1] 
        :param values: [1,5,14,18] 
        :param amount: 28
        :return: 
        """
        n = len(weights)  # 硬币 的个数

        states = zeros((n, amount + 1), dtype=int)  # 选择前k 种硬币, 总额为 y 时 达到重量最轻的 背包重量
        s =  zeros((n, amount + 1), dtype=int)  #  达到重量最轻时，第k 种硬币的个数

        # 1. 子问题的划分 和 计算顺序

        # 初始化
        states[0, :] = [(y // values[0]) * weights[0] for y in
                        range(amount + 1)]  # 只能选 NO.0 种硬币(k=0)时, 总金额 分别为 0,1,..,amount 时 ，背包 达到 最轻的重量

        s[0, :] = [(y // values[0])  for y in
                        range(amount + 1)] # 只能选 NO.0 种硬币(k=0)时,总金额 分别为 0,1,..,amount 时 ，背包 达到 最轻的重量时 第0种 硬币的使用个数


        for k in range(1, n):  # 可选的 硬币  1,..k

            for y in range(0, amount + 1):  # 总金额 y

                # 2. 递推方程

                max_num= y//values[k] # 第k 种零钱 最多 的使用个数

                C_min = float('inf')
                i_C_min = 0

                for i in range(max_num+1): # 第k 种零钱 用i 个, i=0,1,..,max_num

                    C= states[k-1][y - i*values[k]]+i*weights[k] # 第k 种零钱 用i 个 背包的重量

                    if C < C_min:
                        C_min=C
                        i_C_min=i

                states[k][y]=C_min
                s[k][y] = i_C_min # 记录 总重量最小时 使用 第k 种硬币的个数

        print(states)
        print('-------------------')
        print(s)

        # 3.解的追踪

        coins_num = zeros(n, dtype=int)  # 背包中 每一种 硬币的 个数

        bag_value = amount  # 背包 剩余 总金额

        k = n - 1 # 从最后一种硬币开始 计算

        while bag_value > 0:

            num = s[k, bag_value]  # 第k 种硬币 的使用个数

            coins_num[k]=num

            bag_value = bag_value - num*values[k]

            k-=1

        return states[-1][-1], coins_num


    def image_compression_storage(self , P):
        """
        图像的分段压缩存储
        
        图像P 由像素点的 列表组成，
        像素点的 灰度值 范围 [0,255] 可以用一个 B=8bit 表达
        
        图像 P=[10,12,15,255,1,2,1,1,1,2,1,1]
        占用存储空间 N*8 bit=12*8= 96bit (N : 像素点个数)
        
        很多 灰度值较小的像素点 不需要占满 8 bit，所以可以对 P 分段进行存储
        
        分段 = 段头 + 像素点 
        段头 = 像素点个数 l (8 bit) + 像素点占用 位数 b (3 bit) = 11 bit
        
        8bit 可表示 1-256 , 3bit 可表示 1-8 
        
        P 可以 切分为 3 个分段：
        
        P1=[10,12,15] l1=3  最大的像素点为 15 占用的位数为 b1=4  
        P1占用的位数为 11+4*3=23
        
        P2=[255] l2=1 最大的像素点为 255 占用的位数为 b2=8
        P2占用的位数为 11+8*1=19
        
        P3=[1,2,1,1,1,2,1,1] l3=8 最大的像素点为 2 占用的位数为 b2=2
        P3占用的位数为 11+2*8=27
        
        P 占用的位数为 23+19+27=69 显然, 比所有像素点 都占用 8bit 要少
        
        :param P: [10,12,15,255,1,2,1,1,1,2,1,1]
        :return: 
        """

        n=len(P) # 像素点的 个数

        states = zeros(n+1, dtype=int) # P[0:i] 分段后 最少的占用位数
        s = (-1)*ones(n+1, dtype=int) #  P[0:i] 分段后 达到 最少的占用位数时，最后一次分段的切分点

        # 1. 子问题的划分 和 计算顺序

        # 初始化
        states[0]=0
        s[0]=-1

        for i in range(1,n+1):

            C_min=float('inf')
            k_C_min=-1

            # 2. 递推方程
            for k in range(0,i):

                C= states[k] + (11 + (int(log2(max(P[k:i])))+1)*(i-k) ) # numpy 中：log() 是自然对数 即 ln();
                                                                        #           log2() 才是 以2为 底的对数
                if C <= C_min:
                    C_min=C
                    k_C_min=k

            states[i]=C_min
            s[i]=k_C_min


        print(states)
        print(s)

        # 3.解的追踪
        split_list = [] # 记录 中间的 每一个 分段的切分点

        last_split=s[-1] # 当前 最后 一个分段的切分点

        while last_split!=0 and last_split!=-1:
            split_list.append(last_split)

            last_split=s[last_split]

        return states[-1],split_list

    def max_subarray(self, nums):
        """
        最大子数组问题 （股票问题 ; 最大字段和）

        :param nums: [-2,11,-4,13,-5,-2]
        :return: 
        """
        n = len(nums)  # 元素的个数

        states = zeros(n, dtype=int) # states[i]： 以nums[i] 为结尾的 最大子数组的 和
        s = (-1)*ones(n, dtype=int) # 以nums[i] 为结尾的 最大子数组 的开始 的位置

        # 1. 子问题的划分 和 计算顺序

        # 初始化
        states[0]=nums[0]
        s[0]=0

        # 2. 递推方程
        for i in range(1,n):

            C = states[i-1]+nums[i]

            if C >= nums[i]:

                states[i]=C
                s[i]=s[i-1]

            else:
                states[i]=nums[i]
                s[i]=i

        print(states)
        print(s)

        # states 中找到 最大的
        max_C = max(states)

        # 3.解的追踪
        max_C_idx= argmax(states)

        max_C_s=s[max_C_idx]

        return max_C, nums[max_C_s:max_C_idx+1]


    def opt_BST(self,leaf_node,data_node):
        """
        最优二叉检索树
        
        S=[A,B,C,D,E]
        
        所有节点的概率分布为：
        P=[0.04, 0.1, 0.02, 0.3, 0.02, 0.1, 0.05, 0.2, 0.06, 0.1, 0.01]
        
        空隙节点  L=[L0,L1,L2,L3,L4,L5]
        L 的概率分布 leaf_node=[0.04, 0.02, 0.02, 0.05, 0.06, 0.01]
        
        数据节点 S= [ None,A,B,C,D,E ]
        S 的概率分布 data_node=[0, 0.1, 0.3, 0.1, 0.2, 0.1]
        
        :param leaf_node: 
        :param data_node: 
        :return: 
        """
        L=len(leaf_node)

        states = zeros((L,L), dtype=float) # 记录 每一个 子问题的 平均比较次数

        s= zeros((L,L), dtype=int) # 记录 最后一次划分的 切割位置 k

        # 1. 子问题的划分 和 计算顺序
        for r in range(0,L):

            for i in range(1, L - r):

                j = i + r

                # 2. 递推方程
                if r == 0:
                    states[i][j] = data_node[i]+(leaf_node[i-1]+leaf_node[i])

                elif r > 0:  # r==1,2,3,...

                    min_cost = float('inf')
                    min_cost_k=-2

                    for k in range(i , j+1): # k= i,i+1,i+2,..,j-1,j


                        print('r:{},i:{},k:{}'.format(r, i, k))

                        if k==i:
                            cost = states[k+1][j]+ sum(data_node[i:j+1])+sum(leaf_node[i-1:j+1])

                        elif k==j:
                            cost = states[i][k-1] + sum(data_node[i:j + 1]) + sum(leaf_node[i - 1:j + 1])

                        else: # k= i+1,i+2,..,j-1

                            cost= states[i][k-1]+states[k+1][j]+ sum(data_node[i:j+1])+sum(leaf_node[i-1:j+1])

                        if cost<min_cost: # 记录 最少的 元素相乘次数

                            min_cost=cost
                            min_cost_k = k

                    states[i][j]= min_cost
                    s[i][j]=min_cost_k

        print(states)
        print(s)

        res_min_cost = states[1][L - 1]

        # 3.解的追踪

        res_k_list=[]

        def track_k_Recursion(left, right):
            """
            递归 追踪 切分位置 k 

            :param left: 
            :param right: 
            :return: 
            """

            k = s[left][right]

            if k != 0:
                res_k_list.append(k)
                track_k_Recursion(left, k-1)
                track_k_Recursion(k+1, right)

        track_k_Recursion(1,L-1)


        return res_min_cost, res_k_list

    def opt_RNA_structure(self, RNA_list):
        """
        最优 RNA二级结构

        给定 RNA 链，求具有 最多匹配 碱基对数的二级结构（最优结构）
        
        RNA 碱基对匹配规则：
        A-U C-G
        
        :param RNA_list: ['A','U','C','G','A','U','A','G','C','C','G','A','U']
        :return: 
        """
        L = len(RNA_list)

        match_rule={ 'A':'U','U':'A','C':'G','G':'C' } # RNA 碱基对匹配规则

        states = zeros((L, L), dtype=int)  # 记录 每一个 子问题的 最多匹配对数

        s = (-2)*ones((L, L), dtype=int)  # 记录 最后一次划分的 切割位置 k ，初始化-2 代表 不分割

        # 1. 子问题的划分 和 计算顺序
        for r in range(4, L):

            for i in range(0, L - r):

                j = i + r

                # 2. 递推方程
                if r == 4:

                    if match_rule[RNA_list[i]]==RNA_list[j]: # 碱基对 匹配
                        states[i][j] = 1


                else:  # r==5,6,7,...

                    max_matched = 0
                    max_matched_k = -2

                    for k in range(i, (j -4) +1):  # k= i,i+1,i+2,..,j-4

                        if match_rule[RNA_list[k]] == RNA_list[j]:  # RNA_list[k] 与 RNA_list[j] 碱基对 匹配

                            if k == i:
                                matched = states[k + 1][j-1] + 1 # 左子问题 没有了

                            elif k == j-4:
                                matched = states[i][k - 1] + 1 # 右子问题 没有了

                            else:  # k= i+1,i+2,..,j-5

                                matched = states[i][k - 1] + states[k + 1][j-1] + sum(data_node[i:j + 1]) + 1


                        if matched > max_matched:  # 记录 最大的 匹配对数

                            max_matched = matched
                            max_matched_k = k

                    if max_matched > states[i][j-1]:

                        states[i][j] = max_matched # RNA_list[j] 参与 匹配
                        s[i][j] = max_matched_k

                    else:
                        states[i][j] = states[i][j-1] # RNA_list[j] 不参与匹配
                        s[i][j] = s[i][j-1]

        print(states)
        print(s)

        return states[0][-1]

    def Longest_increasing_substring(self, nums):
        """
        最长递增 子序列（LIS）

        有一个数字序列包含 n 个不同的数字，如何求出这个序列中的最长递增子序列长度？
        eg.  2, 9, 3, 6, 5, 1, 7 这样一组数字序列，它的最长递增子序列就是 2, 3, 6, 7，所以最长递增子序列的长度是 4

        M2 简化 :
        states(i)表示 以 数值 nums(i) 为结尾的最长递增子序列的长度

        显然 nums(i) 为以 nums(i)为结尾的 递增子序列中的 最大的数值，

        对于每一个 states_length(i) ：
        遍历在 nums(i)之前 的所有数值，找出 最小的数值 nums(k)，
        case1 nums(i) >  nums(k) ：找满足 大于nums(i) 中最大的数值 nums(j)
                states_length(i)= states_length(j)+1 
        case2 nums(i) <= nums(k) ： states_length(i)=1

        :param nums: 
        :return: 
        """
        L = len(nums)
        nums = array(nums)

        states = zeros(L, dtype=int)  # 以 nums[i] 为结尾的前缀子序列的 LIS 的长度

        prev_ids = zeros(L, dtype=int) # 以 nums[i] 为结尾的前缀子序列的 LIS 的上一个元素的标号

        states[0] = 1
        prev_ids[0] = -2 # -2 代表 None

        # 1. 子问题的划分 和 计算顺序
        for i in range(1, L):

            current = nums[i]

            prev_min = float('inf')

            prev_smaller_max = float('-inf')  # 从 nums[0:i] 中 过滤出比 current 小的元素，并在其中找最大的元素 和 其Index
            prev_smaller_max_id = None

            # 2. 递推方程
            for j in range(i - 1, -1, -1):

                if nums[j] < prev_min:
                    prev_min = nums[j]

                if nums[j] < current and nums[j] > prev_smaller_max:
                    prev_smaller_max = nums[j]
                    prev_smaller_max_id = j

            if current <= prev_min:
                states[i] = 1
                prev_ids[i] = -2

            else:
                states[i] = states[prev_smaller_max_id] + 1
                prev_ids[i] = prev_smaller_max_id

        print(states)
        print(prev_ids)

        max_state = max(states)

        # 3.解的追踪
        LIS = []
        max_id = argmax(states)

        id = max_id

        while id != -2:
            LIS.append(nums[id])
            id = prev_ids[id]

        return max_state, LIS[::-1]


if __name__ == '__main__':

    sol=solutions()
    coin_values=[1,3,5]
    # print(sol.coins_select(coin_values,9))
    # print(sol.coins_select(coin_values, 10))
    # print(sol.coins_select(coin_values, 7))

    nums = [[3], [2, 6], [5, 4, 2], [6, 0, 3, 2]]
    # print(sol.yh_triangle_dp(nums))


    p=[30,35,15,5,10,20]
    # print (sol.matrix_multiplication(p))


    f_matrix=[[0,      0,       0,       0],
             [11,     0,       2,      20],
             [12,     5,       10,     21],
             [13,     10,      30,     22],
             [14,     15,      32,     23],
             [15,     20,      40,     24]]

    m=5
    # print(sol.investment_problem(f_matrix,m))

    weights = [2,3,4,7]
    values = [1,3,5,9]

    # weights = [2,3,4]
    # values = [1,3,5]

    capacity=10

    print(sol.bag_problem(weights,values,capacity))


    P = [10, 12, 15, 255, 1, 2, 1, 1, 1, 2, 1, 1]
    # print(sol.image_compression_storage(P))

    nums= [-2, 11, -4, 13, -5, -2]

    # print(sol.max_subarray(nums))

    leaf_node = [0.04, 0.02, 0.02, 0.05, 0.06, 0.01]
    data_node = [0, 0.1, 0.3, 0.1, 0.2, 0.1]

    # print(sol.opt_BST(leaf_node,data_node))

    RNA_list= ['A', 'U', 'C', 'G', 'A', 'U', 'A', 'G', 'C', 'C', 'G', 'A', 'U']

    # print(sol.opt_RNA_structure(RNA_list))

    # print(sol.Longest_increasing_substring([2, 9, 3, 6, 5, 1, 7]))

    weights= [1, 1, 1, 1]
    values= [1, 5, 14, 18]
    amount= 28

    # print(sol.coins_select_advanced(weights,values,amount))




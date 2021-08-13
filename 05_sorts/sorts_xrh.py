#!/usr/bin/python
# -*- coding: UTF-8 -*-

import timeit
import numpy as np


import random as rand

import re

class solutions:

    def bubble_sort(self,l):
        """
        冒泡排序
        :param l: 
        :return: 
        """

        for i in range(len(l)):
            time=0

            for j in range(1,len(l)):

                if l[j]<l[j-1]: # 保证冒泡排序算法的稳定性，当有相邻的两个元素大小相等的时候，我们不做交换
                    time+=1
                    l[j],l[j-1]=l[j-1],l[j]

            if time ==0: # 当某次冒泡操作已经没有数据交换时，说明已经达到完全有序，不用再继续执行后续的冒泡操作
                break

            # print(i+1,'次冒泡后结果：',l)

        return l


    def insertion_sort(self,l):
        """
         插入排序
        :param l: 
        :return: 
        """
        for i in range(len(l)):
            current=l[i] # current的左侧为已排序 区间

            for j in range(0,i): # 从已排序区间 中找插入的位置 ; for j in range(0,0): 不会遍历任何元素

                if current<l[j]:# 如果 比 current大 则说明 这个位置要给 current
                    l.pop(i) # 弹出i位置的元素，考虑 List长度已 -1 ; pop() 造成 数组中的元素要 进行搬移，有额外的代价
                    l.insert(j,current) # j<i 所以 j 的位置无需变化 ; insert 造成 数组中的元素要 进行搬移，有额外的代价
                    break
            # print(l)

        return l

    def insertion_sort_v2(self,l):
        """
         插入排序 v2
         （1） 避免 pop() 和 insert() 产生的额外 数组元素的搬移 开销；
               但是经过试验， insertion_sort() 更快 ：
                by insertion_sort: 
                time:  0.346201787 s
                by insertion_sort_v2: 
                time:  1.079507429 s
        :param l: 
        :return: 
        """
        if len(l)<=1:
            return l

        for i in range(1,len(l)):
            current=l[i] # current的左侧为 已排序 区间
            j=i-1  # 从尾到头 在 已排序 区间 里面查找插入位置 #TODO: 改进思路：用二分查找 找到插入的位置

            while j >=0 and l[j]>current: # 如果 比 current 大，则不断地把 元素往后移动，以空出插入的位置
                l[j+1]=l[j]
                j-=1
            l[j+1]=current # 稳定性：对于值相同的元素，我们可以选择将后面出现的元素，插入到前面出现元素的后面，这样就可以保持原有的前后顺序不变
            # print(l)

        return l

    def insertion_sort_v3(self,l):
        """
        插入排序 v3
        
        1.用二分查找 在有序的数组中 找到插入的位置
        :param l: 
        :return: 
        """
        pass #TODO

    def merge_sort(self, nums):
        """
        归并排序（分治法）
        
        :param nums: 
        :return: 
        """
        N = len(nums)

        self.nums = nums

        self.__merge_sort(0, N - 1)  # N-1 划重点！

        return self.nums

    def __merge_sort(self, left, right):

        if right > left:  # 划重点！递归注意递归退出条件！

            # 1.分解和递归：左半边有序 和 右半边有序
            mid = (right + left) // 2  # (right + left) 划重点！

            self.__merge_sort(left, mid)
            self.__merge_sort(mid + 1, right)  # mid+1 划重点！ left=0 right=1 mid=0 , self.__process(1 ,1)
            # self.__process(mid , right)  # left=0 right=1 mid=0 会永远陷入 self.__process(0 ,1) 的循环

            # 2. 合并：把左半边有序和右半边有序合起来 ，整个序列有序
            self.__merge(left, mid, right)

    def __merge(self, left, mid, right):

        merge_cache = []

        i = left
        j = mid + 1

        while i <= mid and j <= right:

            if self.nums[i] > self.nums[j]:
                merge_cache.append(self.nums[j])
                j += 1

            else:
                # eg. nums=[2,3,4,|1,5,6]  i=0 j=4 mid+1=3
                #
                merge_cache.append(self.nums[i])

                i += 1

        if i <= mid:
            merge_cache = merge_cache + self.nums[i:mid + 1]

        if j <= right:
            merge_cache = merge_cache + self.nums[j:right + 1]

        print('left:',left,'mid:',mid,'right:',right )
        print('left part:', self.nums[left:mid+1],'right part:',self.nums[mid+1:right+1])
        print('merge_cache:',merge_cache)

        self.nums[left:right + 1] = merge_cache

    def quick_sort(self,nums):
        """
        快速排序 （分治法）
        
        by quick_sort: 
        time:  0.02268848499999998 s

        :param nums: 
        :return: 
        """
        self.N=len(nums)

        self.nums=nums

        self.__quick_sort(0,self.N-1)

        return self.nums

    def __quick_sort(self,left, right):

        if right>left: # 注意递归退出条件

            # 1.分解：找到子问题 的切割方法，保证左右子问题规模近似

            pivot=rand.randint(left,right) # rand.randint(left,right) 取左右 闭区间，即 [left,right]

            q=self.__partition(left,right,pivot)


            # print(self.nums[left:q],self.nums[q],self.nums[q+1:right+1])

            # 2. 递归：求解 左右子问题
            self.__quick_sort(left,q-1)
            self.__quick_sort(q+1,right)



    def __partition(self,left,right,pivot):

        #把 pivot 放到末尾
        self.nums[pivot],self.nums[right]=self.nums[right],self.nums[pivot]

        pivot_num=self.nums[right]

        i=left
        j=left

        while j <right:

            if self.nums[j] < pivot_num:
                self.nums[j],self.nums[i]=self.nums[i],self.nums[j]
                j+=1
                i+=1

            else:
                j+=1

        self.nums[j], self.nums[i] = self.nums[i], self.nums[j]

        return i

    def quick_select(self, nums,K):
        """
        在 O(n) 的时间复杂度内查找一个无序数组中的第 K 大的元素 
        
        :param nums: 
        :return: 
        """
        self.N=len(nums)

        self.nums=nums

        self.k=self.N-K # 第 K 大的元素 相当于 第 (N-K) 小

        # self.k =K # 第K 小的元素

        if 0<=self.k<self.N:

            index=self.__quick_select(0,self.N-1)

            return self.nums[index]

        else:
            print('error: Incorrect assignment of K')
            return None

    def __quick_select(self, left, right):

        if right>left:  # 注意递归退出条件

            # 1.分解：找到子问题 的切割方法，保证左右子问题规模近似
            pivot=rand.randint(left,right) # rand.randint(left,right) 取左右 闭区间，即 [left,right]

            q=self.__partition(left,right,pivot)

            # print('left:',left,'q:',q,'right:',right)
            # print(self.nums[left:q], self.nums[q], self.nums[q + 1:right + 1])

            # 2. 递归：求解 左右子问题
            if q > self.k :
                res=self.__quick_select(left,q-1)

            elif q < self.k:
                res= self.__quick_select(q+1, right)

            else:
                res=q

            return res

        else:  # 注意考虑 递归到原子状况
            return left

    def characters_sorts(self,str1):
        """
        对 D，a，F，B，c，A，z 这个字符串进行排序，要求将其中所有小写字母都排在大写字母的前面，
        但小写字母内部和大写字母内部不要求有序
        
        ref: https://time.geekbang.org/column/article/42038
        :param string: 'DaFBcAz'
        :return: 'aczBDAF'
        """
        str_list=list(str1)
        uppercase = re.compile(r'[a-z]')
        # uppercase.match('A') # 无法匹配 返回None

        i=0
        j=0
        while j<len(str_list):
            if uppercase.match(str_list[j]) !=None: # 匹配到小写字母
                str_list[i],str_list[j]=str_list[j],str_list[i]
                j+=1
                i+=1
            else:
                j+=1

        return ''.join(str_list)


class Heap:

    def __init__(self, input_list): # 构建 大顶堆

        self.heap_list=input_list
        self.heap_list.insert(0,None) # 0 号位置 空置

        self.length=len(self.heap_list)

        for i in range(self.length//2,0,-1): #我们对下标从 n/2 开始到 1 的数据进行堆化,下标是 n/2+1 到 n 的节点是叶子节点，我们不需要堆化

            self.__up_to_down_max_heapfy(i) # 从子树的 根节点开始 往下堆化

    def __up_to_down_max_heapfy(self,p):
        """
        从上往下 堆化，构建大顶堆
        :param p: 
        :return: 
        """

        while p<self.length:

            left_child=2*p
            right_child=2*p+1

            max_index=p

            # 找出 当前节点 左子节点 和 右子节点  中的最大节点
            if left_child<self.length and self.heap_list[left_child] > self.heap_list[max_index]:
                max_index=left_child

            if  right_child <self.length and self.heap_list[right_child] > self.heap_list[max_index]:
                max_index=right_child

            self.heap_list[max_index], self.heap_list[p] = self.heap_list[p], self.heap_list[max_index] # 当前节点与 最大节点进行交换

            if p==max_index: #最大节点 还是 当前节点，说明无需做堆的调整 ，退出循环
                break
            p=max_index

    def sort(self):
        """
        实现 堆排序（逆序） 
        
        by heap_sort: 
        time:  0.04278195799999995 s
        :return: 
        """

        for i in range(self.length-1,0,-1):
            self.heap_list[i],self.heap_list[1]=self.heap_list[1],self.heap_list[i]
            self.length-=1
            self.__up_to_down_max_heapfy(1)

        return self.heap_list[1:]

class solutions2:

    def bucket_sort(self,min_ele,max_ele,person_list):
        """
        桶排序
        
        场景：
        考生的满分是 10 分，最小是 0 分，这个数据的范围很小，所以可以分成 11 个桶，对应分数从 0 分到 10 分。
        根据考生的成绩，我们将这 50 万考生划分到这 11 个桶里。
        桶内的数据都是分数相同的考生，所以并不需要再进行排序。我们只需要依次扫描每个桶，将桶内的考生依次输出到一个数组中，就实现了 50 万考生的排序。
        
        :param min_ele: 0
        :param max_ele: 10
        :param person_list: [('A',8),('B',9),('C',6),('D',8)]
        :return: 
        """
        buckets=[None]*((max_ele-min_ele)+1)

        for ele in person_list:
            l=buckets[ele[1]-min_ele]
            if l==None:
                buckets[ele[1] - min_ele]=[ele]
            else:
                buckets[ele[1] - min_ele].append(ele)

        # print(buckets)

        res=[]
        for bucket in buckets:
            if bucket!=None:
                for ele in bucket:
                    res.append(ele)

        return res


    def counting_sort(self,nums,min_ele=None,max_ele=None):
        """
        计数排序
        （1）保证 nums 中所有的元素为非负整数
        
        :param nums: [2,5,3,0,2,3,0,3]
        :return:  [0 0 2 2 3 3 3 5]
        """

        if min_ele==None or max_ele==None:

            min_ele=min(nums) # min_ele=0

            max_ele=max(nums) # max_ele=5

        C=np.zeros((max_ele-min_ele)+1,dtype=int)

        R=np.zeros(len(nums),dtype=int)

        #1. nums 的 每种元素的个数 记录在 C中，(C的标号-min_ele) 代表 对应元素值
        for ele in nums:
            C[ele-min_ele]+=1

        # print(C) # [2 0 2 3 0 1] , C[0]==2 表示 元素0 有两个

        #2. C[k] 里存储 小于等于 元素值(k+min_ele) 的元素的个数
        for i in range(1,len(C)):
            C[i]=C[i-1]+C[i]

        # print(C) # [2 2 4 7 7 8] , C[3]==7 表示 小于等于 3 的元素的个数有7个

        for i in range(len(nums)-1,-1,-1): #从后到前依次扫描数组 nums ，保证排序的稳定性
            ele=nums[i]
            R[ C[ele-min_ele]-1 ]=ele
            C[ele - min_ele]-=1

        return R

    def create_phone(self,number):

        return ( self.__create_phone() for i in range(number)) # 返回生成器

    def __create_phone(self):
        """
        随机生成 合法的 手机号码
        ref: https://blog.csdn.net/xiaobuding007/article/details/78726833
        :return: 
        """

        # 第二位数字
        second = [3, 4, 5, 7, 8][rand.randint(0, 4)] #[3, 4, 5, 7, 8] 随机取一个

        # 第三位数字
        third = {
            3: rand.randint(0, 9),
            4: [5, 7, 9][rand.randint(0, 2)],
            5: [i for i in range(10) if i != 4][rand.randint(0, 8)],
            7: [i for i in range(10) if i not in [4, 9]][rand.randint(0, 7)],
            8: rand.randint(0, 9),
        }[second]

        # 最后八位数字
        suffix = rand.randint(9999999, 100000000)

        # 拼接手机号
        return "1{}{}{}".format(second, third, suffix)

    def radix_sort(self, telephones):
        """
        基数排序 
        
        :param telephones: 
        :return: 
        """
        length=len(telephones[0]) # 手机号的位数

        for i in range(length-1,-1,-1): # 按照最后一位来排序手机号码，再按照倒数第二位重新排序，最后按照第一位重新排序

            telephone_tuples=map(lambda t:(t,int(t[i])),telephones)

            telephone_tuples=self.bucket_sort(0,9,telephone_tuples) # 每一位 手机号的范围是 0-9

            telephones=map(lambda t:t[0],telephone_tuples)

        return telephones




if __name__ == '__main__':

    sol = solutions()

    ##-------- part1 冒泡排序 插入排序 ------------##

    # print(sol.bubble_sort([4,5,6,3,2,1]))
    # print(sol.bubble_sort([3, 5, 4, 1, 2, 6]))
    # print(sol.bubble_sort([1, 2, 3, 4, 5, 6]))

    # print(sol.insertion_sort([4,5,6,1,3,2]))
    # print(sol.insertion_sort([3, 5, 4, 1, 2, 6]))
    # print(sol.insertion_sort_v2([6, 5, 4, 3, 2, 1 ,1]))


    # 冒泡排序 和 插入排序的 性能大对比
    l=np.random.randint(5000,size=5000)
    l1=list(l)

    # start = timeit.default_timer()
    # print('by bubble_sort: ')
    # sol.bubble_sort(l1)
    # end = timeit.default_timer()
    # print('time: ', end - start, 's')

    l3 = list(l) # bubble_sort 是原地排序算法，直接改变 输入的 list ，因此不能用上次排序过的 l1
    start = timeit.default_timer()
    print('by insertion_sort: ')
    sol.insertion_sort(l3)
    end = timeit.default_timer()
    print('time: ', end-start ,'s')

    # l2 = list(l)
    # start = timeit.default_timer()
    # print('by insertion_sort_v2: ')
    # sol.insertion_sort_v2(l2)
    # end = timeit.default_timer()
    # print('time: ', end-start ,'s')

    ##-------------- part1 end -----------------##


    ##-------- part2 归并排序 快速排序 堆排序 ------------##

    # print(sol.merge_sort([4, 5, 6, 1, 3, 2]))
    # print(sol.quick_sort([11, 8, 3, 9, 7, 1,2,5]))

    # print(sol.quick_select([11, 8, 3, 9, 7, 1,2,5], 1)) # 第一大的元素
    # print(sol.quick_select([11, 8, 3, 9, 7, 1, 2, 5], 3))

    # 性能测试
    l4 = list(l)
    start = timeit.default_timer()
    print('by quick_sort: ')
    sol.quick_sort(l4)
    end = timeit.default_timer()
    print('time: ', end-start ,'s')

    # l5 = list(l)
    # start = timeit.default_timer()
    # print('by quick_select: ')
    # print(sol.quick_select(l5,3))
    # end = timeit.default_timer()
    # print('time: ', end-start ,'s')

    l6 = list(l)
    start = timeit.default_timer()
    print('by heap_sort: ')
    heap = Heap(l6)
    heap.sort()
    end = timeit.default_timer()
    print('time: ', end-start ,'s')



    ##-------------- part2 end -----------------##


    ##-------- part3 桶排序 计数排序 基数排序 ------------##
    sol2 = solutions2()
    # print(sol2.counting_sort([2,5,3,0,2,3,0,3]))

    # print(sol2.bucket_sort(0,10,[('A',8),('B',9),('C',6),('D',8),('E',5)]))

    telephones=list(sol2.create_phone(10))
    # print(list(sol2.radix_sort(telephones)))

    # print(sol.characters_sorts('DaFBcAz'))





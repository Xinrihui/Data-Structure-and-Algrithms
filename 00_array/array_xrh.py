##---- python tips----##

#1.循环遍历删除满足条件的元素

def delete_element(arr):
    L=len(arr)
    for ele in range(L-1,-1,-1):
        if arr[ele] ==2:
            del arr[ele]
    return arr

# print (delete_element([1,2,3,2,5]))


#2. 数组切片的赋值问题
def test1():
    a=[1,2,3]
    a=a[0:2]+[5]+a[1:]
    print(a)

    b=[1,3,5]
    # b.insert(index=1,object:2)
    b.insert(1,2)
    print(b)

    c=[1,3,5]
    c.extend([6])
    print(c)

    c[0],c[1]=c[1],c[0]
    print(c)

# test1()

#2.bisect 的使用
import  bisect

def test2():
    l = [10, 30, 50, 70]
    print(bisect.bisect_left(l,30))
    print(bisect.bisect_left(l, 20))
    print(bisect.bisect_left(l, 80))

    # l.pop()
    # print(l)

    # print('index: ', l.index(70,0,3))
    print('l length: ',len(l))
    print('index: ', l.index(70, 0, 3+1))
    # l.remove(10)
    # print(l)

# test2()
from collections import deque

from queue import Queue

def test3():

    l=[99,203]
    l.insert(1,105)
    print(l)



    l=deque([99,203])
    print(l)
    # print(l[0:])

    # q = Queue([99,203])
    # print(q.get())

# test3()

#3. 二分查找的三种实现:
# http://kuanghy.github.io/2016/06/14/python-bisect
def binary_search_recursion(lst, value, low, high):
    if high < low:
        return None
    mid = (low + high) // 2
    if lst[mid] > value:
        return binary_search_recursion(lst, value, low, mid-1)
    elif lst[mid] < value:
        return binary_search_recursion(lst, value, mid+1, high)
    else:
        return mid

def binary_search_loop(lst,value):
    low, high = 0, len(lst)-1
    while low <= high:
        mid = (low + high) // 2
        if lst[mid] < value:
            low = mid + 1
        elif lst[mid] > value:
            high = mid - 1
        else:
            return mid
    return None

def binary_search_bisect(lst, x):
    from bisect import bisect_left
    i = bisect_left(lst, x)
    if i != len(lst) and lst[i] == x:
        return i
    return None


# import random
# random.seed(1)
# lst = [random.randint(0, 10000) for _ in range(100000)]
# lst.sort()
#
# def test_recursion():
#     binary_search_recursion(lst, 999, 0, len(lst)-1)
#
# def test_loop():
#     binary_search_loop(lst, 999)
#
# def test_bisect():
#     binary_search_bisect(lst, 999)
#
# import timeit
# t1 = timeit.Timer("test_recursion()", setup="from __main__ import test_recursion")
# t2 = timeit.Timer("test_loop()", setup="from __main__ import test_loop")
# t3 = timeit.Timer("test_bisect()", setup="from __main__ import test_bisect")
#
# print("Recursion:", t1.timeit())
# print("Loop:", t2.timeit())
# print("bisect:", t3.timeit())


# 4. 数组 标号访问
def test4():
    a=[1,0,1]
    # print(a[0,1]) #TypeError: list indices must be integers or slices, not tuple
    print(a[ [0,2] ]) #TypeError: list indices must be integers or slices, not list
# test4()


# 5.稀疏矩阵
import numpy as np
import scipy.sparse as sp
import sys

def test5():
    # A = np.array([[1, 0, 2, 0],
    #               [0, 0, 0, 0],
    #               [3, 0, 0, 0],
    #               [1, 0, 0, 4]])
    #
    # A_row = sp.csr_matrix(A)
    #
    # print(A_row)
    #
    # print()
    # A_col= sp.csc_matrix(A)
    # print(A_col)

    a={}
    a[0]='a'
    print(a)
    print('memory_size:', sys.getsizeof(a))
    a[1] = 'b'
    print(a)
    print('memory_size:', sys.getsizeof(a))

# 浮点运算
def test6():
    b=1.5e-20
    print(b*10)

    print(9/3)
    print( 10.0/3 ) #3.3333333333333335
    print(20 / 6)
    print( 20 // 6)
    print( (10.0/3)==(20.0/6))

    # t=2
    # print( 2<=t<=3)

# test6()



from collections import *

# 可变和不可变对象
def test7():
    a = '  ABC  '
    b = a
    a=a.strip()
    print(b)
    print(a)

    c=['a','b']
    d=c
    c.append('c')
    print(d)
    print(c)

# hash 表:  defaultdict
def test8():
    str1 = 'mississippi'
    dict_int = defaultdict(int)
    for s in str1:
        dict_int[s] += 1
    print(sorted(dict_int.items()))

    for (k, v) in dict_int.items():
        print( k,v)

    dic = {'a': 31, 'bc': 5, 'c': 3, 'asd': 4, 'aa': 74, 'd': 0}
    l = sorted(dic.items(), key=lambda d: d[1], reverse=True)
    print(l)

    items = (
        ('A', 1),
        ('B', 2),
        ('C', 3)
    )

    regular_dict = dict(items)
    ordered_dict = OrderedDict(items)

    print('Regular Dict:')
    for k, v in regular_dict.items():
        print(k, v)


    print('Ordered Dict:')
    for k, v in ordered_dict.items():
        print(k, v)


def test9():
    l = ['red', 'blue', 'red', 'green', 'blue', 'blue']
    cnt = Counter(l)
    print(cnt)  #'blue': 3, 'red': 2, 'green': 1}
    print(cnt['blue']) # 3

    import re
    words = re.findall(r'\w+', open('hamlet.txt').read().lower())

    print(Counter(words).most_common(3)) #[('the', 18), ('of', 11), ('to', 10)]


def test10():
    N = 9
    # 1. string to list
    board = ["..9748...", "7........", ".2.1.9...", "..7...24.", ".64.1.59.", ".98...3..", "...8.3.2.", "........6",
             "...2759.."]
    for j in range(N):
        one_row = board[j]
        board[j] = list(one_row)
    print(board)

    # 2. list to np.array
    board = np.array(board)
    print(board)
    #----------------------
    # 3. array to list
    board = board.tolist()
    print(board)

    # 4. list to string
    for j in range(N):
        one_row = (board[j])
        # print one_row
        board[j] = "".join(one_row)
    print(board)


# test8()


def test11():

    dic={}

    a=0
    b=tuple(set([1,2]))
    # print(b)

    # c=(a,b)
    # print(c)

    dic[(a,b)]=100
    print(dic[(a,b)])

    a={}
    a[0]=1
    # print(a[2])

# test11()
from numpy import *

def test12():
    a=[[2, 6], [1, 4], [3, 6], [3, 7], [6, 8], [2, 4], [3, 5]]
    a=array(a)

    b=list(map(lambda x: x[0], a))
    print(b)

    c=2
    def smaller_than(n):
        return n[0]<=c

    print( list( filter(lambda t:t[0]<c,a)))



# test12()

import heapq
def test13():

    char_list=[('a',45),('b',13),('c',12),('d',16),('e',9),('f',5)]

    # char_list = list(map(lambda x: (x[1], x[0]), char_list))

    # h=[]
    # for ele in char_list:
    #     heapq.heappush(h,ele)

    # heapq.heapify(char_list)
    # h=char_list
    #
    # print(h)
    # print([heapq.heappop(h) for i in range(len(h))])

    class ComapreHeap(object):
        def __init__(self, initial=None, key=lambda x: x):
            self.key = key
            if initial:
                self._data = [(key(item), item) for item in initial]
                heapq.heapify(self._data)
            else:
                self._data = []

        def push(self, item):
            heapq.heappush(self._data, (self.key(item), item))

        def pop(self):
            return heapq.heappop(self._data)[1]

    h=ComapreHeap(char_list,  key=lambda x:x[1])
    print([ h.pop() for i in range(len(char_list))])

# test13()
from numpy import *

a=[('<a' ,'</a>' ),('<span>','</span>')]

res= [[] for i in a]

res[0].append('a')
res[1].append('b')
res[0].append('c')

print(res)
##---- end python tips----##








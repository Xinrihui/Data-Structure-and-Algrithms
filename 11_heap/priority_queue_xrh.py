# -*- coding: UTF-8 -*-
from collections import *

import heapq


class Priority_Queue(object):
    """
    
    by XRH 
    date: 2020-05-01 
    
    利用堆实现 优先队列
    提供以下功能：
    1.根据主键直接访问  堆中的元素，包括读取和更新
    2.选择 比较大小 进而做相应的堆调整 的键
    3.当前最小键 从堆中 弹出
    4.元素插入，并调整堆
    
    """

    def __init__(self, initial=None, key_func=lambda x: x, compare_func=lambda x: x):
        """       
        :param initial:  初始的 key-value list，eg.[('a',1),('b',2),...,]
        :param key_func: 指定主键 key 的 lambda 函数，可以根据主键直接访问  堆中的元素
        :param compare_func: 指定 比较键 的lambda 函数，堆 根据此键 来比较元素之间的大小
        """

        self.key_func = key_func

        self.compare_func = compare_func


        self.hash_table = {}
        self._data = []

        if initial:
            self.length = len(initial)

            for item in initial:  # [(key1,value),(key2,value)]

                p_index = [compare_func(item), item]  # p_index 是一个指针，指向了 list 所在的内存地址

                self.hash_table[key_func(item)] = p_index
                self._data.append(p_index)

            heapq.heapify(self._data)

        else:
            self.length =0
            self._data = []

    def __len__(self):

        return self.length

    def has_Key(self, key):
        """
        判断 Key 是否存在
        :param key: 
        :return: 
        """
        return key in self.hash_table

    def get_byKey(self, key):
        """
        通过 key 读取对应的元组
        
        self.hash_table= {'a': [0, ('a', 0)], 'b': [2, ('b', 2)], 'c': [3, ('c', 3)]} 
        
        :param key: 'a'
        :return: ('a', 0) 
        """
        if key in self.hash_table:
            return self.hash_table[key][1]
        else:
            return None

    def push(self, item):
        """
        插入一个 元组，然后调整堆
        :param item: (key,value) 
        :return: 
        """
        self.length += 1

        p_index = [self.compare_func(item), item]

        self.hash_table[self.key_func(item)] = p_index
        heapq.heappush(self._data, p_index)

    def pop(self):
        """
        弹出 堆顶元素，即最小元素 
        :return: 
        """
        self.length -= 1

        ele = heapq.heappop(self._data)
        self.hash_table.pop(self.key_func(ele[1]))

        return ele[1]

    def  update(self, new_tuple):
        """
        从元组中 找到 key 和 用来调整堆的 值
        更新 key 对应的 元组 同时调整堆
        
        eg.
        self.hash_table= {'a': [0, ('a', 0, 'text')], 'b': [2, ('b', 2, 'text')], 'c': [3, ('c', 3, 'text')]}

        new_tuple= ('a', 0, 'text2')
        
        key 为 'a' , 用来调整堆的值为 0 
        
        :param new_tuple: 
        
        :return: 
        """
        key= self.key_func(new_tuple)

        compare_value=self.compare_func(new_tuple)

        self.hash_table[key][0] =compare_value
        self.hash_table[key][1] = new_tuple

        heapq.heapify(self._data)


    def update_byKey(self, key, value):
        """
        通过 key 找到对应的元组，并更新它的值，同时调整堆
        
        self.hash_table= {'a': [0, ('a', 0)], 'b': [2, ('b', 2)], 'c': [3, ('c', 3)]} 
                
        :param key:  'b'
        :param value: 4
        :return: 
        """
        # self.hash_table[key]=[value,(key,value)] # 无法更新 堆中 被引用的 List

        self.hash_table[key][0] = value  # self.hash_table[key] 返回为 List（它也 在堆中被引用） 的内存地址 ，self.hash_table[key][0] 直接更改了 List 中的内容
        self.hash_table[key][1] = (key, value)

        heapq.heapify(self._data)


if __name__ == '__main__':


    # heap = Priority_Queue(key_func=lambda x: x[0], compare_func=lambda x: x[1])  #
    #
    # heap.push(('a', 0))
    # heap.push(('b', 2))
    # heap.push(('c', 3))
    #
    # print(heap.get_byKey('a')) # 拿到键为'a' 的键值对 ('a', 0)
    #
    # print(heap.pop())
    #
    # heap.update_byKey('b', 4)
    # print(heap.pop())


    heap = Priority_Queue( key_func=lambda x: x[0], compare_func=lambda x: x[1])

    heap.push(('a', 0,'text' )) # ('a', 0,'node1')  key='a'  compare_value=0
    heap.push(('b',2,'text'))
    heap.push(('c', 3,'text'))

    print(heap.get_byKey('a')) # 拿到键为'a' 的 tuple

    print(heap.pop())

    heap.update(('b',4,'hhh'))

    print(heap.pop())
    print(heap.pop())









#!/usr/bin/python
# -*- coding: UTF-8 -*-

import timeit
import numpy as np

import sys
import random as rand


class Queue_array:
    """
    顺序队列
    """

    def __init__(self,capacity):

        self._items = [None]*(capacity+1) #最后一个位置 空置
        self._capacity = capacity
        self._head = 0
        self._tail = 0

    def enqueue(self,item):
        """
        入队
        :param item: 
        :return: 
        """
        if self._tail== self._capacity: # 队列的最后一个位置为空置，队尾指针指在此处

            if self._head!=0: # 进行数据的搬移 ，在头部腾出空间，插入新的元素

                self._items[0:self._tail-self._head]=self._items[self._head:self._tail]

                self._tail= self._tail-self._head
                self._head=0

            else: # self._head==0 并且 self._tail== self._capacity 表示 队列已满
                print('the Queue is full!')
                return False

        self._items[self._tail]=item
        self._tail+=1

        return True

    def dequeue(self):
        """
        出队
        :return: 
        """
        if self._head==self._tail: # 队列为空
            print('the Queue is empty!')
            return None

        res=self._items[self._head]

        self._items[self._head]=None
        self._head += 1

        return res


    def __repr__(self):
        return ','.join(self._items[self._head : self._tail])


class CircularQueue:
    """
    循环队列
    """
    def __init__(self,capacity):

        self._items = [None]*(capacity)
        self._capacity = capacity
        self._head = 0
        self._tail = 0

    def enqueue(self,item):
        """
        入队
        循环队列 省略了 数据搬移的 开销
        :param item: 
        :return: 
        """

        if (self._tail+1) % self._capacity==self._head: # (tail+1)% n=head  表示 队列已满
            print('the Queue is full!')
            return False

        self._items[self._tail]=item
        self._tail=(self._tail+1)%self._capacity

        return True

    def dequeue(self):
        """
        出队
        :return: 
        """
        if self._head==self._tail: # 队列为空
            print('the Queue is empty!')
            return None

        res=self._items[self._head]

        self._items[self._head]=None
        self._head = (self._head+1)%self._capacity

        return res

class BlockingQueue:
    """
    阻塞队列
    
    """
    def __init__(self, capacity):
        self._items = []
        self._capacity = capacity

    def producer(self,item): #TODO:多线程调用，然后给队列加锁

        if len(self._items)<=self._capacity:
            self._items.append(item)
            return True
        else:
            print('the Queue is full!')
            return False

    def consumer(self):

        if len(self._items)>0:
            res=self._items.pop()
            return res
        else:
            print('the Queue is empty!')
            return None


if __name__ == '__main__':

    # 1. 顺序队列
    # queue=Queue_array(8)
    # string_list=['a','b','c','d','e','f','g','h']
    #
    # for ele in string_list:
    #     queue.enqueue(ele)
    #
    # print(queue._items)
    #
    # queue.enqueue('i')
    #
    # print('pop:',queue.dequeue())
    # print('pop:', queue.dequeue())
    # print('pop:', queue.dequeue())
    # print(queue._items)
    #
    # queue.enqueue('i')
    # print(queue)

    #2. 循环队列
    queue = CircularQueue(8)

    string_list=['e','f','g','h','i','j']

    for ele in string_list:
        queue.enqueue(ele)

    print(queue._items)

    for i in range(3):
        print('pop:',queue.dequeue())

    print(queue._items)

    queue.enqueue('a')
    queue.enqueue('b')
    print(queue._items)

    queue.enqueue('c')
    queue.enqueue('d')
    print(queue._items)

    queue.enqueue('e')







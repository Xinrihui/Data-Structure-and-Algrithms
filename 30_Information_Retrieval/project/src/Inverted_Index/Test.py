#!/usr/bin/python
# -*- coding: UTF-8 -*-


from collections import *

from numpy import *

import re

import json

from pyspark.sql import SparkSession


from functools import reduce

class Test:



    def test1(self):

        """
        在 遍历dict 的时候 删除 符合条件的 元素
        :return: 
        """
        d = {'a': 1, 'b': 0, 'c': 1, 'd': 0}

        for k in d:  # 遍历dict，发现元素的值是0的话，就删掉
            if d[k] == 0:
                del (d[k])

        # RuntimeError: dictionary changed size during iteration

        for k in d.keys():# 遍历dict，发现元素的值是0的话，就删掉
            if d[k] == 0:
                del (d[k])

        # RuntimeError: dictionary changed size during iteration

        # d = dict( [(k, v) for k, v in d.items() if v != 0] )
        # print(d) # {'a': 1, 'c': 1}

        for k in list(d.keys()):
            print(d[k])
            if d[k] == 0:
                del (d[k])
            print(d)

    def test2(self):
        """
        测试 dict 求和
        :return: 
        """
        d = {'a': 1, 'b': 0, 'c': 1, 'd': 0}

        print(sum(list(d.values())))

        a=reduce(lambda x,y:x+y , d.values())

        print(a)

    def test3(self):
        """
        使用 Counter 进行统计
        :return: 
        """
        a=[(1,1),(1,2),(1,3),(1,2)]

        doc_id_list = [doc_id for term_id, doc_id in a]

        c=Counter(doc_id_list)

        print(list(c.items()))

    def test4(self):

        a=['123','abc']

        print(''.join(a))




if __name__ == '__main__':

    Test = Test()

    Test.test4()
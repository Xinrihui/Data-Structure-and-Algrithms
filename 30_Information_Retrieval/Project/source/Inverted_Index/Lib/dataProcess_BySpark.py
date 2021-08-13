#!/usr/bin/python
# -*- coding: UTF-8 -*-

import os

from collections import *

from numpy import *

import re

import json

from pyspark.sql import SparkSession


from functools import reduce


class Test:

    def test1(self):
        """
        不可见字符 的 打印 和 写入文本
        :return: 
        """

        SOH=chr(0x01) # SOH
        next_line_char=chr(0x0a) # 换行键（LF）
        # print(next_line_char) # 根本看不到 ； 只能写到文本 中使用 notepad++ 查看

        NUL=chr(0x00)
        STX=chr(0x02)
        EXT=chr(0x03)

        a='' #  直接从文本 中 粘SOH 过来
        print(ord(a)) # 输出 ASCII 码 为 1

        # 换行符 不能直接从文本 粘贴过来，这样直接在 IDE中就换行了

        line_str=next_line_char + SOH + NUL + STX + EXT

        data_dir = '../data_test/'
        test_file_dir = os.path.join(data_dir, 'invisible_characters.csv') #

        self.encoding='utf-8'

        with open(test_file_dir , "wb+") as f_test:
            f_test.seek(0, 0)  # 指向 切片文件的 首位

            row_text = line_str

            row_text_bytes = row_text.encode(self.encoding)

            f_test.write(row_text_bytes)


    def test2(self):


        orgin = " '!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~ "

        orgin = " \'!\"#$%&'()*+,-./:;<=>?@[\]^_`{|}~ "

        print('[{0}]'.format(re.escape(orgin)))



class DataProcess:

    """
    by XRH 
    date: 2020-06-14 

    利用 spark 做数据 处理
    
    功能：
    1.对 临时索引文件 tmp_index_spark.bin 按照 termid 进行排序
    2. 在 大文件中 输出 满足 特定 条件的 行
      2.1  读取 使用 0x01 作为分隔符 的 CSV ，并输出 字段数目不匹配的行
    
    """


    def sort_by_termid(self):
        """
        
        对 临时索引文件 tmp_index_spark.bin 按照 termid 进行排序
        
        :return: 
        """
        spark = SparkSession.builder.appName("sort_by_termid").getOrCreate()
        sc = spark.sparkContext

        # nums = sc.parallelize([1, 2, 3, 4])
        # print(nums.map(lambda x: x * x).collect())

        # data_dir = '..\\data\\' # windows 下必须使用 '\\'
        # tmp_index_file_dir = data_dir + 'tmp_index_spark.bin'  # for spark

        data_dir='../data/'
        tmp_index_file_dir=os.path.join(data_dir, 'tmp_index_spark.bin')

        sorted_tmp_index_file_dir=data_dir+'sorted_'+'tmp_index_spark.bin'

        lines = sc.textFile(tmp_index_file_dir,8) # 对大文件 进行切片 sliceNum=8，否则报错
                                                    # 20/05/30 11:54:28 ERROR PythonRunner: This may have been caused by a prior exception:
                                                    # java.net.SocketException: Connection reset by peer: socket write error

        # print(lines.first()) # 看第一条
        # print(lines.take(10)) # 可以看 指定数目的记录

        # for line in lines : #TypeError: 'RDD' object is not iterable
        #     print(line)

        lines=lines.map(lambda line:line.split("\t"))

        lines=lines.sortBy(lambda x: x[0])

        #TODO: 过滤掉 term_id =0 ，因为 term_id==0 为标点符号

        lines=lines.map(lambda line: line[0]+"\t"+line[1]) #采用 分隔符 区分两个 域

        # lines.saveAsTextFile(sorted_tmp_index_file_dir) # 在文件夹下 保存 sliceNum 个切片文件

        lines.coalesce(1, shuffle=True).saveAsTextFile(sorted_tmp_index_file_dir) # 在文件夹下 只有一个 切片

        # lines.saveAsSingleTextFile(sorted_tmp_index_file_dir)

        # list_b=[['1', '1'], ['0', '1'], ['2', '1'], ['3', '1'], ['4', '1'], ['5', '1'], ['6', '1'], ['7', '1'], ['8', '1'],
        #  ['9', '1']]
        # lines = sc.parallelize(list_b)
        # lines=lines.sortBy(lambda x: x[0])
        # print(lines.take(10))


    def __process_oneline(self,line, col_num):
        """
        自定义 Map 中的 lambda 函数
        
        处理 文件中的 每一行 
        :param line: 文件中的一行，已经是字符串了 (str) 
        :param col_num: 预设的字段数
        :return: 
        """
        line_array = line.split("\x01")
        length = len(line_array)

        line_array.append(length)

        res = None

        if length != col_num:  # 实际 字段数目 不符合 预设的 col_num

            res = line_array

        return res  # 每一行 必须都 要有 返回

    def __process_oneslice(self,lines_slice, col_num):
        """
        自定义 mapPartitions 中的 lambda 函数

        这个计算 过程被 分发给了 spark 的计算节点，
        计算节点 使用本地的数据分片进行 计算

        :param lines_slice: 文件的切片，其中有多行数据 （字符串数组）
        :param col_num: 
        :return: 
        """
        res = []

        for line in lines_slice:
            # line 为字符串
            line_array = line.split("\x01")

            length = len(line_array)

            line_array.append(length)  # 记录 总的字段数目

            if length != col_num:  # 找到 字段数目 不符合 col_num 的

                res.append(line + str(line_array))

        return res


    def find_bad_line(self):

        """
        spark 读取 使用 0x01 作为分隔符 的 CSV ，并输出 字段数目不匹配的行
        :return: 
        """

        spark = SparkSession.builder.appName("find_bad_line").getOrCreate()
        sc = spark.sparkContext

        data_dir = '../data_test/'

        test_file_dir = os.path.join(data_dir, '20200614.csv')

        result_file_dir = os.path.join(data_dir, '20200614-result.csv')

        sliceNum = 2
        lines = sc.textFile(test_file_dir, sliceNum)  # 对大文件 进行切片 sliceNum=8，否则报错
        # 20/05/30 11:54:28 ERROR PythonRunner: This may have been caused by a prior exception:
        # java.net.SocketException: Connection reset by peer: socket write error

        # print(lines.take(10))

        col_num = 3

        # linesByMap = lines.map(lambda line: self.__process_oneline(line, col_num))
        # print(linesByMap.take(10))  # [None, ['1', ' abc \x03 ', ' 超哥 ', ' 666 ', 4], None]


        linesByMapPartitions = lines.mapPartitions(lambda lines_slice: self.__process_oneslice(lines_slice, col_num))

        # print(linesByMapPartitions.take(10))

        # 分区合并
        one_slice = linesByMapPartitions.coalesce(1, shuffle=True)

        one_slice.saveAsTextFile(result_file_dir)  # 先删除之前的文件，否则报错

class Join:
    """
    by XRH 
    date: 2020-06-16 

    利用 spark 的基础算子 实现 常见的 join 算法

    功能：
    1. 朴素的 MapReduce 的 join 
    2. 基于 广播变量的 hash join 
    3. 
    
    """

    def common_join(self,table_a_dir,table_b_dir,table_dir):
        """
         利用基本算子 实现 MapReduce 的 join       
        :param table_a: 
        :param table_b: 
        :return: 
        """
        spark = SparkSession.builder.appName("common_Join").getOrCreate()
        sc = spark.sparkContext

        sliceNum = 2
        table_a = sc.textFile(table_a_dir, sliceNum) # 2个分区
        table_b = sc.textFile(table_b_dir, sliceNum) # 2个分区

        table_a = table_a.map(lambda line: line.split(','))
        table_b = table_b.map(lambda line: line.split(','))

        table_a = table_a.map(lambda line: (line[0], line[1:]))  # 只能有 两个元素 ，第1个为 Key; 否则后面的 groupByKey() 报错
        table_b = table_b.map(lambda line: (line[0], line[1:]))

        table = table_a.union(table_b)  # 合并后 分区的数目 也是 两个 RDD 的分区的和

        # table.glom().collect() # 输出 各个分区 的元素 列表

        # [[('1', ['a', '27']), ('2', ['b', '24']), ('3', ['c', '23'])],
        #  [('4', ['d', '21']), ('5', ['e', '22']), ('6', ['f', '20'])],
        #  [('1', ['male']), ('2', ['female'])],
        #  [('4', ['female']), ('5', ['male'])]]
        # 可以看出一共有4个分区

        #重新划分为2个分区, 默认采用 hash 分区, 因此 key 相同的会被 shuffle 到1个分区中
        table = table.partitionBy(2)

        # 1.此处原理与 MapReduce 不同, MapReduce 肯定会做shuffle
        # 一般 1个hdfs 的block对应 1个map task, 在1个map task中:
        # (1) 在环形缓冲区, 数据按照 分区+key 进行快速排序了
        # (2) 环形缓冲区溢出到磁盘, 对每一个分区对应的多个溢出文件进行归并排序, 最后生成 分区文件, 一个分区对应一个文件

        # 1个分区对应 1个reduce task, 在1个reduce task中:
        # (1) 去拉取 map task 在磁盘上的, 我对应要处理的分区文件, 然后进行归并排序
        # (2) 从归并排序后的文件中, 按顺序提取出 (key, key 对应的 value-list ) 输入给reduce 函数,
        #     如果是两张表join, 则此步骤相当于完成了按照key的join 操作

        # 2. 可以看出 spark 相较于 MapReduce ,操作更加灵活, 在spark 中shuffle 是可选的


        # table.glom().collect()

        # [[('1', ['a', '27']), ('4', ['d', '21']), ('1', ['male']), ('4', ['female'])],
        #  [('2', ['b', '24']),
        #   ('3', ['c', '23']),
        #   ('5', ['e', '22']),
        #   ('6', ['f', '20']),
        #   ('2', ['female']),
        #   ('5', ['male'])]]
        # 可以看出一共有2个分区, 并且相同的 key 在同一分区

        def process_oneslice(one_slice, col_num):
            """
            对一个分区的处理

            :param one_slice:
            :param col_num:
            :return:
            """

            res = []

            hash_table = {}

            for line in one_slice:

                key = line[0]
                value = line[1]

                if key not in hash_table:
                    hash_table[key] = value

                else:
                    hash_table[key] = hash_table[key] + value

            for key, value in hash_table.items():

                if len(value) == col_num:  # 这一行的 col 个数 匹配 说明 关联成功

                    res.append([key] + value)

            return res

        col_num = 3  # 最终表 除了 Key 之外 应该有 3 个列（字段）
        table = table.mapPartitions(lambda one_slice: process_oneslice(one_slice, col_num))

        # table.glom().collect()

        table_one_slice = table.map(lambda line: ",".join(line)).coalesce(1, shuffle=True)  # 输出为 一个切片

        table_one_slice.saveAsTextFile(table_dir)


    def hash_join(self,table_a_dir,table_b_dir,table_dir):
        """
        利用 基本 算子 实现 hash join 
        :return: 
        """

        spark = SparkSession.builder.appName("hash_join").getOrCreate()
        sc = spark.sparkContext

        sliceNum = 2
        table_a = sc.textFile(table_a_dir, sliceNum)
        table_b = sc.textFile(table_b_dir, sliceNum)

        table_a = table_a.map(lambda line: line.split(','))  # 大表
        table_b = table_b.map(lambda line: line.split(','))  # 小表

        table_a = table_a.map(lambda line: (line[0], line[1:]))  # 只能有 两个元素 ，第1个为 Key; 否则后面的 groupByKey() 报错
        table_b = table_b.map(lambda line: (line[0], line[1:]))

        table_b = table_b.collect()  # [('1', ['male']), ('2', ['female']), ('4', ['female']), ('5', ['male'])]

        hash_table_b = {} # 把小表 做成 hash 表

        for line in table_b:
            hash_table_b[line[0]] = line[1][0]

        # 把小表 作为 广播变量 分发到各个 计算节点上
        broadcast_table_b = sc.broadcast(hash_table_b)  # SPARK-5063: RDD 不能被广播

        def process_oneslice(big_table_slice):

            res = []

            for line in big_table_slice:

                key = line[0]

                values = line[1]

                if key in broadcast_table_b:
                    res.append([key] + [hash_table_b[key]] + values)

            return res

        table = table_a.mapPartitions(lambda big_table_slice: process_oneslice(big_table_slice))

        # table.collect()

        table_one_slice = table.map(lambda line: ",".join(line)).coalesce(1, shuffle=True)  # 输出为 一个切片

        table_one_slice.saveAsTextFile(table_dir)



    def shuffle_Hash_join(self):
        """
        实现 一个 基于分区 的 Join 
        :return: 
        """
        spark = SparkSession.builder.appName("backet_map_join").getOrCreate()
        sc = spark.sparkContext

        #TODO: 如何 同时 操作 两个分区 中的数据， eg. 把一个 分区中的 数据  放入 内存中 做成 hash 表，与另一个分区 关联



if __name__ == '__main__':

    sol = DataProcess()

    # sol.find_bad_line()

    Test = Test()

    # Test.test2()

    #---------------  join 函数 测试 -------------#
    data_dir = '../data_test/'

    table_a_dir = os.path.join(data_dir, 'table_A')

    table_b_dir = os.path.join(data_dir, 'table_B')

    table_dir = os.path.join(data_dir, 'table')

    sol2=Join()

    sol2.common_join(table_a_dir,table_b_dir,table_dir)

    # sol2.hash_join(table_a_dir, table_b_dir, table_dir)


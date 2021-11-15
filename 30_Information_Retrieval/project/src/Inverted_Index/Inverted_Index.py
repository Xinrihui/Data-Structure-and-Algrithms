#!/usr/bin/python
# -*- coding: UTF-8 -*-

import os

import sys
sys.path.append('../Web_Crawler') # 能找到 此目录下的类

import pickle


from collections import *

import jieba
import re
import timeit


class noDataError(ValueError):
    pass

class fileNotExisitError(ValueError):
    pass

class fileIndexIllegal (ValueError):
    pass


class Inverted_Index_File:
    """
    by XRH 
    date: 2020-06-03 

    倒排索引文件（ index.bin ）

    解析 排序后的临时索引 文件（ sorted_tmp_index.bin），生成 倒排索引文件（index.bin）

    设计详见：《数据结构与算法之美》-> 算法实战（二）：剖析搜索引擎背后的经典数据结构和算法

    功能：

    1. 内存中 维护一个 Hash table （ term_offset.bin ），里面记录 每一个  单词编号在倒排索引文件中的偏移位置 
    2. 内存中 维护一个 Hash table （ doc_nums.bin ）, 里面 记录 每一个文档的  词项的总数 （可以作为 文档的长度）

    """

    def __init__(self, sorted_tmp_index_file_dir='sorted_tmp_index.bin',
                 index_file_dir='index.bin',
                 term_offset_file_dir='term_offset.bin',

                 tmp_index_file_dir='tmp_index.bin',
                 doc_termsNums_dir='doc_termsNums.bin',

                 term_id_len=4, doc_num_len=4,
                 term_num_len=4, doc_id_len=4,
                 encoding='utf-8',slice_max_bytes=10485760):

        """
        :param tmp_index_file_dir: 
        
        :param term_id_len: 词 ID 的字节长度
        :param doc_num_len:  词出现过的 文档数目 的字节长度
        
        :param term_num_len: 词在 该文档中 出现次数 的字节长度
        :param doc_id_len:  文档 ID 的字节长度
        
        :param encoding: 
        :param slice_max_bytes: 内存的阈值，达到这个阈值 即刷写入磁盘，形成切片文件；
                                文件切片的大小为： 10485760 B = 10MB
        """

        self.sorted_tmp_index_file_dir = sorted_tmp_index_file_dir
        self.index_file_dir=index_file_dir
        self.term_offset_file_dir=term_offset_file_dir

        self.tmp_index_file_dir=tmp_index_file_dir
        self.doc_termsNums_dir=doc_termsNums_dir

        self.slice_max_bytes = int(slice_max_bytes)

        self.encoding=encoding

        self.term_id_len=term_id_len
        self.doc_num_len=doc_num_len

        self.term_num_len=term_num_len
        self.doc_id_len=doc_id_len

        # 1.打开 sorted_tmp_index_file.bin 文件

        # 如果存在，则直接读取
        if os.path.exists(self.sorted_tmp_index_file_dir):
            self.f= open(self.sorted_tmp_index_file_dir, "rb+") #  + 表示 打开一个文件进行更新(可读可写)

        else:
            raise fileNotExisitError('the file ', self.sorted_tmp_index_file_dir,'  not exisits')

        #2. 打开 倒排索引文件

        if os.path.exists(self.index_file_dir):
            self.f_index= open(self.index_file_dir, "rb+") #  + 表示 打开一个文件进行更新(可读可写)

        else: # 如果 不存在，则创建一个
            self.f_index = open(self.index_file_dir, "wb+")


        #3.初始化 hash_table： term_offset.bin

        self.hash_term_offset=None

        # 如果存在，则直接读取
        if os.path.exists(self.term_offset_file_dir):
            f_dict= open(self.term_offset_file_dir, "rb")
            self.hash_term_offset = pickle.load(f_dict)
            f_dict.close()

        else: # 如果 不存在，则创建一个
            self.hash_term_offset=defaultdict(str)

        # 5.打开 sorted_tmp_index_file.bin 文件

        # 如果存在，则直接读取
        if os.path.exists(self.tmp_index_file_dir):
            self.f_tmp= open(self.tmp_index_file_dir, "rb+") #  + 表示 打开一个文件进行更新(可读可写)

        else:
            raise fileNotExisitError('the file ', self.tmp_index_file_dir,'  not exisits')


        # 6.初始化 hash_table：doc_nums.bin
        self.hash_doc_num=None

        # 如果存在，则直接读取
        if os.path.exists(self.doc_termsNums_dir):
            f_dict= open(self.doc_termsNums_dir, "rb")
            self.hash_doc_num = pickle.load(f_dict)
            f_dict.close()

        else: # 如果 不存在，则创建一个
            self.hash_doc_num=defaultdict(str)


    def __get_file_tail(self,f=None):
        """
        指针指向文件末尾，同时取得文件 末尾的地址 
        :return: 
        """
        if f==None:
            f=self.f

        index=f.seek(0, 2)

        if index==-1:
            raise fileIndexIllegal

        return f.tell()

    def __readDoc_tuple(self,start_address,f=None):
        """
        根据 起始位置 读取  排序的临时索引文件（ sorted_tmp_index_file.bin）
        
        返回 每一个元组（词项 ID ,文档 ID）
        :param start_address: 起始位置
        :return: 词项 ID ,文档 ID , 当前指针地址
        """
        if f==None:
            f=self.f

        f.seek(start_address, 0)  # 文件指针 指向 start_address

        term_id = int.from_bytes(f.read(self.term_id_len), byteorder='big')

        doc_id = int.from_bytes(f.read(self.doc_id_len), byteorder='big')  # 读取 doc_length 域的值

        return term_id, doc_id, f.tell()


    def __write_Inverted_Index_Doc(self, start_address, term_inverted_list ,f=None):
        """
        在倒排索引文件的 起始位置 写入  一个 词项的 倒排表

        :param start_address: 起始位置
        :param term_inverted_list: 待写入的 词项的 倒排记录表 [1,2,[(1,2),(2,3)]]
        :return: 当前指针地址
        """
        if f==None:
            f=self.f_index

        f.seek(start_address, 0)  # 文件指针 指向 start_address

        term_id = term_inverted_list[0]
        doc_num = term_inverted_list[1]

        inverted_list = term_inverted_list[2]

        term_id_bytes = int(term_id).to_bytes(length=self.term_id_len, byteorder='big')
        f.write(term_id_bytes)

        doc_num_bytes = int(doc_num).to_bytes(length=self.doc_num_len, byteorder='big')
        f.write(doc_num_bytes)

        for doc_id,term_num in inverted_list:

            doc_id_bytes = int(doc_id).to_bytes(length=self.doc_id_len, byteorder='big')
            f.write(doc_id_bytes)

            term_num_bytes = int(term_num).to_bytes(length=self.term_num_len, byteorder='big')
            f.write(term_num_bytes)


        f.flush()  # 将文件缓存中的内容 刷入磁盘

        return f.tell()

    def __read_Inverted_Index_Doc(self, start_address, f=None):
        """
        在倒排索引文件的  某个位置  读取一个 词项的 倒排表

        :param start_address: 指针位置

        :return: term_inverted_list( 词项的 倒排记录表) ，当前指针地址
        """
        if f == None:
            f = self.f_index

        f.seek(start_address, 0)  # 文件指针 指向 start_address

        term_id = int.from_bytes(f.read(self.term_id_len), byteorder='big')

        doc_num = int.from_bytes(f.read(self.doc_num_len), byteorder='big') #

        inverted_list = []

        for i in range(doc_num):

            doc_id = int.from_bytes(f.read(self.doc_id_len), byteorder='big')

            term_num = int.from_bytes(f.read(self.term_num_len), byteorder='big')

            inverted_list.append((doc_id,term_num))


        term_inverted_list=[term_id,doc_num,inverted_list]

        return term_inverted_list,f.tell()

    def count_doc_itemNums(self):
        """
        统计  每一个文档的  词项的总数
        
        :return: 
        """
        start = 0  # 指向 临时索引文件 的开头

        while start != self.__get_file_tail(self.f_tmp):  #  指针指向 临时索引文件 的末尾 即 退出循环

            prev_term_id, prev_doc_id, prev_start = self.__readDoc_tuple(start,self.f_tmp)

            term_num=1
            self.hash_doc_num[prev_doc_id] = term_num

            while True:

                term_id, doc_id,start=self.__readDoc_tuple(prev_start,self.f_tmp)

                if doc_id!=prev_doc_id:# 发现 文档ID 变化 就退出

                    self.hash_doc_num[prev_doc_id]=term_num # 记录 prev_doc_id 的 词项的总数

                    start=prev_start # 读了 下一个 doc ， 指针回到上一个 doc
                    break

                term_num+=1
                prev_term_id, prev_doc_id, prev_start = term_id, doc_id, start

        return True

    def get_hash_doc_num(self):

        return self.hash_doc_num


    def building(self):
        """
        构建 倒排索引
        
        :return: 
        """

        start=0 # 指向 排序的临时索引文件 的开头

        start_index=0 # 指向 倒排索引文件的 开头

        while start!=self.__get_file_tail(self.f): #  指针指向 排序的临时索引文件 的末尾 即 退出循环

            # 1. 读取 排序的临时索引文件 直到 遇到 单词id+1

            one_term_tuple_list=[] # 一个词项的 （词项 ID ,文档 ID） 列表

            prev_term_id, prev_doc_id, prev_start = self.__readDoc_tuple(start)
            one_term_tuple_list.append((prev_term_id, prev_doc_id))


            while True:

                term_id, doc_id,start=self.__readDoc_tuple(prev_start)

                if term_id!=prev_term_id:# 发现 单词ID 变化 就退出
                    start=prev_start # 读了 下一个 term ， 指针回到上一个 term
                    break

                one_term_tuple_list.append((term_id,doc_id))

                prev_term_id,prev_doc_id,prev_start=term_id,doc_id,start

            # 2. 对 每一个 词项 进行处理 ，并写入 倒排索引文件中

            print('one_term_tuple_list: ',one_term_tuple_list[0:20])

            # one_term_tuple_list=sorted(one_term_tuple_list,key=lambda x:x[1])

            doc_id_list=[doc_id for term_id,doc_id in one_term_tuple_list]

            doc_id_nums=Counter(doc_id_list) # 统计 文档ID 的数目

            inverted_list=list(doc_id_nums.items()) # [(doc_id, term_num),...]


            term_id=one_term_tuple_list[0][0]

            if term_id==19010:
                print('stop')

            doc_num=len(inverted_list)

            term_inverted_list=[term_id,doc_num, inverted_list]

            print("term_inverted_list:",term_inverted_list)

            self.hash_term_offset[term_id]=start_index # 记录 term 在倒排索引中的 偏移位置

            start_index=self.__write_Inverted_Index_Doc(start_index,term_inverted_list)


    def search(self, term_id):
        """
        根据 词项 ID 查找 其对应的 倒排列表 
        :param : term_id 
        :return: term_inverted_list：词项的 倒排记录表 [term_id=1,doc_num=2,inverted_list=[(1,2),(2,3)]]
        
        """
        start_address=self.hash_term_offset[term_id]

        term_inverted_list,address=self.__read_Inverted_Index_Doc(start_address)

        return term_inverted_list


    def close_file(self):
        """
        关闭文件流
        最后一定要加上（重要！！！）
        :return: 
        """
        self.f.close()
        self.f_index.close()
        self.f_tmp.close()

        # hash_id_offset 持久化 到磁盘
        f_dict = open(self.term_offset_file_dir, 'wb')
        pickle._dump(self.hash_term_offset, f_dict)
        f_dict.close()

        # hash_doc_num 持久化 到磁盘
        f_dict = open(self.doc_termsNums_dir, 'wb')
        pickle._dump(self.hash_doc_num, f_dict)
        f_dict.close()



if __name__ == '__main__':


    data_dir='data/Aerospace/'

    sorted_tmp_index_file_dir =os.path.join(data_dir, 'sorted_tmp_index.bin')

    index_file_dir =os.path.join(data_dir,'index.bin')

    term_offset_file_dir =os.path.join(data_dir,'term_offset.bin')


    tmp_index_file_dir =os.path.join(data_dir,'tmp_index.bin')
    doc_termsNums_dir =os.path.join(data_dir, 'doc_termsNums.bin')


    sol=Inverted_Index_File(sorted_tmp_index_file_dir,index_file_dir,term_offset_file_dir,tmp_index_file_dir,doc_termsNums_dir)

    start = timeit.default_timer()

    sol.building()
    # print(sol.search(19010)) # term_id=143740 :'RC44' ; doc_id=4898

    sol.count_doc_itemNums()

    # hash_doc_num=sol.get_hash_doc_num()
    # print(hash_doc_num)

    end = timeit.default_timer()
    print('cost time: ', end - start, 's')

    # del sol # 可以手动删除对象 ； Python的内存管理机制 也能自动回收

    sol.close_file()









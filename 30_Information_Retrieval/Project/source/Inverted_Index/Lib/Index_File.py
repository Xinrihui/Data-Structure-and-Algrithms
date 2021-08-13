#!/usr/bin/python
# -*- coding: UTF-8 -*-

import os

import sys
sys.path.append('../../Web_Crawler') # 能找到 此目录下的类

import pickle

from collections import defaultdict

import jieba
import re
import timeit

from Lib.Doc_File import Doc_Raw_File


class noDataError(ValueError):
    pass

class Tmp_Index_File:
    """
    by XRH 
    date: 2020-05-23 
    
    临时索引文件（tmp_index.bin）
    
    解析 爬虫爬取的 原始网页文档文件（doc_raw.bin），生成 存储 临时索引 的文件（tmp_index.bin）
    
    设计详见：《数据结构与算法之美》-> 算法实战（二）：剖析搜索引擎背后的经典数据结构和算法
    
    功能：
    
    1. 文件尾部 追加 （单词 id - 文档 id） 
    2. 内存中 维护一个 Hash table （ term_id.bin ），里面记录 每一个 单词 和 它对应的 单词 ID ，其中 单词 id 的计数器 可以直接 使用 hash 表的 长度
    3. 同理 2，内存中 维护一个 Hash table （inver_term_id.bin） ，里面 记录 每一个 单词 ID 和 它对应的 单词 
    4. 生成 词条字典（ baidubaikeDic.txt ） 并 用于 对 词条文档的分词 
    
    """


    def __init__(self,doc_raw_file_dir,doc_raw_offset_file_dir, tmp_index_file_dir='tmp_index.bin',term_id_file_dir='term_id.bin',inver_term_id_file_dir='inver_term_id.bin',
                 baidubaikeDic_dir='baidubaikeDic.txt',
                 term_id_len=4,doc_id_len=4,
                 encoding='utf-8',max_bytes=1e10):
        """
        
        :param doc_raw_file_dir: 
        :param doc_raw_offset_file_dir: 
        :param tmp_index_file_dir: 
        :param term_id_file_dir: 
        :param inver_term_id_file_dir: 
        :param baidubaikeDic_dir: 
        :param term_id_len: 
        :param doc_id_len: 
        :param encoding: 
        :param max_bytes: 文件 最大的长度（B） 1e10B=10GB
        """

        self.doc_raw_file_dir=doc_raw_file_dir
        self.doc_raw_offset_file_dir=doc_raw_offset_file_dir

        self.tmp_index_file_dir = tmp_index_file_dir

        self.term_id_file_dir=term_id_file_dir
        self.inver_term_id_file_dir=inver_term_id_file_dir

        self.baidubaikeDic_dir=baidubaikeDic_dir

        # 1.打开 tmp_index.bin 文件

        # 如果存在，则直接读取
        if os.path.exists(self.tmp_index_file_dir):
            self.f= open(self.tmp_index_file_dir, "rb+") #  + 表示 打开一个文件进行更新(可读可写)

        else: # 如果 不存在，则创建一个
            self.f = open(self.tmp_index_file_dir, "wb+")

        self.term_id_len=term_id_len
        self.doc_id_len=doc_id_len

        self.f.seek(0, 0)  # 文件指针指向 首位

        self.encoding=encoding
        self.max_bytes=int(max_bytes)

        #2. 初始化 hash_term_id
        self.hash_term_id=None

        # 如果存在，则直接读取
        if os.path.exists(self.term_id_file_dir):
            f_dict= open(self.term_id_file_dir, "rb")
            self.hash_term_id = pickle.load(f_dict)
            f_dict.close()

        else: # 如果 不存在，则创建一个
            self.hash_term_id=defaultdict(int)

        #3. 初始化 hash_id_term
        self.hash_id_term=None

        # 如果存在，则直接读取
        if os.path.exists(self.inver_term_id_file_dir):
            f_dict= open(self.inver_term_id_file_dir, "rb")
            self.hash_id_term = pickle.load(f_dict)
            f_dict.close()

        else: # 如果 不存在，则创建一个
            self.hash_id_term=defaultdict(int)


        #4.生成 百度百科 专有名词词典，用于后面的分词（重要！否则 分词的质量很差）

        self.doc_row_file = Doc_Raw_File(doc_raw_file_dir, doc_raw_offset_file_dir)

        self.__build_baidubaikeDic()

        # 5.配置 正则表达式

        self.reg_punctuation_marks=re.compile('[^0-9A-Za-z\u4e00-\u9fa5]') #  正则匹配 非（数字、大小写字母和中文）即标点符号
                                                                              # TODO:过于粗糙 可以采用停止词典 ;
                                                                              # TODO: 一些符号 要保留 eg. % - ～  最好的方法是 切词的时候 F-22 不要拆成两个词



    def __build_baidubaikeDic(self,rebuild=True):
        """
        读取 原始网页存储文件 doc_raw.bin 中的 每一个 词条页的文档，获得词条，并生成 百度百科专有名词词典
        :param rebuild: 
        :return: 
        """

        is_exisit=os.path.exists(self.baidubaikeDic_dir)

        if is_exisit and rebuild==False : # 若字典存在 且不重构词典 继续往里追加 新的 词条

            f_baidubaikeDic = open(self.baidubaikeDic_dir, "rb+")

        elif  is_exisit and rebuild==True : #若字典存在 且 要重构词典，则直接返回

            print('the baidubaikeDic_dir already exisit!')

            return

        else: # 如果 不存在，则创建一个
            f_baidubaikeDic = open(self.baidubaikeDic_dir, "wb+")

        doc_num=len(self.doc_row_file) #网页的总个数

        f_baidubaikeDic.seek(0, 2)  # 指向文件的末尾

        for i in range(1,doc_num+1): # 遍历 所有的 网页文档

            doc_content=self.doc_row_file.find_doc_byID(i)

            doc_head=doc_content.split('\r\n')[0] # TODO: 不用对 整个文档 做 split

            baidubaikeDic_row=doc_head+' '+'10000'+'\r\n'

            baidubaikeDic_row_bytes=baidubaikeDic_row.encode(self.encoding)

            f_baidubaikeDic.write(baidubaikeDic_row_bytes) #词条页 的词条肯定 不会重复，因此不用去重

        f_baidubaikeDic.close()

        return True


    def __get_file_tail(self):
        """
        指针指向文件末尾，同时取得文件 末尾的地址 
        :return: 
        """
        self.f.seek(0, 2)

        return self.f.tell()

    def __readDoc(self, start_address):
        """
        根据 起始位置 读取  临时索引文件（tmp_index.bin）

        :param start_address: 起始位置
        :return: 词项 ID ,文档 ID , 当前指针地址
        """

        self.f.seek(start_address, 0)  # 文件指针 指向 start_address

        term_id = int.from_bytes(self.f.read(self.term_id_len), byteorder='big')

        doc_id = int.from_bytes(self.f.read(self.doc_id_len), byteorder='big')  # 读取 doc_length 域的值

        return term_id,doc_id, self.f.tell()

    def __writeDoc(self, start_address, term_doc):
        """
        在起始位置 写入 (term_id,doc_id)

        :param start_address: 起始位置
        :param term_doc: 待写入的 term_doc: (term_id,doc_id)
        :return: 当前指针地址
        """
        self.f.seek(start_address, 0)  # 文件指针 指向 start_address

        term_id= term_doc[0]
        doc_id = term_doc[1]

        term_id_bytes = int(term_id).to_bytes(length=self.term_id_len, byteorder='big')
        self.f.write(term_id_bytes)

        doc_id_bytes = int(doc_id).to_bytes(length=self.doc_id_len, byteorder='big')
        self.f.write(doc_id_bytes)

        self.f.flush()  # 将文件缓存中的内容 刷入磁盘

        return self.f.tell()

    def __append(self, term_doc):
        """
        在文件 的尾部 追加 定长字节流：(term_id,doc_id)
        
        :param term_doc: 

        :return: 追加是否成功 ：bool
        """

        tail = self.__get_file_tail()  # 文件的 尾部指针

        if tail >= self.max_bytes:  # 文档的大小
            print('the file:', self.tmp_index_file_dir, ' size exceed ', self.max_bytes, 'B')
            return False

        tail = self.__writeDoc(tail, term_doc)


        return True

    def __append_row(self, term_doc):
        """
        在文件 的尾部 追加 一行文本 ： (term_id,doc_id)
        
        （1）换行符为 
        （2）term_id 和 doc_id 用 分隔符 '\t' 进行区分；
        
        :param term_doc: 
        :return: 
        """
        tail = self.__get_file_tail()  # 文件的 尾部指针

        if tail >= self.max_bytes:  # 文档的大小
            print('the file:', self.tmp_index_file_dir, ' size exceed ', self.max_bytes, 'B')
            return False

        row_text=str(term_doc[0])+'\t'+str(term_doc[1])+'\r\n'

        row_text_bytes=row_text.encode(self.encoding)

        self.f.write(row_text_bytes)

        self.f.flush()

        return True


    def parse_doc(self,doc_str,doc_id,use_spark=False):
        """
        解析 每一个 词条页的文档
        :param doc_str: 
        :param doc_id: 词条页的编号
        :param use_spark: 是否 将解析后的 内容输出为 spark 易读的格式
        :return: 
        """

        # 1. 对 网页文档 分词
        doc_segs=jieba.lcut(doc_str, cut_all=False) #精确模式

        # 2. 构建 hash 词典
        for term in doc_segs:

            if term not in self.hash_term_id and self.reg_punctuation_marks.match(term)==None : # term 不在 hash 表中 ; 并且不是 标点符号

                term_id=len(self.hash_term_id)+1 # 单词的编号，从1开始

                self.hash_term_id[term]=term_id

                self.hash_id_term[term_id]=term

        # 3. tmp_index.bin 文件尾部 追加 单词 id + 文档 id

        # term_id=0 的代表的 是标点符号，统一标记为 'unk'
        self.hash_term_id['unk']=0
        self.hash_id_term[0]='unk'


        for term in doc_segs:

            term_id=self.hash_term_id[term] # self.hash_term_id 中 不存在的term 的 term_id 都是 0

            if use_spark==False:
                self.__append((term_id,doc_id))
            else:
                self.__append_row((term_id,doc_id))


    def parse_all_docs(self,use_spark=False):

        """
        读取 原始网页存储文件 doc_raw.bin 中的 每一个 词条页的文档，生成 临时索引文件 
        
        :param use_spark: 是否 将 生成的 临时索引文件 输出为 spark 易读的格式：利用分隔符和换行符 来 标识字段和行
          
        :return: 
        """

        #初始化 结巴分词 , 载入 自定义的分词词典
        jieba.load_userdict(self.baidubaikeDic_dir)

        doc_num=len(self.doc_row_file) # 词条页的总个数

        for doc_id in range(1,doc_num+1): # 遍历 所有的 词条页文档，词条页的ID 为：[1,doc_num]

            doc_content=self.doc_row_file.find_doc_byID(doc_id)

            print('parsing doc_id=',doc_id)
            print(doc_content[0:50])

            self.parse_doc(doc_content,doc_id,use_spark)

        return True


    def get_hash_table(self):
        """
        返回 两个 hash 表：term -> term_id 和 term_id -> term 
        :return: 
        """

        return self.hash_term_id,self.hash_id_term

    def close_file(self):
        """
        关闭文件流
        最后一定要加上（重要！！！）
        :return: 
        """
        self.doc_row_file.close_file()

        self.f.close()

        # hash_term_id 持久化 到磁盘
        f_dict = open(self.term_id_file_dir, 'wb')
        pickle._dump(self.hash_term_id, f_dict)
        f_dict.close()

        # hash_id_term 持久化 到磁盘
        f_dict = open(self.inver_term_id_file_dir, 'wb')
        pickle._dump(self.hash_id_term, f_dict)
        f_dict.close()



if __name__ == '__main__':


    # 爬虫爬取 原始数据 路径
    data_spider_dir = '../../Web_Crawler/data/Aerospace/'

    doc_raw_file_dir = os.path.join(data_spider_dir,  'doc_raw.bin')
    doc_raw_offset_file_dir = os.path.join(data_spider_dir, 'doc_raw_offset.bin')

    # 与倒排索引 相关的数据 路径
    data_dir='../data/Aerospace/'


    tmp_index_file_dir =os.path.join(data_dir, 'tmp_index.bin')
    # tmp_index_file_dir = os.path.join(data_dir,'tmp_index_spark.bin') # for spark

    term_id_file_dir = os.path.join(data_dir, 'term_id.bin')
    inver_term_id_file_dir =os.path.join(data_dir, 'inver_term_id.bin')
    baidubaikeDic_dir=os.path.join(data_dir,'baidubaikeDic.txt')

    tmp_index_file = Tmp_Index_File(doc_raw_file_dir, doc_raw_offset_file_dir,
                                  tmp_index_file_dir,term_id_file_dir,inver_term_id_file_dir,baidubaikeDic_dir)

    start = timeit.default_timer()

    tmp_index_file.parse_all_docs(use_spark=False)

    # tmp_index_file.parse_all_docs(use_spark=True)

    # hash_term_id, hash_id_term=tmp_index_file.get_hash_table()

    end = timeit.default_timer()
    print('cost time: ', end - start, 's')


    tmp_index_file.close_file()
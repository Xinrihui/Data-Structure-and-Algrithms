#!/usr/bin/python
# -*- coding: UTF-8 -*-

import os

import pickle

from collections import defaultdict

class noDataError(ValueError):
    pass


class Doc_Raw_File:
    """
    by XRH 
    date: 2020-05-01 
    
    存储 爬虫 爬取的 原始网页文档的 文件（doc_raw.bin）
    
    设计详见：《数据结构与算法之美》-> 算法实战（二）：剖析搜索引擎背后的经典数据结构和算法
    
    功能：
    1.文档 ID 的计数器 , 读和写 此计数器
    
    2.文件尾部 追加 新的网页文档
    
    3.内存中 维护一个 Hash table ，里面记录 每一个文档的 ID 和它在 doc_raw.bin 中的开始地址（偏移量）
      可以 根据 文档ID 查找得到 文档内容
      
    4. 内存中 维护一个 Hash table, 记录 每一个 文档ID 和它对应的 url ，根据 文档ID 查找得到 文档的url
    
    上面 3 和 4 的 Hash table 合并到一个 hash_id_offset中，它的结构为 { doc_id : (offset,url) }
    
    """

    def __init__(self, doc_raw_file_dir='doc_raw.bin',doc_raw_offset_file_dir='doc_raw_offset.bin',doc_id_num_len=4,doc_id_len=4,doc_length_len=8,
                 encoding='utf-8',max_bytes=1e10):
        """
        
        :param doc_raw_file_dir: 
        :param doc_id_num_len: 文档ID 计数器域的 字节长度
        :param doc_id_len:  文档ID 域的字节长度
        :param doc_length_len: 文档长度 域的 字节长度
        :param encoding: 
        :param max_bytes: 文件 最大的长度（B） 1e10B=10GB
        """

        flag_exists=True
        # 如果存在，则直接读取
        self.doc_raw_file_dir=doc_raw_file_dir
        self.doc_raw_offset_file_dir=doc_raw_offset_file_dir


        if os.path.exists(doc_raw_file_dir):
            self.f= open(doc_raw_file_dir, "rb+") #  + 表示 打开一个文件进行更新(可读可写)

        else: # 如果 不存在，则创建一个
            flag_exists = False
            self.f = open(doc_raw_file_dir, "wb+")


        self.f.seek(0, 0)  # 文件指针指向 首位

        self.encoding=encoding

        self.doc_id_num_len=doc_id_num_len # doc_id_num 域所占 字节的长度
        self.doc_id_len=doc_id_len # doc_id 域 所占 字节的长度
        self.doc_length_len=doc_length_len   # doc_length 域 所占 字节的长度

        self.doc_id_num_start=0 # doc_id_num 域 开始地址
        self.doc_list_start=self.doc_id_num_start+self.doc_id_num_len # doc_list 域开始地址


        self.max_bytes=int(max_bytes)

        if flag_exists==False: # doc_raw_file 文件是新建的

            self.__write_doc_id_num(0) # 初始 doc_id_num 域的值 为0

        #初始化 hash_table
        self.hash_id_offset=None

        # 如果存在，则直接读取
        if os.path.exists(self.doc_raw_offset_file_dir):
            f_dict= open(self.doc_raw_offset_file_dir, "rb")
            self.hash_id_offset = pickle.load(f_dict)
            f_dict.close()

        else: # 如果 不存在，则创建一个
            self.hash_id_offset=defaultdict(str)


    def __read_doc_id_num(self):
        """
        读取 文档ID 的计数器 的值
        :return: 
        """
        self.f.seek(self.doc_id_num_start, 0)  # 文件指针指向  doc_id_num 域

        doc_id_num=int.from_bytes(self.f.read(self.doc_id_num_len),byteorder='big') # bytes -> int

        return doc_id_num

    def __len__(self):
        return self.__read_doc_id_num()

    def __write_doc_id_num(self,doc_id_num):
        """
        写入 文档 ID 的计数器 
        :param head: 
        :return: 
        """

        self.f.seek(self.doc_id_num_start, 0)  # 文件指针指向 head 域

        doc_id_num_bytes=int(doc_id_num).to_bytes(length=self.doc_id_num_len, byteorder='big')

        self.f.write(doc_id_num_bytes) # write 只接收 bytes 作为参数
        self.f.flush() # 将文件缓存中的内容 刷入磁盘


    def __get_file_tail(self):
        """
        指针指向文件末尾，同时取得文件 末尾的地址 
        :return: 
        """
        self.f.seek(0, 2)

        return  self.f.tell()


    def __readDoc(self,start_address):
        """
        根据 起始位置 读取 doc 并得到其中的 doc_content
        
        :param start_address: 起始位置
        :return: 文档ID ,文档内容,当前指针地址
        """

        self.f.seek(start_address, 0) # 文件指针 指向 start_address

        doc_id=int.from_bytes(self.f.read(self.doc_id_len),byteorder='big')

        doc_length=int.from_bytes(self.f.read(self.doc_length_len),byteorder='big') # 读取 doc_length 域的值

        doc_bytes=self.f.read(doc_length)

        doc_content=doc_bytes.decode(self.encoding,errors='ignore') # 读取 doc_content 域的值 并解码为 字符串

        return doc_id,doc_content,self.f.tell()

    def __writeDoc(self,start_address,doc):
        """
        在起始位置 写入 Content
        （1）强烈建议 起始位置为 文件的末尾 ！
        （2）若起始位置 不为文件末尾，
            注意 新的 doc 的长度 是否和 原有的长度相同，若不相同，会无法正确覆盖 
        
        :param start_address: 起始位置
        :param doc: 待写入的 doc: (doc_id,doc_content)
        :return: 当前指针地址
        """
        self.f.seek(start_address, 0)  # 文件指针 指向 start_address

        doc_id=doc[0]
        doc_content = doc[1]

        doc_id_bytes=int(doc_id).to_bytes(length=self.doc_id_len, byteorder='big')
        self.f.write(doc_id_bytes)

        #self.f.flush()  # 将文件缓存中的内容 刷入磁盘

        doc_content_bytes=doc_content.encode(self.encoding) # 字符串 编码成bytes

        doc_length=len(doc_content_bytes) # 应该为 字符串转换为 字节流后的长度，因为有中文字符

        doc_length_bytes=int(doc_length).to_bytes(length=self.doc_length_len, byteorder='big')

        self.f.write(doc_length_bytes) # 写入 length 域

        #self.f.flush()  # 将文件缓存中的内容 刷入磁盘

        self.f.write( doc_content_bytes ) #  bytes 并写入 URL 域

        self.f.flush()  # 将文件缓存中的内容 刷入磁盘

        return self.f.tell()


    def append(self,doc_content,url):
        """
        在文件 的尾部 追加 新的文档
        同时 记录 新文档 对应的 URL 
        :param doc_content: 
        :param url: 
        :return: 插入是否成功 ：bool
        """

        doc_id_num = self.__read_doc_id_num()
        doc_id_num+=1 # 文档计数器 +1
        self.__write_doc_id_num(doc_id_num)

        # print('wirte doc_id_num success!')

        tail = self.__get_file_tail()  # 文件的 尾部指针

        if tail >= self.max_bytes: #文档的大小
            print('the file:',self.doc_raw_file_dir, ' size exceed ',self.max_bytes,'B')
            return False

        self.hash_id_offset[doc_id_num] = (tail,url)

        tail=self.__writeDoc(tail,(doc_id_num,doc_content)) # 写入 文档ID 和 文档的内容

        # print('wirte doc_content success!')

        return True

    def find_doc_byID(self,doc_id):
        """
        根据文档 ID 获得 文档的内容
        :param doc_id: 
        :return: doc_content
        """
        doc_content=None

        if doc_id in self.hash_id_offset:

            start_address=self.hash_id_offset[doc_id][0]

            _,doc_content,_=self.__readDoc(start_address)

        return doc_content

    def find_url_byID(self,doc_id):
        """
        根据文档 ID 获得 文档的 URL
        :param doc_id: 
        :return: 文档的URL
        """
        url=None

        if doc_id in self.hash_id_offset:

            url=self.hash_id_offset[doc_id][1]

        return url

    def close_file(self):
        """
        关闭文件流
        最后一定要加上（重要！！！）
        :return: 
        """
        self.f.close()

        # hash_id_offset 持久化 到磁盘
        f_dict = open(self.doc_raw_offset_file_dir, 'wb')
        pickle._dump(self.hash_id_offset, f_dict)
        f_dict.close()


if __name__ == '__main__':

    doc_row=Doc_Raw_File()

    doc_row.append('abcd','www.a.com')
    doc_row.append('12345','www.b.com')

    print('the num of doc:', len(doc_row))
    print(doc_row.find_doc_byID(1))
    print(doc_row.find_doc_byID(2))

    print(doc_row.find_url_byID(1))
    print(doc_row.find_url_byID(2))


    doc_row.close_file()












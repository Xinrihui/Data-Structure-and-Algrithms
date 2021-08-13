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


class fileNotExisitError(ValueError):
    pass

class fileIndexIllegal (ValueError):
    pass

class Merge_Sort_onDisk:

    """
    by XRH 
    date: 2020-05-23 
    
    排序的临时索引文件（sorted_tmp_index.bin）
    
    对 磁盘上的 临时索引文件 进行归并排序
    
    步骤：
    1.按照内存的 阈值 对 大文件进行切分 ，得到 临时索引文件 的切片；
    2. 对每一个 切片 放入内存中进行排序，排序后刷写入磁盘中；
    3. 将 这几个 内部已经排好序的小文件  进行一次 基于磁盘的 多路归并 排序 
    
    """

    def __init__(self, tmp_index_file_dir='tmp_index.bin',
                 term_id_len=4, doc_id_len=4,
                 encoding='utf-8',slice_max_bytes=10485760):

        """
        :param tmp_index_file_dir: 
        :param term_id_len: 
        :param doc_id_len: 
        :param encoding: 
        :param slice_max_bytes: 内存的阈值，达到这个阈值 即刷写入磁盘，形成切片文件；
                                文件切片的大小为： 10485760 B = 10MB
        """

        self.tmp_index_file_dir = tmp_index_file_dir

        self.slice_max_bytes = int(slice_max_bytes)

        self.encoding=encoding

        self.term_id_len=term_id_len
        self.doc_id_len=doc_id_len

        # 1.打开 tmp_index.bin 文件

        # 如果存在，则直接读取
        if os.path.exists(self.tmp_index_file_dir):
            self.f= open(self.tmp_index_file_dir, "rb+") #  + 表示 打开一个文件进行更新(可读可写)

        else:
            raise fileNotExisitError('the file ', self.tmp_index_file_dir,'  not exisits')


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

    def __readDoc(self,start_address,f=None):
        """
        根据 起始位置 读取  临时索引文件（tmp_index.bin）

        :param start_address: 起始位置
        :return: 词项 ID ,文档 ID , 当前指针地址
        """
        if f==None:
            f=self.f

        f.seek(start_address, 0)  # 文件指针 指向 start_address

        term_id = int.from_bytes(f.read(self.term_id_len), byteorder='big')

        doc_id = int.from_bytes(f.read(self.doc_id_len), byteorder='big')  # 读取 doc_length 域的值

        return term_id, doc_id, f.tell()

    def __writeDoc(self, start_address, term_doc,f=None):
        """
        在起始位置 写入 (term_id,doc_id)

        :param start_address: 起始位置
        :param term_doc: 待写入的 term_doc: (term_id,doc_id)
        :return: 当前指针地址
        """
        if f==None:
            f=self.f

        f.seek(start_address, 0)  # 文件指针 指向 start_address

        term_id = term_doc[0]
        doc_id = term_doc[1]

        term_id_bytes = int(term_id).to_bytes(length=self.term_id_len, byteorder='big')
        f.write(term_id_bytes)

        doc_id_bytes = int(doc_id).to_bytes(length=self.doc_id_len, byteorder='big')
        f.write(doc_id_bytes)

        f.flush()  # 将文件缓存中的内容 刷入磁盘

        return f.tell()


    def cut_file(self,slice_bytes=None):
        """
        切分文件
        
        1.按照内存的 阈值（10MB） 对 大文件进行切分 ，得到 临时索引文件 的切片
        2. 把每一个 切片 放入内存中进行排序 ，并 输出到 切片文件中 
        每个切片大小估计为 10MB，由于去除了 标点符号，实际每个 切片文件 <10MB
        
        :param slice_bytes: 
        :return: 
        """

        if slice_bytes==None:

            slice_bytes=self.slice_max_bytes

        file_index=self.f.seek(0, 0)  # 文件指针指向 首位

        slice_id=0 # 切片的编号

        while file_index < self.__get_file_tail():

            slice_cache=[]

            while file_index < (slice_id+1)*slice_bytes: # 把 (term_id,doc_id) 放入 切片缓存

                term_id, doc_id,file_index=self.__readDoc(file_index)

                if term_id!=0: # term_id==0 代表 其为 标点符号
                    slice_cache.append((term_id,doc_id))

                if file_index >= self.__get_file_tail():# 最后一个 slice 要退出 循环
                    break

            slice_cache=sorted(slice_cache,key=lambda x:x[0]) # 对切片 排序

            print('slice_id:', slice_id)
            print('slice_cache:',slice_cache[0:10])

            with open(self.tmp_index_file_dir+str(slice_id), "wb+") as f_slice:

                slice_index = f_slice.seek(0, 0) # 指向 切片文件的 首位

                for term_doc in slice_cache: # 排序好的 切片 持久化

                    slice_index=self.__writeDoc(slice_index,term_doc,f_slice)

            slice_id+=1

        return True

    def merge_sort(self):
        """
        对切片文件 进行 N 路归并排序（N= 切片的数量）
        
        :return: 
        """

        #1. 看看有几个切片文件

        folder_dir,file=os.path.split(self.tmp_index_file_dir) #分离 路径 和 文件名

        file_list=os.listdir(folder_dir) # 列出 路径下的 所有文件

        self.reg_file_slice=re.compile('^'+file+'\d{1,}$') #  正则匹配 文件切片的文件名： 原文件名 + 至少1位的数字

        sliceName_list=[] # 切片 的完整文件路径

        for file_name in file_list:

            if self.reg_file_slice.match(file_name)!=None:
                sliceName_list.append(folder_dir+'/'+file_name)


        print('sliceName: ',sliceName_list)

        # 2. 打开所有的 切片文件

        f_slice_list={ os.path.split(sliceName)[1]:open(sliceName, "rb+") for sliceName in sliceName_list}

        sliceIndex_list={ fileName:f_slice.seek(0, 0) for fileName,f_slice in f_slice_list.items() } # 每一个 指针指向 切片文件的 首位

        # 3. 对切片 进行多路归并 排序

        f_sorted_file= open(folder_dir+'/''sorted_'+file, "wb+") # 排序好的结果文件(最终排序文件)
        sorted_file_index=f_sorted_file.seek(0, 0)

        sliceIndex_tail_list={fileName:self.__get_file_tail(f_slice) for fileName,f_slice in f_slice_list.items() } # 记录 每一个 切片文件末尾的指针

        current_sliceIndex_list={ fileName:slice_index for fileName,slice_index in sliceIndex_list.items() } # 当前 切片指针 的集合

        current_f_slice_list={fileName:f_slice for fileName, f_slice in f_slice_list.items()} # 当前 切片文件 句柄的 集合


        while len(current_sliceIndex_list)>1:

            for fileName in list(current_sliceIndex_list.keys()):

                slice_index=current_sliceIndex_list[fileName]

                if slice_index >= sliceIndex_tail_list[fileName]: # 切片 指针到达 切片文件的末尾， 说明此切片所有数据 已经导入 排序好的结果文件

                    current_sliceIndex_list.pop(fileName)
                    current_f_slice_list.pop(fileName)
                    print(fileName,' has loaded into the sorted file')

            tuples=[ (fileName,self.__readDoc(current_sliceIndex_list[fileName],f_slice)) for fileName, f_slice in current_f_slice_list.items()]

            tuples=[(ele[0],ele[1][0],ele[1][1]) for ele in tuples] # (fileName,termId,docId) list

            min_tuple = min(tuples,key=lambda x:x[1]) # (fileName,termId,docId) 排序的键为 termId

            print(min_tuple)

            min_tuple_target=(min_tuple[1],min_tuple[2]) # (termId,docId)

            sorted_file_index=self.__writeDoc(start_address=sorted_file_index,term_doc=min_tuple_target,f=f_sorted_file) # min_tuple_target 写入 最终排序文件中

            current_sliceIndex_list[min_tuple[0]]+=(self.term_id_len+self.doc_id_len) # 更新 最小 tuple 所在切片文件的 指针


        # 剩下最后一个切片文件 last_slice ，全部写入 最终排序文件中

        fileName, last_slice_index=[(fileName,slice_index) for fileName, slice_index in current_sliceIndex_list.items()][0]
        # current_sliceIndex_list 中 只有一个 切片文件 和它的 文件指针
        print('last slice file:',fileName)

        last_slice_index_tail=sliceIndex_tail_list[fileName] # 该切片 的文件末尾位置

        f_last_slice= current_f_slice_list[fileName] # 该切片 的文件 句柄


        f_last_slice.seek(last_slice_index, 0)  # 切片文件 指针 指向 slice_index

        last_slice_bytes=f_last_slice.read(last_slice_index_tail-last_slice_index) # 读出 last_slice 文件 中的剩下的所有字节

        f_sorted_file.seek(sorted_file_index, 0) # 最终排序文件的 文件指针 指向 sorted_file_index

        f_sorted_file.write(last_slice_bytes) # 剩下的所有字节全部写入
        f_sorted_file.flush()

        # 4. 关闭 所有的 切片文件
        for fileName,f_slice in f_slice_list.items():
            f_slice.close()

        # 5. 关闭 最终排序文件
        f_sorted_file.close()

        return True

    def read_sorted_file(self,row_offset):
        """
        根据 行偏移量 读取 最终 排序文件中的 元组，
        元组即一行数据：(term_id , doc_id)
        
        1.若 行偏移量为 正值，则从 文件头部开始读，一共读取 row_offset 行
        2. 若 行偏移量为 负值，则从 （文件尾部- row_offset*元组字节长度） 开始读，一共读取 |row_offset|行
        
        :param row_offset: 
        :return: 
        """

        folder_dir, file = os.path.split(self.tmp_index_file_dir)  # 分离 路径 和 文件名
        f_sorted_file= open(folder_dir+'/''sorted_'+file, "rb") # 排序好的结果文件(最终排序文件)

        f_sorted_file.seek(0, 0)

        if row_offset>=0:
            start_address=0
            end_address=row_offset*(self.term_id_len + self.doc_id_len)

        else:
            row_offset=abs(row_offset)

            end_address=self.__get_file_tail(f_sorted_file)

            start_address=end_address - row_offset*(self.term_id_len + self.doc_id_len)


        print('start_address:',start_address,'end_address:', end_address)

        term_doc_list=[]

        for i in range(row_offset):

            term_id,doc_id,start_address=self.__readDoc(start_address,f_sorted_file)
            term_doc_list.append((term_id,doc_id))

        return term_doc_list


    def close_file(self):
        """
        关闭文件流
        最后一定要加上（重要！！！）
        :return: 
        """
        self.f.close()



if __name__ == '__main__':


    data_dir='../data/Aerospace/'

    tmp_index_file_dir =os.path.join(data_dir,'tmp_index.bin')

    sol=Merge_Sort_onDisk(tmp_index_file_dir)

    start = timeit.default_timer()

    sol.cut_file(10485760*5) # 10485760B =10MB , 切片大小 设置为 50MB

    sol.merge_sort()

    # print(sol.read_sorted_file(100))

    # print(sol.read_sorted_file(-100)) # 输出文件尾部 的100行数据
    # start_address: 45800536 end_address: 45801336 # 从操作系统中可见 sorted_tmp_index.bin 的文件大小 为 45801336 B


    end = timeit.default_timer()
    print('cost time: ', end - start, 's')

    sol.close_file()













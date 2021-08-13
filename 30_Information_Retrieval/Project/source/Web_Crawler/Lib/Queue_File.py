#!/usr/bin/python
# -*- coding: UTF-8 -*-

class File_IO:


    def test1(self):
        """
        测试 seek tell 函数的使用
        :return: 
        """
        f = open("test_data/a.txt", 'rb') # 以普通模式打开，每个数据就是一个字符; 以 b 模式打开，每个数据就是一个字节

        print(f.tell()) # 0
        print(f.read(4)) # 从文件读取指定的字节数; b'http'
        print(f.tell()) # 4

        f.seek(4) # 从文件头 开始 指针向后 移动4个位置
        print(f.tell()) # 4
        print(f.read(3)) # b'://'
        print(f.tell()) # 7

        f.seek(2,1) #从 所在的当前位置开始，向后移动2个位置
        print(f.tell()) # 9
        print(f.read(9)) # b'biancheng'
        print(f.tell()) # 18

        f.seek(-5,2)# 从文件末尾，向前 移动5个位置
        print(f.tell()) # 24-5=19
        print(f.read(10)) # b'net\r\n'
        print(f.tell()) # 24

        f.close()

    def test2(self):

        # 打开文件
        f = open("test_data/file_queue", "rb")
        # print("文件名为: ", f.name)

        print(f.tell())
        line = f.readline() #
        print("读取的数据为:" ,line)
        print(f.tell())

        # 重新设置文件读取指针到开头
        # f.seek(0, 0)
        # print( f.tell())
        # line = f.readline() # 文件指针已经指向下一行
        # print("读取的数据为: " ,line)
        # print(f.tell())

        line = f.readline() #
        print("读取的数据为:" ,line)
        print(f.tell())

        line = f.readline() #
        print("读取的数据为:" ,line)
        print(f.tell())

        # 关闭文件
        f.close()

    def test3(self):

        f = open("test_data/file_queue", "rb+") # + 表示 打开一个文件进行更新(可读可写)

        line = f.readline() #
        print("读取的数据为:" ,line)
        print(f.tell()) # 文件指针已经指向 下一行 为3

        f.seek(0, 2) #指针 指向文件末尾
        print(f.tell())

        a='add\r\n'
        f.write(a.encode('utf-8')) # 在文件末尾 添加
        print(f.tell()) # 随后 指针指向了 文件末尾

        f.close() # 文件关闭之后，才会将 刚刚写入的内容 从文件内存缓存中 刷入磁盘

    def  test4(self):

        f = open("test_data/test4.txt", "wb+")

        f.seek(0, 0)  # 文件指针指向 首位

        a = 97
        b1=a.to_bytes(length=8, byteorder='big')
        print(b1)

        f.write(b1) # write 只接收 bytes 作为参数
        f.flush() # 将文件缓存中的内容 手动刷入磁盘

        f.seek(0,0) # 文件指针指向 首位
        b2=f.read(8) # 读取8个字节
        b2=int.from_bytes(b2,byteorder='big')

        print(b2)

        f.close()


import os

class noDataError(ValueError):
    pass

class Queue_File:
    """
    by XRH 
    date: 2020-05-01 
    
    利用外存（文件）实现一个 队列
    
    设计详见：《数据结构与算法之美》-> 算法实战（二）：剖析搜索引擎背后的经典数据结构和算法
    
    功能：
    1.出队： 获得 head 指针，读取 head 指向的元素（ content），并更新 head指针 
    2.入队：新的元素 追加到文件的 末尾 ，并更新 tail 指针
    
    """

    def __init__(self, queue_file_dir,head_len=8,tail_len=8,num_len=4,length_len=4,
                 encoding='utf-8',max_bytes=1e10):
        """
        
        :param queue_file_dir: 
        :param head_len: 
        :param tail_len: 
        :param num_len:
        :param length_len: 
        :param encoding: 
        :param max_bytes: 文件 最大的长度（B） 1e10B=10GB
        """

        flag_exists=True
        # 如果存在，则直接读取
        if os.path.exists(queue_file_dir):
            self.f= open(queue_file_dir, "rb+") #  + 表示 打开一个文件进行更新(可读可写)

        else: # 如果 不存在，则创建一个
            flag_exists = False
            self.f = open(queue_file_dir, "wb+") #


        self.f.seek(0, 0)  # 文件指针指向 首位

        self.encoding=encoding

        self.head_len=head_len # head 所占 字节的长度
        self.tail_len=tail_len # tail 所占 字节的长度
        self.num_len=num_len   # num 所占 字节的长度
        self.length_len =  length_len # length 所占 字节的长度

        self.head_start=0 # head 域 开始地址
        self.tail_start=self.head_start+self.head_len # tail 域 开始地址
        self.num_start=self.tail_start+self.tail_len  # num 域 开始地址
        self.content_start=self.num_start+self.num_len # content 域 开始地址

        self.max_bytes=int(max_bytes)

        if flag_exists==False: # queue_file 文件是新建的

            self.__writeHead(self.content_start) # 初始 head 域的值
            self.__writeTail(self.content_start) # 初始 tail 域的值 ，初始队列为空

            self.__writeNum(0) # 初始队列为空，元素个数为0


    def __readHead(self):
        """
        读取 head 
        :return: head 域的值
        """
        self.f.seek(self.head_start, 0)  # 文件指针指向 head 域

        head=int.from_bytes(self.f.read(self.head_len),byteorder='big') # bytes -> int

        return head

    def __writeHead(self,head):
        """
        写入 head
        :param head: 
        :return: 
        """

        self.f.seek(self.head_start, 0)  # 文件指针指向 head 域

        head_bytes=int(head).to_bytes(length=self.head_len, byteorder='big')

        self.f.write(head_bytes) # write 只接收 bytes 作为参数
        self.f.flush() # 将文件缓存中的内容 刷入磁盘

    def __readTail(self):
        """
        读取 Tail 
        :return: tail 域的值 
        """
        self.f.seek(self.tail_start, 0)  # 文件指针指向 tail 域

        tail=int.from_bytes(self.f.read(self.tail_len),byteorder='big')

        return tail

    def __writeTail(self,tail):
        """
        写入 tail
        :param tail: 
        :return: 
        """

        self.f.seek(self.tail_start, 0)  # 文件指针指向 tail 域

        tail_bytes=int(tail).to_bytes(length=self.tail_len, byteorder='big')

        self.f.write(tail_bytes) # write 只接收 bytes 作为参数
        self.f.flush() # 将文件缓存中的内容 刷入磁盘

    def __len__(self):
        return self.__readNum()

    def __readNum(self):
        """
        读取 计数器 num  
        :return: tail 域的值 
        """
        self.f.seek(self.num_start, 0)  # 文件指针指向 num 域

        num=int.from_bytes(self.f.read(self.num_len),byteorder='big')

        return num

    def __writeNum(self,num):
        """
        写入 计数器 num 
        :param num: 
        :return: 
        """

        self.f.seek(self.num_start, 0)  # 文件指针指向 num 域

        num_bytes=int(num).to_bytes(length=self.num_len, byteorder='big')

        self.f.write(num_bytes) # write 只接收 bytes 作为参数
        self.f.flush() # 将文件缓存中的内容 刷入磁盘

    def __get_file_tail(self):
        """
        指针指向文件末尾，同时取得文件 末尾的地址 
        :return: 
        """
        self.f.seek(0, 2)

        return  self.f.tell()


    def __readContent(self,start_address):
        """
        根据 起始位置 读取 Content 并得到其中的 URL
        
        :param start_address: 起始位置
        :return: URL,当前指针地址
        """

        self.f.seek(start_address, 0) # 文件指针 指向 start_address

        length=int.from_bytes(self.f.read(self.length_len),byteorder='big') # 读取 length 域的值

        if length==0: # 说明 length域 的值为 空
            raise noDataError("there is no data at the address:",start_address)

        data_string=self.f.read(length).decode(self.encoding) # 读取 URL 域的值 并解码为 字符串

        return data_string,self.f.tell()

    def __writeContent(self,start_address,data_string):
        """
        在起始位置 写入 Content
        （1）强烈建议 起始位置为 文件的末尾 ！
        （2）若起始位置 不为文件末尾，
            注意 新的 data_string 的长度 是否和 原有的长度相同，若不相同，会无法正确覆盖 
        
        :param start_address: 起始位置
        :param data_string: 待写入的 URL 
        :return: 当前指针地址
        """
        self.f.seek(start_address, 0)  # 文件指针 指向 start_address

        data_string_bytes=data_string.encode(self.encoding)

        length=len(data_string_bytes) # 获得字节流的长度

        length_bytes=int(length).to_bytes(length=self.length_len, byteorder='big')

        self.f.write(length_bytes) # 写入 length 域

        self.f.write(data_string_bytes) # 字符串 编码成bytes 并写入 URL 域
        self.f.flush()  # 将文件缓存中的内容 刷入磁盘

        return self.f.tell()

    def popleft(self):
        """
        队列 弹出元素
        :return: 弹出是否成功：bool, 元素
        """

        head=self.__readHead()
        tail=self.__readTail()

        num=self.__readNum()

        if head==tail: # 队列中元素为空
            print('queue is empty!')
            return False,None

        URL,current=self.__readContent(head)

        self.__writeHead(current) # 更新 head 指针

        num-=1 # 队列中的元素个数减1
        self.__writeNum(num)

        return True,URL


    def append(self,ele):
        """
        在队列文件 的尾部 插入新的元素
        :param ele: 
        :return: 插入是否成功 ：bool
        """

        tail=self.__readTail()
        num = self.__readNum()

        if tail >= self.max_bytes:
            print('queue is full!')
            return False

        tail=self.__writeContent(tail,ele)
        self.__writeTail(tail) #更新 tail

        num+=1 # 队列中的元素个数 加1
        self.__writeNum(num)

        return True


    def close_file(self):
        """
        关闭文件流
        最后一定要加上（重要！！！）
        :return: 
        """
        self.f.close()


if __name__ == '__main__':

    # file_IO=File_IO()
    # file_IO.test4()

    file_queue=Queue_File('queue.bin')

    file_queue.append('www.tp.com')
    file_queue.append('www.cs.com')

    print('the num of URL in queue:', len(file_queue))
    print(file_queue.popleft())
    print(file_queue.popleft())

    print('the num of URL in queue:', len(file_queue))
    print(file_queue.popleft())

    file_queue.close_file()












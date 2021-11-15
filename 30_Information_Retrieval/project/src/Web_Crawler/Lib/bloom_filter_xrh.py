

class Bitmap:
    def __init__(self, num_bits):
        """
        位图 能表达 范围在 [1,num_bits] 中的整数
        :param num_bits: 
        """
        self._num_bits = num_bits # 位图的 大小，即二进制位的个数
        self._bytes = bytearray(num_bits // 8 + 1) # 返回一个长度为 (num_bits // 8 + 1)B 的初始化 字节数组
        print('bytearray memory_size:', sys.getsizeof(self._bytes),'B')

    def setbit(self, k):
        """
        k 放入位图中
        :param k: 
        :return: 
        """
        if k > self._num_bits or k < 1: return False # 超出 位图能表达的范围
        self._bytes[k // 8] |= (1 << k % 8)
        return True

    def getbit(self, k):
        """
        k 是否在位图中存在
        :param k: 
        :return: 
        """
        if k > self._num_bits or k < 1: return
        return self._bytes[k // 8] & (1 << k % 8) != 0

import hashlib
import murmurhash
import fnvhash
import math
import sys

import pickle

class Bloom_Filter:

    def __init__(self, input_range=1e9,num_bits=1e8):
        """
        :param input_range :  输入元素的取值范围  
        :param num_bits:  slots槽位数 或者是 比特位数；默认为 1亿个二进制位，大小为 12MB
        """
        self.input_range=input_range
        self.m=int(num_bits)

        self.bitmap=Bitmap(self.m)

        self.hash_func_nums= max(int(( self.input_range / self.m )*(math.log(2))),1)  # 假设 input_range =10亿  m=1亿 则 hash 函数的个数为 6;
                            # hash 函数个数 最少也要有一个

        self.ele_nums=0 # 记录插入 元素的个数
        # TODO:数据个数与位图大小的比例超过某个阈值的时候，将重新申请一个新的位图。后面来的新数据，会被放置到新的位图中

    def __double_hash(self,i,x):
        """
        double Hashing 
        给定两个彼此独立的哈希函数 hasha 和 hashb，可以通过如下的哈希函数创建一个新的哈希函数： 
        hash_i(x, m) = (hasha(x) + i * hashb(x)) mod m
        依次类推，我们可以生成 第 i 个新的哈希函数
        :param i: 
        :param x: 被 Hash 的值 
        :return: 
        """
        if type(x)!=str:
            x=str(x)

        return (murmurhash.hash(x) + i*fnvhash.fnv0_32( bytes(x, encoding = "utf8")))%self.m

    def add(self,key):
        """
        添加元素
        :param key: 
        :return: 
        """
        self.ele_nums+=1

        flag=True

        for i in range(self.hash_func_nums):

            hash_code=self.__double_hash(i,key)
            flag=(flag and self.bitmap.setbit(hash_code))

        return flag

    def has_Key(self, key):
        """
        判断 Key 是否存在
        :param key: 
        :return: 
        """
        flag = True

        for i in range(self.hash_func_nums):
            hash_code = self.__double_hash(i, key)
            flag = (flag and self.bitmap.getbit(hash_code))

        return flag



if __name__ == "__main__":
    # bitmap = Bitmap(10)
    # bitmap.setbit(1)
    # bitmap.setbit(3)
    # bitmap.setbit(6)
    # bitmap.setbit(7)
    # bitmap.setbit(8)
    # for i in range(1, 11):
    #     print(bitmap.getbit(i))

    bloom_filter=Bloom_Filter(input_range=1e9) # 10亿 1e9
    # bytearray memory_size: 12500058 B =12MB

    # TODO: 测试 Bloom_Filter 在大数据量下的 错误率

    # bloom_filter.add(1)
    # bloom_filter.add(3)
    # bloom_filter.add(5)
    # bloom_filter.add(7)
    # bloom_filter.add(9)
    #
    # for i in range(1, 11):
    #     print(bloom_filter.has_Key(i))

    bloom_filter.add('https://baike.baidu.com/item/歼-20/1555348')
    bloom_filter.add('https://baike.baidu.com/item/歼-20/1555349')

    print(bloom_filter.has_Key('https://baike.baidu.com/item/歼-20/1555348'))
    print(bloom_filter.has_Key('https://baike.baidu.com/item/歼-20/1555349'))
    print(bloom_filter.has_Key('https://baike.baidu.com/item/歼-20/'))

    # bloom_filter.add(5)
    # bloom_filter.add(7)
    # bloom_filter.add(9)

    # # bloom_filter 持久化 到磁盘
    # dir='bloom_filter.bin'
    # f = open(dir, 'wb')
    # pickle._dump(bloom_filter,f)
    # f.close()
    #
    # # 反序列化 到 内存对象
    # f = open(dir, 'rb')
    # bloom_filter2=pickle.load(f)
    # f.close()
    #
    # for i in range(1, 11):
    #     print(bloom_filter2.has_Key(i))
















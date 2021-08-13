#!/usr/bin/python
# -*- coding: UTF-8 -*-


# from collections import deque
class Node:
    def __init__(self, c):
        self.key = c
        self.is_ending_char = False
        # 使用有序数组，降低空间消耗，支持更多字符
        # self.children = deque()
        self.children =[]

    def insert_child(self, c):
        self._insert_child(Node(c))

    def _insert_child(self, node):
        """
        我是某个 node ，我要在 children 中插入一个子节点 
        :param c:
        :return:
        
    test=Node('test')
    test.children=[Node('b') ,Node('d'),Node('e'),Node('f')]
    test.insert_child('a')
    test.insert_child('c')
    test.insert_child('g')
        """
        v=ord(node.key)
        result = self._find_insert_idx(v)
        if result[0] == False: #result=[False, 0]
            idx=result[1]
            l=self.children
            if idx>=len(l)-1:
                l.append(node)
            elif idx<=0:
                l.insert(0,node)
            else:
                l.insert(idx+1,node)

    def get_child(self, c):
        """
        我是某个 node ，在children中 搜索 key==c 的子节点并返回
        :param c:
        :return:
        
        """
        v = ord(c) #返回单个字符 对应的 ASCII 数值，或者 Unicode 数值
        result=self._find_insert_idx(v) #

        if result[0]==False: #result=[False, 0]
            return None
        elif result[0]==True: # result=[True, 1, Node(3)]
            return result[2]


    def has_child(self, c):
        return True if self.get_child(c) is not None else False


    def _find_insert_idx(self, v):
        """
        二分查找，找到有序数组的插入位置
        
        还可以直接使用 bisect 实现：
         http://kuanghy.github.io/2016/06/14/python-bisect
        
        :param v:
        :return:
    
    test=Node('test')
    test.children=[Node('b') ,Node('d'),Node('e'),Node('f')]
    print(test._find_insert_idx( 'a'))  #[False, -1]
    print(test._find_insert_idx('c')) #[False,0]
    print(test._find_insert_idx('g')) #[False, 3]
    print(test._find_insert_idx('d')) #[True, 1, Node('d')]
        """
        l=self.children
        left=0
        right=len(l)-1
        flag=False
        idx=0

        while(left<=right):
            mid=((left+right)//2) #  Python 3以后  " / "就表示 浮点数除法，返回浮点结果;" // "表示整数除法。
            if  ord(l[mid].key)==v:
                flag=True
                idx=mid
                break
            elif v>ord(l[mid].key):
                left=mid+1 # mid+1 为了跳出循环
            elif v< ord(l[mid].key):
                right=mid-1 # mid-1 为了跳出循环
        if left>right:
            idx=right
            return [flag,idx]

        return [flag,idx,l[idx]]

class Trie:
    def __init__(self):
        self.root = Node(None)

    def gen_tree(self, string_list):
        """
        创建 trie树

        1. 遍历每个字符串的字符，从根节点开始，如果没有对应子节点，则创建
        2. 每一个串的末尾节点标注为红色(is_ending_char)
        :param string_list:
        :return:
        """
        for string in string_list:
            node = self.root  #每一个单词 都要从树的根开始找
            for i,c in enumerate(string): # string='hello'

                if  node.get_child(c) is None: # 如果找不到 才加入新的节点
                    node.insert_child(c) # c='h'
                    node=node.get_child(c) # 拿到上一步 新加入的 Node('h')
                else:
                    node = node.get_child(c)

                if i==len(string)-1:
                    node.is_ending_char=True


    def search(self, pattern):
        """
        搜索
        1. 遍历模式串的字符，从根节点开始搜索，如果途中子节点不存在，返回False
        2. 遍历完模式串，则说明模式串存在，再检查树中最后一个节点是否为红色，是
           则返回True，否则False
        :param pattern:
        :return:
        """
        assert type(pattern) is str and len(pattern) > 0

        node = self.root
        Flag=False
        for i, c in enumerate(pattern):
            node = node.get_child(c)
            if node==None:
                break
            if i == len(pattern) - 1:
                if node.is_ending_char == True:
                    Flag=True
        return Flag



if __name__ == '__main__':
    string_list = ['abc', 'abd', 'abcc', 'accd', 'acml', 'P@trick', 'data', 'structure', 'algorithm']
    # string_list = ['hello', 'her','hi','how','see','so']

    print('--- gen trie ---')
    print(string_list)
    trie = Trie()
    trie.gen_tree(string_list)
    # trie.draw_img()

    print('\n')
    print('--- search result ---')
    search_string = ['a', 'ab', 'abc', 'abcc', 'abe', 'P@trick', 'P@tric', 'Patrick']
    # search_string = ['hello', 'hell', 'hellow','ho','hi','see']
    for ss in search_string:
        print('[pattern]: {}'.format(ss), '[result]: {}'.format(trie.search(ss)))

    # import time  # 引入time模块
    #
    # ticks_start = time.time()
    #
    #
    # ticks_end = time.time()
    # print(ticks_end-ticks_start)



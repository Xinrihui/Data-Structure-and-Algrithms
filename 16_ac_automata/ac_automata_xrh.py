#!/usr/bin/python
# -*- coding: UTF-8 -*-

from collections import deque

class AcNode:
    def __init__(self, c):
        self.key = c
        self.is_ending_char = False #结尾字符为true

        self.length=-1 # 当isEndingChar=true时，记录模式串长度
        self.fail=None # 失败指针
        self.father=None  #  父亲节点

        # 使用有序数组，降低空间消耗，支持更多字符
        self.children = []

    def insert_child(self, c):
        self._insert_child(AcNode(c))

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
        v = ord(node.key)
        result = self._find_insert_idx(v)
        if result[0] == False:  # result=[False, 0]
            idx = result[1]
            l = self.children
            if idx >= len(l) - 1:
                l.append(node)
            elif idx <= 0:
                l.insert(0, node)
            else:
                l.insert(idx + 1, node)

    def get_child(self, c):
        """
        我是某个 node ，在children中 搜索 key==c 的子节点并返回
        :param c:
        :return:

        """
        v = ord(c)  # 返回单个字符 对应的 ASCII 数值，或者 Unicode 数值
        result = self._find_insert_idx(v)  #

        if result[0] == False:  # result=[False, 0]
            return None
        elif result[0] == True:  # result=[True, 1, Node(3)]
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
        l = self.children
        left = 0
        right = len(l) - 1
        flag = False
        idx = 0

        while (left <= right):
            mid = ((left + right) // 2)  # Python 3以后  " / "就表示 浮点数除法，返回浮点结果;" // "表示整数除法。
            if ord(l[mid].key) == v:
                flag = True
                idx = mid
                break
            elif v > ord(l[mid].key):
                left = mid + 1  # mid+1 为了跳出循环
            elif v < ord(l[mid].key):
                right = mid - 1  # mid-1 为了跳出循环
        if left > right:
            idx = right
            return [flag, idx]

        return [flag, idx, l[idx]]


class AcTrie:
    def __init__(self,pattern_list):
        self.root = AcNode(None)

        self.__gen_tree(pattern_list)
        self._build_fail_pointer()


    def _build_fail_pointer(self):
        """
        层次遍历 树（ 广度优先搜索 BFS ）
        对每一个节点 建立 fail指针 ：
        
        i -> j :  word[i] 的 最长后缀 是 word[j]
        
        ref: https://www.bilibili.com/video/BV1uJ411Y7Eg?p=4
        :return: 
        """
        self.root.fail=self.root # 根节点的 fail指针 指向自己

        queue = deque()
        queue.append(self.root)

        level=0
        while len(queue)!=0:

            N=len(queue)

            # print('level:',level,' nums:',N)

            for i in range(N):
                current=queue.popleft()
                # print(current.key)

                if level!=0: #  root 节点 代表 空字符 ，无需建立 fail指针

                    if level==1: # 第一层 节点 只有一个字符，不可能有后缀，所以 fail指针 指向 root 节点
                        current.fail=self.root

                    elif level>1:
                        father_fail= current.father.fail # 父节点 的 最长后缀 的尾节点

                        target=father_fail.get_child(current.key)

                        while target ==None : #我们假设节点 p 的失败指针指向节点 q，我们看节点 p 的子节点 pc 对应的字符，是否也可以在节点 q 的子节点中找到。
                                            # 如果找到了节点 q 的一个子节点 qc，对应的字符跟节点 pc 对应的字符相同，则将节点 pc 的失败指针指向节点 qc。

                            if father_fail==self.root: #如果节点 q 中没有子节点的字符等于节点 pc 包含的字符，则令 q=q->fail（fail 表示失败指针，这里有没有很像 KMP 算法里求 next 的过程？），
                                                        # 继续上面的查找，直到 q 是 root 为止，如果还没有找到相同字符的子节点，就让节点 pc 的失败指针指向 root。
                                current.fail = self.root
                                break

                            father_fail = father_fail.fail
                            target=father_fail.get_child(current.key)

                        else: # 尾节点 的后面 找到了 有当前的key的节点 ，说明找到了 当前节点的 最长 后缀
                            current.fail=target


                for node in current.children:
                    queue.append(node)

            level+=1


    def __gen_tree(self, string_list):
        """
        创建trie树

        1. 遍历每个字符串的字符，从根节点开始，如果没有对应子节点，则创建
        2. 每一个串的末尾节点标注为红色(is_ending_char)
        :param string_list:
        :return:
        """
        for string in string_list:

            father = self.root  # 每一个单词 都要从树的根开始找
            for i, c in enumerate(string):  # string='hello'

                if father.get_child(c) is None:  # 如果找不到 才加入新的节点
                    father.insert_child(c)  # c='h'
                    node = father.get_child(c)  # 拿到上一步 新加入的 Node('h')

                    node.father=father

                else:
                    node = father.get_child(c)

                if i == len(string) - 1:
                    node.is_ending_char = True
                    node.length=len(string)

                father = node

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
        Flag = False
        for i, c in enumerate(pattern):
            node = node.get_child(c)
            if node == None:
                break
            if i == len(pattern) - 1:
                if node.is_ending_char == True:
                    Flag = True
        return Flag

    def naive_match(self,main_string):
        """
        朴素的多模式 串匹配 ( Tire 树)
        
        1.对敏感词字典进行预处理，构建成 Trie 树结构。
        2.当用户输入一个文本内容后，我们把用户输入的内容作为主串，从第一个字符（假设是字符 C）开始，在 Trie 树中匹配。
        3. 当匹配到 Trie 树的叶子节点，或者中途遇到不匹配字符的时候，我们将主串的开始匹配位置后移一位，也就是从字符 C 的下一个字符开始，重新在 Trie 树中匹配。
        
        ref: https://time.geekbang.org/column/article/72810
        :param main_string: 
        :return: 
        """
        start=0

        N=len(main_string)
        res=[]

        while start< N:

            # string=main_string[start:] # 字符串 复制 有开销
            node = self.root

            for i in range(start,N):

                node = node.get_child(main_string[i])

                if node == None:
                    break

                if node.is_ending_char == True:

                    martched_string_length=node.length
                    martched_string_end=i+1
                    martched_string_start=martched_string_end-martched_string_length

                    martched_string=main_string[ martched_string_start:martched_string_end]

                    res.append((martched_string_start,martched_string_end,martched_string))

                    break

            start+=1

        return res


    def __match_depreature(self, main_string):
        """
        多模式 串匹配 ( 改进的 Tire 树 —— AC自动机)
        
        1.不能找出所有的模式串（弃用）
        
        ref: https://www.bilibili.com/video/BV1uJ411Y7Eg?p=4
        :param main_string: 
        :return: 
        """

        N=len(main_string)
        res=[]

        father = self.root

        start = 0

        while start< N:

            i=start
            while i<N:
            # for i in range(start,N):
                p = father.get_child(main_string[i])
                if p == None:
                    p=father.fail
                    if p==self.root:
                        break
                    else:
                        i-=1

                elif p.is_ending_char == True:

                    martched_string_length=p.length
                    martched_string_end=i+1
                    martched_string_start=martched_string_end-martched_string_length

                    martched_string=main_string[ martched_string_start:martched_string_end]

                    res.append((martched_string_start,martched_string_end,martched_string))

                i += 1

                father=p

            else: # 正常退出 while 说明把 main_string 都遍历了一次
                return res

            start += 1

        return  res

    def match(self, text):
        """
        多模式 串匹配 ( 改进的 Tire 树 —— AC自动机)
        
        ref: https://time.geekbang.org/column/article/72810
        
        :param main_string:  待匹配的主串 eg.'ahishers'
        :return: [(开始位置，结束位置，模式串a),... ] 
                eg.[(1, 3, 'his'), (3, 5, 'she'), (4, 5, 'he'), (4, 7, 'hers')]
        """

        N = len(text)

        p= self.root

        res=[]

        for i in range(N):

            c=text[i]
            while p.get_child(c)==None and p!=self.root: # 如果 p 指向的节点没有等于 b[i]的子节点，那失败指针就派上用场了，我们让 p=p->fai

                p=p.fail

            p=p.get_child(c) #如果 p 指向的节点有一个等于 b[i]的子节点 x，我们就更新 p 指向 x

            if p==None:
                p=self.root

            tmp=p

            while tmp!=self.root: #这个时候我们需要通过失败指针，检测一系列失败指针为结尾的路径是否是模式串

                if tmp.is_ending_char==True:
                    pos=i-tmp.length+1

                    res.append((pos,i,text[pos:i+1]))

                tmp=tmp.fail

        return res



if __name__ == '__main__':


    patterns = ["he", "she", "hers", "his"]
    ac = AcTrie(patterns)

    # print(ac.naive_match('ahishers'))

    print(ac.match('ahishers'))

    # print(ac.match('hershers'))

    patterns = ["at", "art", "oars", "soar"]
    ac = AcTrie(patterns)
    print(ac.match("soarsoars"))

    # patterns = ["aaaa", "bbbb"]
    # ac = AcTrie(patterns)
    # print(ac.match('aaaabbbb'))


    patterns = ["ab"]
    ac = AcTrie(patterns)
    print(ac.match("abxabcabcaby"))





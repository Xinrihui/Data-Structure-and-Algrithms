
import random

#跳跃表节点的类
class SNode:

    def __init__(self,key=None,value=None):

        #键
        self.key=key
        #index表示数组中最末的元素
        self.maxIndex=-1
        #link 使用一个数组  存放全部下节点的索引  link[i]表示第i层的索引
        self.link = []

        self.value=value


class SkipList:
    def __init__(self,size=8,larger=65535):
        #深度
        self.size = size
        '''
        跳跃表的深度和你的数据有关  大约为 log n   
        '''
        #尾节点指针
        self.tial=SNode()
        #头结点指针   存放头结点
        self.head=SNode()
        #存放在插入和删除操作中  每一个链上遇到的最后节点
        self.last=[]
        self.tial.key=larger#表示尾节点

        self.head.key=-65535#表示头节点

        #头结点的全部指针指向尾节点
        for i in range(self.size):

            self.head.link.append(self.tial)

        self.MAX_RAND=self.size


    def randomLevel(self, p=0.25):
            '''
            #define ZSKIPLIST_P 0.25      /* Skiplist P = 1/4 */
            https://github.com/antirez/redis/blob/unstable/src/t_zset.c
            '''
            high = 1
            for _ in range(self.MAX_RAND - 1):
                if random.random() < p:
                    high += 1 #每次有0.25的概率加1，因此 high越大的概率越低, P(high=3)=0.25*0.25
            return high


    def find(self,v):
        FLAGE=False#标准位   查询是否成功
        p=self.head # p开始存放头指针
        prev=self.head
        level=self.size-1 # level 是最大长度-1   因为是从上往下找
        while level>=0: #如果没有越界
            if p.key==v:
                FLAGE=True
                return p,FLAGE
            elif p.key<v:
                prev=p
                p=p.link[level]
            elif p.key>v:
                level=level-1
                p=prev
        return p,FLAGE


    # def insert(self,v,value=None):
    #     """
    #     :param v:
    #     :param value:
    #     :return:
    #     """
    #
    #     high = self.randomLevel()
    #     p=self.head # p开始存放头指针
    #     prev=self.head
    #
    #     newNode = SNode(v, 1)
    #     newNode.link=[None]*high #别忘了！
    #
    #     _,Flag=self.find(v)  #TODO: 解决重复值的插入问题
    #     if Flag==True:
    #         return False
    #
    #     for i in range(high-1,-1,-1):
    #         p = self.head   #每一层都从 head 节点开始找，时间复杂度不满足 O(log(n))
    #         prev = self.head
    #         while ( p.key < v):
    #             prev=p
    #             p=p.link[i]
    #
    #         prev.link[i]=newNode
    #         newNode.link[i]=p
    #     return True

    def insert(self, v, value=None):
        """
        :param v: 
        :param value: 
        :return: 
        """
        high = self.randomLevel()

        newNode = SNode(v, 1)
        newNode.link = [None] * high

        p = self.head  # p开始存放头指针
        prev = self.head

        cache=[self.head]*high

        for level in range(high - 1, -1, -1): #待插入为 v=7 (high - 1)=2
            while ( p.key < v): # level=2 : p=head head.key=-65536<7 prev=head p=node4; node4.key=4<7 prev=node4 p=nodetail; nodetail.key>7
                prev = p
                p = p.link[level]
            cache[level]=prev #prev=node4
            p=prev #p=node4 下层的起点是上一个索引层中小于插入值的最大值对应的节点
                   #level=1: p=node4 node4.key<7 prev=node4 p=node5;node5.key < 7 prev=node5 p=nodetail;nodetail.key>7

        if p.link[0].key == v: #最后一层 有这个节点，说明重复插入了
            return False
        else:
            for  level in range(high):
                newNode.link[level]=cache[level].link[level]
                cache[level].link[level]=newNode

            return True



    def insert_2(self, v, value=None):
        """
        :param v: 
        :param value: 
        :return: 
        """

        high = self.randomLevel()


        newNode = SNode(v, 1)
        newNode.link = [None] * high

        # _, Flag = self.find(v)  # TODO: 重复插入问题,应该有更好的解
        # if Flag == True:
        #     return False


        cache = [self.head] * high
        cur = self.head
        # 在低于随机高度的每一个索引层寻找小于插入值的节点 eg.插入值v=7 我要在每一层找到小于7的节点并缓存起来
        for level in range(high - 1, -1, -1):
            # 每个索引层内寻找小于带插入值的节点
            # 索引层上下是对应的, 下层的起点是上一个索引层中小于插入值的最大值对应的节点，这样不用每次都从head 从左往右找，时间复杂度满足 O(log(n)) 开心！
            while cur.link[level] and cur.link[level].key < v:
                cur = cur.link[level]
            cache[level] = cur

        # 在小于高度的每个索引层中插入新结点
        for i in range(high - 1, -1, -1):

            newNode.link[i] = cache[i].link[i]
            cache[i].link[i] = newNode

        return True


    def remove(self, v):

        high=(self.size) #high=5
        p = self.head
        prev = self.head

        cache=[self.head]*high

        for level in range(high - 1, -1, -1): #待删除为 v=7
            while ( p.key < v): # level=4 : p=head head.key=-65536<7 prev=head p=node4; node4.key=4<7 prev=node4 p=nodetail; nodetail.key>7
                prev = p
                p = p.link[level]
            cache[level]=prev #prev=node4  cache[4]=node4
            p=prev #p=node4 下层的起点是上一个索引层中小于插入值的最大值对应的节点
                   #level=3: ... ;cache[3]=node4 p=node4
                   #level=2: p=node4 node4.key<7 prev=node4 p=node7;node7.key !< 7 ; cache[2]=node4 p=node4
                   #level=1: p=node4 node4.key<7 prev=node4 p=node5;node5.key < 7 ; prev=node5 p=node7;node7.key !< 7 ; cache[1]=node5 p=node5
                   #level=0:... ; cache[0]=node6 p=node6

        if p.link[0].key==v:
            target_node=p.link[0]
            target_node_high=len(target_node.link)
            for level in range(target_node_high):
                cache[level].link[level]=target_node.link[level]
            return True

        else: #找不到删除的对象
            return False



    def remove_2(self, v):

        # cache用来缓存对应索引层中小于插入值的最大节点
        cache = [None] * (self.size)
        cur = self.head
        # 缓存每一个索引层定位小于插入值的节点
        for i in range(self.size - 1, -1, -1):
            while cur.link[i] and cur.link[i].key < v: #
                cur = cur.link[i]
            cache[i] = cur

        # 如果给定的值存在, 更新索引层中对应的节点
        if cur.link[0] and cur.link[0].key == v: # cur指向了第0层的node3
            for i in range(self.size):
                if cache[i].link[i] and cache[i].link[i].key == v:
                    cache[i].link[i] = cache[i].link[i].link[i]

    # 顺序输出跳跃表
    def outpute(self):

        i = self.size - 1
        while i>=0:
           # i是最大长度-1   因为是从上往下找
            p=self.head
            if p.link==None or p.link[i].key == self.tial.key:  # 如果到达最后的尾指针 说明当前层没有要找的节点

                print('head----->tial')
                i -= 1
                continue
            else:

                print('head',end='--->')
                while True:
                    if p.link==None or p.link[i].key == self.tial.key:
                        break
                    print(p.link[i].key, end='--->')
                    p=p.link[i]
                print('tail')
            i -= 1

if __name__ == '__main__':
    s = SkipList(size=5)
    x = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    for i in x:
        s.insert(i, i)
    print('最开始的情况')
    s.outpute()

    print(s.find(5.5))
    print(s.find(6))

    print('删除7')
    print(s.remove(7))
    s.outpute()

    print('插入7')
    print(s.insert(7))
    s.outpute()


    print('插入4')
    print(s.insert(4))
    s.outpute()

    print('插入4')
    print(s.insert(4))
    s.outpute()

    print('删除5.5')
    print(s.remove(5.5))
    s.outpute()


    # print('插入10')
    # print(s.insert(10))
    # s.outpute()

    # print('插入5.5')
    # print(s.insert(5.5))
    # s.outpute()

    # print('删除3')
    # print(s.remove(3))
    # s.outpute()
    # print('删除5')
    # print(s.remove(5))
    # s.outpute()
    # print('删除1')
    # print(s.remove(1))
    # s.outpute()

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
    """
    by XRH 
    date: 2020-04-12 
    """
    def __init__(self,compare_func,size=8):
        """
        需要传入 自定义的比较函数 
        :param compare_func: 自定义的比较函数 , 
                            1.返回为 0 表示 '=='
                            2.返回为 1 表示 '>'
                            3.返回为 2 表示 '<'
        :param size:  跳表的深度 ， 跳跃表的深度和你的数据有关  大约为 log n   
        """

        self.size = size

        self.compare_func=compare_func

        #尾节点指针
        self.tial=SNode()
        #头结点指针   存放头结点
        self.head=SNode()
        #存放在插入和删除操作中  每一个链上遇到的最后节点
        self.last=[]

        # self.tial.key=float('inf')#表示尾节点
        self.tial.key=self.compare_func(mode=1)

        # self.head.key = float('-inf')  # 表示头节点
        self.head.key =self.compare_func(mode=2)

        #头结点的全部指针指向尾节点
        for i in range(self.size):

            self.head.link.append(self.tial)

        self.MAX_RAND=self.size

        self.length=0 # 底层的 有序 数据链表中，有效数据节点的个数

    def __len__(self):

        return  self.length

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
        """
        在 跳表中寻找 是否 存在 key 
        1. key 可以 为 int 和 str
        :param v: 
        :return: 
        """

        FLAGE=False#标准位   查询是否成功
        p=self.head # p开始存放头指针
        prev=self.head
        level=self.size-1 # level 是最大长度-1   因为是从上往下找
        while level>=0: #如果没有越界
            if  self.compare_func(x=p.key,y=v)==0: # p.key==v
                FLAGE=True
                return p,FLAGE
            elif self.compare_func(x=p.key,y=v)==2: # p.key<v
                prev=p
                p=p.link[level]
            elif self.compare_func(x=p.key,y=v)==1 : # p.key>v
                level=level-1
                p=prev
        return p,FLAGE

    def rangeQuery(self): # TODO 实现 范围查询
        pass


    def add(self, v, value=None):
        """
        跳表 插入 key 
        :param v: 
        :param value: 
        :return: 
        """

        high = self.randomLevel()

        newNode = SNode(v, 1) #TODO： 只记录key ，value 均为默认值 1 ，后面可以利用value 记录其他信息
        newNode.link = [None] * high

        p = self.head  # p开始存放头指针
        prev = self.head

        cache=[self.head]*high

        for level in range(high - 1, -1, -1): #待插入为 v=7 (high - 1)=2
            while ( self.compare_func(x=p.key,y=v)==2  ): # p.key < v
                                # level=2 : p=head head.key=-65536<7 prev=head p=node4; node4.key=4<7 prev=node4 p=nodetail; nodetail.key>7
                prev = p
                p = p.link[level]
            cache[level]=prev #prev=node4
            p=prev #p=node4 下层的起点是上一个索引层中小于插入值的最大值对应的节点
                   #level=1: p=node4 node4.key<7 prev=node4 p=node5;node5.key < 7 prev=node5 p=nodetail;nodetail.key>7

        if self.compare_func(x=p.link[0].key,y=v)==0 : #p.link[0].key == v
                                                    #最后一层 有这个节点，说明重复插入了
            return False
        else:
            for  level in range(high):
                newNode.link[level]=cache[level].link[level]
                cache[level].link[level]=newNode

            self.length+=1 # 底层 数据链表中加入 一个有效的数据节点

            return True


    def remove(self, v):
        """
        跳表 删除 key
        :param v: 
        :return: 
        """

        high=(self.size) #high=5
        p = self.head
        prev = self.head

        cache=[self.head]*high

        for level in range(high - 1, -1, -1): #待删除为 v=7
            while ( self.compare_func(x=p.key,y=v)==2 ): # p.key < v
                                # level=4 : p=head head.key=-65536<7 prev=head p=node4; node4.key=4<7 prev=node4 p=nodetail; nodetail.key>7
                prev = p
                p = p.link[level]
            cache[level]=prev #prev=node4  cache[4]=node4
            p=prev #p=node4 下层的起点是上一个索引层中小于插入值的最大值对应的节点
                   #level=3: ... ;cache[3]=node4 p=node4
                   #level=2: p=node4 node4.key<7 prev=node4 p=node7;node7.key !< 7 ; cache[2]=node4 p=node4
                   #level=1: p=node4 node4.key<7 prev=node4 p=node5;node5.key < 7 ; prev=node5 p=node7;node7.key !< 7 ; cache[1]=node5 p=node5
                   #level=0:... ; cache[0]=node6 p=node6

        if self.compare_func(x=p.link[0].key,y=v)==0: # p.link[0].key==v
            target_node=p.link[0]
            target_node_high=len(target_node.link)
            for level in range(target_node_high):
                cache[level].link[level]=target_node.link[level]

            self.length -= 1  # 底层 数据链表中 少了 一个有效的数据节点
            return True

        else: #找不到删除的对象
            return False



    def print_all(self):
        """
        从 上往下 打印整个 跳跃表 包括 所有的 指针
     
        :return:  
        """

        i = self.size - 1
        while i>=0:
           # i是最大长度-1   因为是从上往下找
            p=self.head
            if p.link==None or self.compare_func(x=p.link[i].key,y=self.tial.key)==0 : #p.link[i].key == self.tial.key
                                                        # 如果到达最后的尾指针 说明当前层没有要找的节点

                print('head----->tial')
                i -= 1
                continue
            else:

                print('head',end='--->')
                while True:
                    if p.link==None or self.compare_func(x=p.link[i].key,y=self.tial.key)==0: # p.link[i].key == self.tial.key
                        break

                    print(p.link[i].key, end='--->')
                    p=p.link[i]
                print('tail')
            i -= 1

    def output(self):
        """
        顺序 记录 最下面一层 即 数据 链表 的所有节点 的key ，并输出 

        :return:  数据 链表 的所有节点的 key（已排序）
        """

        res=[]

        i = self.size - 1
        while i >= 0:
            # i是最大长度-1   因为是从上往下找
            p = self.head
            if p.link == None or self.compare_func(x=p.link[i].key,y=self.tial.key)==0:  # 如果到达最后的尾指针 说明当前层没有要找的节点

                i -= 1
                continue
            else:

                while True:
                    if p.link == None or self.compare_func(x=p.link[i].key,y=self.tial.key)==0:
                        break

                    if i==0:
                        res.append(p.link[i].key)

                    p = p.link[i]
            i -= 1

        return res

if __name__ == '__main__':

    def compare_func(x=None,y=None,mode=0):
        """
        比较函数
        mode=0: 比较模式
        mode=1：返回最大值 
        mode=2: 返回最小值
        :param x: 
        :param y: 
        :param mode: 
        :return: 
        """
        if mode==0:
            res=-1
            if x==y:
                res=0
            elif x>y:
                res=1
            elif x<y:
                res=2
            return res

        elif mode==1:
            return float('inf')

        elif mode==2:
            return float('-inf')

    s = SkipList(compare_func,size=5)
    x = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    for i in x:
        s.add(i, i)
    print('最开始的情况')
    s.print_all()

    print(s.find(5.5))
    print(s.find(6))

    print('删除7')
    print(s.remove(7))
    s.print_all()

    print('插入7')
    print(s.add(7))
    s.print_all()


    print('插入4')
    print(s.add(4))
    s.print_all()

    print('插入4')
    print(s.add(4))
    s.print_all()

    print('删除5.5')
    print(s.remove(5.5))
    s.print_all()

    print(s.output())
    print(len(s))


    # print('插入10')
    # print(s.add(10))
    # s.print_all()

    # print('插入5.5')
    # print(s.add(5.5))
    # s.print_all()

    # print('删除3')
    # print(s.remove(3))
    # s.print_all()
    # print('删除5')
    # print(s.remove(5))
    # s.print_all()
    # print('删除1')
    # print(s.remove(1))
    # s.print_all()

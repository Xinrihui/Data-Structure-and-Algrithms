# Definition for singly-linked list.
class DbListNode(object):
    def __init__(self, x, y):
        self.key = x
        self.val = y
        self.next = None
        self.prev = None

class LRUCache_v1:
    '''
    leet code: 146
        运用你所掌握的数据结构，设计和实现一个  LRU (最近最少使用) 缓存机制。
        它应该支持以下操作： 获取数据 get 和 写入数据 put 。
    
    LRUCache_v1  使用双向链表实现 ，查找的 时间 复杂度 O(n) 
    LRUCache_v2  使用 哈希表+双向链表 ，降低 查找的时间 复杂度为 O(1)
    Author: xrh
    '''

    def __init__(self, capacity: int):
        self.head=DbListNode('head',None)
        self.capacity=capacity #缓存容量 的上限
        self.current=0 # 记录当前 占用的 缓存容量
        self.tail=DbListNode('tail',None)

        self.head.next=self.tail
        self.tail.prev=self.head

    def __find(self,key):
        """
        在双向链表中 查找特定的key ,找到的话返回 DbListNode，找不到返回 None
        :param key: 
        :return: 
        """
        p=self.head
        res=None
        while(p!= self.tail):
            if p.key==key:
                res=p
                break
            p=p.next
        return res

    def get(self, key: int) -> int:
        """
        获取数据 get(key) - 如果密钥 (key) 存在于缓存中，则获取密钥的值（总是正数），否则返回 -1。
        :param key: 
        :return: 
        """
        p=self.__find(key)
        if p==None:
            return -1
        else:
            #从原来的位置删除
            # 删除链表结点时，一定要记得手动释放内存空间，像 Java 这种虚拟机自动管理内存 就无所谓了
            # python 的内存管理 https://www.cnblogs.com/xybaby/p/7491656.html
            front=p.prev
            behind=p.next
            front.next=behind
            behind.prev=front

            #插到 链表的最前面
            tmp=self.head.next
            self.head.next=p
            p.prev=self.head

            p.next=tmp
            tmp.prev=p


            return p.val


    def put(self, key: int, value: int) -> None:
        """
        写入数据 put(key, value) - 如果密钥不存在，则写入其数据值。
            当缓存容量达到上限时，它应该在写入新数据之前删除最近最少使用的数据值，从而为新的数据值留出空间
        :param key: 
        :param value: 
        :return: 
        """
        p=self.__find(key)
        if p !=None:
            print("The key already exitsis" )
            return
        p=DbListNode(key,value) # p 是 None 则新建一个 node
        if self.current< self.capacity:
            # p 插到 链表的最前面
            tmp = self.head.next
            self.head.next = p
            p.prev = self.head

            p.next = tmp
            tmp.prev = p

            self.current=self.current+1

        else: #达到了 缓存的上限
            #删除链表最后的 一个元素
            p1=self.tail.prev.prev
            p1.next=self.tail
            self.tail.prev=p1

            # p 插到 链表的最前面
            tmp = self.head.next
            self.head.next = p
            p.prev = self.head

            p.next = tmp
            tmp.prev = p


    def __repr__(self):
        vals = []
        p = self.head
        while p:
            vals.append(str(p.key))
            p = p.next
        return '->'.join(vals)


class LRUCache_v2:
    '''
    leet code: 146
         设计和实现一个  LRU (最近最少使用) 缓存机制。
        它应该支持以下操作： 获取数据 get 和 写入数据 put 。

    LRUCache_v1  使用双向链表实现 ，查找的 时间 复杂度 O(n) 
    LRUCache_v2  使用 哈希表+双向链表 ，降低 查找的时间 复杂度为 O(1)
    Author: xrh
    '''

    def __init__(self, capacity: int):
        self.head = DbListNode('head', None)
        self.capacity = capacity  # 缓存容量 的上限
        self.current = 0  # 记录当前 占用的 缓存容量
        self.tail = DbListNode('tail', None)

        self.hash_cache={} # 记录 key 和 key 对应的节点指针

        #head 和 tail 节点要先连接到一起
        self.head.next = self.tail
        self.tail.prev = self.head

    def __find(self, key):
        """
        在hash 表中 查找特定的key ,找到的话返回 DbListNode，找不到返回 None
        :param key: 
        :return: 
        """
        res=None
        if key  in self.hash_cache :
            res=self.hash_cache[key]

        return res

    def get(self, key: int) -> int:
        """
        获取数据 get(key) - 如果密钥 (key) 存在于缓存中，则获取密钥的值（总是正数），否则返回 -1。
        :param key: 
        :return: 
        """
        p = self.__find(key)

        if p == None:
            return -1
        else:
            # 从原来的位置删除
            # 删除链表结点时，一定要记得手动释放内存空间，像 Java 这种虚拟机自动管理内存 就无所谓了
            # python 的内存管理 https://www.cnblogs.com/xybaby/p/7491656.html
            front = p.prev
            behind = p.next
            front.next = behind
            behind.prev = front

            # 插到 链表的最前面
            tmp = self.head.next
            self.head.next = p
            p.prev = self.head

            p.next = tmp
            tmp.prev = p

            return p.val

    def put(self, key: int, value: int) -> None:
        """
        写入数据 put(key, value) - 如果密钥不存在，则写入其数据值。
            当缓存容量达到上限时，它应该在写入新数据之前删除最近最少使用的数据值，从而为新的数据值留出空间
        :param key: 
        :param value: 
        :return: 
        """
        p = self.__find(key)
        if p != None:
            print("The key already exitsis")
            return
        p = DbListNode(key, value)  # p 是 None 则新建一个 node
        self.hash_cache[key] = p

        if self.current < self.capacity:
            # p 插到 链表的最前面
            tmp = self.head.next
            self.head.next = p
            p.prev = self.head

            p.next = tmp
            tmp.prev = p

            self.current = self.current + 1

        else:  # 达到了 缓存的上限
            # 删除链表最后的 一个元素
            key_del=self.tail.prev.key
            del self.hash_cache[key_del]

            p1 = self.tail.prev.prev
            p1.next = self.tail
            self.tail.prev = p1

            # p 插到 链表的最前面
            tmp = self.head.next
            self.head.next = p
            p.prev = self.head

            p.next = tmp
            tmp.prev = p

    def __repr__(self):
        vals = []
        p = self.head
        while p:
            vals.append(str(p.key))
            p = p.next
        return '->'.join(vals)


if __name__ == '__main__':
    cache = LRUCache_v2(2)
    cache.put(1, 1)
    cache.put(2, 2)
    print(cache)
    print(cache.get(1) ) # 返回  1
    cache.put(3, 3)  # 该操作会使得密钥 2 作废
    print(cache)
    print( cache.get(2))  # 返回 -1 (未找到)
    cache.put(4, 4)  # 该操作会使得密钥 1 作废
    print(cache)
    print( cache.get(1) ) # 返回 -1 (未找到)
    print(cache.get(3))  # 返回  3
    print(cache)
    print(cache.get(4))  # 返回  4
    print(cache)

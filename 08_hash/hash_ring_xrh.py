import hashlib
import bisect

# class Node(object):
#     def __init__(self, hostname,key_list):
#         self.hostname=hostname
#         self.key_list=key_list


class HashRing(object):

    def __init__(self, nodes=None, n_number=3):
        """
        :param nodes:           所有的节点
        :param n_number:        一个节点对应多少个虚拟节点
        :return:
        """
        self.node_cycle_list=[]
        self.nodes=nodes
        self.n_number=n_number

        self.hashTable={}

        for node in nodes: #["127.0.0.1", "192.168.1.1","192.168.10.2"]
            for i in range(self.n_number):
                node_extend=node+'#'+str(i) #"127.0.0.1#0"
                hash_node_extend=self._gen_key(node_extend)

                bisect.insort_left(self.node_cycle_list,hash_node_extend) # 二分法 快速找到插入有序数组的位置并插入

                self.hashTable[hash_node_extend]=node_extend

    def add_node(self, node):
        """
        添加node，首先要根据虚拟节点的数目，创建所有的虚拟节点，保持虚拟节点hash值的顺序
        :param node:
        :return:
        """
        flag = False
        if node not in self.nodes:
            self.nodes.append(node) #先把实体节点加上
            for i in range(self.n_number):
                node_extend = node + '#' + str(i)  # "127.0.0.1#0"
                hash_node_extend = self._gen_key(node_extend)

                bisect.insort_left(self.node_cycle_list, hash_node_extend)
                                # self.node_cycle_list= [10, 30, 50, 70] hash_node_extend=20

                self.hashTable[hash_node_extend] = node_extend

            flag=True
        return flag

    def remove_node(self, node):
        """
        一个节点的退出，需要将这个节点的所有的虚拟节点都删除
        :param node:
        :return:
        """
        flag=False
        if node in self.nodes:
            self.nodes.remove(node) # 先把实体节点删除

            for i in range(self.n_number): #删除所有的虚拟节点
                node_extend = node + '#' + str(i)  # "127.0.0.1#0"
                hash_node_extend = self._gen_key(node_extend)
                index = bisect.bisect_left(self.node_cycle_list, hash_node_extend) #肯定能找到
                                                            # self.node_cycle_list= [10, 30, 50, 70] ，hash_node_extend=30 ,index=1

                self.node_cycle_list.pop(index)

                del self.hashTable[hash_node_extend]

            flag = True
        return flag

    def get_node(self, key_str):
        """
        返回这个字符串应该对应的node，这里先求出字符串的hash值，然后找到第一个小于等于的虚拟节点，然后返回node
        如果hash值大于所有的节点，那么用第一个虚拟节点
        :param :
        :return:
        """
        hash_key=self._gen_key(key_str)
        index=bisect.bisect_left(self.node_cycle_list, hash_key)
        result='None'

        hash_node=self.node_cycle_list[index]
        if hash_key <= hash_node :
            result=self.hashTable[ hash_node ]

        elif self.node_cycle_list[index]==len(self.node_cycle_list): # 如果hash值大于所有的节点，那么用第一个虚拟节点
            result = self.hashTable[self.node_cycle_list[0]]

        return result.split('#')[0] #去掉虚拟节点的后缀  eg. 127.0.0.1#0 -> 127.0.0.1

    @staticmethod
    def _gen_key(key_str):
        """
        通过key，返回当前key的 hash值，这里采用 md5
        :param key:
        :return:
        """
        md5_str = hashlib.md5(key_str.encode("utf-8")).hexdigest()
        return md5_str


if __name__ == "__main__":
    # data = "你好"
    # m = hashlib.md5(data.encode("gb2312"))
    # print(m.hexdigest())
    # m = hashlib.md5(data.encode("utf-8"))
    # print(m.hexdigest())

    hash_ring = HashRing(["127.0.0.1", "192.168.1.1","192.168.10.2"])
    print(hash_ring.get_node("a"))

    print(hash_ring.remove_node("192.168.1.1"))
    print(hash_ring.get_node("a"))

    print(hash_ring.add_node("192.168.1.1"))
    print(hash_ring.get_node("a"))

    print(hash_ring.remove_node("192.168.1.2"))

    #TODO: 模拟 整个 一致性哈希的存储的过程
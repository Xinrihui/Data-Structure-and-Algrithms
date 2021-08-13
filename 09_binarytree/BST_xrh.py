
from queue import Queue
import math

class TreeNode:
    def __init__(self, val=None):
        self.val = val
        self.left = None
        self.right = None
        self.parent = None


class BinarySearchTree:
    def __init__(self, val_list=[]):
        if len(val_list)!=0:
            self.root = TreeNode(val_list[0])

            for index in range(1,len(val_list)) :
                self.insert(val_list[index])
        else:
            print('val_list is empty ! ')

    def _search(self, data):
        """
        找到插入的 位置,返回插入位置的父节点
        1.支持重复数据：如果碰到一个节点的值，与要插入数据的值相同，我们就将这个要插入的数据放到这个节点的右子树
        :param data: 
        :return: 
        """
        p=self.root
        parent=None
        while p:
            if data >= p.val: # 支持重复数据
                parent=p
                p=p.right
            elif data < p.val:
                parent = p
                p=p.left

        return parent

    def insert(self, data):
        """
        插入
        :param data:
        :return:
        """
        assert (isinstance(data, int))
        # print('data:',data)
        parent=self._search(data)
        # print('parent:', parent.val)
        p = TreeNode(data)
        if data >= parent.val: # data 比父节点大 挂在右边
            parent.right=p
        elif data < parent.val:  # data 比父节点小 挂在左边
            parent.left=p
        p.parent = parent

    def search(self, data):
        """
        搜索
        返回bst中所有值为data的节点列表
        1.支持重复数据：当要查找数据的时候，遇到值相同的节点，我们并不停止查找操作，而是继续在右子树中查找，直到遇到叶子节点，才停止。
        :param data:
        :return:
        """
        assert (isinstance(data, int))
        res=[]
        p=self.root
        while p:
            if data ==p.val:
                res.append(p)
                p = p.right
            elif data > p.val:
                p=p.right
            elif data < p.val:
                p=p.left

        return res


    def delete(self, data):
        """
        删除
        :param data:
        :return:
        """
        assert (isinstance(data, int))

        # 通过搜索得到需要删除的节点
        del_list = self.search(data)

        for n in del_list:
            # 父节点为空，又不是根节点，已经不在树上，不用再删除
            if n.parent is None and n != self.root:
                continue
            else:
                self._del(n)

    def _del(self, node):
        """
        删除
        所删除的节点N存在以下情况：
        1. 没有子节点：直接删除N的父节点指针
        2. 有一个子节点：将N父节点指针指向N的子节点
        3. 有两个子节点：找到右子树的最小节点M，将值赋给N，然后删除M
        :param data:
        :return:
        """
        # 1.
        parent=node.parent
        if node.left is None and node.right is None: # node.val=55  parent.val=51
           if node.val >= parent.val:
               parent.right=None
           elif node.val < parent.val:
               parent.left=None

        #2.
        elif node.left and node.right is None:
            if node.val >= parent.val:
                parent.right = node.left

            elif node.val < parent.val:
                parent.left = node.left

            node.left.parent = parent

        elif node.left is None and node.right : # node.val=13  parent.val=16
            if node.val >= parent.val:
                parent.right = node.right
            elif node.val < parent.val:
                parent.left = node.right

            node.right.parent=parent

        #3.
        elif node.left and node.right: # node.val=18 parent.val=16
            M_node=self.get_min(node.right)
            node.val=M_node.val
            self._del(M_node)


    def get_min(self,root):
        """
        返回最小值节点
        :return:
        """
        p = root
        while p.left:

            p=p.left
        return p


    def get_max(self):
        """
        返回最大值节点
        :return:
        """
        p = self.root
        while p.right:
            p = p.right
        return p

    def in_order(self):
        """
        中序遍历
        :return:
        """
        if self.root is None:
            return []

        return self._in_order(self.root)

    def _in_order(self, node):
        if node is None:
            return []

        ret = []
        n = node
        ret.extend(self._in_order(n.left))
        ret.append(n.val)
        ret.extend(self._in_order(n.right))

        return ret

    def __repr__(self):
        # return str(self.in_order())
        # print(str(self.in_order()))
        return self._draw_tree()

    def _bfs(self):
        """
        bfs
        通过父子关系记录节点编号
        :return:
        """
        if self.root is None:
            return []

        ret = []
        q = Queue()
        # 队列[节点，编号]
        q.put((self.root, 1))

        while not q.empty():
            n = q.get()

            if n[0] is not None:
                ret.append((n[0].val, n[1]))
                q.put((n[0].left, n[1]*2))
                q.put((n[0].right, n[1]*2+1))

        return ret


    def _draw_tree(self):
        """
        可视化
        :return:
        """
        nodes = self._bfs()

        if not nodes:
            print('This tree has no nodes.')
            return

        layer_num = int(math.log(nodes[-1][1], 2)) + 1

        prt_nums = []

        for i in range(layer_num):
            prt_nums.append([None] * 2 ** i)

        for v, p in nodes:
            row = int(math.log(p, 2))
            col = p % 2 ** row
            prt_nums[row][col] = v

        prt_str = ''
        for l in prt_nums:
            prt_str += str(l)[1:-1] + '\n'

        return prt_str


if __name__ == '__main__':
    ## 插入、查找
    # nums = [33, 17, 50, 13, 18,34,58,16,25,51,66,19,27]
    # bst = BinarySearchTree(nums)
    # bst.insert(55)
    # bst.insert(18)
    # print(bst.in_order())
    # print(bst)
    # print( [node.val for node in bst.search(18)]  )
    # print( (bst.get_max()).val)
    # print( (bst.get_min(bst.root)).val)


    # # 删除
    nums = [33, 16, 50, 13, 18, 34, 58, 15, 17, 25, 51, 66, 19, 27, 55]
    bst = BinarySearchTree(nums)
    print(bst)

    # bst.delete(55)
    # bst.delete(13)
    # bst.delete(18)
    # print(bst)



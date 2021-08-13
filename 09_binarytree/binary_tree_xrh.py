from collections import deque


class TreeNode(object):

    def __init__(self,item):
        self.val=item
        self.left=None
        self.right=None
        # self.height=None


# Codec
class DFS_Serialize:
    """
    基于 DFS 的 二叉树 的序列化 和 反序列化

    递归实现

    ref:
    https://blog.csdn.net/Shenpibaipao/article/details/108378093
    https://zhuanlan.zhihu.com/p/164408048

    """

    def __serialize_preorder(self, p):
        """
        将 树 序列化为 前序序列
        :param p:
        :return:
        """

        if p != None:

            c = p.val
            self.preorder_list.append(c)

            self.__serialize_preorder(p.left)
            self.__serialize_preorder(p.right)

        else:
            self.preorder_list.append('#')  # 使用 '#' 表示空节点

    def serialize(self, root):
        """
        将 树 序列化为 前序序列
        :type preorder: str
        """
        if root is None:
            return ''

        self.preorder_list = []

        self.__serialize_preorder(root)

        preorder_list = [str(ele) for ele in self.preorder_list]

        res = ','.join(preorder_list)

        return res

    def __preorder_deSerialize(self, preorder):
        """
        递归方法

        :param preorder:
        :param prev: 父亲节点
        :return:
        """

        c = preorder.popleft()

        if c != '#':
            p = TreeNode(c)

            p.left = self.__preorder_deSerialize(preorder)
            p.right = self.__preorder_deSerialize(preorder)

            return p

    def deserialize(self, str1):
        """

        由 先序序列 反序列化 出 BST 二叉搜索树 (递归)


        :param preorder:
        :return:
        """
        if len(str1) == 0:
            return None

        preorder = str1.split(',')

        preorder = [ele.strip() for ele in preorder]

        preorder = deque(preorder)

        root = self.__preorder_deSerialize(preorder)

        return root


# Codec
class DFS_Serialize_Stack:
    """
    基于 DFS 的 二叉树 的序列化 和 反序列化

    非递归实现

    """

    def serialize(self, root):
        """
        将 二叉树 序列化为 前序序列

        """
        if root is None:
            return ''

        preorder_list = []

        p = root

        stack = []
        stack.append(p)

        while len(stack) > 0:

            current = stack.pop()

            if current is not None:  # 当前节点不是 空节点

                preorder_list.append(current.val)

                if current.right is not None:
                    stack.append(current.right)
                else:
                    stack.append(None)  # 空节点 入栈

                if current.left is not None:
                    stack.append(current.left)
                else:
                    stack.append(None)

            else:  # 当前节点 是空节点
                preorder_list.append('#')  # 使用 '#' 表示空节点

        preorder_list = [str(ele) for ele in preorder_list]  # leetcode 的树 的节点是 int

        preorder_str = ','.join(preorder_list)

        return preorder_str

    def deserialize(self, preorder_str):
        """
        由 先序序列 反序列化 出 二叉树 ( 非递归 )

        ref: https://blog.csdn.net/cyuyanenen/article/details/51589945

        :param preorder:
        :return:
        """
        if len(preorder_str) == 0:
            return None

        preorder = preorder_str.split(',')

        preorder = [ele.strip() for ele in preorder]

        i = 0
        root = TreeNode(preorder[i])

        stack = []
        stack.append(root)

        i += 1

        flag = 1
        """
        flag = 1 表示现在需要创建当前节点的左孩子，
        flag = 2 表示需要创建右孩子，
        flag = 3 则表示当前节点的左右孩子都已经创建完毕，需要执行出栈操作，直到出栈节点不是当前栈顶节点的右孩子为止。
        """

        while i < len(preorder):

            if flag == 1:
                if preorder[i] == '#':
                    flag = 2
                else:
                    child_left = TreeNode(preorder[i])
                    current = stack[-1]
                    current.left = child_left

                    stack.append(child_left)

                    flag = 1

            elif flag == 2:
                if preorder[i] == '#':
                    flag = 3
                else:
                    child_right = TreeNode(preorder[i])
                    current = stack[-1]
                    current.right = child_right

                    stack.append(child_right)

                    flag = 1

            elif flag == 3:

                top_ele = stack.pop()

                while len(stack) > 0 and stack[-1].right == top_ele:
                    top_ele = stack.pop()

                i -= 1
                flag = 2

            i += 1

        return root


# Codec
class BFS_Serialize:
    """
    基于 BFS 的 二叉树 的序列化 和 反序列化

    """

    def serialize(self, root):
        """
        将 二叉树 序列化为  层次序列

        """
        if root is None:
            return ''

        h_list = []

        p = root

        queue = deque()
        queue.append(p)

        while len(queue) > 0:

            current = queue.popleft()

            if current is not None:  # 当前节点不是 空节点

                h_list.append(current.val)

                if current.left is not None:
                    queue.append(current.left)
                else:
                    queue.append(None)

                if current.right is not None:
                    queue.append(current.right)
                else:
                    queue.append(None)  # 空节点 入栈


            else:  # 当前节点 是空节点
                h_list.append('#')  # h_list 使用 '#' 表示空节点

        h_list = [str(ele) for ele in h_list]  # leetcode 的树 的节点是 int

        h_str = ','.join(h_list)  # ',' 作为 分隔符

        return h_str

    def deserialize(self, h_str):
        """
        由 先序序列 反序列化 出 二叉树 ( 非递归 )

        :param preorder:
        :return:
        """
        if len(h_str) == 0:
            return None

        h_list = h_str.split(',')
        h_list = [ele.strip() for ele in h_list]

        i = 0
        root = TreeNode(h_list[i])
        i += 1

        queue = deque()
        queue.append(root)

        while i < len(h_list) and len(queue) > 0:

            current = queue.popleft()

            if h_list[i] != '#':
                left_child = TreeNode(h_list[i])
                current.left = left_child

                queue.append(left_child)

            i += 1

            if h_list[i] != '#':
                right_child = TreeNode(h_list[i])
                current.right = right_child

                queue.append(right_child)

            i += 1

        return root




class Solution1(object):
    """
    二叉树的链式存储法 表达二叉树
    """
    def buildTree(self, preorder,inorder):
        """
        用树的前序和中序遍历的结果来构建树
        :type preorder:  ['a','b','c','e','d']
        :type inorder:   ['c','b','e','a','d']
        :rtype: TreeNode
        """
        self.preorder = deque(preorder)
        self.inorder = deque(inorder)
        return self._buildTree(0, len(inorder))

    def _buildTree(self, start, end):
        if start<end:
            root_val=self.preorder.popleft()
            print("root: ",root_val )
            root=TreeNode(root_val)

            index=self.inorder.index(root_val,start,end) # 在数组的位置范围： [start,end) 中寻找 root_val
            root.left=self._buildTree(start,index)
            root.right=self._buildTree(index+1,end)

            return root

    def pre_order(self,root):
        if root is not None:
            print(root.val)
            self.pre_order(root.left)
            self.pre_order(root.right)

        return


    def in_order_depreatured(self,root):
        """
        非递归 实现树的中序遍历
        :param root: 
        :return: 
        """
        stack=[root]
        p=root
        res=[]

        while len(stack)!=0 :

            while (p!=None) and (p.left!=None) and (p.val not in res): #访问过的节点不要再入栈
                p = p.left
                stack.append(p)

            p=stack.pop()
            res.append(p.val)

            if p.right!=None:
                p=p.right
                stack.append(p)

        return res

    def in_order(self, root):
        """
        非递归 实现树的中序遍历
        :param root: 
        :return: 
        """
        stack = []
        p = root
        res = []

        while p!=None or len(stack)!=0:

            if p!=None: # p 不为空就入栈
                stack.append(p)
                p=p.left #指向左节点

            else: # 如果p 为空就弹出
                p=stack.pop() # 访问中间节点
                res.append(p.val)
                p=p.right  # 指针指向右子树

        return res

    def _depth_recursion(self,root):
        if root is None:
            return 0
        left_depth= self._depth_recursion(root.left)
        right_depth=self._depth_recursion(root.right)

        return max(left_depth,right_depth)+1


    def _depth(self, root):
        """
        改进层次遍历 ，把树的各个层都切分出来，并能输出树的高度
        :type root: TreeNode
        :rtype: int
        """
        Queue = deque()
        Queue.append(root)
        depth = 0
        while (len(Queue) != 0):
            depth += 1
            n = len(Queue)
            for i in range(n):  # Stratified according to depth

                target = Queue.popleft()
                print(target.val)
                print('depth: ', depth)

                if target.left != None:
                    Queue.append(target.left)
                if target.right != None:
                    Queue.append(target.right)

        return depth

class Solution2(object):
    """
    基于数组的顺序存储法 表达二叉树
    """
    def pre_order(self, tree_array):
        """
        前序遍历 中->左->右
        :param tree_array: 
        :return: 
        """
        stack=[]
        i=1
        node=[tree_array[i],i]
        stack.append(node)
        result=[]
        while ( len(stack)!=0 ):
            current=stack.pop()
            # print(current)
            result.append(current[0])
            i=current[1]

            if 2*i+1<len(tree_array) and tree_array[2*i+1]!=None: # tree_array 越界 访问检查 : 2*i+1<len(tree_array)
                node=[tree_array[2*i+1],2*i+1]
                stack.append(node)
            if  2*i<len(tree_array) and tree_array[2*i]!=None:
                node = [tree_array[2 * i ], 2 * i]
                stack.append(node)
        return result

    def post_order(self, tree_array):
        """
        前序遍历 ：中->左->右
        前序遍历反过来 ：中->右->左
        前序遍历反过来再逆序 ： 左 -> 右 ->中 （后序遍历）
        
        https://www.cnblogs.com/bjwu/p/9284534.html  
        :param tree_array: 
        :return: 
        """
        stack=[]
        i=1
        node=[tree_array[i],i]
        stack.append(node)
        result=[]
        while ( len(stack)!=0 ):
            current=stack.pop()
            # print(current)
            result.append(current[0])
            i=current[1]

            if  2*i<len(tree_array) and tree_array[2*i]!=None:
                node = [tree_array[2 * i ], 2 * i]
                stack.append(node)

            if 2*i+1<len(tree_array) and tree_array[2*i+1]!=None: # tree_array 越界 访问检查 : 2*i+1<len(tree_array)
                node=[tree_array[2*i+1],2*i+1]
                stack.append(node)

        return result[::-1]  # 逆序输出即为 后序遍历

    def in_order_deprecated(self, tree_array):
        stack=[]
        i=1
        result=[]
        while ( i < len(tree_array) and tree_array[i] != None)  or (len(stack) != 0):  #   ( i < len(tree_array) and tree_array[i] != None) 等价于 p != None
            while (i < len(tree_array) and tree_array[i] != None):
                node = [tree_array[i],  i]
                stack.append(node)
                i = 2 * i  # 左子树全部进栈

            if (len(stack) != 0) :
                current = stack.pop()  #
                # print(current)
                result.append(current[0])
                i = current[1]
                i= 2*i+1 #尝试去访问右子树

        return result

    def in_order(self, tree_array):
        """
        好理解 
        :param tree_array: 
        :return: 
        """
        stack=[]
        i=1
        result=[]
        while ( i < len(tree_array) and tree_array[i] != None)  or (len(stack) != 0):
            if (i < len(tree_array) and tree_array[i] != None):
                node = [tree_array[i],  i]
                stack.append(node)
                i = 2 * i  # 左子树全部进栈
            else:
                current = stack.pop()  #
                # print(current)
                result.append(current[0])
                i = current[1]
                i= 2*i+1 #尝试去访问右子树

        return result

    def hierarchy_order(self, tree_array):
        """
        树的层次遍历 （广度优先遍历）
        :param tree_array: 
        :return: 
        """
        fifo=deque()
        i=1
        node=[tree_array[i],i]
        fifo.appendleft(node)

        result=[]

        while ( len(fifo)!=0 ):
            current=fifo.pop()
            # print(current)
            result.append(current[0])
            i=current[1]

            if  2*i<len(tree_array) and tree_array[2*i]!=None: # 左边
                node = [tree_array[2 * i ], 2 * i]
                fifo.appendleft(node)

            if 2*i+1<len(tree_array) and tree_array[2*i+1]!=None: # 右边
                node=[tree_array[2*i+1],2*i+1]
                fifo.appendleft(node)

        return result

class Test:

    def test_DFS(self):
        sol = DFS_Serialize_Stack()

        preorder = '1,2,#,#,3,4,#,5,#,#,#'

        tree = sol.deserialize(preorder)

        print(sol.serialize(tree))

        assert sol.serialize(tree) == preorder

        preorder = ''
        tree = sol.deserialize(preorder)

        assert sol.serialize(tree) == preorder

        preorder = '9,3,4,#,5,#,#,1,#,#,#'

        tree = sol.deserialize(preorder)

        print(sol.serialize(tree))

        assert sol.serialize(tree) == preorder

    def test_BFS(self):
        sol = BFS_Serialize()

        preorder = '8,6,10,5,7,9,11,#,#,#,#,#,#,#,#'

        tree = sol.deserialize(preorder)

        print(sol.serialize(tree))

        assert sol.serialize(tree) == preorder

        preorder = ''
        tree = sol.deserialize(preorder)

        assert sol.serialize(tree) == preorder

        preorder = '8,6,10,#,#,9,11,#,#,#,#'

        tree = sol.deserialize(preorder)

        print(sol.serialize(tree))

        assert sol.serialize(tree) == preorder


    def test_solution1(self):

        # solution1
        preorder=['a','b','c','e','d']
        inorder= ['c','b','e','a','d']

        preorder=['A','B','D','F','G','C','E','H']
        inorder=['F','D','G','B','A','E','H','C']
        postorder= ['F','G','D','B','H','E','C','A']

        solution=Solution1()
        root=solution.buildTree(preorder,inorder)
        solution.pre_order(root)
        print(solution.in_order(root))
        print(solution._depth(root))
        print(solution._depth_recursion(root))

    def test_solution2(self):

        tree_array=[None,'A','B','C','D',None,'E',None,'F','G',None,None,None,'H']
        solution2 = Solution2()
        print('preorder: ',solution2.pre_order(tree_array))
        print('inorder: ',solution2.in_order(tree_array))
        print('postorder: ', solution2.post_order(tree_array))
        print('hierarchy_order: ', solution2.hierarchy_order(tree_array))




if __name__ == "__main__":

    t = Test()

    t.test_BFS()







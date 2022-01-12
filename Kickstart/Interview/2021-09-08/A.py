#!/usr/bin/python
# -*- coding: UTF-8 -*-

import timeit
from collections import *
import numpy as np


class TreeNode(object):

    def __init__(self, item):

        self.val = item
        self.left = None
        self.right = None

        # for Q2
        # 正向(递增)路径
        self.pos_path_len = 0
        # 反向(递减)路径
        self.neg_path_len = 0



# Codec
class BFS_Serialize:
    """
    基于 BFS 的 二叉树 的序列化 和 反序列化

    """

    def serialize(self, root):
        """
        将 二叉树 序列化为 层次序列 ;

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
        由 层次序列 反序列化 出 二叉树 ( 非递归 )

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


class Solution1:

    # 二叉树上的最长等差数列 Q1
    def solve_Q1(self, root, diff):

        self.max_len = 0

        p = root
        c_len = 0  # 当前等差数列的长度
        prev_val = int(root.val) - diff  # 数列中上一个元素的值: eg. 2-2=0

        self.__process(p, c_len, prev_val, diff)

        return self.max_len

    def __process(self, p, c_len, prev_val, diff):

        # 递归结束条件
        if p is None:
            return

        if int(p.val) - prev_val == diff:

            c_len += 1

            if c_len > self.max_len:
                self.max_len = c_len

            self.__process(p.left, c_len, int(p.val), diff)
            self.__process(p.right, c_len, int(p.val), diff)


class Solution2:

    # 二叉树上的最长等差数列 Q2
    def solve_Q2(self, root, diff):

        p = root
        c_len = 0  # 当前等差数列的长度
        prev_val = int(root.val) - diff  # prev_val: 数列中上一个元素的值:
        # root.val=2  diff=2  prev_val= 2-2 = 0
        self.__process_pos(p, c_len, prev_val, diff)

        prev_val = int(root.val) + diff
        # root.val=2  diff=2  prev_val = 2+2 = 4
        self.__process_neg(p, c_len, prev_val, diff)

        self.max_path_len = 0
        self.__process_merge(p)

        return self.max_path_len

    def __process_pos(self, p, c_len, prev_val, diff):
        """
        找出最长的递增(正向)路径

        :param p: 当前节点
        :param c_len: 当前等差数列的长度
        :param prev_val: 上一个节点的值
        :param diff: 差值
        :return:
        """
        # 递归结束条件
        if p is None:
            return c_len

        # 延续之前的路径
        if int(p.val) - prev_val == diff:  # 当前节点的值 - 上一个节点的值 == 差值

            c_len += 1

            # 向左走
            left_len = self.__process_pos(p.left, c_len, int(p.val), diff)

            # 向右走
            right_len = self.__process_pos(p.right, c_len, int(p.val), diff)

            p.pos_path_len = max(left_len, right_len)

            return p.pos_path_len

        # 从当前p 开始新的路径
        else:
            root = p
            c_len = 0
            prev_val = int(root.val) - diff
            self.__process_pos(root, c_len, prev_val, diff)

            return 0

    def __process_neg(self, p, c_len, prev_val, diff):
        """
       找出最长的递减(反向)路径

       :param p: 当前节点
       :param c_len: 当前等差数列的长度
       :param prev_val: 上一个节点的值
       :param diff: 差值
       :return:
        """

        # 递归结束条件
        if p is None:
            return c_len

        # 延续之前的路径
        if (int(p.val) - prev_val) == -diff:

            c_len += 1

            # 从p 开始向左走
            left_len = self.__process_neg(p.left, c_len, int(p.val), diff)

            # 从p 开始向右走
            right_len = self.__process_neg(p.right, c_len, int(p.val), diff)

            p.neg_path_len = max(left_len, right_len)

            return p.neg_path_len

        # 从当前p 开始新的路径
        else:
            root = p
            c_len = 0
            prev_val = int(root.val) + diff
            self.__process_neg(root, c_len, prev_val, diff)

            return 0

    def __process_merge(self, p):
        """
        对于 某个节点, 把它的最长递增路径 和 最长递减路径 进行拼接,
        得到这个节点的最长总路径

        :param p:
        :return:
        """

        # 递归结束条件
        if p is None:
            return

        merge_path_len = (p.pos_path_len + p.neg_path_len) - 1

        if merge_path_len > self.max_path_len:
            self.max_path_len = merge_path_len

        self.__process_merge(p.left)
        self.__process_merge(p.right)



class Test:

    def test_build_tree(self):
        sol = BFS_Serialize()

        # Case1
        h_str = '8,6,10,5,7,9,11,#,#,#,#,#,#,#,#'

        tree = sol.deserialize(h_str)

        print(sol.serialize(tree))

        assert sol.serialize(tree) == h_str

        h_str = ''
        tree = sol.deserialize(h_str)

        assert sol.serialize(tree) == h_str

        # Case2
        h_str = '8,6,10,#,#,9,11,#,#,#,#'

        tree = sol.deserialize(h_str)

        print(sol.serialize(tree))

        assert sol.serialize(tree) == h_str

    def test_solve_Q1(self):

        build = BFS_Serialize()

        sol = Solution1()

        # Case1
        h_str = '2,0,4,-2,10,6,9,9,-4,#,9,8,2,9,2,#,#,#,#,#,#,#,#,#,#,#,#,#,#'
        tree = build.deserialize(h_str)
        # assert build.serialize(tree) == h_str

        assert sol.solve_Q1(tree, 2) == 4

        # Case2
        h_str = '2,#,#'
        tree = build.deserialize(h_str)

        assert sol.solve_Q1(tree, 2) == 1

    def test_solve_Q2(self):

        build = BFS_Serialize()

        sol = Solution2()

        # Case1
        h_str = '3,0,4,-2,2,6,6,9,-4,4,#,8,2,8,10,#,#,#,#,#,#,#,#,#,#,#,#,#,#'
        tree = build.deserialize(h_str)

        # print(sol.solve_Q2(tree, 2))
        assert sol.solve_Q2(tree, 2) == 5

        # Case2
        h_str = '2,0,4,-2,10,6,9,9,-4,#,9,8,2,2,2,#,#,#,#,#,#,#,#,#,#,#,#,#,#'
        tree = build.deserialize(h_str)

        assert sol.solve_Q2(tree, 2) == 7

        # Case3
        h_str = '2,#,#'
        tree = build.deserialize(h_str)

        assert sol.solve_Q2(tree, 2) == 1

if __name__ == '__main__':
    # IDE 测试 阶段：

    test = Test()

    # test.test_solve_Q1()

    test.test_solve_Q2()
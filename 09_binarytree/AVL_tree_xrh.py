

# upgrade the BST tree to balanced tree
# ref:
# http://www.cnblogs.com/linxiyue/p/3659448.html

class Node(object):
    def __init__(self,key):
        self.val=key
        self.left=None
        self.right=None
        self.height=0 # the height of the son-tree +1

class AVLTree(object):
    def __init__(self):
        self.root=None
    def find(self,key):
        if self.root is None:
            return None
        else:
            return self._find(key,self.root)
    def _find(self,key,node):
        if node is None:
            return None
        elif key<node.key:
            return self._find(key,self.left)
        elif key>node.key:
            return self._find(key,self.right)
        else:
            return node
    def findMin(self):
        if self.root is None:
            return None
        else:
            return self._findMin(self.root)
    def _findMin(self,node):
        if node.left:
            return self._findMin(node.left)
        else:
            return node
    def findMax(self):
        if self.root is None:
            return None
        else:
            return self._findMax(self.root)
    def _findMax(self,node):
        if node.right:
            return self._findMax(node.right)
        else:
            return node
    def height(self,node):
        if node is None:
            return -1
        else:
            return node.height

    def roateLeft(self, node):
        K1 = node.left
        node.left = K1.right
        K1.right = node

        node.height = max(self.height(node.left), self.height(node.right)) + 1
        K1.height = max(self.height(K1.left), self.height(K1.right)) + 1

        return K1

    def roateRight(self, node):
        K1 = node.right
        node.right = K1.left
        K1.left = node
        node.height = max(self.height(node.left), self.height(node.right)) + 1
        K1.height = max(self.height(K1.left), self.height(K1.right)) + 1

        return K1

    def doubleleft(self, node):
        node.left = self.roateRight(node.left)
        return self.roateLeft(node)

    def doubleright(self, node):
        node.right = self.roateLeft(node.right)
        return self.roateRight(node)

    def put(self, key):
        if not self.root:
            self.root = Node(key)
        else:
            self.root = self._put(self.root,key)

    def _put(self,node,key):
        if node is None:
            node = Node(key)
            # node.height=1

        elif key <= node.val:
            node.left = self._put(node.left, key)
            if self.height(node.left)-self.height(node.right)==2: #
                if self.height(node.left.left)>=self.height(node.left.right):
                    node=self.roateLeft(node)
                else:
                    node=self.doubleleft(node)

        else:
            node.right =self. _put(node.right, key)
            if self.height(node.right)-self.height(node.left) == 2:
                if  self.height(node.right.right)>=self.height(node.right.left):
                    node = self.roateRight(node)
                else:
                    node = self.doubleright(node)

        node.height=max(self.height(node.left),self.height(node.right))+1
        return node

    def delete(self, key):
        self.root = self.remove(self.root,key)

    def remove(self, node, key):

        if node is None:

            raise Exception('Error,key not in tree')

        if key < node.val:
            node.left = self.remove(node.left, key)
            if self.height(node.left) - self.height(node.right) == 2:  #
                if self.height(node.left.left)>=self.height(node.left.right):
                    node = self.roateLeft(node)
                else:
                    node = self.doubleleft(node)
            node.height = max(self.height(node.left), self.height(node.right)) + 1
        elif key>node.val:
            node.right = self.remove(node.right, key)
            if self.height(node.right) - self.height(node.left) == 2:
                if self.height(node.right.right)>=self.height(node.right.left):
                    node = self.roateRight(node)
                else:
                    node = self.doubleright(node)
            node.height = max(self.height(node.left), self.height(node.right)) + 1
        elif key==node.val:
            if node.left and node.right:
                if self.height(node.left)>= self.height(node.right):
                    left_max = self._findMax(node.left)
                    node.val=left_max.val
                    node.left =self.remove(node.left,left_max.val)
                elif self.height(node.left) < self.height(node.right):
                    right_min = self._findMin(node.right)
                    node.val = right_min.val
                    node.right =self.remove(node.right, right_min.val)
                node.height = max(self.height(node.left), self.height(node.right)) + 1
            elif node.left:
                node=node.left
            elif node.right:
                node=node.right
            else:
                node=None

        return node

def preorder(tree, nodelist=None):
    if nodelist is None:
        nodelist = []
    if tree:
        nodelist.append(tree.val)
        preorder(tree.left, nodelist)
        preorder(tree.right, nodelist)
    return nodelist

if __name__ == '__main__':

    nums = [16,3,7,11,9,26,18,14,15]
    AVL_obj=AVLTree()

    for i in range(0,len(nums)):
        ele=nums[i]
        AVL_obj.put(ele)

    result=preorder(AVL_obj.root)
    print(result)

    AVL_obj.delete(16)
    AVL_obj.delete(15)
    AVL_obj.delete(11)

    result=preorder(AVL_obj.root)
    print(result)

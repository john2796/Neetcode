# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right


class Solution:
    def invertTree(self, root: Optional[TreeNode]) -> Optional[TreeNode]:
        if not root:
            return root
        # swap
        root.left, root.right = root.right, root.left

        self.invertTree(root.left)
        self.invertTree(root.right)
        return root


# 3 ways to solve maxDepth
# Recursive DFS (template)
class Solution:
    def maxDepth_recursive_DFS(self, root: TreeNode) -> int:
        if not root:
            return 0
        return 1 + max(
            self.maxDepth_recursive_DFS(root.left),
            self.maxDepth_recursive_DFS(root.right),
        )
  

# Iterative DFS (template)
class Solution:
    def maxDepth_iterative_DFS(self, root: TreeNode) -> int:
        stack = [[root, 1]]
        res = 0

        while stack:
            node, depth = stack.pop()
            if node:
                res = max(res, depth)  # add logic here
                stack.append([node.left, depth + 1])
                stack.append([node.right, depth + 1])
        return res


# BFS (template)
class Solution:
    def maxDepth_BFS(self, root: TreeNode) -> int:
        q = deque()
        if root:
            q.append(root)
        level = 0
        while q:
            for i in range(len(q)):
                node = q.popleft()
                if node.left:
                    q.append(node.left)
                if node.right:
                    q.append(node.right)
            level += 1  # add logic for level here
        return level


class Solution:
    def diameterOfBinaryTree(self, root: Optional[TreeNode]) -> int:
        res = 0

        def dfs(root):
            nonlocal res
            if not root:
                return 0
            # add height of left+right
            left = dfs(root.left)
            right = dfs(root.right)
            res = max(res, left + right)
            return 1 + max(left, right)

        dfs(root)
        return res


class Solution:
    def isBalanced(self, root: Optional[TreeNode]) -> bool:
        def dfs(root):
            if not root:
                return [True, 0]
            left = dfs(root.left)
            right = dfs(root.right)
            balanced = left[0] and right[0] and abs(left[1] - right[1]) <= 1
            return [balanced, 1 + max(left[1], right[1])]

        return dfs(root)[0]


class Solution:
    def isSameTree(self, p: Optional[TreeNode], q: Optional[TreeNode]) -> bool:
        if not p and not q:
            return True
        if p and q and p.val == q.val:
            return self.isSameTree(p.left, q.left) and self.isSameTree(p.right, q.right)
        else:
            return False

class Solution:
    def isSubtree(self, r: Optional[TreeNode], s: Optional[TreeNode]) -> bool:
        if not s:
            return True
        if not r:
            return False
        if self.isSameTree(r, s):
            return True
        return self.isSubtree(r.left, s) or self.isSubtree(r.right, s)
    
    def isSameTree(self, p: Optional[TreeNode], q: Optional[TreeNode]) -> bool:
        if not p and not q:
            return True
        if p and q and p.val == q.val:
            return self.isSameTree(p.left, q.left) and self.isSameTree(p.right, q.right)
        return False


# inorder travelsal + bst
class Solution:
    def balanceBST(self, root: TreeNode) -> TreeNode:
        self.sortedArr = []
        self.inorderTraverse(root)
        return self.sortedArrToBST(0, len(self.sortedArr) - 1)

    def inorderTraverse(self, root) -> None:
        if not root:
            return
        self.inorderTraverse(root.left)
        self.sortedArr.append(root)
        self.inorderTraverse(root.right)
    # bst
    def sortedArrToBST(self, l: int, r: int) -> TreeNode:
        if l > r:
            return None
        m = (l+r) // 2
        root = self.sortedArr[m]
        root.left = self.sortedArrToBST(l, m - 1)
        root.right = self.sortedArrToBST(m + 1, r)
        return root


# BFS level order traversal
class Solution:
    def levelOrder(self, root: Optional[TreeNode]) -> List[List[int]]:
        # bfs
        q = deque()
        if root:
            q.append(root)
        res = []
        while q:
            lvl = []
            for i in range(len(q)):
                node = q.popleft()
                lvl.append(node.val)
                if node.left:
                    q.append(node.left)
                if node.right:
                    q.append(node.right)
            res.append(lvl)
        return res

class Solution:
    def rightSideView(self, root: Optional[TreeNode]) -> List[int]:
        q = deque()
        if root:
            q.append(root)

        res = []
        while q:
            rightSide = None
            for i in range(len(q)):
                node = q.popleft()
                if node:
                    rightSide = node
                    q.append(node.left)
                    q.append(node.right)
            # store value from every level
            if rightSide:
                res.append(rightSide.val)
        return res


# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
        while True:
            if root.val < p.val and root.val < q.val:
                root = root.right
            elif root.val > p.val and root.val > q.val:
                root = root.left
            else:
                return root

# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def goodNodes(self, root: TreeNode) -> int:
        def dfs(node, maxVal):
            if not node:
                return 0
            res = 1 if node.val >= maxVal else 0 # increment good nodes if current node is greater than max val
            maxVal = max(maxVal, node.val) # store the max val

            # count the good nodes
            res += dfs(node.left, maxVal)
            res += dfs(node.right, maxVal)
            return res
        return dfs(root, root.val)

# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right

# Given root of binary tree, determine if it's valid (left all < curr, right all > curr)
# Inorder traversal & check if prev >= curr, recursive

class Solution:
    def isValidBST(self, root: Optional[TreeNode]) -> bool:
        def dfs(node, left, right):
            if not node:
                return True
            if not (node.val < right and node.val > left):
                return False
            return dfs(node.left, left , node.val) and dfs(node.right, node.val , right)
        return dfs(root, float("-inf"), float("inf"))
    
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def kthSmallest(self, root: Optional[TreeNode], k: int) -> int:
        stack = []
        curr = root
        while stack or root:
            while curr:
                stack.append(curr)
                curr = curr.left
            curr = stack.pop()
            k -= 1
            if k == 0:
                return curr.val
            curr = curr.right
        
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right

# preorder -> will always be the root 
# inorder -> go through three in order left to right
"""
preorder = [x, x, x, x, 7]
                       r
inorder = [x, x, x, x, 7]
                      m
                  l       r
   3 
9   20
    / \
   15   7
"""
class Solution:
    def buildTree(self, preorder: List[int], inorder: List[int]) -> Optional[TreeNode]:
        if not preorder or not inorder:
            return None
        
        root = TreeNode(preorder[0])
        mid = inorder.index(preorder[0])
        root.left = self.buildTree(preorder[1 : mid + 1], inorder[:mid])
        root.right = self.buildTree(preorder[mid + 1 :], inorder[mid+1 :])
        return root


# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def maxPathSum(self, root: Optional[TreeNode]) -> int:
        res = [root.val]
        def dfs(node):
            if not node:
                return 0
            l = max(dfs(node.left), 0)
            r = max(dfs(node.right), 0)

            res[0] = max(res[0], node.val + l + r)
            return node.val + max(l, r)
            
        dfs(root)
        return res[0]

# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
# searalize, dfs throught root and store values in res then return a joined one string , to deserialize split the join string dfs to create new bst
class Codec:

    def serialize(self, root):
        """Encodes a tree to a single string.
        
        :type root: TreeNode
        :rtype: str
        """
        res = []
        def dfs(node):
            if not node:
                res.append("N")
                return
            res.append(str(node.val))
            dfs(node.left)
            dfs(node.right)
        dfs(root)
        return ",".join(res)
        

    def deserialize(self, data):
        """Decodes your encoded data to tree.
        
        :type data: str
        :rtype: TreeNode
        """
        vals = data.split(",")
        self.i = 0
        def dfs():
            if vals[self.i] == "N":
                self.i += 1
                return None
            node = TreeNode(int(vals[self.i]))
            self.i += 1
            node.left = dfs()
            node.right = dfs()
            return node
        return dfs()
        

# Your Codec object will be instantiated and called as such:
# ser = Codec()
# deser = Codec()
# ans = deser.deserialize(ser.serialize(root))
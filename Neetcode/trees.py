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
       self.right = right

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
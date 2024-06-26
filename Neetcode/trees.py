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

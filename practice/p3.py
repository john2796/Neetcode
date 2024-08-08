# ---- Arrays & Hashing
# contains duplicate
class Solution:
    def containsDuplicate(self, nums: List[int]) -> bool:
        s = {}
        for n in nums:
            if n in s:
                return True
            else:
                s[n] = n
        return False


# valid anagram
# two sum
# group anagram
# top k frequent elements
# encode and decode strings
# product of array except self
# valid sudoku
# longest consecutive sequence

# ---- Two Pointers
# valid palindrome
# two sum II input array is sorted
# 3sum
# container with most water
# trapping rain water

# ---- Sliding Window
# best time to buy and sell stock
# longest substring without repeating characters
# longest repeating character replacement
# permutation in string
# minimum window substring
# sliding window maximum

# ---- Stack
# valid parenthesis
class Solution:
    def isValid(self, s:str) -> bool:
        map = {")":"(","]":"[","}":"{"}
        stack = []
        for c in s:
            if c in map:
                if stack and stack[-1] == map[c]:
                    stack.pop()
                else:
                    return False
            else:
                stack.append(c)
        return True if not stack else False
# min stack
class MinStack:
    def __init__(self):
        self.s = []
        self.min_stack = []

    def push(self, val:int) -> None:
        self.s.append(val)
        min_val = min(val, self.min_stack[-1] if self.min_stack else val)
        self.min_stack.append(min_val)

    def pop(self) -> None:
        self.s.pop()
        self.min_stack.pop()

    def top(self) -> int:
        return self.s[-1]
    
    def getMin(self) -> int:
        return self.min_stack[-1]

# evaluate reverse polish notation
class Solution:
    def evalRPN(self, tokens: List[str]) -> int:
        s = []
        for t in tokens:
            if t == "+":
                v1, v2 = s.pop(), s.pop()
                s.append(v1 + v2)
            elif t == "-":
                v1, v2 = s.pop(), s.pop()
                s.append(v2 - v1)
            elif t == "*":
                v1, v2 = s.pop(), s.pop()
                s.append(int(v2) * int(v1))
            elif t == "/":
                v1, v2 = s.pop(), s.pop()
                s.append(int(float(v2) / v1))
            else:
                s.append(int(t))
        return s[0]

# generate parenthesis
class Solution:
    def generateParenthesis(self, n:int) -> List[int]:
        s = []
        res = []
        def backtrack(openN, closedN):
            if openN == closedN == n:
                res.append("".join(s))
                return
            if openN < n:
                s.append("(")
                backtrack(openN + 1, closedN)
                s.pop()
            if closedN < openN:
                s.append(")")
                backtrack(openN, closedN + 1)
                s.pop()
            backtrack(0,0)
            return res
# daily temperatures
class Solution:
    def dailyTemperatures(self, temperatures:List[int]) -> List[int]:
        s =[] # pair: [temp, indx]
        res = [0] * len(temperatures)
        for i, t in enumerate(temperatures):
            while s and t > s[-1][0]:
                sT, sI = s.pop()
                res[sI] = i - sI
            s.append((t, i))
        return res
# car fleet
class Solution:
    def carFleet(self, target:int, position:List[int], speed: List[int]) -> int:
        pair = [(p,s) for p,s in zip(position, speed)]
        pair.sort(reverse=True)
        stack = []
        for p, s in pair: # reversed sorted order
            stack.append((target - p) / s)
            if len(stack) >= 2 and stack[-1] <= stack[-2]:
                stack.pop()
        return len(stack)
# largest rectangle in historgram
class Solution:
    def largestRectangleArea(self, heights:List[int]) -> int:
        max_area = 0
        stack = []
        for i, h in enumerate(heights):
            idx = i
            if not stack:
                stack.append([i, h])
                continue
            while stack and stack[-1][1] > h:
                idx, he = stack.pop()
                max_area = max(max_area, (i - idx) * he)
            stack.append([idx, h])
        while stack:
            idx, he = stack.pop()
            max_area = max(max_area, (i - idx + 1) * he)
        return max_area

# ----- Binary Search -----
# binary search
class Solution:
    def search(self, nums:List[int], target: int) -> int:
        l, r = 0, len(nums) - 1
        while l <= r:
            m = (r+l) // 2
            if nums[m] == target:
                return m
            elif nums[m] < target:
                l = m + 1
            else:
                r = m - 1
        return -1
# search a 2d matrix
class Solution:
    def searchMatrix(self, matrix: List[List[int]], target: int) -> bool:
        if not matrix: 
            return False
        m, n = len(matrix), len(matrix[0])
        l, r = 0, m * n - 1
        while l <= r:
            mid = (l + r) // 2
            mid_row, mid_col = divmod(mid, n)
            if matrix[mid_row][mid_col] == target:
                return True
            elif matrix[mid_row][mid_col] < target:
                l = m + 1
            else:
                r = m - 1
        return False
# koko eating banana
class Solution:
    def minEatingSpeed(self, piles: List[int], h: int) -> int:
        l, r = 1, max(piles)
        def isSufficientSpeed(cnt):
            return sum(ceil(i/cnt) for i in piles) <= h
        while l < r:
            m = (l + r) // 2
            if isSufficientSpeed(m):
                r = m
            else:
                l = m + 1
        return l
# find minimum in rotated sorted array
class Solution:
    def findMin(self, nums: List[int]) -> int:
        l, r = 0, len(nums) - 1
        while l < r:
            m = l + (r - l) // 2
            if nums[m] < nums[r]:
                r = m
            else:
                l = m + 1
        return nums[l]
# search in rotated sorted array
class Solution:
    def search(self, nums: List[int], target: int) -> int:
        l, r = 0, len(nums) - 1
        while l <= r:
            m = (l+r) // 2 
            if nums[m] == target:
                return m
            if nums[l] <= nums[m]:
                if nums[l] <= target < nums[m]:
                    r = m - 1
                else:
                    l = m + 1
            else:
                if nums[m] < target <= nums[r]:
                    l = m + 1
                else:
                    r = m - 1
        return -1
# time based key value store
class TimeMap:
    def __init__(self):
        self.dic = {}
    def set(self, key:str, value:str, timestamp:int) ->None:
        if key not in self.dic:
            self.dic[key] = []
        self.dic[key].append([value, timestamp])
    def get(self, key:str, timestamp:int) -> str:
        res = ""
        values = self.dic.get(key, [])
        l, r = 0, len(values) - 1
        while l <= r:
            m = (l+r) // 2
            if values[m][1] <= timestamp:
                l = m + 1
                res = values[m][0]
            else:
                r = m - 1
        return res
# median of two sorted arrays
class Solution:
    def findMedianSortedArrays(self, nums1:List[int],nums2:List[int]) -> float:
        nums = nums1 + nums2
        nums.sort()
        l,r = 0, len(nums) - 1
        m = (r+l) // 2
        if (len(nums) % 2) != 0:
            return nums[m]
        else:
            return (nums[m] + nums[m+1]) / 2
# -----Linked List
# reverse linked list
class Solution:
    def reverseList(self,head:Optional[ListNode]) -> Optional[ListNode]:
        prev, curr = None, head
        while curr:
            tmp = curr.next
            curr.next = prev
            prev = curr
            curr = tmp
        return prev
# merge two sorted lists
class Solution:
    def mergeTwoLists(self, l1:Optional[ListNode],l2:Optional[ListNode]) -> Optional[ListNode]:
        dummy = node = ListNode()
        while l1 and l2:
            if l1.val < l2.val:
                node.next = l1
                l1 = l1.next
            else:
                node.next = l2
                l2 = l2.next
            node = node.next
        node.next = l1 or l2
        return dummy.next
# reorder list
class Solution:
    def reorderList(self, head:Optional[ListNode]) -> None:
        slow, fast = head, head
        while fast.next and fast.next.next:
            slow = slow.next
            fast = fast.next.next
        prev,cur = None, slow.next
        while cur:
            tmp = cur.next
            cur.next = prev
            prev = cur
            cur = tmp
        slow.next = None
        first, second = head, prev
        while second:
            tmp1, tmp2 = first.next, second.next
            first.next = second
            second.next = tmp1
            first, second = tmp1, tmp2
# remove nth node from end of list
class Solution:
    def removeNthFromEnd(self, head:Optional[ListNode],n:int) -> Optional[ListNode]:
        dummy = ListNode(0, head)
        left = dummy
        cur = head
        while n > 0:
            cur = cur.next
            n -= 1
        while cur:
            cur = cur.next
            left = left.next
        left.next = left.next.next
        return dummy.next
# copy list with random pointer
class Solution:
    def copyRandomList(self,head:Optional[Node]) -> Optional[Node]:
        oldToCopy = {None:None}
        cur = head
        while cur:
            copy = Node(cur.val)
            oldToCopy[cur] = copy
            cur = cur.next
        cur = head
        while cur:
            copy = oldToCopy[cur]
            copy.next = oldToCopy[cur.next]
            copy.random = oldToCopy[cur.random]
            cur = cur.next
        return oldToCopy[head]
# add two numbers
class Solution:
    def addTwoNumbers(self, l1:Optional[ListNode],l2:Optional[ListNode]) -> Optional[ListNode]:
        dummy = ListNode()
        cur = dummy
        carry = 0
        while l1 or l2 or carry:
            v1 = l1.val if l1 else 0
            v2 = l2.val if l2 else 0
            val = v1 + v2 + carry
            carry = val // 10
            val = val % 10
            cur.next = ListNode(val)
            cur = cur.next
            l1 = l1.next if l1 else None
            l2 = l2.next if l2 else None
        return dummy.next
# linked list cycle
class Solution:
    def hasCycle(self, head:Optional[ListNode]) -> bool:
        fast, slow = head, head
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next
            if slow == fast:
                return True
        return False
# find the duplicate number
class Solution:
    def findDuplicate(self, nums:List[int]) -> int:
        slow, fast = 0, 0
        while True:
            slow = nums[slow]
            fast = nums[numst[fast]]
            if slow == fast:
                break
        slow2 = 0
        while True:
            slow = nums[slow]
            slow2=nums[slow2]
            if slow2 == slow:
                return slow
# lru cache
class Node:
    def __init__(self, key, val):
        self.key, self.val = key, val
        self.prev = self.next = None
class LRUCache:
    def __init__(self, capacity:int):
        self.cap = capacity
        self.cache = {}
        self.left, self.right = Node(0, 0), Node(0, 0)
        self.left.next, self.right.prev = self.right, self.left
    def remove(self, node):
        prev, nxt = node.prev, node.next
        prev.next, nxt.prev = nxt, prev
    def insert(self, node):
        prev, nxt = self.right.prev, self.right
        prev.next = nxt.prev = node
        node.next, node.prev = nxt, prev
    def get(self, key:int) -> int:
        if key in self.cache:
            self.remove(self.cache[key])
            self.insert(self.cache[key])
            return self.cache[key].val
        return -1
    def put(self, key:int, value:int) -> None:
        if key in self.cache:
            self.remove(self.cache[key])
        self.cache[key] = Node(key, value)
        self.insert(self.cache[key])
        if len(self.cache) > self.cap:
            lru = self.left.next
            self.remove(lru)
            del self.cache[lru.key]
# merge k sorted lists
# reverse nodes in k group
# ----- Trees
# invert binary tree
class Solution:
    def invertTree(self, root: Optional[TreeNode]) -> Optional[TreeNode]:
        if not root:
            return root
        root.left, root.right = root.right, root.left
        self.invertTree(root.left)
        self.invertTree(root.right)
# max depth 3 ways:
class Solution:
    # recursive dfs
    def maxdepth_recursive_dfs(self,root:TreeNode) -> int:
        if not root:
            return 0
        return 1 + max(
                self.maxdepth_recursive_dfs(root.left),
                self.maxdepth_recursive_dfs(root.right),
                )
    # iterative dfs
    def maxdepth_iterative_dfs(self, root: TreeNode) -> int:
        stack = [[root, 1]]
        res =  0
        while stack:
            node, depth = stack.pop()
            if node:
                res = max(res, depth)
                stack.append([node.left, depth + 1])
                stack.append([node.right, depth + 1])
        return res
    # bfs
    def maxdepth_bfs(self, root:TreeNode) -> int:
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
            level += 1
        return level
# diameter of binary tree
class Solution:
    def diameterOfBinaryTree(self, root:Optional[TreeNode]) -> int:
        res = 0
        def dfs(root):
            nonlocal res
            if not root:
                return 0
            left = dfs(root.left)
            right = dfs(root.right)
            res = max(res, left + right)
            return 1 + max(left, right)
        dfs(root)
        return res
# balanced binary tree
class Solution:
    def isBalanced(self, root:Optional[TreeNode]) -> bool:
        def dfs(root):
            if not root:
                return [True, 0]
            left = dfs(root.left)
            right = dfs(root.right)
            balanced = left[0] and right[0] and abs(left[1] - right[1]) <= 1
            return [balanced, 1 + max(left[1], right[1])]
        return dfs(root)[0]
# same tree
class Solution:
    def isSameTree(self,p:Optional[TreeNode], q:Optional[TreeNode]) -> bool:
        if not p and not q:
            return True
        if p and q and p.val == q.val:
            return self.isSameTree(p.left, q.left) and self.isSameTree(p.right, q.right)
        else:
            return False
# subtree of another tree
class Solution:
    def isSubtree(self, r:Optional[TreeNode],s:Optional[TreeNode]) -> bool:
        if not s:
            return True
        if not r:
            return False
        if self.isSameTree(r, s):
            return True
        return self.isSubtree(r.left, s) or self.isSubtree(r.right, s)
    def isSameTree(self, p:Optional[TreeNode],q:Optional[TreeNode]) -> bool:
        if not p and not q:
            return True
        if p and q and p.val == q.val:
            return self.isSameTree(p.left,q.left) and self.isSameTree(p.right, q.right)
        return False
# lowest common ancestor of a binary search tree
class Solution:
    def lowestCommonAncestor(self,root:TreeNode,p:TreeNode,q:TreeNode) -> TreeNode:
        while True:
            if root.val < p.val and root.val < q.val:
                root = root.right
            elif root.val > p.val and root.val > q.val:
                root = root.left
            else:
                return root
# balanced bst
class Solution: # inorder traversal + bst
    def balancedBST(self,root:TreeNode) -> TreeNode:
        self.sortedArr = []
        self.inorderTraverse(root)
        return sortedArrToBST(0, len(self.sortedArr) - 1)
    def inorderTraverse(self, root) -> None:
        if not root:
            return
        self.inorderTraverse(root.left)
        self.sortedArr.append(root)
        self.inorderTraverse(root.right)
    def sortedArrToBST(self, l:int, r:int) -> TreeNode:
        if l > r:
            return None
        m = (l+r) // 2
        root = self.sortedArr[m]
        root.left = self.sortedArrToBST(l, m - 1)
        root.right = self.sortedArrToBST(m + 1, r)
        return root
# binary tree level order travelsal
class Solution:
    def levelOrder(self, root: Optional[TreeNode]) -> List[List[int]]:
        # BFS level order traversal
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
# binary tree right side view
class Solution:
    def rightSideView(self, root:Optional[TreeNode]) -> List[int]:
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
            if rightSide:
                res.append(rightSide.val)
        return res
# count good nodes in binary tree
class Solution:
    def goodNodes(self, root:TreeNode) -> int:
        def dfs(node, maxVal):
            if not node:
                return 0
            res = 1 if node.val >= maxVal else 0
            maxVal = max(maxVal, node.val)
            res += dfs(node.left, maxVal)
            res += dfs(node.right, maxVal)
            return res
        return dfs(root, root.val)
# validate binary tree
class Solution:
    def isValidBST(self, root:Optional[TreeNode]) -> bool:
        def dfs(node,left,right):
            if not node:
                return True
            if not (node.val < right and node.val > left):
                return False
            return dfs(node.left, left, node.val) and dfs(node.right, node.val, right)
        return dfs(root, float("-inf"), float("inf"))
# kth smallest element in a bst
class Solution:
    def kthSmallest(self, root:Optional[TreeNode], k:int) -> int:
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
# construct binary tree from preorder and inorder travelsal
class Solution:
    def buildTree(self, preorder: List[int], inorder:List[int]) -> Optional[TreeNode]:
        if not preorder or not inorder:
            return None
        root = TreeNode(preorder[0])
        mid = inorder.index(preorder[0])
        root.left = self.buildTree(preorder[1:mid+1], inorder[mid+1:])
        root.right = self.buildTree(preorder[mid+1:], inorder[mid+1:])
        return root
# binary tree maximum path sum
class Solution:
    def maxPathSum(self, root:Optional[TreeNode]) -> int:
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
# serialize and deserialize binary tree
class Codec:
    def serialize(self, root):
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

# ----- Heap / Priority Queue
# kth largest element in a stream
class KthLargest:
    def __init__(self, k:int, nums:List[int]):
        self.minHeap, self.k = nums, k
        heapq.heapify(self.minHeap)
        while len(self.minHeap) > k:
            heapq.heappop(self.minHeap)
    def add(self, val:int) -> int:
        heapq.heappush(self.minHeap, val)
        if len(self.minHeap) > self.k:
            heapq.heappop(self.minHeap)
        return self.minHeap[0]
# last stone weight
class Solution:
    def lastStoneWeight(self, stones:List[int]) -> int:
        stones = [-s for s in stones]
        heapq.heapify(stones)
        while len(stones) > 1:
            x = heapq.heappop(stones)
            y = heapq.heappop(stones)
            if y > x:
                heapq.heappush(stones, x - y)
        stones.append(0)
        return abs(stones[0])
# k closest points to origin
class Solution:
    def kClosest(self, points:List[List[int]], k:int) -> List[List[int]]:
        minHeap = []
        for x,y in points:
            dist = (x ** 2) + (y ** 2)
            minHeap.append((dist, x, y))
        heapq.heapify(minHeap)
        res = []
        for _ in range(k):
            _, x, y = heapq.heappop(minHeap)
            res.append((x, y))
        return res
# Kth Largest element in the array
class Solution:
    def findKthLargest(self, nums:List[int], k:int) -> int:
        heap = nums[:k]
        heapq.heapify(heap)
        for num in nums[k:]:
            if num > heap[0]:
                heapq.heappop(heap)
                heapq.heappush(heap, num)
        return heap[0]
# task scheduler
class Solution:
    def leastInterval(self, tasks: List[str], n:int) ->int:
        c = Counter(tasks)
        maxHeap = [-cnt for cnt in c.values()]
        heapq.heapify(maxHeap)
        time = 0
        q = deque()
        while maxHeap or q:
            time += 1
            if maxHeap:
                cnt = 1 + heapq.heappop(maxHeap)
                if cnt:
                    q.append([cnt, time + n])
            if q and q[0][1] == time:
                heapq.heappush(maxHeap, q.popleft()[0])
        return time
# design twitter
# find median data stream
# ----- Backtracking
# subsets
class Solution:
  def subsets(self, nums: List[int]) -> List[List[int]]:
    res = []
    subset = []
    def dfs(i):
      if i >= len(nums):
        res.append(subset.copy())
        return
      subset.append(nums[i])
      dfs(i + 1)
      subset.pop()
      dfs(i + 1)
    dfs(0)
    return res
# combination sum
class Solution:
  def combinationSum(self, candidates: List[int], target: int) -> List[List[int]]:
    res = []
    def dfs(i, cur, total):
      if total == target:
        res.append(cur.copy())
        return
      if i >= len(candidates) or total > target:
        return
      cur.append(candidates[i])
      dfs(i, cur, total + candidates[i])
      cur.pop()
      dfs(i + 1, cur, total)
    dfs(0, [], 0)
    return res
# permutation
class Solution:
  def permute(self, nums: List[int]) -> List[List[int]]:
    res = []
    if len(nums) == 1:
      return [nums[:]]
    for i in range(len(nums)):
      n = nums.pop(0)
      perms = self.permute(nums)
      for perm in perms:
        perm.append(n)
      res.extend(perms)
      nums.append(n)
    return res
# subsets II
class Solution:
  def subsetsWithDup(self, nums: List[int]) -> List[List[int]]:
    res = []
    subset = []
    nums.sort()
    def dfs(i):
      if i >= len(nums):
        res.append(subset.copy())
        return 
      subset.append(nums[i])
      dfs(i + 1)
      subset.pop()
      while i + 1 < len(nums) and nums[i] == nums[i + 1]:
        i += 1
      dfs(i + 1)
    dfs(0)
    return res
# combination sum II
class Solution:
  def combinationSum2(self, c: List[int], t:int) -> List[List[int]]:
    c.sort()
    res = []
    def backtrack(cur, pos, target):
      if target == 0:
        res.append(cur.copy())
        return
      if target <= 0:
        return
      prev = -1
      for i in range(pos, len(c)):
        if c[i] == prev:
          continue
        cur.append(c[i])
        backtrack(cur, i + 1, target - c[i])
        cur.pop()
        prev = c[i]
    backtrack([], 0, t)
    return res
# word search
class Solution:
    def exist(self, board: List[List[str]], word: str) -> bool:
        ROWS, COLS  =len(boar), len(board[0])
        path = set()
        def dfs(r, c, i):
            if i == len(word):
                return True
            if (
                min(r, c) < 0
                or r >= ROWS
                or c >= COLS
                or word[i] != board[r][c]
                or (r, c) in path
            ):
                return False
            path.add((r, c))
            res = (
                dfs(r + 1, c, i + 1)
                or dfs(r - 1, c, i + 1)
                or dfs(r, c + 1, i + 1)
                or dfs(r, c - 1, i + 1)
            )
            path.remove((r, c))
            return res
        count = defaultdict(int, sum(map(Counter, board), Counter()))
        if count[word[0]] > count[word[-1]]:
            word = word[::-1]
        for r in range(ROWS):
            for c in range(COLS):
                if dfs(r, c, 0):
                    return True
        return False

# palindrome partitioning
class Solution:
  def partition(self, s:str) -> List[List[str]]:
    res, part = [], []
    def dfs(i):
      if i >= len(s):
        res.append(part.copy())
        return
      for j in range(i, len(s)):
        if self.isPali(s,i,j):
          part.append(s[i:j+1])
          dfs(j+1)
          part.pop()
    dfs(0)
    return res
  def isPali(self,s,l,r):
    while l < r:
      if s[l] != s[r]:
        return False
      l, r = l+1, r-1
    return True
# letter combination of a phone number
class Solution:
  def letterCombinations(self, digits:str) -> List[str]:
    res = []
    digitToChar = {
      "2":"abc",
      "3":"def",
      "4":"ghi",
      "5":"jkl",
      "6":"mno",
      "7":"qprs",
      "8":"tuv",
      "9":"wxyz",
    }
    def backtrack(i, curStr):
      if len(curStr) == len(digits):
        res.append(curStr)
        return
      for c in digitToChar[digits[i]]:
        backtrack(i + 1, curStr + c)
    if digits:
      backtrack(0, "")
    return res
# n queens
class Solution:
  def solveNQueens(self, n:int) -> List[List[str]]:
    col = set()
    posDiag = set()
    negDiag = set()
    res = []
    board = [["."] * n for i in range(n)]
    def backtrack(r):
      if r == n:
        copy = ["".join(row) for row in board]
        res.append(copy)
        return
      for c in range(n):
        if c in col or (r + c) in posDiag or (r - c) in negDiag:
          continue
        col.add(c)
        posDiag.add(r + c)
        negDiag.add(r - c)
        board[r][c] = "Q"
        backtrack(r + 1)
        col.remove(c)
        posDiag.remove(r + c)
        negDiag.remove(r - c)
        board[r][c] = "."
    backtrack(0)
    return res
# ----- Tries
# implement trie prefix
class TrieNode:
  def __init__(self):
    self.children = [None] * 26
    self.end = False
class Trie:
  def __init__(self):
    self.root = TrieNode()

  def insert(self, word: str) -> None:
    curr = self.root
    for c in word:
      i = ord(c) - ord("a")
      if curr.children[i] == None:
        curr.children[i] = TrieNode()
      curr = curr.children[i]
    curr.end = True

  def search(self, word:str) -> bool:
    curr = self.root
    for c in word:
      i = ord(c) - ord("a")
      if curr.children[i] == None:
        return False
      curr = curr.children[i]
    return curr.end

  def startsWith(self, prefix) -> bool:
    curr = self.root
    for c in prefix:
      i = ord(c) - ord("a")
      if curr.children[i] == None:
        return False
      curr = curr.children[i]
    return True

# design add and search words data
class TrieNode2:
  def __init__(self):
    self.children = {}
    self.end = False
class WordDictionary:
  def __init__(self):
    self.root = TrieNode2()
    
  def addWord(self, word:str) -> None:
    cur = self.root
    for c in word:
      if c not in cur.children:
        cur.children[c] = TrieNode2()
      cur = cur.children[c]
    cur.word = True
    
  def search(self, word:str) -> bool:
    def dfs(j, root):
      cur = root
      for i in range(j, len(word)):
        c = word[i]
        if c == ".":
          for child in cur.children.values():
            if dfs(i + 1, child):
              return True
          return False
        else:
          if c not in cur.children:
            return False
          cur = cur.children[c]
      return cur.word
    return dfs(0, self.root)
# word search II
class TrieNode3:
  def __init__(self):
    self.children = {}
    self.isWord = False
    self.refs = 0
  
  def addWord(self, word):
    cur = self
    cur.refs += 1
    for c in word:
      if c not in cur.children:
        cur.children[c] = TrieNode3()
      cur = cur.children[c]
      cur.refs += 1
    cur.isWord = True
  
  def removeWord(self, word):
    cur = self
    cur.refs -= 1
    for c in word:
      if c in cur.children:
        cur = cur.children[c]
        cur.refs -= 1
class Solution:
  def findWords(self, board:List[List[str]], words:List[str]) -> List[str]:
    # dfs + trie
    root = TrieNode3()
    for w in words:
      root.addWord(w)
    rows, cols = len(board), len(board[0])
    res, visit = set(), set()
    def dfs(r, c, node, word):
      if (
        r not in range(rows) 
        or c not in range(cols) 
        or board[r][c] not in node.children
        or node.children[board[r][c]].refs < 1
        or (r, c) in visit
      ):
        return
      visit.add((r, c))
      node = node.children[board[r][c]]
      word += board[r][c]
      if node.isWord:
        node.isWord = False
        res.add(word)
        root.removeWord(word)
      dfs(r + 1, c, node, word)
      dfs(r - 1, c, node, word)
      dfs(r, c + 1, node, word)
      dfs(r, c - 1, node, word)
      visit.remove((r, c))

    for r in range(rows):
      for c in range(cols):
        dfs(r, c, root, "")
    return list(res)

# ----- Graphs
# number of islands
class Solution:
  def numIsland(self, grid:List[List[str]]) -> int:
    if not grid or not grid[0]:
      return 0
    islands = 0
    visit = set()
    rows, cols = len(grid), len(grid[0])
    def dfs(r, c):
      if (
        r not in range(rows)
        or c not in range(cols)
        or grid[r][c] == "0"
        or (r, c) in visit
      ):
        return
      visit.add((r, c))
      directions = [[0, 1], [0, -1], [1, 0], [-1, 0]]
      for dr, dc in directions:
        dfs(r + dr, c + dc)
    for r in range(rows):
      for c in range(cols):
        if grid[r][c] == "1" and (r, c) not in visit:
          islands += 1
          dfs(r, c)
    return islands
# max area of island
class Solution:
  def maxAreaOfIsland(self, grid: List[List[int]]) -> int:
    rows, cols = len(grid), len(grid[0])
    visit = set()
    def dfs(r, c):
      if (
        r not in range(rows)
        or c not in range(cols)
        or grid[r][c] == 0
        or (r, c) in visit
      ): 
        return 0
      visit.add((r, c))
      return 1 + dfs(r + 1, c) + dfs(r - 1, c) + dfs(r, c + 1) + dfs(r, c - 1)
    area =0 
    for r in range(rows):
      for c in range(cols):
        area = max(area, dfs(r, c))
    return area
# clone graph
def cloneGraph(self, node: Node) -> Node:
  oldToNew ={}
  def dfs(node):
    if node in oldToNew:
      return oldToNew[node]
    copy = Node(node.val)
    oldToNew[node] = copy
    for nei in node.neighbors:
      copy.neighbors.append(dfs(nei))
    return copy
  return dfs(node) if node else None
# walls and gates
def walls_and_gates(self, rooms: List[List[int]]):
  rows, cols = len(rooms), len(rooms[0])
  visit = set()
  q = deque()
  def addRooms(r, c):
    if (
      r not in range(rows)
      or c not in range(cols)
      or rooms[r][c] == -1
      or (r, c) in visit
    ):
      return
    visit.add((r, c))
    q.append([r, c])

  for r in range(rows):
    for c in range(cols):
      if rooms[r][c] == 0:
        q.append([r, c])
        visit.add((r, c))
  dist = 0
  while q:
    for i in range(len(q)):
      r, c = q.popleft()
      addRooms(r + 1, c)
      addRooms(r - 1, c)
      addRooms(r, c + 1)
      addRooms(r, c - 1)
    dist += 1
# rotting oranges
def orangesRotting(self, grid: List[List[int]]) -> int:
  rows, cols = len(grid), len(grid[0])
  q = collections.deque()
  fresh = 0
  time = 0
  for r in range(rows):
    for c in range(cols):
      if grid[r][c] == 1:
        fresh += 1
      if grid[r][c] == 2:
        q.append((r, c))
  directions = [[0, 1], [0, -1], [1, 0], [-1, 0]]
  while q and fresh > 0:
    for i in range(len(q)):
      r, c = q.popleft()
      for dr, dc in directions:
        row, col = r + dr, c + dc
        if (
          row in range(rows)
          and col in range(cols)
          and grid[row][col] == 1
        ):
          grid[row][col] = 2
          q.append((row, col))
          fresh -= 1
    time += 1
  return time if fresh == 0 else -1
# pacific atlantic water flow
def pacificAtlantic(self, heights: List[List[int]]) -> List[List[int]]:
  rows, cols = len(heights), len(heights[0])
  pac, atl = set(), set()
  def dfs(r, c, visit, prevHeight):
    if (
      r not in range(rows)
      or c not in range(cols)
      or heights[r][c] < prevHeight
      or (r, c) in visit
    ):
        return
    visit.add((r, c))
    dfs(r + 1, c, visit, heights[r][c])
    dfs(r - 1, c, visit, heights[r][c])
    dfs(r, c + 1, visit, heights[r][c])
    dfs(r, c - 1, visit, heights[r][c])

  for c in range(cols):
    dfs(0, c, pac, heights[0][c])
    dfs(rows - 1, c, atl, heights[rows - 1][c])
  
  for r in range(rows):
    dfs(r, 0, pac, heights[r][0])
    dfs(r, cols - 1, atl, heights[r][cols - 1])
  res = []
  for r in range(rows):
    for c in range(cols):
      if (r, c) in pac and (r, c) in atl:
        res.append([r, c])
  return res
# surrounded regions
def solve(self, board:List[List[int]]) -> None:
  rows, cols = len(board), len(board[0])
  def dfs(r, c):
    if (
      r not in range(rows)
      or c not in range(cols)
      or board[r][c] != "O"
    ):
      return
    board[r][c] = "T"
    dfs(r + 1, c)
    dfs(r - 1, c)
    dfs(r, c + 1)
    dfs(r, c - 1)
  for r in range(rows):
    for c in range(cols):
      if board[r][c] == "O" and (r in [0, rows - 1] or c in [0, cols - 1]):
        dfs(r, c)
  
  for r in range(rows):
    for c in range(cols):
      if board[r][c] == "O":
        board[r][c] = "X"

  for r in range(rows):
    for c in range(cols):
      if board[r][c] == "T":
        board[r][c] = "O"
# course schedule
def canFinish(self, numCourses:int, prerequisites: List[List[int]]) -> bool:
  preMap = {i: [] for i in range(numCourses)}
  for crs, pre in prerequisites:
    preMap[crs].append(pre)
  visiting = set()
  def dfs(crs):
    if crs in visiting:
      return False
    if preMap[crs] == []:
      return True
    visiting.add(crs)
    for nei in preMap[crs]:
      if not dfs(nei):
        return False
    visiting.remove(crs)
    preMap[crs] = []
    return True
  for c in range(numCourses):
    if not dfs(c):
      return False
  return True
# course schedule II
def findOrder(self, numCourses: int, prerequisites: List[List[int]]) -> List[int]:
  prereq = {i: [] for i in range(numCourses)}
  for crs, pre in prerequisites:
    prereq[crs].append(pre)
  output = []
  visit, cycle = set(), set()
  def dfs(crs):
    if crs in cycle:
      return False
    if crs in visit:
      return True
    cycle.add(crs)
    for nei in prereq[crs]:
      if dfs(nei) == False:
        return False
    cycle.remove(crs)
    visit.add(crs)
    output.append(crs)
    return True
  for c in range(numCourses):
    if dfs(c) == False:
      return []
  return output
# graph valid tree
def validTree(self, n, edges):
  if not n:
    return True
  adj = {i: [] for i in range(n)}
  for n1, n2 in edges:
    adj[n1].append(n2)
    adj[n2].append(n1)
  visit = set()
  def dfs(i, prev):
    if i in visit:
      return False
    visit.add(i)
    for j in adj[i]:
      if j == prev:
        continue
      if not dfs(j, i):
        return False
    return True
  return dfs(0, -1) and n == len(visit)
# number of connected components in an undirected graph
# redundant connection
# word ladder

# ---- Advanced Graphs
# reconstruct itinerary
# min cost to connect all points
# network delay time
# swim in rising water
# alien dictionary
# cheapest flights within k stops

# ---- 1-D DP
# climbing stairs
def stairs(self, n: int) -> int:
   if n <= 3:
      return n
   n1, n2 = 2, 3
   for i in range(4, n + 1):
      temp = n1 + n2
      n1 = n2
      n2 = temp
   return n2
# min cost climbing stairs
def minCostClimbingStairs(self, cost: List[int]) -> int:
   for i in range(len(cost) -3, -1, -1):
      cost[i] += min(cost[i + 1], cost[i + 2])
   return min(cost[0], cost[1])
# house robber
def rob(self, nums:List[int]) -> int:
   rob1, rob2 = 0, 0
   for n in nums:
      temp = max(n + rob1, rob2)
      rob1 = rob2
      rob2 = temp
   return rob2
# house robber II
class Solution:
    def rob2(self, nums:List[int]) -> int:
        return max(nums[0], self.helper(nums[1:]), self.helper(nums[:-1]))
    def helper(self, nums):
        rob1, rob2 = 0, 0
        for n in nums:
            newRob = max(rob1 + n, rob2)
            rob1 = rob2
            rob2 = newRob
        return rob2
# longest palindromic substring
class Solution:
    def longestPalindrome(self, s:str) -> str:
        res = ""
        resLen = 0
        for i in range(len(s)):
            l, r = i, i
            while l >= 0 and r < len(s) and s[l] == s[r]:
                if (r - l + 1) > resLen:
                    res = s[l:r+1]
                    resLen = r - l + 1
                l -= 1
                r += 1
            l, r = i, i + 1
            while l >= 0 and r < len(s) and s[l] == s[r]:
                if (r - l + 1) > resLen:
                    res = s[l : r + 1]
                    resLen = r - l + 1
                l -= 1
                r += 1
        return res
# palindromic substrings
class Solution:
    def countSubstrings(self, s:str) -> int:
        res = 0
        for i in range(len(s)):
            res += self.countPali(s, i, i)
            res += self.countPali(s, i, i + 1)
        return res
    def countPali(self, s, l, r):
        res = 0
        while l >= 0 and r < len(s) and s[l] == s[r]:
            res += 1
            l -= 1
            r += 1
        return res
# decode ways
class Solution:
    def numDecoding(self, s:str) -> int:
        # memoization
        dp = {len(s): 1}
        def dfs(i):
            if i in dp:
                return dp[i]
            if s[i] == "0":
                return 0
            res = dfs(i + 1)
            if i + 1 < len(s) and (
                s[i] == "1" or s[i] == "2" and s[i + 1] in "0123456"
            ):
                res += dfs(i + 2)
                dp[i] = res
                return res
        return dfs(0)
    def dpNumDecoding(self, s:str) -> int:
        # dynamic programming
        dp = {len(s): 1}
        for i in range(len(s) -1, -1, -1):
            if s[i] == "0":
                dp[i] = 0
            else:
                dp[i] = dp[i + 1]
            if i + 1 < len(s) and (
                s[i] == "1" or s[i] == "2" and s[i + 1] in "0123456"
            ):
                dp[i] += dp[i + 1]
        return dp[0]
# coin change
class Solution:
    def coinChange(self, coins:List[List[int]], amount: int) -> int:
        dp = [amount + 1] * (amount + 1)
        dp[0] = 0
        for a in range(1, amount + 1):
            for c in coins:
                if a - c >= 0:
                    dp[a] = min(dp[a], 1 + dp[a - c])
        return dp[amount] if dp[amount] != amount + 1 else -1
# maximum product subarray
class Solution:
    def maxProduct(self, nums:List[int]) -> int:
        res = nums[0]
        curMin, curMax = 1, 1
        for n in nums:
            tmp = curMax * n 
            curMax = max(n * curMax, n * curMin, n)
            curMin = min(tmp, n * curMin, n)
            res = max(res, curMax)
        return res
# word break
class Solution:
    def wordBreak(self, s:str, wordDict: List[str]) -> bool:
        dp = [False] * (len(s) + 1)
        dp[len(s)] = True
        for i in range(len(s) -1, -1, -1):
            for w in wordDict:
                if (i + len(w)) <= len(s) and s[i : i + len(w)] == w:
                    dp[i] = dp[i + len(w)]
                if dp[i]:
                    break
        return dp[0]
# longest increasing subsequence
class Solution:
    def lengthOfLIS(self, nums: List[int]) -> int:
        LIS = [1] * len(nums)
        for i in range(len(nums) - 1, -1, -1):
            for j in range(i + 1, len(nums)):
                if nums[i] < nums[j]:
                    LIS[i] = max(LIS[i], 1 + LIS[j])
        return max(LIS)
# partition equal subset sum

# ---- 2-D DP
# unique paths
# longest common subsequence
# best time to buy and sell stock with cooldown
# coin change II
# target sum
# interleaving string
# longest increasing path in a matrix
# distinct subsequences
# edit distance
# burst balloons
# regular expression matching

# ---- Greedy
# Maximum subbaray
# jump game
# jump game II
# gas station
# hand of straights
# merge triplets to form target triplet
# partition labels
# valid parenthesis string

# ---- Intervals
# insert interval
# merge intervals
# non overlapping intervals
# meeting rooms
# meeting rooms II
# minimum interval to include each query

# ---- Math & Geometry
# rotate image
# spiral mamtrix
# set matrix zeroes
# happy number
# plus one
# pow(x, n)
# multiply strings
# detect squares

# ---- Bit Manipulation
# single number
# number of 1 bits
# counting bits
# reverse bits
# missing number
# sum of two integers
# reverse integer

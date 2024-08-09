
from collections import deque
import heapq
from typing import Counter, List 

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
# design twitter: use hash maps and heap to merge tweets from followed users, keeping track of the most recent tweets.
class Twitter:
  def __init__(self):
    self.count = 0
    self.tweetMap = defaultdict(list)
    self.followMap = defaultdict(list)
    
  def postTweet(self, userId:int, tweetId:int) -> None:
    self.tweetMap[userId].append([self.count, tweetId])
    self.count -= 1
    
  def getNewsFeed(self, userId:int) -> List[int]:
    res = []
    minHeap = []
    self.followMap[userId].add(userId)
    for followeeId in self.followMap[userId]:
      if followeeId in self.tweetMap:
        index = len(self.tweetMap[followeeId]) - 1
        count, tweetId = self.tweetMap[followeeId][index]
        heapq.heappush(minHeap, [count, tweetId, followeeId, index - 1])
    while minHeap and len(res) < 10:
      count, tweetId, followeeId, index = heapq.heappop(minHeap)
      res.append(tweetId)
      if index >= 0:
        count, tweetId = self.tweetMap[followeeId][index]
        heapq.heappush(minHeap, [count, tweetId, followeeId, index - 1])
    return res
  
  def follow(self,followerId:int, followeeId: int) -> None:
    self.followMap[followerId].add(followeeId)
  
  def unfollow(self, followerId: int, followeeId:int) -> None:
    self.followMap[followerId].remove(followeeId)

# find median data stream: use two heaps (max-heap and min-heap) to balance the lower and upper halves of the data to find the median efficiently
class MedianFinder:
  def __init__(self):
    self.small, self.large = [], []
  
  def addNum(self, num: int) -> None:
    if self.large and num > self.large[0]:
      heapq.heappush(self.large, num)
    else:
      heapq.heappush(self.small, -1 * num)
    if len(self.small) > len(self.large) + 1:
      val = -1 * heapq.heappop(self.small)
      heapq.heappush(self.large, val)
    if len(self.large) > len(self.small) + 1:
      val = heapq.heappop(self.large)
      heapq.heappush(self.small, -1 * val)
      
  def findMedian(self) -> float:
    if len(self.small) > len(self.large):
      return -1 * self.small[0]
    elif len(self.large) > len(self.small):
      return self.large[0]
    return (-1 * self.small[0] + self.large[0]) / 2.0
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
class UnionFind:
  def __init__(self):
    self.f = {}
  def findParent(self, x):
    y = self.f.get(x,x)
    if x != y:
      y = self.f[x] = self.findParent(y)
    return y
  def union(self, x, y):
    self.f[self.findParent(x)] = self.findParent(y)

class Solution:
  def countComponents(self, n:int, edges: List[List[int]]) -> int:
    dsu = UnionFind()
    for a, b in edges:
      dsu.union(a, b)
    return len(set(dsu.findParent(x) for x in range(n)))
# redundant connection
class Solution:
  def findRedundantConnection(self, edges: List[List[int]]) -> List[int]:
    par = [i for i in range(len(edges) + 1)]
    rank = [1] * (len(edges) + 1)
    def find(n):
      p = par[n]
      while p != par[p]:
        par[p] = par[par[p]]
        p = par[p]
      return p
    def union(n1, n2):
      p1, p2 = find(n1), find(n2)
      if p1 == p2:
        return False
      if rank[p1] > rank[p2]:
        par[p2] = p1
        rank[p1] += rank[p2]
      else:
        par[p1] = p2
        rank[p2] += rank[p1]
      return True
    for n1, n2 in edges:
      if not union(n1, n2):
        return [n1, n2]
# word ladder
def ladderLength(self, beginWord: str, endWord: str, wordList: List[str]) -> int:
  if endWord not in wordList:
    return 0
  nei = collections.defaultdict(list)
  wordList.append(beginWord)
  for word in wordList:
    for j in range(len(word)):
      pattern = word[:j] + "*" + word[j + 1:]
      nei[pattern].append(word)
  visit = set([beginWord])
  q = deque([beginWord])
  res = 1
  while q:
    for i in range(len(q)):
      word = q.popleft()
      if word == endWord:
        return res
      for j in range(len(word)):
        pattern = word[:j] + "*" + word[j + 1:]
        for neiWord in nei[pattern]:
          if neiWord not in visit:
            visit.add(neiWord)
            q.append(neiWord)
    res += 1
  return 0
# ---- Advanced Graphs
# reconstruct itinerary
class Solution:
  def findItinerary(self, tickets:List[List[str]]) -> List[str]:
    graph = defaultdict(list)
    for src, dst in sorted(tickets, reverse=True):
      graph[src].append(dst)
    itinerary = []
    def dfs(airport):
      while graph[airport]:
        dfs(graph[airport].pop())
      itinerary.append(airport)
    dfs("JFK")
    return itinerary[::-1]
# min cost to connect all points
class Solution:
  def minCostConnectPoints(self, points: List[List[int]]) -> int:
    N = len(points)
    adj = {i: [] for i in range(N)}
    for i in range(N):
      x1, y1 = points[i]
      for j in range(i + 1, N):
        x2, y2 = points[j]
        dist = abs(x1 - x2) + abs(y1 - y2)
        adj[i].append([dist, j])
        adj[i].append([dist, i])
    # prims
    res = 0
    visit = set()
    minH = [[0, 0]] # [cost, point]
    while len(visit) < N:
      cost, i = heapq.heappop(minH)
      if i in visit:
        continue
      res += cost
      visit.add(i)
      for neiCost, nei in adj[i]:
        if nei not in visit:
          heapq.heappush(minH, [neiCost, nei])
    return res
# network delay time
class Solution:
  def networkDelayTime(self, times: List[List[int]], n: int, k:int) -> int:
    # create adjacency list + prims (uses minheap, visited, add result explore neighbors if not in visit add to minHeap) 
    edges = collections.defaultdict(list)
    for u,v,w in times:
      edges[u].append((v, w))
    minHeap = [(0, k)]
    visit = set()
    t = 0
    while minHeap:
      w1, n1 = heapq.heappop(minHeap)
      if n1 in visit:
        continue
      visit.add(n1)
      t1 = w1
      for n2, w2 in edges[n1]:
        if n2 not in visit:
          heapq.heappush(minHeap, (w1 + w2, n2))
    return t if len(visit) == n else -1
# swim in rising water
class Solution:
  def swimInWater(self, grid:List[List[int]]) -> int:
    N = len(grid)
    visit = set()
    minH = [[grid[0][0], 0, 0]] # (time/max-height, r, c)
    directions = [[0,1], [0,-1], [1,0], [-1, 0]]
    visit.add((0, 0))
    while minH:
      t, r, c = heapq.heappop(minH)
      if r == N - 1 and c == N - 1:
        return t
      for dr, dc in directions:
        neiR, neiC = r + dr, c + dc
        if (
          neiR < 0
          or neiC < 0
          or neiR == N
          or neiC == N
          or (neiR, neiC) in visit
        ):
          continue
        visit.add((neiR, neiC))
        heapq.heappush(minH, [max(t, grid[neiR][neiC]), neiR, neiC])
    return 0 # to get error line out
# alien dictionary
class Solution:
  def alienOrder(self, words: List[str]) -> str:
    adj = {char: set() for word in words for char in word}
    for i in range(len(words) - 1):
      w1, w2 = words[i], words[i + 1]
      minLen = min(len(w1), words[i + 1])
      if len(w1) > len(w2) and w1[:minLen] == w2[:minLen]:
        return ""
      for j in range(minLen):
        if w1[j] != w2[j]:
          adj[w1[j]].add(w2[j])
          break
    visited = {}
    res = []
    def dfs(char):
      if char in visited:
        return visited[char]
      visited[char] = False
      res.append(char)
    for char in adj:
      if dfs(char):
        return ""
    res.reverse()
    return "".join(res)
# cheapest flights within k stops
class Solution:
  def findCheapestPrice(self,n:int,flights:List[List[int]], src:int, dst:int,k:int) -> int:
    prices = [float("inf")] * n
    prices[src] = 0
    for i in range(k + 1):
      tmpPrices = prices.copy()
      for s,d,p in flights: # s=source, d=dest, p=price
        if prices[s] == float("inf"):
          continue
        if prices[s] + p < tmpPrices[d]:
          tmpPrices[d] = prices[s] + p
      prices = tmpPrices
    return -1 if prices[dst] == float("inf") else prices[dst]
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
# partition equal subset sum: Use DP to determine if the array can be partitioned into two subsets with equal sums.
class Solution:
  def canPartition(self, nums: List[int]) -> bool:
    if sum(nums) % 2:
      return False
    dp = set()
    dp.add(0)
    target = sum(nums) // 2
    for i in range(len(nums) -1, -1, -1):
      nextDP = set()
      for t in dp:
        if (t + nums[i]) == target:
          return True
        nextDP.add(t + nums[i])
        nextDP.add(t)
      dp = nextDP
    return False
# ---- 2-D DP
# unique paths: use DP to calculate the number of ways to reach each cell by summing the ways from the top and left cells.
class Solution:
  def uniquePaths(self, m:int, n:int) -> int:
    row = [1] * n
    for i in range(m - 1):
      newRow = [1] * n
      for j in range(n - 2, -1, -1):
        newRow[j] = newRow[j + 1] + row[j]
      row = newRow
    return row[0] # O(n * m) O(n)
# longest common subsequence: Use DP to compare two strings and build the longest subsequence common to both
class Solution:
  def longestCommonSubsequence(self, text1:str, text2:str) -> int:
    dp = [[0 for j in range(len(text2) + 1)] for i in range(len(text1) + 1)]
    for i in range(len(text1) -1, -1, -1):
      for j in range(len(text2) -1, -1, -1):
        if text1[i] == text2[j]:
          dp[i][j] = 1 + dp[i + 1][j + 1]
        else:
          dp[i][j] = max(dp[i][j +1], dp[i + 1][j])
    return dp[0][0]
# best time to buy and sell stock with cooldown: Use dp to maximize profit while considering the cooldown period after selling
class Solution:
  def maxProfit(self, prices: List[int]) -> int:
    dp = {}
    def dfs(i, buying):
      if i >= len(prices):
        return 0
      if (i, buying) in dp:
        return dp[(i, buying)]
      cooldown = dfs(i + 1, buying)
      if buying:
        buy = dfs(i + 1, not buying) - prices[i]
        dp[(i, buying)] = max(buy, cooldown)
      else:
        sell = dfs(i + 2, not buying) - prices[i]
        dp[(i, buying)] = max(sell, cooldown)
      return dp[(i, buying)]
    return dfs(0, True)
# coin change II: Use DP to count the number of ways to make the amount with the given coins, considering combinations
class Solution:
  def change_memo(self, amount:int, coins:List[int]) -> int:
    cache = {}
    def dfs(i, a):
      if a == amount:
        return 1
      if a > amount:
        return 0
      if i == len(coins):
        return 0
      if (i, a) in cache:
        return cache[(i, a)]
      cache[(i, a)] = dfs(i, a + coins[i]) + dfs(i + 1)
      return cache[(i, a)]
    return dfs(0, 0)
  
  def change_dp1(self, amount: int, coins: List[int]) -> int:
      dp = [[0] * (len(coins) + 1) for i in range(amount + 1)]
      dp[0] = [1] * (len(coins) + 1)
      for a in range(1, amount + 1):
          for i in range(len(coins) -1, -1, -1):
              dp[a][i] = dp[a][i + 1]
              if a - coins[i] >= 0:
                  dp[a][i] += dp[a - coins[i]][i]
      return dp[amount][0]

  def change_dp2(self, amount:int, coins:List[int]) -> int:
    dp = [0] * (amount + 1)
    dp[0] = 1
    for i in range(len(coins) -1, -1, -1):
      nextDP = [0] * (amount + 1)
      nextDP[0] = 1
      for a in range(1, amount + 1):
        nextDP[a] = dp[a]
        if a - coins[i] >= 0:
          nextDP[a] += nextDP[a - coins[i]]
      dp = nextDP
    return dp[amount]
# target sum: Use DP to calculate the number of ways to assign symbols to make the sum equal to the target.
class Solution:
  def findTargetSumWays(self, nums:List[int], target:int) -> int:
    dp = {} # (index, total) -> # of ways
    def backtrack(i, total):
      if i == len(nums):
        return 1 if total == target else 0
      if (i, total) in dp:
        return dp[(i, total)]
      dp[(i, total)] = backtrack(i + 1, total + nums[i]) + backtrack(i + 1, total - nums[i])
      return dp[(i, total)]
    return backtrack(0, 0)
# interleaving string: Use DP to check if the third string is an interleaving of the other two by maintaining a 2D table.
class Solution:
  def isInterleave(self, s1:str, s2:str, s3:str) -> bool:
    if len(s1) + len(s2) != len(s3):
      return False
    dp = [[False] * (len(s2 + 1) for i in range(len(s1) + 1))]
    dp[len(s1)][len(s2)] = True
    for i in range(len(s1), -1, -1):
      for j in range(len(s2), -1, -1):
        if i < len(s1) and s1[i] == s3[i + j] and dp[i + 1][j]:
          dp[i][j] = True
        if j < len(s2) and s2[j] == s3[i + j] and dp[i][j + 1]:
          dp[i][j] = True
    return dp[0][0]
# longest increasing path in a matrix: Use DP with memoization to find the longest increasing path in the matrix by exploring all directions.
class Solution:
  def longestIncreasingPath(self, matrix:List[List[int]]) -> int:
    rows, cols = len(matrix), len(matrix[0])
    dp = {} # (r, c) -> LIP
    def dfs(r, c, prevVal):
      if r < 0 or r == rows or c < 0 or c == cols or matrix[r][c] <= prevVal:
        return 0
      if (r, c) in dp:
        return dp[(r, c)]
      res = 1
      res = max(res, 1 + dfs(r + 1, c, matrix[r][c]))
      res = max(res, 1 + dfs(r -1, c, matrix[r][c]))
      res = max(res, 1 + dfs(r, c + 1, matrix[r][c]))
      res = max(res, 1 + dfs(r, c - 1, matrix[r][c]))
      dp[(r, c)] = res
      return res
    for r in range(rows):
      for c in range(cols):
        dfs(r, c, -1)
    return max(dp.values())
# distinct subsequences: Use DP to count the number of distinct subsequences of one string that equals the other string.
class Solution:
  def numDistinct(self, s:str, t:str) -> int:
    cache = {}
    for i in range(len(s) + 1):
      cache[(i, len(t))] = 1
    for j in range(len(t)):
      cache[(len(s), j)] = 0
    for j in range(len(s) -1, -1, -1):
      for j in range(len(t) -1, -1, -1):
        if s[i] == t[j]:
          cache[(i, j)] = cache[(i + j, j + 1)] + cache[(i + 1, j)]
        else:
          cache[(i, j)] = cache[(i + 1, j)]
    return cache[(0, 0)]
# edit distance: Use DP to calculate the minimum number of operations required to convert one string into another.
class Solution:
  def minDistance(self, word1: str, word2: str) -> int:
    dp = [[float("inf")] * (len(word2) + 1) for i in range(len(word1) + 1)]
    for j in range(len(word2) + 1):
      dp[len(word1)][j] = len(word2) - j
    for i in range(len(word1) + 1):
      dp[i][len(word2)] = len(word1) - 1
    for i in range(len(word1) -1, -1, -1):
      for j in range(len(word2) -1, -1, -1):
        if word1[i] == word2[j]:
          dp[i][j] = dp[i + 1][j + 1]
        else:
          dp[i][j] = 1 + min(dp[i + 1][j], dp[i][j + 1], dp[i + 1][j + 1])
    return dp[0][0]
# burst balloons: Use DP to calculate the maximum coins that can be obtained by strategically bursting balloons. 
class Solution:
  def maxCoins(self, nums: List[int]) -> int:
    cache = {}
    nums = [1] + nums + [1]
    for offset in range(2, len(nums)):
      for left in range(len(nums) - offset):
        right = left + offset
        for pivot in range(left + 1, right):
          coins = nums[left] * nums[pivot] * nums[right]
          coins += cache.get((left,pivot), 0) + cache.get((pivot, right), 0)
          cache[(left, right)] = max(coins, cache.get((left, right), 0))
    return cache.get((0, len(nums) -1), 0)
# regular expression matching: Use DP to match a string against a pattern that includes '.' and '\*' as wildcards. bottom-up dp
class Solution:
  # Bottom up
  def isMatch(self, s:str, p:str) -> bool:
    cache = [[False] * (len(p) + 1) for i in range(len(s) + 1)] 
    cache[len(s)][len(p)] = True
    for i in range(len(s), -1, -1):
      for j in range(len(p) -1, -1, -1):
        match = i < len(s) and (s[i] == p[j] or p[j] == ".")
        if (j + 1) < len(p) and p[j + 1] == "*":
          cache[i][j] = cache[i][j + 2]
          if match:
            cache[i][j] = cache[i + 1][j] or cache[i][j]
        elif match:
          cache[i][j] = cache[i + 1][j + 1]
    return cache[0][0]
  # Top Down
  def isMatch_topdown(self, s:str, p:str) -> bool:
    cache = {}
    def dfs(i, j):
      if (i, j) in cache:
        return cache[(i, j)]
      if i >= len(s) and j >= len(p):
        return True
      if j >= len(p):
        return False
      match = i < len(s) and (s[i] == p[j] or p[j] == ".")
      if (j + 1) < len(p) and p[j + 1] == "*":
        cache[(i, j)] = dfs(i, j + 2) or (match and dfs(i + 1, j))
        return cache
      if match:
        cache[(i, j)] = dfs(i + 1, j + 1)
        return cache[(i, j)]
      cache[(i, j)] = False
      return False
    return dfs(0, 0)
  
# ---- Greedy
<<<<<<< HEAD
# Maximum subbaray: Use a greedy approach (Kadane's algorithm) to find the subarray with the maximum sum.
class Solution:
    def maxSubarray(self, nums: List[int]) -> int:
        res = nums[0]
        total = 0
        for n in nums:
            total += n
            res = max(res, total)
            if total < 0:
                total = 0
        return res
# jump game: Use a greedy approach to keep track of the farthest position
class Solution:
    def canJump(self, nums:List[int]) -> bool:
        goal = len(nums) - 1
        for i in range(len(nums) -2, -1, -1):
            if 1 + nums[i] >= goal:
                goal = i
        return goal == 0
# jump game II: Use a greedy approach to minimize the number of jumps needed to reach the end by always jumping to the farthest reachable position.
class Solution:
    def jumpII(self, nums:List[int]) -> int:
        l, r = 0, 0
        res = 0
        while r < (len(nums) - 1):
            maxJump = 0
            for i in range(l, r + 1):
                maxJump = max(maxJump, i + nums[i])
            l = r + 1
            r = maxJump
            res += 1
        return res
# gas station: Use a greedy approach to find the starting point where you can complete the circuit by checking if the total gas is sufficient.
class Solution:
    def canCompleteCircuit(self, gas: List[int], cost:List[int]) -> int:
        start, end = len(gas) - 1, 0
        total = gas[start] - cost[start]
        while start >= end:
            while total > 0 and start >= end:
                start -= 1
                total += gas[start] - cost[start]
            if start == end:
                return start
            total += gas[end] - cost[end]
            end += 1
        return -1
# hand of straights: Use a greedy approach with a frequency map to form hands by consecutively grouping cards.
class Solution:
    def isNSStraightHand(self, hand:List[int], groupSize:int) -> bool:
        if len(hand) % groupSize:
            return False
        count = {}
        for n in hand:
            count[n] = 1 + count.get(n, 0)
        minH = list(count.keys())
        heapq.heapify(minH)
        while minH:
            first = minH[0]
            for i in range(first, first + groupSize):
                if i not in count:
                    return False
                count[i] -= 1
                if count[i] == 0:
                    if i != minH[0]:
                        return False
                    heapq.heappop(minH)
        return True
# merge triplets to form target triplet: Use a greedy approach to check if the target can be formed by merging valid triplets.
# partition labels: Use a greedy approach to partition the string such that each letter appreas in at most one part, based on its last occurrence.
class Solution:
    def mergeTriplets(self, triplets: List[List[int]], target:List[int]) -> bool:
        good = set()
        for t in triplets:
            if t[0] > target[0] or t[1] > target[1] or t[2] > target[2]:
                continue
            for i,v in enumerate(t):
                if v == target[i]:
                    good.add(i)
        return len(good) == 3
# valid parenthesis string: Use a greedy approach to ensure the string can be balanced by tracking the possible number of open parentheses.
class Solution:
    def checkValidString(self, s:str) -> bool:
        dp = {(len(s), 0): True} # key=(i, leftCount) -> isValid
        def dfs(i, left):
            if i == len(s) or left < 0:
                return left == 0
            if (i, left) in dp:
                return dp[(i, left)]
            if s[i] == "(":
                dp[(i, left)] = dfs(i + 1, left + 1)
            elif s[i] == ")":
                dp[(i, left)] = dfs(i + 1, left - 1)
            else:
                dp[(i, left)] = (
                    dfs(i + 1, left + 1) or fs(i + 1, left - 1) or 
                    dfs(i + 1, left)
                )
            return dp[(i, left)]
        return dfs(0, 0)
    
=======
# Maximum subbaray: use greedy approach (Kadan'es algorithm) to find the subarray with the maximum sum.
# jump game: use a greedy approach to keep track of the farthest position you can reach and see if you can reach the end.
# jump game II: Use a greedy approach to minimize the number of jumps needed to reach th end by always jumping to the farthest reachable position.
# gas station: Use greedy approach to find the starting point where you can complete the circuit by checking if the total gas is sufficient.
# hand of straights: Use a greedy approach with a frequency map to form hands by consecutively grouping cards.
# merge triplets to form target triplet: Use a greedy
# partition labels
# valid parenthesis string

>>>>>>> 33a443b (Manual Scan)
# ---- Intervals
# insert interval: Use a greedy approach to merge the new interval into the existing list, adjusting overlaps.
class Solution:
    def insert(
        self, intervals:List[List[int]], newInterval:List[int]
    ) -> List[List[int]]:
        res = []
        for i in range(len(intervals)):
            if newInterval[1] < intervals[i][0]:
                res.append(newInterval)
                return res + intervals[i:]
            elif newInterval[0] > intervals[i][1]:
                res.append(intervals[i])
            else:
                newInterval = [
                    max(newInterval[0], intervals[i][0]),
                    max(newInterval[1], intervals[i][1])
                ]
        res.append(newInterval)
        return res
# merge intervals: Sort intervals by start time, then merge overlapping intervals by comparing end times.
class Solution:
    def merge(self, intervals: List[List[int]]) -> List[List[int]]:
        intervals.sort(key=lambda x:x[0])
        merged = [intervals[0]]
        for current in intervals[1:]:
            last = merged[-1]
            if current[0] <= last[1]:
                last[1] = max(last[1], current[1])
            else:
                merged.append(current)
        return merged
# non overlapping intervals: Use a greedy approach to remove the minimum number of intervals to avoid overlap, prioritizing intervals that end the earliest.
class Solution:
    def eraseOverlappingIntervals(self, intervals: List[List[int]]) -> int:
        intervals.sort(key=lambda x:x[1])
        count = 0
        end = float("-inf")
        for start, stop in intervals:
            if start >= end:
                end = stop
            else:
                count += 1
        return count
# meeting rooms: Sort the intervals by start time and check for overlaps to determine if all meetings can be attended.
class Solution:
    def canAttendMeetings(self, intervals):
        intervals.sort(key=lambda x:[x])
        for i in range(1, len(intervals)):
            if intervals[i][0] < intervals[i-1][1]:
                return False
        return True
# meeting rooms II: Use a min-heap to track the end times of meetings and determine the minimum of rooms required.
def minMeetingRooms(self, intervals:List[List[int]]) -> int:
    intervals.sort(key=lambda x:x[0])
    heap = []
    for interval in intervals:
        if heap and heap[0] <= interval[0]:
            heapq.heappop(heap)
        heap.heappush(heap, interval[1])
    return len(heap)
# minimum interval to include each query: Use a combination of sorting and a priority queue to find the smallest interval containing each query.
class Solution:
    def minInterval(self, intervals:List[List[int]], queries:List[int]) -> List[int]:
        intervals.sort()
        minHeap = []
        res = {}
        i = 0
        for q in sorted(queries):
            while i < len(intervals) and intervals[i][0] <= q:
                l, r = intervals[i]
                heapq.heappush(minHeap, (r - l + 1, r))
                i += 1
            while minHeap and minHeap[0][1] < q:
                heapq.heappop(minHeap)
            res[q] = minHeap[0][0] if minHeap else -1
        return [res[q] for q in queries]

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

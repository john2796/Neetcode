# ---- Arrays & Hashing
from typing import Counter, List

# contains duplicate
"""
Problem: return True if the value in nums are duplicate otherwise false 
Approach: set
"""


class Solution1:
    def containsDuplicate(self, nums: List[int]) -> bool:
        s = set()
        for n in nums:
            if n not in s:
                s.add(n)
            else:
                return True
        return False


# valid anagram
"""
Problem: True if t is an anagram of s otherwise false
Approach: check if character count of t equal s
"""


class Solution2:
    def isAnagram(self, s: str, t: str) -> bool:
        if len(s) != len(t):
            return False
        ct, cs = {}, {}

        for i in range(len(s)):
            cs[s[i]] = 1 + cs.get(s[i], 0)
            ct[t[i]] = 1 + ct.get(t[i], 0)

        return cs == ct


# two sum
"""
Problem: return the indices of two number that adds up to target, you may not use the element twice  
Approach: hashset, store current n in dictionary, then check whether t=(target - current_n) already in hashset
"""


class Solution3:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        s = {}
        for i in range(len(nums)):
            t = target - nums[i]
            if t in s:
                return [s[t], i]
            else:
                s[nums[i]] = i
        return []


# group anagram
"""
Problem: group anagrams together 
Approach: hashset + 26 alpha technique
"""


class Solution4:
    def groupAnagrams(self, strs: List[str]) -> List[List[str]]:
        # defaultdict + 26
        # [[0, 1, 0]: [""]
        c = collections.defaultdict(list)
        res = []
        for words in strs:
            a = [0] * 26
            for w in words:
                i = ord("a") - ord(w)
                a[i] += 1
            c[tuple(a)].append(words)
        return c.values()


# top k frequent elements
"""
Problem: Given an integer array nums and an integer k, return the k most frequent elements. You may return the answer in any order

Approach: count + freq + store n in freq count index + loop in reverse to get the most freq res
"""


class Solution5:
    # store count frequency in arr , loop in reverse to get most frequent when res len equal to k return values
    def topKFrequent(self, nums: List[int], k: int) -> List[int]:
        count = {}
        freq = [[] for i in range(len(nums) + 1)]

        for n in nums:
            count[n] = 1 + count.get(n, 0)
        for n, c in count.items():
            freq[c].append(n)
        # print(count, freq) #{1: 3, 2: 2, 3: 1} [[], [3], [2], [1], [], [], []]
        res = []
        for i in range(len(freq) - 1, 0, -1):
            for n in freq[i]:
                res.append(n)
                if len(res) == k:
                    return res
        return []


# product of array except self
"""
Problem: Given an integer array nums, return an array answer such that answer[i] is equal to the product of all the elements of nums except nums[i].

Approach: compute prefix sum (res[i] = res[i] * nums[i-1]), update res using postfix loop nums in reverse index
"""


class Solution:
    def productExceptSelf(self, nums: List[int]) -> List[int]:
        res = [1] * (len(nums))  # [1, 1, 1, 1]
        for i in range(1, len(nums)):
            res[i] = res[i - 1] * nums[i - 1]
        # print(res) [1, 1, 2, 6]


# valid sudoku
"""
Problem: Determine if a 9 x 9 Sudoku board is valid. Only the filled cells need to be validated according to the following rules:

Approach: use defaultdict(set) store m[r], n[c] , sub boxes
"""


class Solution:
    def isValidSudoku(self, board: List[List[str]]) -> bool:
        rows, cols = collections.defaultdict(set), collections.defaultdict(set)
        square = collections.defaultdict(set)

        for r in range(9):
            for c in range(9):
                if board[r][c] == ".":
                    continue
                if (
                    board[r][c] in rows[r]
                    or board[r][c] in cols[c]
                    or board[r][c] in square[(r // 3, c // 3)]
                ):
                    return False
                rows[r].add(board[r][c])
                cols[c].add(board[r][c])
                square[(r // 3, c // 3)].add(board[r][c])
        return True


# longest consecutive sequence
"""
Problem: Given an unsorted array of integers nums, return the length of the longest consecutive elements sequence. You must write an algorithm that runs in O(n) time.

Approach: set(nums) + find starting point (n-1) not in set, found (n+length) in s, get max(longest, length)
"""


class Solution:
    def longestConsecutive(self, nums: List[int]) -> int:
        s = set(nums)
        longest = 0
        for n in nums:
            if (n - 1) not in s:
                length = 0
                while (n + length) in s:
                    length += 1
                longest = max(length, longest)
        return longest


# ---- Two Pointers
# valid palindrome
"""
Problem: 
A phrase is a palindrome if, after converting all uppercase letters into lowercase letters and removing all non-alphanumeric characters, it reads the same forward and backward. Alphanumeric characters include letters and numbers.

Given a string s, return true if it is a palindrome, or false otherwise.

Approach:
use two pointer l=0 and r=len(s) - 1, compare value and move pointers while not s[l].isalnum() and l < r:
"""


class Solution:
    def isPalindrome(self, s: str) -> bool:
        l, r = 0, len(s) - 1

        while l < r:
            # edge case: lower case, non alpha
            while not s[l].isalnum() and l < r:
                l += 1
            while not s[r].isalnum() and l < r:
                r -= 1

            if s[l].lower() != s[r].lower():
                return False
            l += 1
            r -= 1

        return True


# two sum II input array is sorted
"""
Problem:
Given integers numbers already sorted in non-decreasing order, 
find two numbers such that they add up to a specific target number. Let these two numbers be numbers[index1] and numbers[index2] where 1 <= index1 < index2 <= numbers.length.
must use only constant extra space.

Approach:
two pointer + move pointer depending on the size of sum compared to target if sum is lower than target move left pointer otherwise move right pointer -1
"""


class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        l, r = 0, len(nums) - 1

        while l < r:
            s = nums[r] + nums[l]
            if s == target:
                return [l + 1, r + 1]
            elif s < target:
                l += 1
            else:
                r -= 1
        return []


# 3sum
"""
Problem:
return all the triplets [nums[i], nums[j], nums[k]] such that 
i != j, i != k, and j != k, and 
nums[i] + nums[j] + nums[k] == 0.

must not contain duplicate triplets.

Approach:
sort + skip n[i] == n[i -1] , 2sum, also check n[l] == n[l -1] and l <r l+=1.
"""


class Solution:
    def threeSum(self, n: List[int]) -> List[List[int]]:
        res = []
        n.sort()
        for i in range(len(n)):
            l = i + 1
            r = len(n) - 1
            if i > 0 and n[i] == n[i - 1]:
                continue
            while l < r:
                triplets = n[i] + n[l] + n[r]
                if triplets == 0:
                    res.append([n[i], n[l], n[r]])
                    l += 1
                    r -= 1
                    while n[l] == n[l - 1] and l < r:
                        l += 1
                elif triplets < 0:
                    l += 1
                else:
                    r -= 1
        return res


# container with most water
"""
Problem:
given array height of length n. n vertical lines drawn such that the two endpoints of the ith line are 
(i, 0) and (i, height[i]).

Find two lines that together with the x-axis form a container, such that the container contains the most water.

Return the maximum amount of water a container can store.
you may not slant the container.

Approach:
two pointer + calulate area + edge case h[r] <= h[l]: r-=1, both side will check if they're less than then move pointers l or r.
"""


class Solution:
    def maxArea(self, h: List[int]) -> int:
        l, r = 0, len(h) - 1
        res = 0
        while l < r:
            res = max(res, (r - l) * min(h[l], h[r]))
            if h[l] < h[r]:
                l += 1
            elif h[r] <= h[l]:
                r -= 1
        return res


# trapping rain water
"""
Problem:
    given non integers representing an elevation each bar 1,compute how much water it can trap after raining
Approach:
    use two pointer, track leftmax and right max,leftMax < rightMax, for both side res += max - height[pointer]
"""


class Solution:
    def trap(self, h: List[int]) -> int:
        if not h:
            return 0
        l, r = 0, len(h) - 1
        leftMax, rightMax = h[l], h[r]
        res = 0

        while l < r:
            if leftMax < rightMax:
                l += 1
                leftMax = max(leftMax, h[l])
                res += leftMax - h[l]
            else:
                r -= 1
                rightMax = max(rightMax, h[r])
                res += rightMax - h[r]
        return res


# ---- Sliding Window
# best time to buy and sell stock
"""
Problem:
array prices[i]  price of a given stock on the ith day.
You want to maximize your profit by choosing a single day to buy one stock and choosing a different day in the future to sell that stock.
Return the maximum profit you can achieve from this transaction. If you cannot achieve any profit, return 0.

Approach:
"""
# longest substring without repeating characters

"""
https://leetcode.com/problems/longest-repeating-character-replacement/

Approach: sliding window , expand increment char count and max frequency, shrink window while (r - l + 1) - maxf > k *window - max_frequency.
"""

# longest repeating character replacement
"""
https://leetcode.com/problems/longest-repeating-character-replacement/description/

Approach: 
sliding window expand track max frequency fill count dictionary shrink while (r - l + 1) - maxf > k 
"""
# permutation in string
"""
Problem: https://leetcode.com/problems/permutation-in-string/
Approach: 
base edge case len(s1)>len(s2) return false,
store s1 and s2 count in [0] * 26, 
track matches loop through 26 to count matches s1Count and s2Count, 
sliding window count matches for both s[r] and s[l]
"""
# minimum window substring
"""
https://leetcode.com/problems/minimum-window-substring/
Approach:
base case t == "" return "" 
use sliding window track countT, window, have ,need, res, resLen 
expand store char count, increment have if char in countT and value in countT and window equal
shrink while have == need, update result, pop from the left window
"""

# sliding window maximum

"""
Approach:
Instead of recalculating maximum for each window, we can utilize a double-ended queue (deque). The beauty of deques is their ability to add or remove elements from both ends in constant time, making them perfect for this scenerio.

1. Initialization: Begin by defining an empty deque and a result list.
2. Iterate over nums:
    - For each number, remove indices from the front of the deque if they are out of the current window's bounds.
    - Next, remove indices form the back if the numbers they point to are smaller than the current number. This ensures our deque always has the maximum of the current window at its front.
    - Add the current index to the deque.
    - If the current index indicates that we've seen at least k numbers, add the front of the deuque (i.e, the current window's maximum) to the result list.
3. Return the list
"""
class Solution:
    def maxSlidingWindow(self, nums: List[int], k: int) -> List[int]:
        if not nums:
            return []
        if k == 1:
            return nums
        deq = deque()
        res = []
        for i in range(len(nums)):
            while deq and deq[0] < (i - k + 1):
                deq.popleft()
            while deq and nums[i] > nums[deq[-1]]:
                deq.pop()
            deq.append(i)
            if i >= k - 1:
                res.append(nums[deq[0]])
        return res

# ---- Stack
# valid parenthesis
class Solution:
    def isValid(self, s: str) -> bool:
        Map = {")": "(", "]": "[", "}": "{"}
        stack = []
        for c in s:
            if c not in Map:
                stack.append(c)
                continue
            if not stack or stack[-1] != Map[c]:
                return False
            stack.pop()
        return not stack

# min stack
# evaluate reverse polish notation
# generate parenthesis
    
# daily temperatures
class Solution:
    def dailyTemperatures(self, temperatures: List[int]) -> List[int]:
        res = [0] * len(temperatures)
        stack = [] # [temp , index]
        for i, t in enumerate(temperatures):
            while stack and t > stack[-1][0]:
                stackT, stackI = stack.pop()
                res[stackI] = i - stackI
            stack.append((t, i))
        return res
    
# car fleet
class Solution:
    def carFleet(self, target: int, position: List[int], speed: List[int]) -> int:
        pair = [(p, s) for p, s in zip(position, speed)]
        pair.sort(reverse=True)
        stack = []

        for p, s in pair:
            # [(10, 2), (8, 4), (5, 1), (3, 3), (0, 1)]
            # [1.0]
            # [1.0, 1.0]
            # [1.0, 7.0]
            # [1.0, 7.0, 3.0]
            # [1.0, 7.0, 12.0]
            stack.append((target - p) / s)
            if len(stack) >= 2 and stack[-1] <= stack[-2]:
                stack.pop()
        return len(stack)
    
# largest rectangle in historgram
class Solution:
    def largestRectangleArea(self, heights: List[int]) -> int:
        maxArea = 0 
        stack = [] # [index, height]
        # maxArea = max(maxArea, height * (i - index))
        for i, h in enumerate(heights):
            start = i
            while stack and stack[-1][1] > h:
                index, height = stack.pop()
                maxArea = max(maxArea, height * (i - index))
                start = index
            stack.append((start, h))
        for i, h in stack:
            maxArea = max(maxArea, h * (len(heights) - i))
        return maxArea
    
# ----- Binary Search -----
# binary search
class Solution:
    def search(self, nums: List[int], target: int) -> int:
        l, r = 0, len(nums) - 1
        while l <= r:
            m = l + ((r - l) // 2)
            if nums[m] > target:
                r = m - 1
            elif nums[m] < target:
                l = m + 1
            else:
                return m
        return -1

# search a 2d matrix
class Solution:
    def searchMatrix(self, matrix: List[List[int]], target: int) -> bool:
        # matrix[0][n] > matrix[0][n-1]
        m, n = len(matrix), len(matrix[0])
        left, right = 0, m * n - 1
        while left <= right:
            mid = (left + right) // 2
            mid_row, mid_col = divmod(mid, n)
            if matrix[mid_row][mid_col] == target:
                return True
            elif matrix[mid_row][mid_col] < target:
                left = mid + 1
            else:
                right = mid - 1
        return False

# koko eating banana
class Solution:
    def minEatingSpeed(self, piles: List[int], h: int) -> int:
        # clasic allocated books problem : https://www.algotree.org/algorithms/binary_search/allot_books/
        l, r = 1, max(piles)
        def isEnough(cnt):
            return sum(ceil(i/cnt) for i in piles) <= h
        while l < r:
            m = (l + r) // 2
            if isEnough(m):
                r = m 
            else:
                l = m + 1
        return l     
    
# find minimum in rotated sorted array
class Solution:
    def findMin(self, nums: List[int]) -> int:
        l, r = 0, len(nums) - 1
        res = float("inf")
        while l < r:
            m = l + ((r-l) // 2)
            res = min(res, nums[m])
            # right has the min
            if nums[m] > nums[r]:
                l = m + 1
            else:
                # left has the min
                r = m - 1
        return min(res, nums[l])
    
# search in rotated sorted array
class Solution:
    def search(self, nums: List[int], target: int) -> int:
        l, r = 0, len(nums) - 1
        while l <= r:
            m = l + ((r-l) // 2)
            if nums[m] == target:
                return m
            # left sorted portion
            if nums[l] <= nums[m]:
                if target > nums[m] or target < nums[l]:
                    l = m + 1
                else:
                    r = m - 1
            # right sorted portion
            else:
                if target < nums[m] or target > nums[r]:
                    r = m - 1
                else:
                    l = m + 1
        return -1
    
# time based key value store
class TimeMap:

    def __init__(self):
        self.keyStore = {} # [val, timestamp]

    def set(self, key: str, value: str, timestamp: int) -> None:
        if key not in self.keyStore:
            self.keyStore[key] = []
        self.keyStore[key].append([value, timestamp])

    def get(self, key: str, timestamp: int) -> str:
        res, values = "", self.keyStore.get(key, [])
        l, r = 0, len(values) - 1
        while l <= r:
            m = (l + r) // 2
            if values[m][1] <= timestamp:
                res = values[m][0]
                l = m + 1
            else:
                r = m - 1
        return res
    
# median of two sorted arrays
"""
Approach: Binary search
- Use binary search to parition the smaller of the two input arrays into two parts.
- Find the partition of the larger array such that the sum of elements on the left side of the partition in both arrays is half of the total elements.
- Check if this partition is valid by verifying if the largest number on the left side is smaller than the smallest number on the right side.
- If the partition is valid, calculate and return the median
"""
class Solution:
    def findMedianSortedArrays(self, nums1: List[int], nums2: List[int]) -> float:
        n1 = len(nums1)
        n2 = len(nums2)
        # ensure nums1 is the smaller array for simplicity
        if n1 > n2:
            return self.findMedianSortedArrays(nums2, nums1)
        n = n1 + n2
        left = (n1 + n2 + 1) // 2 # Calculate the left partition size
        low = 0
        high = n1
        while low <= high:
            mid1 = (low + high) // 2 # calculate mid index for nums1
            mid2 = left - mid1 # calculate mid index for nums2 
            l1 = float('-inf')
            l2 = float('-inf')
            r1 = float('inf')
            r2 = float('inf')
            # Determine values of l1, l2, r1, and r2
            if mid1 < n1:
                r1 = nums1[mid1]
            if mid2 < n2:
                r2 = nums2[mid2]
            if mid1 - 1 >= 0:
                l1 = nums1[mid1 - 1]
            if mid2 - 1 >= 0:
                l2 = nums2[mid2 - 1]
            
            if l1 <= r2 and l2 <= r1:
                # the partition is correct, we found the median
                if n % 2 == 1:
                    return max(l1, l2)
                else:
                    return (max(l1, l2) + min(r1, r2)) / 2.0
            elif l1 > r2:
                # move towards the left side of nums1
                high = mid1 - 1
            else:
                # move towards the right side of nums1
                low = mid1 + 1
        return 0 # if the code reaches here, the input arrays were not sorted.
    
# -----Linked List
# reverse linked list
# https://leetcode.com/problems/reverse-linked-list/
class Solution:
    def reverseList(self, head: Optional[ListNode]) -> Optional[ListNode]:
        prev, curr = None, head
        while curr:
            temp = curr.next
            curr.next = prev
            prev = curr
            curr = temp
        return prev
    
# merge two sorted lists
# return the head of the merged linked list
class Solution:
    def mergeTwoLists(self, l1: Optional[ListNode], l2: Optional[ListNode]) -> Optional[ListNode]:
        dummy = node = ListNode()
        while l1 and l2:
            if l1.val < l2.val:
                node.next = l1
                l1 = l1.next
            else:
                node.next = l2
                l2 = l2.next
            node = node.next
        # if theres still value from l1 or l2 add to list
        node.next = l1 or l2
        return dummy.next
    
# reorder list
# https://leetcode.com/problems/reorder-list/
class Solution:
    def reorderList(self, head: Optional[ListNode]) -> None:
        # find middle
        slow, fast = head, head.next
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next
        # reverse second half
        second = slow.next
        prev = slow.next = None
        while second:
            tmp = second.next
            second.next = prev
            prev = second
            second = tmp
        # merge two halfs
        first, second = head, prev
        while second:
            tmp1, tmp2 = first.next, second.next
            first.next = second
            second.next = tmp1
            first, second = tmp1, tmp2

# remove nth node from end of list
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def removeNthFromEnd(self, head: Optional[ListNode], n: int) -> Optional[ListNode]:
        dummy = ListNode(0, head)
        left = dummy
        right = head
        while n > 0:
            right = right.next 
            n -= 1
        while right:
            left = left.next
            right = right.next
        # delete node
        left.next = left.next.next
        return dummy.next

# copy list with random pointer
# Approach: Use a hash map to store the mapping of old nodes to new nodes and copy the list.
# Return the head of the copied linked list.
class Solution:
    def copyRandomList(self, head: 'Optional[Node]') -> 'Optional[Node]':
        oldToCopy = {None: None}
        curr = head
        while curr:
            copy = Node(curr.val)
            oldToCopy[curr] = copy
            curr = curr.next
        curr = head
        while curr:
            copy = oldToCopy[curr]
            copy.next = oldToCopy[curr.next]
            copy.random = oldToCopy[curr.random]
            curr = curr.next
        return oldToCopy[head]
    
# add two numbers
class Solution:
    def addTwoNumbers(self, l1: Optional[ListNode], l2: Optional[ListNode]) -> Optional[ListNode]:
        # Approach: Use two pointers to add digits and handle carry.
        dummy = ListNode()
        c = dummy
        carry = 0
        while l1 or l2 or carry:
            v1 = l1.val if l1 else 0
            v2 = l2.val if l2 else 0
            # new digit
            val = v1 + v2 + carry
            carry = val // 10
            val = val % 10
            c.next = ListNode(val)
            # update ptrs
            c = c.next
            l1 = l1.next if l1 else None
            l2 = l2.next if l2 else None 
        return dummy.next

# linked list cycle
class Solution:
    def hasCycle(self, head: Optional[ListNode]) -> bool:
        # Approach: Use two pointers to detect if a cycle exists in the list.
        slow, fast = head, head
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next
            if slow == fast:
                return True
        return False
    
# find the duplicate number
class Solution:
    def findDuplicate(self, nums: List[int]) -> int:
        # Approach: Use Floyd's Tortoise and Hare (Cycle Detection) to find the duplicate number.
        slow, fast = 0, 0
        while True:
            slow = nums[slow]
            fast = nums[nums[fast]]
            if slow == fast:
                break
        slow2 = 0
        while True:
            slow = nums[slow]
            slow2 = nums[slow2]
            if slow == slow2:
                return slow

# lru cache
# Approach: Use a hash map and a doubly linked list to implement LRU cache.
# Problem: LRU least recently used cache
class Node:
    def __init__(self, key, val):
        self.key, self.val = key, val
        self.prev = self.next = None

class LRUCache:
    def __init__(self, capacity: int):
        self.cap = capacity
        self.cache = {} # map key to node

        self.left, self.right = Node(0, 0), Node(0, 0)
        self.left.next, self.right.prev = self.right, self.left
    
    def remove(self, node):
        prev, nxt = node.prev, node.next
        prev.next, nxt.prev = nxt, prev
    
    def insert(self, node):
        prev, nxt = self.right.prev, self.right
        prev.next = nxt.prev = node
        node.next, node.prev = nxt, prev

    def get(self, key: int) -> int:
        if key in self.cache:
            self.remove(self.cache[key])
            self.insert(self.cache[key])
            return self.cache[key].val
        return -1

    def put(self, key: int, value: int) -> None:
        if key in self.cache:
            self.remove(self.cache[key])
        self.cache[key] = Node(key, value)
        self.insert(self.cache[key])

        if len(self.cache) > self.cap:
            # remove from the list and delete the LRU from hashmap
            lru = self.left.next
            self.remove(lru)
            del self.cache[lru.key]
# Your LRUCache object will be instantiated and called as such:
# obj = LRUCache(capacity)
# param_1 = obj.get(key)
# obj.put(key,value)
            

# merge k sorted lists
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next

# Approach: Use a priority queue (min-heap) to merge k sorted linked lists.
class Solution:
    def mergeKLists(self, lists: List[Optional[ListNode]]) -> Optional[ListNode]:
        # create an empty heap
        heap = []
        for l in lists:
            pointer = l 
            # traverse the current linked list and push each node's value onto the heap
            while pointer:
                heappush(heap, pointer.val)
                pointer = pointer.next
        # create a dummy node to serve as the head of the merged linked list
        head = ListNode(0)
        cur = head
        # pop values from the heap and create nodes in the merged linked list
        while len(heap) != 0:
            cur.next = ListNode(heappop(heap))
            cur = cur.next
        # return the next node of the dummy node, which is the head of the meregd linked list
        return head.next

# reverse nodes in k group
# Approach: Reverse nodes in groups of k using iterative or recursive approach.
class Solution:
    def reverseKGroup(self, head: Optional[ListNode], k: int) -> Optional[ListNode]:
        dummy = ListNode(0, head)
        groupPrev = dummy
        while True:
            kth = self.getKth(groupPrev, k)
            if not kth:
                break
            groupNext = kth.next
            # reverse group
            prev, curr = kth.next, groupPrev.next
            while curr != groupNext:
                tmp = curr.next
                curr.next = prev
                prev = curr
                curr = tmp
            tmp = groupPrev.next
            groupPrev.next = kth
            groupPrev = tmp
        return dummy.next
    def getKth(self, curr, k):
        while curr and k > 0:
            curr = curr.next
            k -= 1
        return curr            

# ----- Trees
# invert binary tree
# diameter of binary tree
# balanced binary tree
# same tree
# subtree of another tree
# lowest common ancestor of a binary search tree
# binary tree level order travelsal
# binary tree right side view
# count good nodes in binary tree
# validate binary tree
# kth smallest element in a bst
# construct binary tree from preorder and inorder travelsal
# binary tree maximum path sum
# serialize and deserialize binary tree

# ----- Heap / Priority Queue
# kth largest element in a stream
# last stone weight
# k closest points to origin
# task scheduler
# design twitter
# find median data stream

# ----- Backtracking
# subsets
# combination sum
# permutation
# subsets II
# combination sum II
# word search
# palindrome partitioning
# letter combination of a phone number
# n queens

# ----- Tries
# implement trie prefix
class TrieNode:
    def __init__(self):
        self.children = [None] * 26
        self.end = False
class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, words: str) -> None:
        curr = self.root
        for c in words:
            i = ord("a") - ord(c)
            if curr.children[i] == None:
                curr.children[i] = TrieNode()
            curr = curr.children[i]
        curr.end = True

    def search(self, words: str) -> bool:
        curr = self.root
        for c in words:
            i = ord("a") - ord(c)
            if curr.children[i] == None:
                return False
            curr = curr.children[i] 
        return curr.end

    def startsWith(self, prefix) -> bool:
        curr = self.root
        for c in prefix:
            i = ord("a") - ord(c)
            if curr.children[i] == None:
                return False
            curr = curr.children[i]
        return True

# design add and search words data
"""
design data structure that supports adding new words and finding if a string matches previously added string

Input:
["WordDictionary","addWord","addWord","addWord","search","search","search","search"]
[[],["bad"],["dad"],["mad"],["pad"],["bad"],[".ad"],["b.."]]
Output:
[null,null,null,null,false,true,true,true]
"""
class TrieNode:
    def __init__(self):
        self.children = {}  # a : TrieNode
        self.word = False

class WordDictionary:
    def __init__(self):
        self.root = TrieNode()

    def addWord(self, word: str) -> None:
        cur = self.root
        for c in word:
            if c not in cur.children:
                cur.children[c] = TrieNode()
            cur = cur.children[c]
        cur.word = True

    def search(self, word: str) -> bool:
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
"""
return all the word from the board 

Input: board = 
[["o","a","a","n"],["e","t","a","e"],["i","h","k","r"],["i","f","l","v"]], words = ["oath","pea","eat","rain"]

Output: ["eat","oath"]
"""
class TrieNode:
    def __init__(self):
        self.children = {}
        self.isWord = False
        self.refs = 0

    def addWord(self, word):
        cur = self
        cur.refs += 1
        for c in word:
            if c not in cur.children:
                cur.children[c] = TrieNode()
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
    def findWords(self, board: List[List[str]], words: List[str]) -> List[str]:
        # dfs + trie
        root = TrieNode()
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
"""
Given an m x n 2D binary grid grid which represents a map of '1's (land) and '0's (water), return the number of islands.

An island is surrounded by water and is formed by connecting adjacent lands horizontally or vertically. You may assume all four edges of the grid are all surrounded by water.

Example 1:
Input: grid = [
  ["1","1","1","1","0"],
  ["1","1","0","1","0"],
  ["1","1","0","0","0"],
  ["0","0","0","0","0"]
]
Output: 1
"""
class Solution:
    def numIslands(self, grid: List[List[str]]) -> int:
        rows, cols = len(grid), len(grid[0])
        visit = set()
        def dfs(r, c):
            if (
                r not in range(rows)
                or c not in range(cols) 
                or grid[r][c] == "0"
                or (r, c) in visit
            ):
                return False
            visit.add((r, c))
            dfs(r + 1, c)
            dfs(r - 1, c)
            dfs(r, c + 1)
            dfs(r, c - 1)
            return True

        island = 0
        for r in range(rows):
            for c in range(cols):
                if dfs(r, c):
                    island += 1
        return island


# max area of island
"""
You are given an m x n binary matrix grid. An island is a group of 1's (representing land) connected 4-directionally (horizontal or vertical.) You may assume all four edges of the grid are surrounded by water.

The area of an island is the number of cells with a value 1 in the island.

Return the maximum area of an island in grid. If there is no island, return 0.
"""
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
        area = 0
        for r in range(rows):
            for c in range(cols):
                area = max(area, dfs(r, c)) 
        return area

# clone graph
"""
Given a reference of a node in a connected undirected graph.
Return a deep copy (clone) of the graph.
Each node in the graph contains a value (int) and a list (List[Node]) of its neighbors.
class Node {
    public int val;
    public List<Node> neighbors;
}
 
Test case format:
For simplicity, each node's value is the same as the node's index (1-indexed). For example, the first node with val == 1, the second node with val == 2, and so on. The graph is represented in the test case using an adjacency list.
An adjacency list is a collection of unordered lists used to represent a finite graph. Each list describes the set of neighbors of a node in the graph.
The given node will always be the first node with val = 1. You must return the copy of the given node as a reference to the cloned graph.

Input: adjList = [[2,4],[1,3],[2,4],[1,3]]
Output: [[2,4],[1,3],[2,4],[1,3]]
Explanation: There are 4 nodes in the graph.
1st node (val = 1)'s neighbors are 2nd node (val = 2) and 4th node (val = 4).
2nd node (val = 2)'s neighbors are 1st node (val = 1) and 3rd node (val = 3).
3rd node (val = 3)'s neighbors are 2nd node (val = 2) and 4th node (val = 4).
4th node (val = 4)'s neighbors are 1st node (val = 1) and 3rd node (val = 3).
"""
class Solution:
    def cloneGraph(self, node: Optional['Node']) -> Optional['Node']:
        oldToNew = {}
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
# rotting oranges
# pacific atlantic water flow
# surrounded regions
# course schedule
# course schedule II
# graph valid tree
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
# min cost climbing stairs
# house robber
# house robber II
# longest palindromic substring
# palindromic substrings
# decode ways
# coin change
# maximum product subarray
# word break
# longest increasing subsequence
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

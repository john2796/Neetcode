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
Approach:
"""
# longest substring without repeating characters
"""
Problem:
Approach:
"""
# longest repeating character replacement
"""
Problem:
Approach:
"""
# permutation in string
"""
Problem:
Approach:
"""
# minimum window substring
"""
Problem:
Approach:
"""
# sliding window maximum
"""
Problem:
Approach:
"""

# ---- Stack
# valid parenthesis
# min stack
# evaluate reverse polish notation
# generate parenthesis
# daily temperatures
# car fleet
# largest rectangle in historgram

# ----- Binary Search -----
# binary search
# search a 2d matrix
# koko eating banana
# find minimum in rotated sorted array
# search in rotated sorted array
# time based key value store
# median of two sorted arrays

# -----Linked List
# reverse linked list
# merge two sorted lists
# reorder list
# remove nth node from end of list
# copy list with random pointer
# add two numbers
# linked list cycle
# find the duplicate number
# lru cache
# merge k sorted lists
# reverse nodes in k group

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
# design add and search words data
# word search II

# ----- Graphs
# number of islands
# max area of island
# clone graph
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

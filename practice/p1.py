
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
        ct,cs = {}, {}

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
        if len(strs) == 0:
            return [[""]]
        hs = {} # {(a:1, b:0,...z:0): ["nat", "tan", "etc"]}
        for words in strs:
            for char in words:
                key = ord("a") - ord(char)
                print(key)

        
        return [["aa"]]

# top k frequent elements
"""
Problem:
Approach:
"""
# encode and decode strings
"""
Problem:
Approach:
"""
# product of array except self
"""
Problem:
Approach:
"""
# valid sudoku
"""
Problem:
Approach:
"""
# longest consecutive sequence
"""
Problem:
Approach:
"""

#---- Two Pointers
# valid palindrome
"""
Problem:
Approach:
"""
# two sum II input array is sorted
"""
Problem:
Approach:
"""
# 3sum
"""
Problem:
Approach:
"""
# container with most water 
"""
Problem:
Approach:
"""
# trapping rain water
"""
Problem:
Approach:
"""

#---- Sliding Window
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

#---- Stack
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

#----- Tries 
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

#---- Advanced Graphs 
# reconstruct itinerary
# min cost to connect all points
# network delay time
# swim in rising water
# alien dictionary
# cheapest flights within k stops

#---- 1-D DP
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

#---- 2-D DP
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

#---- Greedy
# Maximum subbaray
# jump game
# jump game II
# gas station
# hand of straights
# merge triplets to form target triplet
# partition labels
# valid parenthesis string

#---- Intervals
# insert interval 
# merge intervals
# non overlapping intervals
# meeting rooms
# meeting rooms II
# minimum interval to include each query

#---- Math & Geometry
# rotate image
# spiral mamtrix
# set matrix zeroes
# happy number
# plus one
# pow(x, n)
# multiply strings
# detect squares

#---- Bit Manipulation
# single number
# number of 1 bits
# counting bits
# reverse bits
# missing number
# sum of two integers
# reverse integer

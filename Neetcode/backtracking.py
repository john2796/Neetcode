# Backtracking


# Subsets
"""
return all possible subsets the solution set must not contain duplicate subsets return the solution in any order
Input: nums = [1,2,3]
Output: [[],[1],[2],[1,2],[3],[1,3],[2,3],[1,2,3]]
"""
class Solution:
    def subsets(self, nums: List[int]) -> List[List[int]]:
        res = []
        subset = []
        def dfs(i):
            if i >= len(nums):
                res.append(subset.copy())
                return
            # decision to include nums[i]
            subset.append(nums[i])
            dfs(i + 1)
            # decision NOT to include nums[i]
            subset.pop()
            dfs(i + 1)
        dfs(0)
        return res

# Combination Sum
"""
return a list of all unique combination of candidates where the chosen numbers sum to target

Input: candidates = [2,3,6,7], target = 7
Output: [[2,2,3],[7]]
Explanation:
2 and 3 are candidates, and 2 + 2 + 3 = 7. Note that 2 can be used multiple times.
7 is a candidate, and 7 = 7.
These are the only two combinations.
"""
class Solution:
    def combinationSum(self, candidates: List[int], target: int) -> List[List[int]]:
        res = []

        def dfs(i, cur, total):
            if total == target:
                res.append(cur.copy())
                return
            # condition
            if i >= len(candidates) or total > target:
                return
            # take it
            cur.append(candidates[i])
            dfs(i, cur, total + candidates[i])
            # leave it
            cur.pop()
            dfs(i + 1, cur, total)
        dfs(0, [], 0)
        return res

# Permutations
"""
Given an array nums of distinct integers, return all the possible permutations. You can return the answer in any order
Input: nums = [1,2,3]
Output: [[1,2,3],[1,3,2],[2,1,3],[2,3,1],[3,1,2],[3,2,1]]
"""
class Solution:
    def permute(self, nums: List[int]) -> List[List[int]]:
        res = []
        # base case
        if len(nums) == 1:
            return [nums[:]] # nums[:] is a deep copy
        for i in range(len(nums)):
            n = nums.pop(0)
            perms = self.permute(nums)

            for perm in perms:
                perm.append(n)
            res.extend(perms)
            nums.append(n)
        return res

# Subsets II
"""
Given an integer arry nums that may contain duplicates, return all possible subsets (the power set).
The solution oset must not contain duplicate subsets.
return the solution in any order
Input: nums = [1,2,2]
Output: [[],[1],[1,2],[1,2,2],[2],[2,2]]
"""
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

# Combination Sum II
"""
Given a collection of candidate numbers (candidates) and a target number (target), find all unique combinations in candidates where the candidate numbers sum to target.

Each number in candidates may only be used once in the combination.

Note: The solution set must not contain duplicate combinations.
Example 1:

Input: candidates = [10,1,2,7,6,1,5], target = 8
Output: 
[
[1,1,6],
[1,2,5],
[1,7],
[2,6]
]
"""
class Solution:
    def combinationSum2(self, c: List[int], t: int) -> List[List[int]]:
        c.sort() # sort the candidates to handle duplicates efficiently
        res = [] # list to store the resulting combination
        def bactrack(cur, pos, target):
            if target == 0:
                # if target is achieved (sum of current combination is t), add it to the result
                res.append(cur.copy()) # make a copy of current combination and add to result
                return
            if target <= 0:
                # if target is less than 0, stop further processing (invalid path)
                return

            prev = -1 # variable to keep track of the previous element to handle duplicates
            for i in range(pos, len(c)):
                if c[i] == prev:
                    # skip duplicates at the same level to avoid duplicate combinations
                    continue
                cur.append(c[i]) # choose current element c[i] and add to current combination
                bactrack(cur, i + 1, target - c[i]) # recursively call backtrack with updated parameters
                cur.pop() # backtrack: remove the last element to backtrack
                prev = c[i] # update prev to current element c[i] for the next iteration
        bactrack([], 0, t) # start backtrack with an empty current combination, starting position 0, and target t
        return res # return the resulting combinations
    
# Word Search
"""
Given an m x n grid of characters board and a string word, return true if word exists in the grid.

The word can be constructed from letters of sequentially adjacent cells, where adjacent cells are horizontally or vertically neighboring. The same letter cell may not be used more than once.

 Input: board = [["A","B","C","E"],["S","F","C","S"],["A","D","E","E"]], word = "ABCCED"
Output: true
"""
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

        # To preven TLE,reverse the word if frequency of the first letter is more than the last letter's
        count = defaultdict(int, sum(map(Counter, board), Counter()))
        if count[word[0]] > count[word[-1]]:
            word = word[::-1]

        for r in range(ROWS):
            for c in range(COLS):
                if dfs(r, c, 0):
                    return True
        return False
        # O(n * m * 4^n)

# Palindrome Partition
"""
Given a string s, partition s such that every 
substring
 of the partition is a 
palindrome
. Return all possible palindrome partitioning of s.
Example 1:

Input: s = "aab"
Output: [["a","a","b"],["aa","b"]]
"""
class Solution:
    def partition(self, s: str) -> List[List[str]]:
        res, part = [], []
        def dfs(i):
            if i >= len(s):
                res.append(part.copy())
                return
            for j in range(i, len(s)):
                if self.isPali(s, i, j):
                    part.append(s[i : j + 1])
                    dfs(j + 1)
                    part.pop()
        dfs(0)
        return res
    def isPali(self, s, l, r):
        while l < r:
            if s[l] != s[r]:
                return False
            l, r = l + 1, r - 1
        return True

# Letter Combination of a Phone Number
"""
Given a string containing digits from 2-9 inclusive, return all possible letter combinations that the number could represent. Return the answer in any order.

A mapping of digits to letters (just like on the telephone buttons) is given below. Note that 1 does not map to any letters
Example 1:

Input: digits = "23"
Output: ["ad","ae","af","bd","be","bf","cd","ce","cf"]
"""
class Solution:
    def letterCombinations(self, digits: str) -> List[str]:
        res = []
        digitToChar = {
            "2": "abc",
            "3": "def",
            "4": "ghi",
            "5": "jkl",
            "6": "mno",
            "7": "qprs",
            "8": "tuv",
            "9": "wxyz",
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

# N queens
"""
The n-queens puzzle is the problem of placing n queens on an n x n chessboard such that no two queens attack each other.

Given an integer n, return all distinct solutions to the n-queens puzzle. You may return the answer in any order.

Each solution contains a distinct board configuration of the n-queens' placement, where 'Q' and '.' both indicate a queen and an empty space, respectively.

Input: n = 4
Output: [[".Q..","...Q","Q...","..Q."],["..Q.","Q...","...Q",".Q.."]]
Explanation: There exist two distinct solutions to the 4-queens puzzle as shown above
"""
class Solution:
    def solveNQueens(self, n: int) -> List[List[str]]:
        col = set()
        posDiag = set() # (r + c)
        negDiag  =set() # (r - c)

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
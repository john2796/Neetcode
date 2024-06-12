# https://leetcode.com/problems/best-time-to-buy-and-sell-stock/
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        buy = prices[0]
        profit = 0

        for sell in prices[1:]:
            if sell > buy:
                profit = max(profit, sell - buy)
            else:
                buy = sell
        return profit


# https://leetcode.com/problems/longest-substring-without-repeating-characters/
class Solution:
    def lengthOfLongestSubstring(self, s: str) -> int:
        l = 0
        c = set()
        longest = 0
        for r in range(len(s)):
            while s[r] in c:
                c.remove(s[l])
                l += 1
            c.add(s[r])
            longest = max(longest, r - l + 1)
        return longest


# https://leetcode.com/problems/longest-repeating-character-replacement/
class Solution:
    def characterReplacement(self, s: str, k: int) -> int:
        charSet = {}
        l = 0
        res = 0

        for r in range(len(s)):
            # expand
            if s[r] not in charSet:
                charSet[s[r]] = 1
            else:
                charSet[s[r]] += 1
            res = max(res, charSet[s[r]])

            # shrink
            if (r - l + 1) - res > k:
                charSet[s[l]] -= 1
                l += 1
        return r - l + 1


# https://leetcode.com/problems/permutation-in-string/
"""
- create a hashmap with the count of every character in the string s1
- then we slide a window over the strings s2 and decrease the counter for the characters that occured in the window
- as soon as all counters in the hashmap get to zero that means we encountered the permutation
"""


class Solution:
    def checkInclusion(self, s1: str, s2: str) -> bool:
        cntr, w, match = Counter(s1), len(s1), 0

        for i in range(len(s2)):
            if s2[i] in cntr:
                if not cntr[s2[i]]:
                    match -= 1
                cntr[s2[i]] -= 1
                if not cntr[s2[i]]:
                    match += 1

            if i >= w and s2[i - w] in cntr:
                if not cntr[s2[i - w]]:
                    match -= 1
                cntr[s2[i - w]] += 1
                if not cntr[s2[i - w]]:
                    match += 1
            if match == len(cntr):
                return True
        return False

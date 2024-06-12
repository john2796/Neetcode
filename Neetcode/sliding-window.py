""" https://leetcode.com/problems/best-time-to-buy-and-sell-stock/
     b
[ 7, 1, 5, 3, 6, 4 ]
              s 
max_profit=-6
"""


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

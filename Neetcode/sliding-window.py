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


""" https://leetcode.com/problems/longest-substring-without-repeating-characters/

Input: s = "abcabcbb"
Output: 3
Explanation: The answer is "abc", with the length of 3.
"""


class Solution:
    def lengthOfLongestSubstring(self, s: str) -> int:
        n = len(s)
        max_length = 0
        char_map = {}
        left = 0

        for right in range(n):
            # If the current character is not in the map or its index is less than left, it means it is a new unique character.
            if s[right] not in char_map or char_map[s[right]] < left:
                char_map[s[right]] = right
                max_length = max(max_length, right - left + 1)
            else:
                # If the character is repeating within the current substring, we move the left pointer to the next position after the last occurrence of the character.
                left = char_map[s[right]] + 1
                char_map[s[right]] = right
        return max_length

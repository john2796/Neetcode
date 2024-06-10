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

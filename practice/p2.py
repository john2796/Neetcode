# Two-Pointer Approach
"""
42. Trapping Rain Water
Compute how much water it can trap after raining

                                       rm
Input: height = [0,1,0,2,1,0,1,3,2,1,2,1]
                  lm
output: 6
"""
class Solution:
    def trap(self, h: List[int]) -> int:
        # use two pointer track maxVal from left and right, subtrack maxVal with current pointer position value
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



"""
11. Container with most water
return the maximum amount of water container can store.
"""
# use two pointer calc max_area=( window * the minimum between left and right), move pointer depending whichever one is smaller.
class Solution:
    def maxArea(self, height: List[int]) -> int:
        l, r = 0, len(height) - 1
        max_area = 0
        while l < r:
            current_area = min(height[l], height[r]) * (r - l)
            max_area = max(max_area, current_area)
            if height[l] < height[r]:
                l += 1
            else:
                r -= 1
        return max_area

"""
sort nums, two pointer + loop , check whether total == 0, check for dups value to skip
"""
class Solution:
    def threeSum(self, n: List[int]) -> List[List[int]]:
        n.sort()
        res = []
        for i in range(len(n) - 1):
            if i > 0 and n[i] == n[i - 1]:  # skip dups i
                continue
            j = i + 1
            k = len(n) - 1
            while j < k:
                total = n[i] + n[j] + n[k]
                if total > 0:
                    k -= 1
                elif total < 0:
                    j += 1
                else:
                    res.append([n[i], n[j], n[k]])
                    j += 1  # move left pointer 1over

                    while n[j] == n[j - 1] and j < k:
                        j += 1  # also moved j if dups
        return res

# two pointer , move pointer when not isalnum() move both if same
class Solution:
    def isPalindrome(self, s: str) -> bool:
        l, r = 0, len(s) - 1
        while l < r:
            if not s[l].isalnum():
                l += 1
            elif not s[r].isalnum():
                r -= 1
            elif s[l].lower() == s[r].lower():
                l += 1
                r -= 1
            else:
                return False
        return True

 # array is sorted , two-pointer 
class Solution:
    def twoSum(self, n: List[int], target: int) -> List[int]:
        l, r = 0, len(n) - 1
        while l < r:
            t = n[r] + n[l]
            if t < target:
                l += 1
            elif t > target:
                r -= 1
            else:
                return [l + 1, r + 1]
        return []

# sliding window
class Solution:
    def maxProfit(self, p: List[int]) -> int:
        buy = p[0]
        profit = 0
        for sell in p[1:]:
            if sell > buy: # 1 > 7, 5>1, 3>5
                profit = max(profit, sell - buy) # 5-1=4
            else: # buy=1, buy=
                buy = sell
        return profit        

        """
             s  b
            [7, 1, 5, 3, 6, 4]
            | sell | buy  | profit|
            |  1   |  7   |   0   |
            |  5   |  1   |   4   |
            |  3   |  1   |   2   |
            |  6   |  1   |   5   | <-- answer
            |  4   |  1   |   3   |

                    """

class Solution:
    def lengthOfLongestSubstring(self, s: str) -> int:
        longest = 0
        l = 0
        c = set()
        for r in range(len(s)):
            while s[r] in c:
                c.remove(s[l])
                l += 1
            c.add(s[r])
            longest = max(longest, r-l+1)
        return longest
    
class Solution:
    def characterReplacement(self, s: str, k: int) -> int:
        c = collections.defaultdict()
        l = 0
        res = 0
        for r in range(len(s)):
            if s[r] not in c:
                c[s[r]] = 1
            else:
                c[s[r]] += 1
            res = max(res, c[s[r]])
            if (r - l + 1) - res > k:
                c[s[l]] -= 1
                l += 1
        return r - l + 1
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


# https://leetcode.com/problems/minimum-window-substring/
"""
- track t count
- expand window add right char into window, increment have if char in countT and window and countT val are same
- shrink window when have and need are equal , update result when window length < resLen, pop from the left window
"""


class Solution:
    def minWindow(self, s: str, t: str) -> str:
        if t == "":
            return ""
        countT, window = {}, {}
        for c in t:
            countT[c] = 1 + countT.get(c, 0)

        have, need = 0, len(countT)
        res, resLen = [-1, -1], float("infinity")
        l = 0

        for r in range(len(s)):
            c = s[r]
            window[c] = 1 + window.get(c, 0)

            if c in countT and window[c] == countT[c]:
                have += 1

            while have == need:
                # update our result
                if (r - l + 1) < resLen:
                    res = [l, r]
                    resLen = r - l + 1
                # pop from the left of our window
                window[s[l]] -= 1
                if s[l] in countT and window[s[l]] < countT[s[l]]:
                    have -= 1
                l += 1
        l, r = res
        return s[l : r + 1] if resLen != float("infinity") else ""


# https://leetcode.com/problems/sliding-window-maximum/
class Solution:
    def maxSlidingWindow(self, nums: List[int], k: int) -> List[int]:
        q = collections.deque()
        l = r = 0
        output = []

        while r < len(nums):
            # pop smaller values from q
            while q and nums[q[-1]] < nums[r]:
                q.pop()
            q.append(r)

            # remove left val from window
            if l > q[0]:
                q.popleft()

            if (r + 1) >= k:
                output.append(nums[q[0]])
                l += 1
            r += 1
        return output

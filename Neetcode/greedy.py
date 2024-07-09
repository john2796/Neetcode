# Maximum Subarray
class Solution:
    def maxSubarray(self, nums: List[in]) -> int:
        res = nums[0]

        total = 0
        for n in nums:
            total += n
            res = max(res, total)
            if total < 0:
                total = 0
        return res

# Jump Game
class Solution:
    def canJump(self, nums: List[int]) -> bool:
        goal = len(nums) - 1

        for i in range(len(nums) - 2, -1, -1):
            if i + nums[i] >= goal:
                goal = i
        return goal == 0

# Jump Game II
class Solution:
    def jump(self, nums: List[int]) -> int:
        l, r = 0, 0 
        res = 0

        while r < (len(nums) - 1):
            maxJump = 0
            for i in range(l, r + 1):
                maxJump = max(maxJump, i + nums[i])
            l = r + 1
            r = maxJump
            res += 1
        return res

# Gas Station
class Solution:
    def canCompleteCircuit(self, gas: List[int], cost: List[int]) -> int:
        start, end = len(gas) - 1, 0
        total = gas[start] - cost[start]

        while start >= end:
            while total < 0 and start >= end:
                start -= 1
                total += gas[start] - cost[start]
            if start == end:
                return start
            total += gas[end] - cost[end]
            end += 1
        return -1

# Hand of Straights
class Solution:
    def isNSStraightHand(self, hand: List[int], groupSize: int) -> bool:
        if len(hand) % groupSize:
            return False
        count = {}

        for n in hand:
            count[n] = 1 + count.get(n, 0)
        
        minH = list(count.keys())
        heapq.heapify(minH)

        while minH:
            first = minH[0]
            for i in range(first, first + groupSize):
                if i not in count:
                    return False
                count[i] -= 1
                if count[i] == 0:
                    if i != minH[0]:
                        return False
                    heapq.heappop(minH)
        return True

# Merge Triplets to Form Targt Triplet
class Solution:
    def mergeTriplets(self, triplets: List[List[int]], target: List[int]) -> bool:
        good = set()

        for t in triplets:
            if t[0] > target[0] or t[1] > target[1] or t[2] > target[2]:
                continue

            for i, v in enumerate(t):
                if v == target[i]:
                    good.add(i)
        return len(good) == 3

# Partition Labels
class Solution:
    def partitionLabels(self, S: str) -> List[int]:
        count = {}
        res = []
        i, length = 0, len(S)
        for j in range(length):
            c = S[j]
            count[c] = j
        
        curLen = 0
        goal = 0
        while i < length:
            c = S[i]
            goal = max(goal, count[c])
            curLen += 1

            if goal == i:
                res.append(curLen)
                curLen = 0
            i += 1
        return res

# Valid Parenthesis String
class Solution:
    def checkValidString(self, s: str) -> bool:
        dp = {(len(s), 0): True} # key=(i, leftCount) -> isValid

        def dfs(i, left):
            if i == len(s) or left < 0:
                return left == 0
            if (i, left) in dp:
                return dp[(i, left)]
            if s[i] == "(":
                dp[(i, left)] = dfs(i + 1, left + 1)
            elif s[i] == ")":
                dp[(i, left)] = dfs(i + 1, left - 1)
            else:
                dp[(i, left)] = (
                    dfs(i + 1, left + 1) or dfs(i + 1, left - 1) or dfs(i + 1, left)
                )
            return dp[(i, left)]
        return dfs(0, 0)
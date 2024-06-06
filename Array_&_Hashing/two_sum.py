# https://leetcode.com/problems/two-sum/submissions/1279101622/


class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        # {2: 0, 7: 1} 9 - 2 7 in the set return index [currentIdx, set_value_index]
        hashset = {}
        for i, n in enumerate(nums):
            print(n, i)
            t = target - n
            if t in hashset:
                return [hashset[t], i]
            else:
                hashset[n] = i

# https://leetcode.com/problems/two-sum-ii-input-array-is-sorted/description/

class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        l, r = 0, len(nums) - 1

        while l < r:
            sum = nums[r] + nums[l]
            if target == sum:
                return [l + 1, r + 1]
            elif sum < target:
                l += 1
            else:
                r -= 1
            

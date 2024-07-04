class Solution:
    def threeConsecutiveOdds(self, arr: List[int]) -> bool:
        count = 0

        for n in arr:
            if n % 2 == 0:
                count = 0
            else:
                count += 1
            if count == 3:
                return True
        return False


class Solution:
    def intersect(self, nums1: List[int], nums2: List[int]) -> List[int]:
        nums1.sort()
        nums2.sort()
        res = []
        i, j = 0, 0
        while i < len(nums1) and j < len(nums2):
            if nums1[i] < nums2[j]:
                i += 1
            elif nums1[i] > nums2[j]:
                j += 1
            else:
                res.append(nums1[i])
                i += 1
                j += 1
        return res


# {4: 1, 9: 1, 5: 1}
# {9: 2, 4: 2, 8: 1}
        
class Solution:        
    def minDifference(self, nums: List[int]) -> int:
        n = len(nums)
        if n <= 4:
            return 0

        nums.sort()
        # evaluate the minimum difference possible with at most 3 moves
        min_diff = min(
            nums[n-1] - nums[3], # change 3 smallest element
            nums[n-2] - nums[2], # change 2 smallest and 1 largest element
            nums[n-3] - nums[1], # change 1 smallest and 2 largest elements
            nums[n-4] - nums[0]  # change 3 largest element
        )
        return min_diff
        

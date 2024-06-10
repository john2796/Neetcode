""" https://leetcode.com/problems/valid-palindrome/
return true if it is a palindrome, or false otherwise
Input: s = "A man, a plan, a canal: Panama"
Output: true
Explanation: "amanaplanacanalpanama" is a palindrome.

two-pointer approach:
 l 
 amanaplanacanalpanama
                     r
"""


class Solution:
    def isPalindrome(self, s: str) -> bool:
        l, r = 0, len(s) - 1

        while l < r:
            # move pointer if its not alphanumeric
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


""" https://leetcode.com/problems/two-sum-ii-input-array-is-sorted/description/
  l
[ 2, 7, 11, 15]
            r
l+r = t
[l,r]
"""


class Solution:
    def twoSum(self, numbers: List[int], target: int) -> List[int]:
        l, r = 0, len(numbers) - 1

        while l < r:
            s = numbers[r] + numbers[l]
            if target == s:
                return [l + 1, r + 1]
            elif s < target:
                l += 1
            else:
                r -= 1


""" https://leetcode.com/problems/3sum/description/
  i
[-1, 0, 1, 2, -1, -4]
     j.            k
-1+0+1=
"""


class Solution:
    def threeSum(self, nums: List[int]) -> List[List[int]]:
        nums.sort()
        res = []
        for i in range(len(nums) - 1):
            if i > 0 and nums[i] == nums[i - 1]:  # skip dups i
                continue
            j = i + 1
            k = len(nums) - 1
            while j < k:
                total = nums[i] + nums[j] + nums[k]
                if total > 0:
                    k -= 1
                elif total < 0:
                    j += 1
                else:
                    res.append([nums[i], nums[j], nums[k]])
                    j += 1  # move left pointer 1over

                    while nums[j] == nums[j - 1] and j < k:
                        j += 1  # also moved j if dups
        return res


""" https://leetcode.com/problems/container-with-most-water/
Return the maximum amount of water a container can store.
area = (r-l) * min(r, l)

  0. 1  2  3. 4. 5  6. 7. 8
     l
[ 1, 8, 6, 2, 5, 4, 8, 3, 7 ]
                          r
max_area=8
a = (8-0) * min(1,7)  - 8*1 = 8
a = (8-1) * min(7,8)  - 7*7=49
                
"""


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

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

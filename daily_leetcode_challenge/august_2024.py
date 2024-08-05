
# https://leetcode.com/problems/number-of-senior-citizens/?envType=daily-question&envId=2024-08-01
class Solution:
    def countSeniors(self, details: List[str]) -> int:
        # 11
        # 12 and 13
        res = 0
        for detail in details:
            age = detail[11:13]
            if int(age) > 60:
                res += 1
        return res

"""
Overview
In this problem, we're given a binary circular array where each element is either 0 or 1. The curcular nature of the array means that the last element is considered adjacent to the first. Our task is to find the minimum number of swaps needed to group the 1s. A swap involves exchanging values between two distinct positions.

The circularr property of the array open up more possible groupings to consider compared to a linear array.

Input: nums = [0,1,1,1,0,0,1,1,0]
output: 2

two swaps are required to group all 1s together, either forming 
[1,1,1,0,0,0,0,1,1]
[1,1,1,1,1,0,0,0,0]


Approach 2: Using Sliding Window
by creating a sliding window equal to the number of 1s in the array and using it to identify the grouping with the highest concentration of 1s. Then, we'll use this to determine how many values need to be swapped to group all 1s in the array together.

we'll determin the size of our sliding window by counting the number of 1s in the given array. Next, we'll initialize the window to be this size and count the number of 1s within it. This gives us a baseline count, representing how many 1s are already in place within the first possible grouping. This step is important because it sets the stage for comparison as we slide the dinwo across the array.

As we slide the window, we'll dynamically update our count of 1s by subtracting the value at the window's starting edge and adding the value at the window's new trailing edge. This step is crucial because it allows us to track the number of 1s in each potential group without re-scanning the entire window. The circular nature of the array is naturally handled because the sliding window can wrap around from the end to the beginning of the array.

Finally, we'll find the difference between the total number of 1s in the array and the grouping with the highest concentration of 1s to find the minimum of swaps required to group the 1s. 

"""
class Solution:
    def minSwaps(self, nums: List[int]) -> int:
        # Calculate the minimum swaps needed to group all 1s or all 0s together
        op1 = self.min_swaps_helper(nums, 0)  # Grouping all 0s together
        op2 = self.min_swaps_helper(nums, 1)  # Grouping all 1s together
        return min(op1, op2)
    def min_swaps_helper(self, data: List[int], val: int) -> int:
        length = len(data)
        total_val_count = 0
        # Count the total number of `val` in the array
        for i in range(length - 1, -1, -1):
            if data[i] == val:
                total_val_count += 1
        # If there is no `val` or the array is full of `val`, no swaps are needed
        if total_val_count == 0 or total_val_count == length:
            return 0
        start = 0
        end = 0
        max_val_in_window = 0
        current_val_in_window = 0
        # Initial window setup: count the number of `val` in the first window of size `total_val_count`
        while end < total_val_count:
            if data[end] == val:
                current_val_in_window += 1
            end += 1
        max_val_in_window = max(max_val_in_window, current_val_in_window)
        # Slide the window across the array to find the maximum number of `val` in any window
        while end < length:
            if data[start] == val:
                current_val_in_window -= 1
            start += 1
            if data[end] == val:
                current_val_in_window += 1
            end += 1
            max_val_in_window = max(max_val_in_window, current_val_in_window)
        # Minimum swaps are the total `val` minus the maximum found in any window
        return total_val_count - max_val_in_window
    
# https://leetcode.com/problems/make-two-arrays-equal-by-reversing-subarrays/description/?envType=daily-question&envId=2024-08-03
class Solution:
    def canBeEqual(self, target: List[int], arr: List[int]) -> bool:
        seen = Counter(arr)
        t = Counter(target)
        return seen == t
      
      
# https://leetcode.com/problems/kth-distinct-string-in-an-array/description/?envType=daily-question&envId=2024-08-05
class Solution:
    def kthDistinct(self, arr: List[str], k: int) -> str:
        c = Counter(arr)
        res = ""
        for char, val in c.items():
            if val >= 2:
                continue
            if val <= 1:
                k -= 1
            if k == 0:
                res = char
        if k > 0:
            return ""
        else:
            return res

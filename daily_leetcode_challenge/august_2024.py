
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
<<<<<<< HEAD
      
      
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

# https://leetcode.com/problems/minimum-number-of-pushes-to-type-word-ii/description/?envType=daily-question&envId=2024-08-06
"""
Approach 1: Greedy Sorting
Intuition
To solve this problem, we use a greedy algorithm approach combined with sorting. Keeping in mind that we have 8 keys available (2-9), the primary intuition is to remap the keys so the 8 most frequently occurring characters in the given string are assigned as first key presses, the next most common 8 characters as second key presses, and so on.

We begin by counting the occurrences of each letter using a counter, which provides the frequency of each distinct letter. Next, we sort these frequencies in descending order.

Since there are 8 possible key assignments, we'll divide the frequency rank by 8 to group it as a first, second, or third key press. Note that dividing the frequencies by 8 will result in 0, 1, and 2. We must add 1 to this group number to get the actual number of presses required for letters in that group. Multiplying this by the number of times the character appears in the given string yields the total number of presses for that letter.

Finally, we will sum the total presses required to type the word.

This greedy way, combined with sorting by frequency, ensures that each decision (assignment of letters to keys) is optimal for minimizing key presses.


Understand Intuition:
8 possible key assignments
we'll divide the frequency rank by 8 to group it as a first, second, or third key press.
We must add 1 to this group number to get the actual number of presses required for letters in that group.
Multiplying this by the number of times the character appears in the given string yields the total number of presses for that letter.
"""
class Solution:
    def minimumPushes(self, word: str) -> int:
        # Frequency list to store count of each letter 
        frequency = [0] * 26
        # Count occurences of each letter
        for c in word:
            frequency[ord(c) - ord("a")] += 1
        # Sort Frequencies in descending order
        frequency.sort(reverse=True)
        total_pushes = 0
        # Calculate the number of pressesj
        for i in range(26):
            if frequency[i] == 0:
                break
            total_pushes += (i // 8 + 1) * frequency[i]
        return total_pushes


# https://leetcode.com/problems/integer-to-english-words/?envType=daily-question&envId=2024-08-07
# Recursive approach
class Solution:
    below_ten = ["", "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine"]
    below_twenty = ["Ten", "Eleven", "Twelve", "Thirteen", "Fourteen", "Fifteen", "Sixteen", "Seventeen", "Eighteen", "Nineteen"]
    below_hundred = ["", "Ten", "Twenty", "Thirty", "Forty", "Fifty", "Sixty", "Seventy", "Eighty", "Ninety"]
    def numberToWords(self, num: int) -> str:
        if num == 0:
            return "Zero"
        return self._convert_to_words(num)
    def _convert_to_words(self, num: int) -> str:
        if num < 10:
            return self.below_ten[num]
        if num < 20:
            return self.below_twenty[num - 10]
        if num < 100:
            return self.below_hundred[num // 10] + (" " + self._convert_to_words(num % 10) if num % 10 != 0 else "")
        if num < 1000:
            return self._convert_to_words(num // 100) + " Hundred" + (" " + self._convert_to_words(num % 100) if num % 100 != 0 else "")
        if num < 1000000:
            return self._convert_to_words(num // 1000) + " Thousand" + (" " + self._convert_to_words(num % 1000) if num % 1000 != 0 else "")
        if num < 1000000000:
            return self._convert_to_words(num // 1000000) + " Million" + (" " + self._convert_to_words(num % 1000000) if num % 1000000 != 0 else "")
        return self._convert_to_words(num // 1000000000) + " Billion" + (" " + self._convert_to_words(num % 1000000000) if num % 1000000000 != 0 else "")
=======
    
# https://leetcode.com/problems/range-sum-of-sorted-subarray-sums/?envType=daily-question&envId=2024-08-04
# binary search and sliding window
"""
Approach: Priority Queue
Intuition:
We can maintain the sorted order of subarray sums using a pq, which stores elements in a sorted order using a heap data structure. By inserting all the subarray sums into the pq, we ensure that the smallest sums are always easily accessible.

Inserting all subarray sums into the pq results in the same time and space complexity as the previous approach, but it's possible to refine this strateg to optimize space complexity.

In our first approach, we created an array to store all possible subarray sums. In this approach, we'll use the pq to store pairs. The first element of each pair will reperesent the sum of the current subarray and the second element will represent the end index of that subarray. We'll initialize the pq with pairs representing all one-sized subarrays

As we process the queue, we repeatedly pop the smallest element, which represents the smallest subarray sum. However, this subarray could be part of a larger subarray. To account for this, we expand the subarray by one element (incrementing the end index), update its sum, and push the updated pair back into the pq.

Once we have performed exactly left pop operations, we start accumulating the subarray sums. The process continues until we return the accumulated sum.
"""
class Solution:
    import heapq
    def rangeSum(self, nums, n, left, right):
        pq = []
        for i in range(n):
            heapq.heappush(pq, (nums[i], i))
        ans = 0
        mod = 1e9 + 7
        for i in range(1, right + 1):
            p = heapq.heappop(pq)
            # If the current index is greater than or equal to left, add the
            # value to the answer.
            if i >= left:
                ans = (ans + p[0]) % mod
            # If index is less than the last index, increment it and add its
            # value to the first pair value.
            if p[1] < n - 1:
                p = (p[0] + nums[p[1] + 1], p[1] + 1)
                heapq.heappush(pq, p)
        return int(ans)
>>>>>>> adc7e57 (rangeSum)

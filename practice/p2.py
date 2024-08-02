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


from collections import deque


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
            nums[n - 1] - nums[3],  # change 3 smallest element
            nums[n - 2] - nums[2],  # change 2 smallest and 1 largest element
            nums[n - 3] - nums[1],  # change 1 smallest and 2 largest elements
            nums[n - 4] - nums[0],  # change 3 largest element
        )
        return min_diff


class Solution:
    def mergeNodes(self, head: Optional[ListNode]) -> Optional[ListNode]:
        if not head.next:
            return None
        ptr = head.next
        sum = 0
        while ptr.val != 0:
            sum += ptr.val
            ptr = ptr.next
        head.next.val = sum
        head.next.next = self.mergeNodes(ptr)
        return head.next


# https://leetcode.com/problems/find-the-minimum-and-maximum-number-of-nodes-between-critical-points/description/?envType=daily-question&envId=2024-07-05
class Solution:
    def nodesBetweenCriticalPoints(self, head: Optional[ListNode]) -> List[int]:
        pre = head
        cur = head.next
        ans = [-1, -1]
        prePosition, curPosition, firstPosition, position = -1, -1, -1, 0

        while cur.next is not None:
            if (cur.val < pre.val and cur.val < cur.next.val) or (
                cur.val > pre.val and cur.val > cur.next.val
            ):
                # local
                prePosition = curPosition
                curPosition = position

                if firstPosition == -1:
                    firstPosition = position
                if prePosition != -1:
                    if ans[0] == -1:
                        ans[0] = curPosition - prePosition
                    else:
                        ans[0] = min(ans[0], curPosition - prePosition)
                    ans[1] = position - firstPosition
            position += 1
            pre = cur
            cur = cur.next
        return ans


# https://leetcode.com/problems/pass-the-pillow/?envType=daily-question&envId=2024-07-06
class Solution:
    def passThePillow(self, n: int, time: int) -> int:
        return n - abs(n - 1 - time % (n * 2 - 2))


# https://leetcode.com/problems/water-bottles/?envType=daily-question&envId=2024-07-07
class Solution:
    def numWaterBottles(self, nb: int, ne: int) -> int:
        # 9 + 3 + 1 = 13
        return nb + (nb - 1) // (ne - 1)


# https://leetcode.com/problems/find-the-winner-of-the-circular-game/?envType=daily-question&envId=2024-07-08
class Solution:
    def findTheWinner(self, n: int, k: int) -> int:
        def recursion(n, k):
            if n == 1:
                return 0
            return (recursion(n - 1, k) + k) % n

        return recursion(n, k) + 1


# https://leetcode.com/problems/average-waiting-time/submissions/1315589100/?envType=daily-question&envId=2024-07-09
class Solution:
    def averageWaitingTime(self, c: List[List[int]]) -> float:
        at = 0
        tw = 0

        for a, t in c:
            at = max(at, a) + t
            tw += at - a
        return tw / len(c)

        # at = (0, 1) + 2 = 3
        # tw = 0 + 3 - 1 =  2

        # at = (2, 2) + 5 = 7
        # tw = 2 + 7 - 2 =  7


# https://leetcode.com/problems/crawler-log-folder/?envType=daily-question&envId=2024-07-10


# Return the minimum number of operations needed to go back to the main folder after the change folder operations.
class Solution:
    def minOperations(self, logs: List[str]) -> int:
        # "../" if main folder remain in the same folder

        # "./" remain in the same folder

        # "x/". move to child named x
        lvl = 0
        for l in logs:
            if lvl < 1:
                lvl = 0
            if "../" == l:
                lvl -= 1
            elif "./" == l:
                continue
            else:
                lvl += 1
        return max(0, lvl)


class Solution:
    def reverseParentheses(self, s: str) -> str:
        # Initialize a stack to keep track of the indices of '(' characters
        stack = deque()

        # Initialize a list to build the result string
        res = []

        # Iterate through each character in the input string
        for char in s:
            if char == "(":
                # If the character is '(', push the current length of the result list onto the stack
                # This keeps track of the position where the '(' was found
                stack.append(len(res))
            elif char == ")":
                # If the character is ')', pop the top index from the stack
                # This index indicates where the matching '(' was found
                idx = stack.pop()
                # Reverse the substring in the result list that is enclosed by these parentheses
                res[idx:] = res[idx:][::-1]
            else:
                # If the character is not a parenthesis, append it to the result list
                res.append(char)

        # Join all the characters in the result list to form the final output string
        return "".join(res)


# https://leetcode.com/problems/maximum-score-from-removing-substrings/?envType=daily-question&envId=2024-07-12
class Solution:
    def maximumGain(self, s: str, x: int, y: int) -> int:
        # Greedy with stack
        res = 0
        if y > x:
            top = "ba"
            top_score = y
            bot = "ab"
            bot_score = x
        else:
            top = "ab"
            top_score = x
            bot = "ba"
            bot_score = y
        
        # removing first top substring cause they give more points
        stack = []
        for char in s:
            if char == top[1] and stack and stack[-1] == top[0]:
                res += top_score
                stack.pop() # delete the first char of this substring
            else:
                stack.append(char)
        
        # removing bot substring cause they give less or equal amount of scores
        new_stack = []
        for char in stack:
            if char == bot[1] and new_stack and new_stack[-1] == bot[0]:
                res += bot_score
                new_stack.pop()
            else:
                new_stack.append(char)
        return res



# https://leetcode.com/problems/robot-collisions/description/?envType=daily-question&envId=2024-07-13
"""
Problem: 2751. Robot Collisions
There are n 1-indexed robots, each having a position on a line, health, and movement direction.

You are given 0-indexed integer arrays positions, healths, and a string directions (directions[i]) is either 'L' for left or 'R' for right). All integers in positions are unique.

All robots start moving on the line simultaneously at the same speed in their given direction. If two robots ever share the same position while moving, they will collide.

If two robots collide, the robot with lower health is removed from the line, and the health of the other robot decreases by one. The surviving robot continues in the same direction it was going. If both robots have the same health, they are both removed from the line.

Your task is to determine the health of the robots that survive the collisions, in the same order that the robots were given, i.e final health of robot 1 (if survived), final health of robot2 (if survived), and so on. If there are no survivors, return an empty array.

Return an array containing the health of the remaining robtos (in the order they were given in the input), after no further collisions can occur.

Note: The positions may be unsorted
"""

class Solution:
    def survivedRobotsHealths(self, positions: List[int], healths: List[int], directions: str) -> List[int]:
        n = len(positions) # number of robots
        indices = list(range(n))  # list of indices from 0 to n-1
        res = [] # list to store the healt of surviving robots
        stack = deque() # stack to store the indices of right-moving robots

        # sort indices based on their positions
        indices.sort(key=lambda x: positions[x])

        # iterate through each robot based on their sorted positions
        for current_index in indices:
            if directions[current_index] == "R":
                # if the current robot is moving right, add its index to the stack
                stack.append(current_index)
            else:
                # if the current robot is moving left, check for collisions with right-moving robots
                while stack and healths[current_index] > 0:
                    top_index = stack.pop()

                    if healths[top_index] > healths[current_index]:
                        # top robot survives, current robot is destroyed
                        healths[top_index] -= 1
                        healths[current_index] = 0
                        stack.append(top_index) # re-add top robot to the stack as it survived
                    elif healths[top_index] < healths[current_index]:
                        # current robot survives, top robot is destroyed
                        healths[current_index] -= 1
                        healths[top_index] = 0
                    else:
                        # both robots are destroyed if they have the same health
                        healths[current_index] = 0
                        healths[top_index] = 0

        # collect surviving robot's healths
        for i in range(n):
            if healths[i] > 0:
                res.append(healths[i])
        return res
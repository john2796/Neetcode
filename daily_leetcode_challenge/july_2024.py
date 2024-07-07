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

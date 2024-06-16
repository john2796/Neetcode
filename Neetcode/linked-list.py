
#https://leetcode.com/problems/reverse-linked-list/

# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def reverseList(self, head: Optional[ListNode]) -> Optional[ListNode]:
        prev, curr = None, head
        while curr:
            tmp = curr.next
            curr.next = prev
            prev = curr
            curr = tmp
        return prev

# https://leetcode.com/problems/merge-two-sorted-lists/

# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def mergeTwoLists(self, l1: Optional[ListNode], l2: Optional[ListNode]) -> Optional[ListNode]:
        dummy = node = ListNode()
        while l1 and l2:
            if l1.val < l2.val:
                node.next = l1
                l1 = l1.next
            else:
                node.next = l2
                l2 = l2.next
            node = node.next
        node.next = l1 or l2
        return dummy.next
    

# https://leetcode.com/problems/reorder-list/
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def reorderList(self, head: Optional[ListNode]) -> None:
        """
        Do not return anything, modify head in-place instead.
        """
        # find the middle of the list
        slow, fast=head,head
        while fast.next and fast.next.next:
            slow = slow.next
            fast = fast.next.next

        # reverse the second half of the list
        prev, cur = None, slow.next
        while cur:
            tmp = cur.next
            cur.next = prev
            prev = cur
            cur = tmp
        slow.next = None

        # merge 2 halves
        first ,second = head, prev
        while second:
            tmp1, tmp2 = first.next, second.next
            first.next =second
            second.next = tmp1
            first, second = tmp1, tmp2
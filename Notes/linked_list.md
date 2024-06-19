# Linked List

- A linedlist is a fundamental data structure used in computer science to represent a sequence of elements. Each element in a linked list is called a node, and each node contains two main components:

1. Data: the actual value or data stored in the node.
2. Pointer (or Reference): A referencce to the next node in the sequence

Unlike arrays, linked list do not require contigous memory location. Instead, the elements (nodes) can be scattered throughout memory, with the pointers connecting them. This structure allows for efficient insertion and deletion because these operations only involve updating the pointers

## Types of Linked Lists

1. **Singly Linked List**: Each node has a single pointer to the next node. The last node's pointer points to `null` or `None`, indicating the end of the list.
2. **Doubly Linked List**: Each node has two pointers, one pointing to the next node and one pointing to the previous node. This allows for traversal in both directions
3. **Circular Linked List**: The last node points back to the first node, forming a circle. This can be implemented as either a singly or doubly linked list.

## Advantages of Linked Lists

- Dynamic Size: Linked list can grow or shrink in size dynamically, as nodes are added or removed.
- Efficient Insertions/Deletions: Inserting or deleting a node does not require shifting elements, as is necessary in arrays.

## Disadvantages of Linked Lists

- Memory Overhead: Each node requires additional memory for the pointer(s).
- Sequential Access: Linked lists do not support direct access to elements by index. To access a specific element, you must traverse the list from the head node.
- Cache Locality: Linked lists may exhibit poor cache performance compared to arrays, due to non-contiguous memory allocation.

## Basic Operations

1. Traversal: Visiting each node in the list, typically starting from the head node.
2. Insertion: Adding a newe node to the list at a specified position.
3. Deletion: Removing a node from the list
4. Search: Finding a node with a specific value.

```python
class Node:
    def __init__(self, data):
        self.data = data
        self.next = None

class LinkedList:
    def __init__(self):
        self.head = None

    def append(sef, data):
        new_node = Node(data)
        if not self.head:
            self.head = new_node
            return
        curr = self.head
        while curr:
            curr = curr.head
        curr.next = new_node

    def delete_node(self, key):
        current = self.head
        # If the node to be deleted is the head node
        if current and current.data == key:
            self.head = current.next
            current = None
            return
        # Search for the key to be deleted
        prev = None
        while current and current.data != key:
            prev = current
            current = current.next
        # If the key was not present in the linked list
        if current is None:
            return
        # Unlink the node from the linked list
        prev.next = current.next
        current = None

    def delete_node_at_pos(self, pos):
        if self.head is None:
            return
        curr = self.head
        # If the head needs to be removed
        if pos == 0:
            self.head = curr.next
            curr = None
            return
        # Find the key to be deleted
        prev = None
        count = 0
        while current and count != pos:
            prev = curr
            curr = curr.next
            count += 1
        # If the position is greater than the number of nodes
        if curr is None:
            return
        # Unlink the node from the linked list
        prev.next = curr.next
        curr = None

    def search(self, key):
        curr = self.head
        while curr:
            if curr.data == key:
                return True
            curr = curr.next
        return False

    def print_list(self):
        curr = self.head
        while current:
            print(curr.data, end=" -> ")
            curr = curr.next
        print("None")
```

## Most common ways to apporach LeetCode Linked list problems

1. **Two Pointers (Fast and Slow)**
   This technique uses two pointers that move at different speeds to solve problems related to cycles, middle elements, and more.

Example: Linked List Cycle

```python
class Solution:
    def hasCycle(self, head: ListNode) -> bool:
        slow, fast = head, head
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next
            if slow == fast:
                return True
        return False
```

Example: Middle of the Linked List

```python
class Solution:
    def middleNode(self, head: ListNode) -> ListNode:
        slow, fast = head, head
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next
        return slow
```

2. **Reverse a Linked List**
   Reversing a linked list is a common subroutine for many linked list problems.

Example: Reverse Linked List

```python
class Solution:
    def reverseList(self, head: ListNode) -> ListNode:
        prev = None
        curr = head
        while curr:
            next_temp = curr.next
            curr.next = prev
            prev = curr
            curr = next_temp
        return prev
```

3. **Merging Two Linked Lists**
   Combining two sorted linked lists into one sorted linked list is a frequent task.

Example: Merge Two sorted Lists

```python
class Solution:
    def mergeTwoLists(self, l1: ListNode, l2: ListNode) -> ListNode:
        dummy = ListNode()
        tail = dummy

        while l1 and l2:
            if l1.val < l2.val:
                tail.next = l1
                l1 = l1.next
            else:
                tail.next = l2
                l2 = l2.next
            tail = tail.next
        tail.next = l1 if l1 else l2
        return dummy.next
```

4. **Detect and Remove Cycles**
   Detecting and removing cycle is essential for problems related to cyclic linked lists.

Example: Linked List Cycle II

```python
class Solution:
    def detectCycle(selef, head: ListNode) -> ListNode:
        slow, fast = head, head
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next
            if slow == fast:
                slow = head
                while slow != fast:
                    slow = slow.next
                    fast = fast.next
                return slow
        return None
```

5. **Finding the Intersection of Two Linked Lists**
   Identifying the intersection node between two linked lists.

Example: Intersection of Two Linked Lists

```python
class Solution:
    def getIntersectionNode(self, headA: ListNode, headB: ListNode) -> ListNode:
        if not headA or not headB:
            return None
        pA, pB = headA, headB
        while pA != pB:
            pA = headB if not pA else pA.next
            pB = headA if not pB else pB.next
        return pA
```

6. **Removing N-th Node from End**
   This involves find the n-th node from the end and removing it.

Example: Remove N-t Node from the End of list

```python
class Solution:
    def removeNthFromEnd(self, head: ListNode, n: int) -> ListNode:
        dummy = ListNode(0, head)
        slow = fast = dummy
        for _ in range(n + 1):
            fast = fast.next
        while fast:
            slow = slow.next
            slow = fast.next
        slow.next = slow.next.next
        return dummy.next
```

7. Recursion Solutions
   Sometimes recursion provides a more elegant solution, especially for problems involving nested or hierarchical structures.

Example: Reverse Linked List II

```python
class Solution:
    def reverseBetween(self, head: ListNode, left: int, right: int) -> ListNode:
        if not head:
            return None

        def reverseN(node, n):
            if n == 1:
                return node
            last = reverseN(node.next, n-1)
            node.next.next = node
            return last

        dummy = ListNode(0, head)
        prev = dummy

        for _ in range(left - 1):
            prev = prev.next

        prev.next = reverseN(prev.next, right - left + 1)
        prev.next.next = None

        return dummy.next
```

## Tips for solving Linked List Problems:

1. Understand the Problem: Carefully read the problem statement and understand the requirements. Draw diagrams if necessary.
2. Edge Cases: Consider edge case such as empty lists, single-node lists, and lists with cycles.
3. Pointer Manipulation: Be cautious with pointer manipulation to avoid issues like null pointer dereference.
4. Dummy Nodes: Use dummy nodes to simplify edge cases, especially when dealing with head nodes.
5. Practice: Regular practice with a variety of linked list problems helps in recognizing patterns and common techniues.

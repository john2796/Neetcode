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

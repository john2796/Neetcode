Certainly! Here's a brief overview of how to solve each problem and the approach
used:

### Arrays & Hashing

1. **Contains Duplicate**: Use a hash set to track seen elements; if an element
   is already in the set, return true.
2. **Valid Anagram**: Sort both strings and compare, or use a hash map to count
   character frequencies and compare.
3. **Two Sum**: Use a hash map to store indices of elements, and check if the
   complement of the current element exists in the map.
4. **Group Anagrams**: Use a hash map where the key is a sorted version of the
   string and the value is a list of anagrams.
5. **Top K Frequent Elements**: Use a hash map to count frequencies and then a
   heap (or bucket sort) to find the top K elements.
6. **Encode and Decode Strings**: For encoding, prepend the length of each
   string, and for decoding, extract based on the length.
7. **Product of Array Except Self**: Use two arrays to track the product of all
   elements to the left and right of each index, then multiply them.
8. **Valid Sudoku**: Use hash sets to track numbers seen in each row, column,
   and 3x3 sub-box.
9. **Longest Consecutive Sequence**: Use a hash set to store all numbers, then
   iterate through, checking the length of sequences that start with the
   smallest number.

### Two Pointers

1. **Valid Palindrome**: Use two pointers, one starting from the beginning and
   the other from the end, to compare characters while skipping
   non-alphanumerics.
2. **Two Sum II (Input Array Is Sorted)**: Use two pointers, one at the start
   and one at the end, and move them based on the sum comparison.
3. **3Sum**: Sort the array, and use a fixed pointer with a two-pointer approach
   to find pairs that sum to the negative of the fixed element.
4. **Container with Most Water**: Use two pointers at the start and end, moving
   the pointer with the smaller height to maximize area.
5. **Trapping Rain Water**: Use two pointers to track the maximum heights from
   both ends, accumulating trapped water based on the shorter height.

### Sliding Window

1. **Best Time to Buy and Sell Stock**: Track the minimum price and calculate
   the maximum profit as you iterate through the prices.
2. **Longest Substring Without Repeating Characters**: Use a sliding window with
   a hash map to track characters and their indices.
3. **Longest Repeating Character Replacement**: Use a sliding window to track
   character frequencies and maximize the window size under the constraint.
4. **Permutation in String**: Use a sliding window with two hash maps to compare
   frequencies of characters in the current window and the target string.
5. **Minimum Window Substring**: Use a sliding window with a hash map to track
   characters needed and minimize the window size when all required characters
   are found.
6. **Sliding Window Maximum**: Use a deque to keep track of indices of maximum
   elements within the current window.

### Stack

1. **Valid Parenthesis**: Use a stack to track opening brackets and match them
   with closing ones.
2. **Min Stack**: Use two stacks, one to track the actual values and another to
   track the current minimum.
3. **Evaluate Reverse Polish Notation**: Use a stack to evaluate expressions by
   pushing numbers and popping the top two elements for operations.
4. **Generate Parenthesis**: Use backtracking with a stack to explore all valid
   combinations of parentheses.
5. **Daily Temperatures**: Use a stack to track indices of temperatures, and for
   each temperature, pop the stack when a warmer day is found.
6. **Car Fleet**: Sort cars by position and use a stack to track fleets based on
   their arrival times.
7. **Largest Rectangle in Histogram**: Use a stack to keep track of bars and
   calculate the largest rectangle by popping bars and computing areas.

### Binary Search

1. **Binary Search**: Use the classic binary search algorithm by repeatedly
   dividing the search interval in half.
2. **Search a 2D Matrix**: Treat the 2D matrix as a flattened sorted array and
   apply binary search.
3. **Koko Eating Bananas**: Use binary search to determine the minimum eating
   speed by checking if she can finish within the given hours.
4. **Find Minimum in Rotated Sorted Array**: Use binary search to find the
   inflection point where the rotation occurs.
5. **Search in Rotated Sorted Array**: Use binary search while adjusting for the
   rotation to find the target.
6. **Time Based Key Value Store**: Use binary search to retrieve the largest
   timestamp that is less than or equal to the query timestamp.
7. **Median of Two Sorted Arrays**: Use binary search to partition the two
   arrays so that the combined halves are correctly ordered.

### Linked List

1. **Reverse Linked List**: Iteratively reverse the pointers of each node or use
   recursion to reverse the list.
2. **Merge Two Sorted Lists**: Use a dummy node and iteratively compare and
   merge the nodes of both lists.
3. **Reorder List**: Split the list into two halves, reverse the second half,
   and merge the two halves.
4. **Remove Nth Node From End of List**: Use two pointers, with one starting n
   steps ahead, and then move both until the first reaches the end.
5. **Copy List with Random Pointer**: Use a hash map to store the mapping of
   original to copied nodes, or interleave the list and then separate it.
6. **Add Two Numbers**: Traverse both lists, adding corresponding nodes, and
   handle carry over with a new linked list.
7. **Linked List Cycle**: Use two pointers (slow and fast) to detect if they
   meet, indicating a cycle.
8. **Find the Duplicate Number**: Use two pointers in a cycle detection method
   (Floyd’s Tortoise and Hare) to find the duplicate.
9. **LRU Cache**: Use a combination of a doubly linked list and a hash map to
   manage the cache and access times efficiently.
10. **Merge K Sorted Lists**: Use a min-heap to keep track of the smallest
    elements across all lists and merge them.
11. **Reverse Nodes in k-Group**: Reverse every k nodes in the linked list using
    an iterative approach with a dummy node.

### Trees

1. **Invert Binary Tree**: Recursively swap the left and right children of every
   node.
2. **Diameter of Binary Tree**: Use depth-first search (DFS) to calculate the
   height of the tree while keeping track of the maximum diameter found.
3. **Balanced Binary Tree**: Recursively check the height of the left and right
   subtrees and ensure the difference is not more than one.
4. **Same Tree**: Use DFS to recursively compare each node of both trees.
5. **Subtree of Another Tree**: Use DFS to compare each subtree in the main tree
   with the given subtree.
6. **Lowest Common Ancestor of a Binary Search Tree**: Use the properties of the
   BST to recursively find the split point where one node is in the left subtree
   and the other is in the right.
7. **Binary Tree Level Order Traversal**: Use a queue to perform breadth-first
   search (BFS) and collect nodes level by level.
8. **Binary Tree Right Side View**: Use BFS and collect the last node at each
   level, which represents the rightmost view.
9. **Count Good Nodes in Binary Tree**: Use DFS to traverse the tree and count
   nodes that are greater than or equal to all previous nodes on the path.
10. **Validate Binary Search Tree**: Use DFS to ensure that each node's value is
    within a valid range (defined by its ancestors).
11. **Kth Smallest Element in a BST**: Use an in-order traversal to visit nodes
    in ascending order and return the kth element.
12. **Construct Binary Tree from Preorder and Inorder Traversal**: Use the
    preorder list to determine the root and recursively construct the tree using
    the inorder list for left and right subtrees.
13. **Binary Tree Maximum Path Sum**: Use DFS to explore all paths, keeping
    track of the maximum sum by considering both the node and the path.
14. **Serialize and Deserialize Binary Tree**: Use BFS or DFS to serialize the
    tree into a string and then deserialize it back by reconstructing the nodes
    in the same order.

### Heap / Priority Queue

1. **Kth Largest Element in a Stream**: Maintain a min-heap of size k to track
   the k largest elements.
2. **Last Stone Weight**: Use a max-heap to repeatedly smash the two largest
   stones until one or none is left.
3. **K Closest Points to Origin**: Use a max-heap to track the k closest points,
   pushing and popping based on the distance.
4. **Task Scheduler**: Use a max-heap to manage task frequencies and a queue to
   handle cooldown periods.
5. **Design Twitter**: Use hash maps and heap to merge tweets from followed
   users, keeping track of the most recent tweets.
6. **Find Median Data Stream**: Use two heaps (max-heap and min-heap) to balance
   the lower and upper halves of the data to find the median efficiently.

### Backtracking

1. **Subsets**: Use backtracking to explore all combinations by either including
   or excluding each element.
2. **Combination Sum**: Use backtracking to explore combinations that sum to the
   target, reusing elements if necessary.
3. **Permutation**: Use backtracking to generate all possible permutations by
   swapping elements.
4. **Subsets II**: Similar to Subsets but skip duplicates by checking the
   previous element.
5. **Combination Sum II**: Like Combination Sum, but skip duplicates and each
   element can only

be used once. 6. **Word Search**: Use DFS to explore all possible paths on the
board to find the word, marking cells as visited. 7. **Palindrome
Partitioning**: Use backtracking to explore all ways to partition the string
into palindromic substrings. 8. **Letter Combinations of a Phone Number**: Use
backtracking to generate all possible letter combinations from the digit
mapping. 9. **N Queens**: Use backtracking to place queens on the board such
that no two queens threaten each other.

### Tries

1. **Implement Trie (Prefix Tree)**: Build a trie structure with nodes for each
   character to support insert, search, and startsWith operations.
2. **Design Add and Search Words Data Structure**: Extend the trie to handle '.'
   as a wildcard character, enabling flexible searches.
3. **Word Search II**: Use a trie to store words and then DFS on the board to
   find all words in the trie.

### Graphs

1. **Number of Islands**: Use BFS or DFS to traverse the grid and mark all
   connected '1's as visited for each island.
2. **Max Area of Island**: Use DFS to explore each island and calculate its area
   by counting connected '1's.
3. **Clone Graph**: Use BFS or DFS with a hash map to clone nodes and their
   neighbors.
4. **Walls and Gates**: Use BFS starting from all gates simultaneously to fill
   in the shortest distance to each gate.
5. **Rotting Oranges**: Use BFS starting from all rotten oranges simultaneously
   to calculate the minimum time to rot all fresh oranges.
6. **Pacific Atlantic Water Flow**: Use DFS or BFS from both oceans and find the
   cells reachable from both.
7. **Surrounded Regions**: Use DFS to mark 'O's connected to the boundary and
   then flip the rest.
8. **Course Schedule**: Use DFS to detect cycles in the graph (course
   prerequisites) to determine if all courses can be finished.
9. **Course Schedule II**: Use topological sorting (DFS or BFS) to determine the
   order in which courses should be taken.
10. **Graph Valid Tree**: Use DFS to check for cycles and whether the graph is
    connected to validate if it's a tree.
11. **Number of Connected Components in an Undirected Graph**: Use DFS or BFS to
    count all connected components in the graph.
12. **Redundant Connection**: Use Union-Find to detect and remove the edge that
    creates a cycle in the graph.
13. **Word Ladder**: Use BFS to transform the word step by step and track the
    transformation sequence.

### Advanced Graphs

1. **Reconstruct Itinerary**: Use DFS to build the itinerary in reverse,
   ensuring lexical order by sorting destinations.
2. **Min Cost to Connect All Points**: Use Kruskal's or Prim's algorithm to find
   the minimum spanning tree that connects all points.
3. **Network Delay Time**: Use Dijkstra's algorithm to calculate the shortest
   time for all nodes to receive the signal.
4. **Swim in Rising Water**: Use binary search combined with BFS/DFS to
   determine the minimum time to swim from top-left to bottom-right.
5. **Alien Dictionary**: Use topological sorting to derive the order of
   characters from the alien language.
6. **Cheapest Flights Within K Stops**: Use BFS or Dijkstra's algorithm to find
   the cheapest flight route within k stops.

### 1-D Dynamic Programming (DP)

1. **Climbing Stairs**: Use DP to calculate the number of ways to reach the top
   by summing the ways to reach the previous two steps.
2. **Min Cost Climbing Stairs**: Use DP to find the minimum cost to reach the
   top, considering the cost of each step.
3. **House Robber**: Use DP to decide whether to rob each house based on
   maximizing the amount of money without robbing adjacent houses.
4. **House Robber II**: Similar to House Robber but account for the circular
   arrangement of houses.
5. **Longest Palindromic Substring**: Use DP to track palindromic substrings and
   expand around centers to find the longest one.
6. **Palindromic Substrings**: Use DP or expand around centers to count all
   palindromic substrings.
7. **Decode Ways**: Use DP to calculate the number of ways to decode a string,
   considering single and double digits.
8. **Coin Change**: Use DP to find the minimum number of coins needed to make
   the amount by trying all possibilities.
9. **Maximum Product Subarray**: Use DP to track the maximum and minimum
   products up to each element, updating the result accordingly.
10. **Word Break**: Use DP to determine if the string can be segmented into
    valid words in the dictionary.
11. **Longest Increasing Subsequence**: Use DP to build the longest subsequence
    where each element is greater than the previous one.
12. **Partition Equal Subset Sum**: Use DP to determine if the array can be
    partitioned into two subsets with equal sums.

### 2-D Dynamic Programming (DP)

1. **Unique Paths**: Use DP to calculate the number of ways to reach each cell
   by summing the ways from the top and left cells.
2. **Longest Common Subsequence**: Use DP to compare two strings and build the
   longest subsequence common to both.
3. **Best Time to Buy and Sell Stock with Cooldown**: Use DP to maximize profit
   while considering the cooldown period after selling.
4. **Coin Change II**: Use DP to count the number of ways to make the amount
   with the given coins, considering combinations.
5. **Target Sum**: Use DP to calculate the number of ways to assign symbols to
   make the sum equal to the target.
6. **Interleaving String**: Use DP to check if the third string is an
   interleaving of the other two by maintaining a 2D table.
7. **Longest Increasing Path in a Matrix**: Use DP with memoization to find the
   longest increasing path in the matrix by exploring all directions.
8. **Distinct Subsequences**: Use DP to count the number of distinct
   subsequences of one string that equals the other string.
9. **Edit Distance**: Use DP to calculate the minimum number of operations
   required to convert one string into another.
10. **Burst Balloons**: Use DP to calculate the maximum coins that can be
    obtained by strategically bursting balloons.
11. **Regular Expression Matching**: Use DP to match a string against a pattern
    that includes '.' and '\*' as wildcards.

### Greedy

1. **Maximum Subarray**: Use a greedy approach (Kadane's algorithm) to find the
   subarray with the maximum sum.
2. **Jump Game**: Use a greedy approach to keep track of the farthest position
   you can reach and see if you can reach the end.
3. **Jump Game II**: Use a greedy approach to minimize the number of jumps
   needed to reach the end by always jumping to the farthest reachable position.
4. **Gas Station**: Use a greedy approach to find the starting point where you
   can complete the circuit by checking if the total gas is sufficient.
5. **Hand of Straights**: Use a greedy approach with a frequency map to form
   hands by consecutively grouping cards.
6. **Merge Triplets to Form Target Triplet**: Use a greedy approach to check if
   the target can be formed by merging valid triplets.
7. **Partition Labels**: Use a greedy approach to partition the string such that
   each letter appears in at most one part, based on its last occurrence.
8. **Valid Parenthesis String**: Use a greedy approach to ensure the string can
   be balanced by tracking the possible number of open parentheses.

### Intervals

1. **Insert Interval**: Use a greedy approach to merge the new interval into the
   existing list, adjusting overlaps.
2. **Merge Intervals**: Sort intervals by start time, then merge overlapping
   intervals by comparing end times.
3. **Non-Overlapping Intervals**: Use a greedy approach to remove the minimum
   number of intervals to avoid overlap, prioritizing intervals that end the
   earliest.
4. **Meeting Rooms**: Sort intervals by start time and check for overlaps to
   determine if all meetings can be attended.
5. **Meeting Rooms II**: Use a min-heap to track the end times of meetings and
   determine the minimum number of rooms required.
6. **Minimum Interval to Include Each Query**: Use a combination of sorting and
   a priority queue to find the smallest interval containing each query.

### Math & Geometry

1. **Rotate Image**: Transpose the matrix and then reverse each row to rotate
   the image 90 degrees clockwise.
2. **Spiral Matrix**: Simulate the spiral order by iterating over the matrix in
   layers, adjusting boundaries after completing each layer.
3. **Set Matrix Zeroes**: Use the first row and column as markers to set entire
   rows and columns to zero based on the original matrix.
4. **Happy Number**: Use a hash set to detect cycles in the sequence of sums of
   squares of digits.
5. **Plus One**: Start from the last digit, add one, and handle carry over until
   the beginning of the array.
6. **Pow(x, n)**: Use exponentiation by squaring to efficiently calculate the
   power, handling both positive and negative exponents.
7. **Multiply Strings**: Simulate multiplication as you would by hand, storing
   intermediate results in an array and adjusting for carry.
8. **Detect Squares**: Use a hash map to store points and their frequencies, and
   check possible squares by comparing distances between points.

### Bit Manipulation

1. **Single Number**: Use XOR to cancel out all pairs, leaving the single
   number.
2. **Number of 1 Bits**: Use a bitwise AND operation with n-1 repeatedly

to count the number of 1s in the binary representation. 3. **Counting Bits**:
Use a dynamic programming approach with bitwise operations to count the number
of 1s for each number. 4. **Reverse Bits**: Use bit manipulation to reverse the
bits by shifting and masking. 5. **Missing Number**: Use XOR to find the missing
number, as XORing all numbers with their indices will cancel out all but the
missing number. 6. **Sum of Two Integers**: Use bitwise operations to simulate
addition without using the '+' operator, handling carry with AND and XOR. 7.
**Reverse Integer**: Reverse the digits of the number and check for overflow by
comparing the reversed number with the limits. 8. **Decode XORed Permutation**:
Use XOR to reconstruct the permutation from the encoded array, leveraging the
properties of XOR.

This concise summary provides the core approach for solving each type of problem
across various topics.

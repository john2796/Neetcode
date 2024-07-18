# ---- Arrays & Hashing

# contains duplicate
# Approach: Use a set to track seen elements. If an element is already in the set, return True. If we finish the loop, return False.

# valid anagram
# Approach: Count the frequency of each character in both strings and compare the counts. If they match, the strings are anagrams.

# two sum
# Approach: Use a dictionary to store the difference between the target and the current element. If the current element is in the dictionary, we found a pair.

# group anagram
# Approach: Use a dictionary to group words that are anagrams by sorting each word and using the sorted word as the key.

# top k frequent elements
# Approach: Use a dictionary to count frequencies and a heap to keep track of the top k elements.

# encode and decode strings
# Approach: Use length-prefix encoding to handle arbitrary characters, including delimiters.

# product of array except self
# Approach: Use two passes to calculate the product of all elements to the left and right of each element.

# valid sudoku
# Approach: Use sets to track seen numbers in each row, column, and subgrid.

# longest consecutive sequence
# Approach: Use a set to track elements and find the longest sequence by checking the length of sequences starting from each number.

#---- Two Pointers

# valid palindrome
# Approach: Use two pointers to compare characters from the beginning and end of the string, moving inward.

# two sum II input array is sorted
# Approach: Use two pointers, one at the beginning and one at the end, and adjust them based on the sum comparison with the target.

# 3sum
# Approach: Sort the array and use a pointer for each element combined with two pointers for the remaining part of the array.

# container with most water
# Approach: Use two pointers at the ends of the array and move them inward, keeping track of the maximum area.

# trapping rain water
# Approach: Use two pointers to traverse the array while maintaining the maximum heights seen so far from both ends.

#---- Sliding Window

# best time to buy and sell stock
# Approach: Track the minimum price seen so far and calculate the maximum profit by comparing the current price with the minimum price.

# longest substring without repeating characters
# Approach: Use a sliding window with a set to track characters in the current substring.

# longest repeating character replacement
# Approach: Use a sliding window and a frequency map to keep track of the most frequent character and the size of the window.

# permutation in string
# Approach: Use a sliding window and two frequency maps to compare the current window with the target string's frequency.

# minimum window substring
# Approach: Use two pointers to expand and contract the window while maintaining a count of required characters.

# sliding window maximum
# Approach: Use a deque to keep track of the indices of maximum elements in the current window.

#---- Stack

# valid parenthesis
# Approach: Use a stack to keep track of opening brackets and match them with closing brackets.

# min stack
# Approach: Use two stacks, one to store all elements and another to store the minimums.

# evaluate reverse polish notation
# Approach: Use a stack to evaluate the expression by pushing operands and applying operators to the top elements.

# generate parenthesis
# Approach: Use backtracking to generate all valid combinations of parentheses.

# daily temperatures
# Approach: Use a stack to keep track of temperatures' indices and calculate the days until a warmer temperature.

# car fleet
# Approach: Sort cars by position and use a stack to determine the number of fleets based on their arrival times.

# largest rectangle in histogram
# Approach: Use a stack to keep track of heights and calculate the maximum rectangle area.

# ----- Binary Search -----

# binary search
# Approach: Use the binary search algorithm to find the target by repeatedly dividing the search interval in half.

# search a 2d matrix
# Approach: Treat the 2D matrix as a sorted array and use binary search.

# koko eating banana
# Approach: Use binary search to find the minimum eating speed by checking if Koko can eat all bananas within the given hours.

# find minimum in rotated sorted array
# Approach: Use binary search to find the minimum element in a rotated sorted array.

# search in rotated sorted array
# Approach: Use binary search to find the target in a rotated sorted array.

# time based key value store
# Approach: Use binary search to find the value corresponding to the latest timestamp that is less than or equal to the given timestamp.

# median of two sorted arrays
# Approach: Use binary search to partition the arrays into two halves and find the median.

# -----Linked List

# reverse linked list
# Approach: Use iterative or recursive approach to reverse the links between nodes.

# merge two sorted lists
# Approach: Use a dummy node to merge two sorted linked lists.

# reorder list
# Approach: Split the list, reverse the second half, and merge both halves.

# remove nth node from end of list
# Approach: Use two pointers to find the nth node from the end and remove it.

# copy list with random pointer
# Approach: Use a hash map to store the mapping of old nodes to new nodes and copy the list.

# add two numbers
# Approach: Use two pointers to add digits and handle carry.

# linked list cycle
# Approach: Use two pointers to detect if a cycle exists in the list.

# find the duplicate number
# Approach: Use Floyd's Tortoise and Hare (Cycle Detection) to find the duplicate number.

# lru cache
# Approach: Use a hash map and a doubly linked list to implement LRU cache.

# merge k sorted lists
# Approach: Use a priority queue (min-heap) to merge k sorted linked lists.

# reverse nodes in k group
# Approach: Reverse nodes in groups of k using iterative or recursive approach.

# ----- Trees 

# invert binary tree
# Approach: Use recursive or iterative approach to invert the left and right children of each node.

# diameter of binary tree
# Approach: Use depth-first search to calculate the diameter at each node.

# balanced binary tree
# Approach: Use depth-first search to check if the height difference between left and right subtrees is not more than 1.

# same tree
# Approach: Use depth-first search to compare nodes of both trees.

# subtree of another tree
# Approach: Use depth-first search to check if one tree is a subtree of another.

# lowest common ancestor of a binary search tree
# Approach: Use the properties of BST to find the lowest common ancestor.

# binary tree level order traversal
# Approach: Use breadth-first search to traverse the tree level by level.

# binary tree right side view
# Approach: Use breadth-first search to capture the rightmost element at each level.

# count good nodes in binary tree
# Approach: Use depth-first search to count nodes that are greater than or equal to all ancestor nodes.

# validate binary search tree
# Approach: Use depth-first search to validate that all nodes follow BST properties.

# kth smallest element in a BST
# Approach: Use in-order traversal to find the kth smallest element.

# construct binary tree from preorder and inorder traversal
# Approach: Use recursion to build the tree by selecting the root from preorder and dividing the tree using inorder.

# binary tree maximum path sum
# Approach: Use depth-first search to calculate the maximum path sum passing through each node.

# serialize and deserialize binary tree
# Approach: Use pre-order traversal to serialize and deserialize the binary tree.

# ----- Heap / Priority Queue 

# kth largest element in a stream
# Approach: Use a min-heap to keep track of the kth largest element in a stream of numbers.

# last stone weight
# Approach: Use a max-heap to repeatedly smash the two heaviest stones.

# k closest points to origin
# Approach: Use a max-heap to keep track of the k closest points to the origin.

# task scheduler
# Approach: Use a max-heap to schedule tasks and a queue to track cooldown periods.

# design twitter
# Approach: Use hash maps and heaps to implement the Twitter design.

# find median data stream
# Approach: Use two heaps to maintain the lower and upper halves of the data stream.

# ----- Backtracking 

# subsets
# Approach: Use backtracking to generate all possible subsets of a given set.

# combination sum
# Approach: Use backtracking to find all combinations of numbers that sum up to a target.

# permutation
# Approach: Use backtracking to generate all permutations of a given set.

# subsets II
# Approach: Use backtracking to generate all possible subsets of a given set, handling duplicates.

# combination sum II
# Approach: Use backtracking to find all unique combinations of numbers that sum up to a target.

# word search
# Approach: Use backtracking to search for a word in a grid by exploring all possible paths.

# palindrome partitioning
# Approach: Use backtracking to partition a string into all possible palindromic substrings.

# letter combination of a phone number
# Approach: Use backtracking to generate all possible letter combinations for a phone number.

# n queens
# Approach: Use backtracking to place queens on a chessboard so that no two queens threaten each other.

#----- Tries 

# implement trie prefix
# Approach: Use a trie data structure to store and search for prefixes efficiently.

# design add and search words data
# Approach: Use a trie data structure to store and search for words with support for '.' wildcard.

# word search II
# Approach: Use a trie and backtracking to find all words in a grid that match words in a dictionary.

# ----- Graphs

# number of islands
# Approach: Use depth-first search or breadth-first search to count the number of connected components of land.

# max area of island
# Approach: Use depth-first search or breadth-first search to find the maximum area of connected land.

# clone graph
# Approach: Use depth-first search or breadth-first search to clone a graph.

# walls and gates
# Approach: Use breadth-first search to fill each empty room with the distance to its nearest gate.

# rotting oranges
# Approach: Use breadth-first search to find the minimum time required for all oranges to rot.

# pacific atlantic water flow
# Approach: Use depth-first search or breadth-first search to find cells that can flow to both the Pacific and Atlantic oceans.

# surrounded regions
# Approach: Use depth-first search or breadth-first search to capture all regions surrounded by 'X'.

# course schedule 
# Approach: Use depth-first search to detect cycles in a directed graph to determine if all courses can be finished.

# course schedule II
# Approach: Use depth-first search or breadth-first search to find a topological order of courses.

# graph valid tree
# Approach: Use depth-first search to check if a graph is a single connected component with no cycles.

# number of connected components in an undirected graph
# Approach: Use depth-first search or breadth-first search to count connected components in a graph.

# redundant connection  
# Approach: Use union-find to detect and return the redundant connection in a graph.

# word ladder                                    
# Approach: Use breadth-first search to find the shortest transformation sequence from start word to end word.

#---- Advanced Graphs 

# reconstruct itinerary
# Approach: Use depth-first search to reconstruct the itinerary in lexical order.

# min cost to connect all points
# Approach: Use Kruskal's or Prim's algorithm to find the minimum cost to connect all points.

# network delay time
# Approach: Use Dijkstra's algorithm to find the shortest time for all nodes to receive a signal from the source.

# swim in rising water
# Approach: Use binary search and depth-first search to find the minimum time to swim from the top left to bottom right corner.

# alien dictionary
# Approach: Use depth-first search to find the order of letters in an alien language based on given words.

# cheapest flights within k stops
# Approach: Use breadth-first search with a priority queue to find the cheapest flight with at most k stops.

#---- 1-D DP

# climbing stairs
# Approach: Use dynamic programming to count the number of ways to reach the top.

# min cost climbing stairs
# Approach: Use dynamic programming to find the minimum cost to reach the top.

# house robber
# Approach: Use dynamic programming to find the maximum amount of money that can be robbed without robbing adjacent houses.

# house robber II
# Approach: Use dynamic programming to find the maximum amount of money that can be robbed in a circular array of houses.

# longest palindromic substring
# Approach: Use dynamic programming to find the longest palindromic substring.

# palindromic substrings
# Approach: Use dynamic programming to count the number of palindromic substrings.

# decode ways 
# Approach: Use dynamic programming to count the number of ways to decode a string of digits.

# coin change
# Approach: Use dynamic programming to find the minimum number of coins needed to make up a given amount.

# maximum product subarray
# Approach: Use dynamic programming to find the maximum product of a contiguous subarray.

# word break
# Approach: Use dynamic programming to determine if a string can be segmented into words from a dictionary.

# longest increasing subsequence
# Approach: Use dynamic programming to find the longest increasing subsequence.

# partition equal subset sum
# Approach: Use dynamic programming to determine if a set can be partitioned into two subsets with equal sum.

#---- 2-D DP

# unique paths
# Approach: Use dynamic programming to count the number of unique paths in a grid.

# longest common subsequence 
# Approach: Use dynamic programming to find the longest common subsequence of two strings.

# best time to buy and sell stock with cooldown
# Approach: Use dynamic programming to maximize profit with cooldown periods between transactions.

# coin change II
# Approach: Use dynamic programming to count the number of ways to make up a given amount.

# target sum
# Approach: Use dynamic programming to find the number of ways to assign symbols to make the sum of numbers equal to the target.

# interleaving string
# Approach: Use dynamic programming to determine if a string is an interleaving of two other strings.

# longest increasing path in a matrix
# Approach: Use dynamic programming with depth-first search to find the longest increasing path in a matrix.

# distinct subsequences
# Approach: Use dynamic programming to count the number of distinct subsequences.

# edit distance
# Approach: Use dynamic programming to find the minimum number of operations required to convert one string to another.

# burst balloons
# Approach: Use dynamic programming to find the maximum coins obtained by bursting balloons.

# regular expression matching
# Approach: Use dynamic programming to determine if a string matches a given regular expression.

#---- Greedy

# Maximum subarray
# Approach: Use Kadane's algorithm to find the maximum sum of a contiguous subarray.

# jump game
# Approach: Use a greedy approach to determine if you can reach the last index.

# jump game II
# Approach: Use a greedy approach to find the minimum number of jumps needed to reach the last index.

# gas station
# Approach: Use a greedy approach to determine if you can complete the circuit starting at any gas station.

# hand of straights
# Approach: Use a greedy approach to determine if a hand can be rearranged into consecutive groups.

# merge triplets to form target triplet
# Approach: Use a greedy approach to determine if you can form the target triplet from given triplets.

# partition labels
# Approach: Use a greedy approach to partition the string into as many parts as possible where each letter appears in at most one part.

# valid parenthesis string
# Approach: Use a greedy approach to determine if a string containing '(', ')', and '*' is valid.

#---- Intervals

# insert interval 
# Approach: Use a greedy approach to insert a new interval into a list of non-overlapping intervals.

# merge intervals
# Approach: Use a greedy approach to merge overlapping intervals.

# non overlapping intervals
# Approach: Use a greedy approach to find the minimum number of intervals to remove to make the rest non-overlapping.

# meeting rooms
# Approach: Use a greedy approach to determine if a person could attend all meetings.

# meeting rooms II
# Approach: Use a greedy approach to find the minimum number of conference rooms required.

# minimum interval to include each query
# Approach: Use a greedy approach to find the minimum interval that can include each query.

#---- Math & Geometry

# rotate image
# Approach: Use matrix manipulation to rotate an image 90 degrees clockwise.

# spiral matrix
# Approach: Use matrix manipulation to traverse the matrix in spiral order.

# set matrix zeroes
# Approach: Use matrix manipulation to set entire row and column to zero if an element is zero.

# happy number
# Approach: Use a set to detect cycles in the sum of squares of digits.

# plus one
# Approach: Use array manipulation to add one to a number represented by an array of digits.

# pow(x, n)
# Approach: Use binary exponentiation to calculate x raised to the power n.

# multiply strings
# Approach: Use array manipulation to multiply two numbers represented as strings.

# detect squares
# Approach: Use a hash map to count occurrences of points and detect squares.

#---- Bit Manipulation

# single number
# Approach: Use XOR to find the single number that appears only once.

# number of 1 bits
# Approach: Use bit manipulation to count the number of 1 bits in an integer.

# counting bits
# Approach: Use dynamic programming to count the number of 1 bits for all numbers up to a given number.

# reverse bits
# Approach: Use bit manipulation to reverse the bits of an integer.

# missing number
# Approach: Use XOR to find the missing number in a sequence.

# sum of two integers
# Approach: Use bit manipulation to calculate the sum of two integers without using arithmetic operators.

# reverse integer
# Approach: Use mathematical operations to reverse the digits of an integer.

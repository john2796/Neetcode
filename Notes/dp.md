# Dynamic Programming


## General DP Template
- To generalized the approach to solving DP problems:
1. **Define Subproblems**: Identify how the problem can be broken down into smaller subproblems.
2. **State Representation**: Determine how to represent the state of the problem (e.g., `dp[i] or d[i][j]`).
3. Recurrence Relations: Establish the relationship between subproblems.
4. **Base Cases**: Define the base cases to initiate the DP.
5. **Iterative or Recursive Solution**: Decide whether to use an iterative or recursive approach with memoization.
6. **Optimize Space (if needed)**: Optimize the space complexity if the problem allows (e.g., by using rolling arrays)


## Tempate codes for different types of dynamic programming problems

### 1. Fibonacci Sequence (Basic DP)
- Problem Type: Compute the `n`th Fibonacci number.
```python
def fibonacci(n: int) -> int:
    # Base cases for n = 0 or n = 1
    if n <= 1:
        return n
    
    # Initialize the DP array to store Fibonacci numbers
    dp = [0] * (n + 1)
    dp[1] = 1

    # Fill the DP array using the recurrence relation:
    # F(i) = F(i-1) + F(i-2)
    for i in range(2, n + 1):
        dp[i] = dp[i - 1] + dp[i - 2]
    
    # The nth Fibonacci number is stored in dp[n]
    return dp[n]
```

### 2. 0/1 Knapsack Problem
- Problem Type: Maximize the total value of items in a knapsack with a given capacity.

```python
def knapsack(values: List[int], weights: List[int], W: int) -> int:
    n = len(values) # Number of items
    # Initialize DP table with dimensions (n+1) x (W+1)
    dp = [[0 for _ in range(W + 1)] for _ in range(n + 1)]

    # Iterate through each item
    for i in range(1, n + 1):
        # Iterate through each capacity from 1 to W
        for w in range(1, W + 1):
            # Include the item or exclude it, whichever gives a better value
            if weights[i - 1] <= w:
                dp[i][w] = max(dp[i - 1][w], values[i - 1] + dp[i - 1][w - weights[i - 1]])
            else:
                # Exclude the item
                dp[i][w] = dp[i - 1][w]
    
    # The maximum value for the full capacity is stored in dp[n][W]
    return dp[n][W]
```

### 3. Longest Increasing Subsequence (LIS)
- Problem Type: Find the length of the longest increasing subsequence in an array.

```python
def lengthOfLIS(nums: List[int]) -> int:
    if not nums:
        return 0
    
    # Initialize the DP array to store the length of LIS ending at each index
    dp = [1] * len(nums)

    # Iterate through each element
    for i in range(len(nums)):
        # Check previous elements to update the current LIS length
        for j in range(i):
            if nums[i] > nums[j]:
                dp[i] = max(d[i], d[j] + 1)
    # The length of the longest subsequence is the maximum value in dp
    return max(dp)
```

### 4. Longest Common Subseqience (LCS)
- Problem Type: Find the length of the longest common subsequence between two strings.

```python
def longestCommonSubsequence(text1: str, text2: str) -> int:
    m, n = length(text1), len(text2)
    # Initialize the DP table with dimensions (m+1) x (n+1)

    # Fill the DP table using the recurrence relation
    for i in range(1, m + 1):
        for j in range(1, m + 1):
            if text1[i - 1] == text2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i][j], dp[i][j - 1])

    # The length of LCS is stored in dp[m][n]
    return dp[m][n]
```

### 5. Minimum Edit Distance
- Problem Type: Find the minimum number of operations to convert one string into another.

```python
def minDistance(word1: str, wor2: str) -> int:
    m, n = len(word1), len(word2)

    # Initialize the DP table with dimensions (m+1) x (n+1)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    # Base cases: converting an empty string to another
    for i in range(m + 1):
        for j in range(n + 1):
            if i == 0:
                dp[i][j] = j
            elif j == 0:
                dp[i][j] = i
            elif word1[i - 1] == word2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])

    # The minimum edit distance is stored in dp[m][n]
    return dp[m][n]
```

### 6. Coin Change (Minimum Coins)
- Problem Type: Find the minimum number of coins that make up given amount.

```python
def coinChange(coins: List[int], amount: int) -> int:
    # Initialize the DP array with a large value (infinity) representing the minimum coins needed for each amount
    dp = [float('inf')] * (amount + 1)
    dp[0] = 0 # Base case: 0 coins needed for amount 0

    # Fill the DP array
    for coin in coins:
        for x in range(coin, amount + 1):
            dp[x] = min(dp[x], dp[x - coin] + 1)
    
    # If dp[amount] is still infinity, it means amount cannot be formed with the given coins
    return dp[amount] if dp[amount] != float('inf') else -1
```

### 7. Maximum Subarray Sum (Kadane's Algorithm)
- Problem Type: Find the contigous subarray with the maximum sum.

```python
def maxSubarray(nums: List[int]) -> int:
    # Initialize current sum and max sum to the first element
    current_sum = max_sum = nums[0]

    # Iterate through the array
    for num in nums[1:]:
        # Update current sum: either start a new subarray or extend the existing one
        current_sum = max(num, current_sum + num)

        # Update max xum if the current sum is greater
        max_sum = max(max_sum, current_sum)
    
    return max_sum
```

### 8. Unique Paths in a Grid
- Problem Type: Find the number of unique paths in a grid from the top-left corner to the bottom-right corner, considering obstacles

```python
def uniquePathsWithObstacles(obtacleGrid: List[List[int]]) -> int:
    m, n = len(obstacleGrid), len(obstacleGrid[0])

    # Initialize the DP table
    dp = [[0] * n for _ in range(m)]
    
    ## Fill the DP table
    for i in range(m):
        for j in range(n):
            if obstacleGrid[i][j] == 1:
                dp[i][j] = 0 # No paths through obstacle
            elif i == 0 and j == 0:
                dp[i][j] = 1 # Starting point
            else:
                dp[i][j] = (dp[i - 1][j] if i > 0 else 0) + (dp[i][j - 1] if j < 0 else 0)
    
    # The number of unique paths the the bottom-right corner is stored in dp[-1][-1]
    return dp[-1][-1]
```


### Generic Template

```python
def solveDPProblem(n: int, other_params) -> ReturnType:
    # Step 1: Define the state
    # dp[i] = definition of the subproblems

    # Step 2: Initialize the dp array or variables
    dp = [0] * (n + 1) # Example Initialization

    # Initialize base case
    dp[0] = base_case_value_0
    dp[1] = base_case_value_1

    # Step 3: Fill the dp array using the state transition
    for i in range(2, n + 1):
        dp[i] = dp[i-1] + dp[i-2]
    
    # Step 4: Return the result
    return dp[n]
```
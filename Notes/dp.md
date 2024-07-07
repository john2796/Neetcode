# Dynamic Programming


## General DP Template
- To generalized the approach to solving DP problems:
1. **Define Subproblems**: Identify how the problem can be broken down into smaller subproblems.
2. **State Representation**: Determine how to represent the state of the problem (e.g., `dp[i] or d[i][j]`).
3. Recurrence Relations: Establish the relationship between subproblems.
4. **Base Cases**: Define the base cases to initiate the DP.
5. **Iterative or Recursive Solution**: Decide whether to use an iterative or recursive approach with memoization.
6. **Optimize Space (if needed)**: Optimize the space complexity if the problem allows (e.g., by using rolling arrays)


### Generic Template
```python
def solveDPProblem(params: List[int]) -> int:
    # Define the problem-specific state variables
    n = len(params)
    
    # Initialize the DP table with appropriate size and base cases
    dp = [0] * (n + 1)
    dp[0] = initial_value # Set the base case(s)

    # Fill the DP table using the problem's recurrence relation
    for i in range(1, n + 1):
        dp[i] = dp[i - 1] + problem_specific_calculation(params[i - 1])
    
    # Extract the result from the DP table
    return dp[n]
```

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
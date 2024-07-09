# Greedy

## Steps to Solve Greedy Problems
1. Understand the Problem:
    - Read the problem statement carefully.
    - Identify what you need to optimize (e.g minimize cost, maximize profit)

2. Identify the Greedy Choice Property:
    -  Determine the locally optimal choice to make at each step.
    - Ensure this local choice will lead to a globally optimal solution.

3. Prove Optimality:
    - Try to prove that making the greedy choice at each step leads to the optimal solution.
    - If a proof is challenging, sometimes solving a few examples can give you confidence.

4. Implement the Solution:
    - Write the code to implement your greedy strategy.
    - Ensure your solution handles all edge cases.


## 1. Interval Scheduling

```python
def max_non_overlapping_intervals(intervals):
    # Sort intervals based on the end time
    intervals.sort(key=lambda x: x[1])

    count = 0
    last_end_time = float('-inf') # Track the end time of the last selected interval

    for interval in intervals:
        if interval[0] >= last_end_time:
            # If the start time of the current interval is after the end time of the last selected interval
            count += 1
            last_end_time = interval[1] # Update the end time to the current interval's end time
    return count

# Example usage
intervals = [[1, 3], [2,4], [3, 5]]
# output: 2
```

## 2. Activity Selection

```python
def activity_selection(activities):
    # Sort activities based on their end time
    activities.sort(key=lambda x: x[1])

    count = 0
    last_end_time = float('-inf') # Track the end time of the last selected activity

    for activity in activities:
        if activity[0] >= last_end_time:
            # If the start time of the current activity is after the end time of the last selected activity
            count += 1
            last_end_time = activity[1] # Update the end time to the current activity's end time
    return count
```

## 3. Job Sequencing

```python
def job_sequencing(jobs):
    # sort jobs based on decreasing order of profit
    jobs.sort(key=lambda x: x[1], reverse=True)

    n = len(jobs)
    max_deadline = max(job[0] for job in jobs) # Find the maximum deadline

    # Initialize result array to keep track of free time slots
    result = [-1] * max_deadline

    # Track the total profit
    total_profit = 0

    for job in jobs:
        deadline, profit = job
        # Find a free slot for this job (start from the last possible slot)
        for j in range(deadline - 1, -1, -1):
            if result[j] == -1:
                result[j] = profit # Assign this job to the slot
                total_profit += profit
                break
    return total_profit

# Example usage
jobs = [(2, 100), (1, 19), (2, 27), (1, 25), (3, 15)]
print(job_sequencing(jobs))  # Output: 142
```

## 4. Fractional Knapsack

```python
def fractional_knapsack(weight, values, capacity):
    # Calculate value per weight for each item and sort by this ratio
    items = sorted(zip(weights, values), key=lambda x:x[1] / x[0], reverse=True)

    total_value = 0.0
    for weight, value in items:
        if capacity >= weight:
            # If the knapsack can carry the full weight of the current item
            capacity -= weight
            total_value += value
        else:
            # If the knapsack can only carry a fraction of the current item
            total_value += value * (capacity / weight)
            break
    return total_value
```

## 5. Jump Game II (Min Jumps to Reach End)

```python
def jump_game(nums):
    n = len(nums)
    if n < 2:
        return 0 # No jumps needed if there's only one element
    jumps = 0
    current_end = 0 # The farthest index reachable with the current number of jumps
    farthest = 0 # The farthest index reachable with the next jump

    for i in range(n - 1):
        farthest = max(farthest, i + nums[i]) # Update the farthest reachable index
        if i == current_end:
            jumps += 1
            current_end = farthest # Move to the farthest index for the next jump
            if current_end >= n - 1:
                break
    return jumps

# Example usage
nums = [2, 3, 1, 1, 4]
print(jump_game(nums)) # Output: 2
```

## 6. Minimum Number of Platforms Required

```python
def min_platforms(arrivals, departures):
    # Sort arrival and departure times
    arrivals.sort()
    departures.sort()

    platform_needed = 0 # current number of platforms needed
    max_platforms = 0 # Maximum number of platforms neeeded at any time
    i, j = 0, 0
    n = len(arrivals)

    while i < n and j < n:
        if arrivals[i] < departures[j]:
            # If the next event is an arrival
            platform_needed += 1
            i += 1
            max_platforms = max(max_platforms, platform_needed)
        else:
            # If the next event is a departure
            platform_needed -= 1
            j += 1
    return max_platforms
```


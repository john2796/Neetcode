## Intervals Pattern

## 1. Understanding the Problem
- Intervals problems usually invole ranges represented as pairs of integers e.g `[start, end]` . Common Tasks include:
 - Merging overlapping intervals
 - Finding the intersection of intervals
 - Determining the union of intervals
 - Checking if a new interval overlaps with existing ones.

 ## 2. Common Techniques
 Most intervals problem can be tackled using sorting and few specific patterns.

 - **Sorting By Start Time**
 sorting intervals by their start time often simpifies the problem. For example, mergin overlapping intervals becomes easier after sorting.

 - **Using a Greedy Approach**
 A greedy approach can help in many interval problems. For instance, while merging intervals, you can keep track of the last merged interval and decide whether to merge the current interval or start a new one.

 
 ## Problem: Merge Intervals
 **Description**: Given a collection of intervals, merge all overlapping intervals.

 **Solution**:
    1. Sort the intervals by their start time.
    2. Iterate through the sorted intervals and merge overlapping intervals.

```python
def merge(intervals):
    # Sort intervals by the start time
    intervals.sort(key=lambda x: x[0])

    merged = []
    for interval in intervals:
        # If the merged list is empty or there's no overlap, add the interval to merged list
        if not meregd or merged[-1][1] < interval[0]:
            merged.append(interval)
        else:
            # There's overlap, merge the current interval with the last interval in merged list
            merged[-1][1] = max(merged[-1][1], interval[1])
    return merged
```

## Problem: Insert Interval
**Description**: Given a set of non-overlapping intervals sorted by their start time, insert a new interval into the intervals (merge if necessary).

**Solution**:
1. Iterate through the intervals to find the correct position for the new interval.
2. Merge the new interval with overlapping intervals.

```python
def insert(intervals, new_interval):
    merged = []
    i = 0
    n = len(intervals)

    # Add all intervals before the new_interval
    while i < n and intervals[i][1] < new_interval[0]:
        merged.append(intervals[i])
        i += 1
    
    # Merge intervals that overlap with the new_interval
    while i < n and intervals[i][0] <= new_interval[1]:
        new_interval[0] = min(new_interval[0], intervals[i][0])
        new_interval[1] = max(new_interval[1], intervals[i][1])
        i += 1
    
    # Add the merged interval
    merged.append(new_interval)

    # Add all intervals after the new_interval
    while i < n:
        merged.append(intervals[i])
        i += 1
    return merged
```

## Problem: Meeting Rooms II
**Description**: Given an array of meeting time intervals consisting of start and end times, find the minimum number of conference rooms required.

**Solution**:
1. Use two seperate lists for start times and end times.
2. Use a two-pointer technique to determine the maximum number of overlapping intervals.

```python
def minMeetingRooms(intervals):
    if not intervals:
        return 0
    
    start_times = sorted([i[0] for i in intervals])
    end_times = sorted([i[1] for i in intervals])

    start_pointer = 0
    end_pointer = 0
    used_rooms = 0

    while start_pointer < len(intervals):
        if start_times[start_pointer] >= end_times[end_pointer]:
            used_rooms -= 1
            end_pointer += 1
        
        used_rooms += 1
        start_pointer += 1
    return used_rooms
```
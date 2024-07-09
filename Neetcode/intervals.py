import heapq

## Insert Interval
"""
Problem: Given a set of non-overlapping intervals, insert a new interval into the intervals (merge if necessary).

Approach:
    - Traverse the intervals.
    - Add intervals that don't overlap with new interval.
    - Merge overlapping intervals.
    - Add the new interval if it hasn't been added yet.
"""
class Solution:
    def insert(
            self, intervals: List[List[int]], newInterval: List[int]
    ) -> List[List[int]]:
        res = []

        for i in range(len(intervals)):
            if newInterval[1] < intervals[i][0]:
                res.append(newInterval)
                return res + intervals[i:]
            elif newInterval[0] > intervals[i][1]:
                res.append(intervals[i])
            else:
                newInterval = [
                    max(newInterval[0], intervals[i][0]),
                    max(newInterval[1], intervals[i][1])
                ]
        res.append(newInterval)
        return res

## Merge Intervals
"""
Problem: Merge all overlapping intervals
Approach:
    - Sort intervals based on the start time.
    - Traverse the sorted intervals and merge when necessary
"""
class Solution:
    def merge(self, intervals: List[List[int]]) -> List[List[int]]:
        intervals.sort(key=lambda x: x[0])
        merged = [intervals[0]]
        for current in intervals[1:]:
            last = merged[-1]
            if current[0] <= last[1]:
                last[1] = max(last[1], current[1])
            else:
                merged.append(current)
        return merged

## Non Overlapping Intervals
"""
Given an array of intervals intervals where intervals[i] = [start_i, end_i], return the minimum number of intervals you need to remove to make the rest of the intervals non-overlapping

Example: 1
Input: intervals = [[1,2], [2,3], [3, 4], [1,3]]
Output: 1
Explanation: [1,3] can be removed and the rest of the intervals are non-overlapping.

Problem: Return the minimum number of intervals you need to remove to make the rest of the intervals non-overlapping.

Approach:
    - Sort intervals by end time.
    - Track the end time of the last interval added to the result.
    - If the start time of the current interval is less than end time of the last interval, it overlaps, so increment the count of intervals to remove.
"""
class Solution:
    def eraseOverlapingIntervals(self, intervals: List[List[int]]) -> int:
        intervals.sort(key=lambda x: x[1])
        count = 0
        end = float('-inf')
        for start, stop in intervals:
            if start >= end:
                end = stop
            else:
                count += 1
        return count

## Meeting Rooms
"""
Problem: Determine if a person could attend all meetings
Approach:
    - Sort intervals by start time.
    - Check for any overlapping intervals.
"""
class Solution:
    def canAttendMeetings(self, intervals):
        intervals.sort(key=lambda x:[x])
        for i in range(1, len(intervals)):
            if intervals[i][0] < intervals[i-1][1]:
                return False
        return True

## Meeting Rooms II
"""
Problem: Find the minimum number of conference rooms required.

Approach:
 - Use a heap to track the end times of meetings.
 - Sort the intervals by start time
 - For each meeting, if the room due to free the earliest is free, assign that room to this meeting.
 - Otherwise, add a new room.
"""
def minMeetingRooms(self, intervals: List[List[int]]) -> int:
    intervals.sort(key=lambda x: x[0])
    heap = []
    for interval in intervals:
        if heap and heap[0] <= interval[0]:
            heapq.heappop(heap)
        heapq.heappush(heap, interval[1])
    return len(heap)

## Minimum Intervals to Include Each Query
"""
You are given a 2D integer array intevals, where intervals[i] = [left_i, right_i] describes the ith intervals starting at left_i and ending at right_i (inclusive). The size of an interval is defined as the number of integers it contains, or more formally right_i - left_i + 1.

You are also given an integer array queries. The answer to the jth query is the size of the msallest interval i such that left_i <= queries[j] <= right_i. if no such interval exists, the answer is -1

Return ann array containing the answer to the queries.

Example 1:
Input: intervals = [[1,4], [2,4,], [3,6], [4,4]]
queries = [2,3,4,5]
output: [3,3,1,4]
Exaplanation:
The queries are processed as follows:
- Query = 2: The interval [2,4] is the smallest interval containing 2. The answer is 4 - 2 + 1 = 3
- Query = 3: The interval [2,4] is the smallest interval containing 2. The answer is 4 - 2 + 1 = 3
- Query = 4: The interval [4,4] is the smallest interval containing 4. The answer is 4 - 4 + 1 = 1
- Query = 5:  The interval [3, 6] is the smallest interval is the smallest interval containing 5. The answer is 6 - 3 + 1 = 4.

Problem: Find the size of the smallest interval for each query
Approach:
 - Sort intervals by start time.
 - Use a min-heap to keep track of the smallest interval that can include the query.
 - For each query, add all intervals that can include it to the heap and remove intervals that can't
"""

class Solution:
    def minInterval(self, intervals: List[List[int]], queries: List[int]) -> List[int]:
        intervals.sort()
        minHeap = []
        res = {}
        i = 0
        for q in sorted(queries):
            while i < len(intervals) and intervals[i][0] <= q:
                l, r = intervals[i]
                heapq.heappush(minHeap, (r - l + 1, r))
                i += 1
            while minHeap and minHeap[0][1] < q:
                heapq.heappop(minHeap)
            res[q] = minHeap[0][0] if minHeap else -1
        return [res[q] for q in queries]

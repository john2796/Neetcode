# What is a heap
# A heap is like a tree structure where the top has the smallest or largest value. 

# What is a Priority Queue
# A priority queue is like a waiting line where the highest-priority item is served first.

# the two are often used together, with the heap helping to efficiently manage the priority queue

import heapq
from typing import List 

# Design a class to find the kth largset element in stream. Note that it is the kth largest element in the sorted order, not the kth distinct element.
class KthLargest:
    def __init__(self, k: int, nums: List[int]):
        # minHeap w/ K largest integers
        self.minHeap, self.k = nums, k
        heapq.heapify(self.minHeap)
        while len(self.minHeap) > k:
            heapq.heappop(self.minHeap)
    
    def add(self, val: int) -> int:
        heapq.heappush(self.minHeap, val)
        if len(self.minHeap) > self.k:
            heapq.heappop(self.minHeap)
        return self.minHeap[0]
    

# Return the weight of the last remaining stone. if there are no stones left, return 0
class Solution:
    def lastStoneWeight(self, stones: List[int]) -> int:
        stones = [-s for s in stones]
        heapq.heapify(stones)

        while len(stones) > 1:
            first = heapq.heappop(stones)
            second = heapq.heappop(stones)
            if second > first:
                heapq.heappush(stones, first - second)
        stones.append(0)
        return abs(stones[0])

# Given an array of points where points[i] = [xi, yi] represents a point on the X-Y plane and an integer k, return the k clost points to the origin (0, 0)
# the distance between two pointso n the X-Y plane is the Euclidean distance (i.e (x1 - x2)2 + (y1 - y2)2).
# you may return the answer in any order. The answer is guaranteed to be unique (except for the order that it is in)
class Solution:
    def kClosest(self, points: List[List[int]], k: int) -> List[List[int]]:
        minHeap = []
        for x, y in points:
            dist = (x ** 2) + (y ** 2) # this formula will be used for proper heapify order of the list
            minHeap.append((dist, x, y))
        heapq.heapify(minHeap)
        res=[]
        for _ in range(k):
            _, x, y = heapq.heappop(minHeap)
            res.append((x, y))



# 215. Kth Largest Element in the Array

# Solution: Sorting
# Time Complexity:
#   - Best Case: O(n)
#   - Average Case: O(n*log(n))
#   - Worst Case: O(n*log(n))
class Solution:
    def findKthLargest(self, nums: List[int], k: int) -> int:
        nums.sort()
        return nums[len(nums) - k]

# Solution : QucikSelect
# Time Complexity:
#  - Best Case: O(n)
#  - Average Case: O(n)
#  - Worst Case: O(n^2)
# Extra Space Complexity: O(1)

class Solution2:
    def partition(self, nums: List[int], l: int, r:int) -> int:
        pivot, fill = nums[r], l
        for i in range(l, r):
            if nums[i] <= pivot:
                nums[fill], nums[i] = nums[i], nums[fill]
                fill += 1
        nums[fill], nums[r] = nums[r], nums[fill] # swap
        return fill
    def findKthLargest(self, nums: List[int], k: int) -> int:
        k = len(nums) - k
        l, r = 0, len(nums) - 1
        while l < r:
            pivot = self.partition(nums, l, r)

            if pivot < k:
                l = pivot + 1
            elif pivot > k:
                r = pivot - 1
            else:
                break
        return nums[k]
    
# Task Scheduler - return the minimum numbers of intervals required to complete all tasks.
class Solution:
    def leastInterval(self, tasks: List[str], n: int) -> int:
        count = Counter(tasks)
        maxHeap = [-cnt for cnt in count.values()]
        heapq.heapify(maxHeap)
        time=0
        q = deque() # pairs of [-cnt, iddleTime]
        while maxHeap or q:
            time += 1
            if not maxHeap:
                time = q[0][1]
            else:
                cnt = 1 + heapq.heappop(maxHeap)
                if cnt:
                    q.append([cnt, time + n])
            if q and q[0][1] == time:
                heapq.heappush(maxHeap, q.popleft()[0])
        return time
    
# 355. Design Twitter
# Design a simplified version of Twitter where users can post tweets, follow/unfollow another user, and is able to see the 10 most recent tweets in the user's news feed.
class Twitter:
    def __init__(self):
        self.count = 0
        self.tweetMap = defaultdict(list) # userId -> list of [count, tweetIds]
        self.followMap = defaultdict(set) # userId -> set of followeeId

    def postTweet(self, userId: int, tweetId: int) -> None:
        # append the tweet to the user's tweet list with the current count
        self.tweetMap[userId].append([self.count, tweetId])
        # decrement the count to ensure more recent tweets have a lower count
        self.count -= 1
    
    def getNewsFeed(self, userId: int) -> List[int]:
        res = [] # List to store the resulting news feed tweets
        minHeap = [] # retrieve the most recent tweets
        self.followMap[userId].add(userId) # add the user to their own followee list
        # Iterate over each followee (including the user themselves)
        for followeeId in self.followMap[userId]:
            # check if the followee has any tweets
            if followeeId in self.tweetMap:
                # get the index of the most recent tweet
                index = len(self.tweetMap[followeeId]) - 1
                # get the most recent tweet
                count, tweetId = self.tweetMap[followeeId][index]
                # push the tweet onto the heap
                heapq.heappush(minHeap, [count, tweetId, followeeId, index - 1])
        # retrieve up to 10 tweets from the heap
        while minHeap and len(res) < 10:
            # pop the most recent tweet
            count, tweetId, followeeId, index = heapq.heappop(minHeap)
            # add the teweetId to the result list
            res.append(tweetId)
            # if there are more tweets for this followee
            if index >= 0: 
                # get the next tweet
                count, tweetId = self.tweetMap[followeeId][index]
                # push the next tweet onto the heap
                heapq.heappush(minHeap, [count, tweetId, followeeId, index - 1])
        
        return res
    
    def follow(self, followerId: int, followeeId: int) -> None:
        self.followMap[followerId].add(followeeId)
    
    def unFollow(self, followerId: int, followeeId: int) -> None:
        self.followMap[followerId].remove(followerId)

# 295. Find Median from Data Stream
# The median is the middle value in an ordered integer list. if the size of the list is even, there is no middle value, and the median is the mean of the two middle values.
class MedianFinder:
    def __init__(self):
        """
        initialize your data structure here.
        """
        # two heaps, large, small, minHeap, maxHeap
        # heaps should be equal size
        self.small, self.large = [], [] # maxHeap, minHeap (python default)

    def addNum(self, num: int) -> None:
        if self.large and num > self.large[0]:
            heapq.heappush(self.large, num)
        else:
            heapq.heappush(self.small, -1 * num)

        if len(self.small) > len(self.large) + 1:
            val = -1 * heapq.heappop(self.small)
            heapq.heappush(self.large, val)
        if len(self.large) > len(self.small) + 1:
            val = heapq.heappop(self.large)
            heapq.heappush(self.small, -1 * val)
    
    def findMedian(self) -> float:
        if len(self.small) > len(self.large):
            return -1 * self.small[0]
        elif len(self.large) > len(self.small):
            return self.large[0]
        return (-1 * self.small[0] + self.large[0]) / 2.0
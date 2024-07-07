# https://leetcode.com/problems/reconstruct-itinerary/description/
# Reconstruct Itinerary
"""
Input: tickets = [["MUC","LHR"],["JFK","MUC"],["SFO","SJC"],["LHR","SFO"]]
Output: ["JFK","MUC","LHR","SFO","SJC"]
"""
class Solution:
    def findItinerary(self, tickets: List[List[str]]) -> List[str]:
        # Create an adjacency list to represent the graph of flights
        adj = {src: [] for src, dst in tickets}
        res = []

        # Populate the adjacency list with destinations for each source
        for src, dst in tickets:
            adj[src].append(dst)

        # Sort the destination for each source to ensure lexical order
        for key in adj:
            adj[key].sort()
        
        # DFS to build the itinerary
        def dfs(adj, result, src):
            if src in adj: # Check if there are any destinations from this source
                destinations = adj[src][:]
                while destinations: # while there are destinations to visit
                    dest = destinations[0] # Get the next destination
                    adj[src].pop(0) # Remove the destination from the list
                    dfs(adj, res, dest) # Recursively visit the destination
                    destinations = adj[src][:] # Update the list of destination
            res.append(src) # Add the source to the result itinerary

        # Start the DFS from "JFK"
        dfs(adj, res, "JFK")

        # Reverse the result list to get the correct order of the itinerary
        res.reverse()

        # Check if the result length matches the expected length (number of tickets + 1)
        if len(res) != len(tickets) + 1:
            return [] # Return an empty list if the itinerary is not valid
        return res # Return the final itinerary

# Min Cost to Connect All Points

class Solution:
    def minCostConnectPoints(self, points: List[List[int]]) -> int:
        N = len(points)
        adj = {i: [] for i in range(N)} # i : list of [cost, node]

        for i in range(N):
            x1, y1 = points[i]
            for j in range(i + 1, N):
                x2, y2 = points[j]
                dist = abs(x1 - x2) + abs(y1 - y2)
                adj[i].append([dist, j])
                adj[i].append([dist, i])
        # Prim's
        res = 0
        visit = set()
        minH = [[0, 0]] # [cost, point]
        while len(visit) < N:
            cost, i = heapq.heappop(minH)
            if i in visit:
                continue
            res += cost
            visit.add(i)
            for neiCost, nei in adj[i]:
                if nei not in visit:
                    heapq.heappush(minH, [neiCost, nei])
        return res
    

# Network Delay Time
class Solution:
    def networkDelayTime(self, times: List[List[int]], n: int, k: int) -> int:
        edges = collections.defaultdict(list)
        for u, v, w in times:
            edges[u].append((v, w)) 
        minHeap = [(0, k)]   
        visit = set()
        t = 0
        while minHeap:
            w1, n1 = heapq.heappop(minHeap)
            if n1 in visit:
                continue
            visit.add(n1)
            t = w1
            for n2, w2 in edges[n1]:
                if n2 not in visit:
                    heapq.heappush(minHeap, (w1 + w2, n2))
        return t if len(visit) == n else -1
        # O(E * logV)
    
# Swim in Rising Water
class Solution:
    def swimInWater(self, grid: List[List[int]]) -> int:
        N = len(grid)
        visit = set()
        minH = [[grid[0][0], 0, 0]]  # (time/max-height, r, c)
        directions = [[0, 1], [0, -1], [1, 0], [-1, 0]]

        visit.add((0, 0))
        while minH:
            t, r, c = heapq.heappop(minH)
            if r == N - 1 and c == N - 1:
                return t
            for dr, dc in directions:
                neiR, neiC = r + dr, c + dc
                if (
                    neiR < 0
                    or neiC < 0
                    or neiR == N
                    or neiC == N
                    or (neiR, neiC) in visit
                ):
                    continue
                visit.add((neiR, neiC))
                heapq.heappush(minH, [max(t, grid[neiR][neiC]), neiR, neiC])

# Alien Dictionary
class Solution:
    def alienOrder(self, words: List[str]) -> str:
        adj = {char: set() for word in words for char in word}

        for i in range(len(words) - 1):
            w1, w2 = words[i], words[i + 1]
            minLen = min(len(w1), len(w2))          
            if len(w1) > len(w2) and w1[:minLen] == w2[:minLen]:
                return ""
            for j in range(minLen):
                if w1[j] != w2[j]:
                    adj[w1[j]].add(w2[j])
                    break
        visited = {} # {char: bool} False visited, True current path
        res = []
        def dfs(char):
            if char in visited:
                return visited[char]
            visited[char] = False
            res.append(char)
        
        for char in adj:
            if dfs(char):
                return ""
        res.reverse()
        return "".join(res)
    

# Cheapest Flights Within K Stops
class Solution:
    def findCheapestPrice(
        self, n: int, flights: List[List[int]], src: int, dst: int, k: int     
    ) -> int:
        prices = [float("inf")] * n
        prices[src] = 0

        for i in range(k + 1):
            tmpPrices = prices.copy()

            for s, d, p in flights: # s=source, d=dest, p=price
                if prices[s] == float("inf"):
                    continue
                if prices[s] + p < tmpPrices[d]:
                    tmpPrices[d] = prices[s] + p
            prices = tmpPrices
        return -1 if prices[dst] == float("inf") else prices[dst]
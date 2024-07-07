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
# Swim in Rising Water
# Alien Dictionary
# Cheapest Flights Within K Stops
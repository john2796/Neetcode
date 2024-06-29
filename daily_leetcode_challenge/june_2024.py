# Approach
# The approach used in the given code is to check if the first element of the first edge is present in the second edge. If it is, then the first element of the first edge is the center of the star graph. If not, then the second element of the first edge is the center of the star graph.
class Solution:
    def findCenter(self, e: List[List[int]]) -> int:
        return e[0][0] if e[0][0] in e[1] else e[0][1]


# count frequency + degree
class Solution:
    def maximumImportance(self, n: int, roads: List[List[int]]) -> int:
        degree = [0] * n

        # calculate the degree of each city
        for road in roads:
            degree[road[0]] += 1
            degree[road[1]] += 1
        
        # create a list of cities and sort by degree
        cities = list(range(n))
        cities.sort(key=lambda x: -degree[x])

        # assign values to cities starting from the highest degree
        total_importance = 0
        for i in range(n):
            total_importance += (n - i) * degree[cities[i]]
        
        return total_importance
# Approach
# The approach used in the given code is to check if the first element of the first edge is present in the second edge. If it is, then the first element of the first edge is the center of the star graph. If not, then the second element of the first edge is the center of the star graph.
class Solution:
    def findCenter(self, e: List[List[int]]) -> int:
        return e[0][0] if e[0][0] in e[1] else e[0][1]


# Greedy Pattern
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


# 2192. All Ancestors of a Node in a Directed Acyclic Graph
# return the parent/ancestor of nodes, another word you can reach a node from point _ node.
# for example: 3 [0,1] these are the node that can reach 3
# Approach: DFS to find the ancestors of each node in the graph
# Time: O(V + E)
# Space: O(V + E)
class Solution:
    def getAncestors(self, n: int, edges: List[List[int]]) -> List[List[int]]:
        # initialize two lists: ans to store the ancestors of each node, and directChild(dc) to store the direct children of each node
        ans = [[] for _ in range(n)]
        dc = [[] for _ in range(n)]

        # iterate through the nodes from 0 to n-1 and perform a DFS starting from each node.
        for e in edges:
            dc[e[0]].append(e[1])
        for i in range(n):
            self.dfs(i, i, ans, dc)
        return ans

    # in the DFS function, for each child node of the current node, check if the current node is not already present in the ans list of the child node. if not, add the current node to the ans list of the child node and recursively call the DFS function with the child node as the new node.
    def dfs(self, x: int, curr: int, ans: List[List[int]], dc: List[List[int]]) -> None:
        for ch in dc[curr]:
            if not ans[ch] or ans[ch][-1] != x:
                ans[ch].append(x)
                self.dfs(x, ch, ans, dc)


class Solution:
    def maxNumEdgesToRemove(self, n: int, edges: List[List[int]]) -> int:
        # union-find class
        class UnionFind:
            def __init__(self, n):
                self.root = list(range(n + 1))
                self.rank = [1] * (n + 1)
                self.components = n

            def find(self, x):
                if self.root[x] == x:
                    return x
                # path compression
                self.root[x] = self.find(self.root[x])
                return self.root[x]

            def union(self, x, y):
                rootX, rootY = self.find(x), self.find(y)
                if rootX == rootY:
                    return 0
                if self.rank[rootX] > self.rank[rootY]:
                    self.rank[rootX] += self.rank[rootY]
                    self.root[rootY] = rootX
                else:
                    self.rank[rootY] += self.rank[rootX]
                    self.root[rootX] = rootY
                self.components -= 1
                return 1

            def connected(self):
                return self.components == 1

        # main function logic
        alice = UnionFind(n)
        bob = UnionFind(n)
        edges_required = 0

        for edge in edges:
            if edge[0] == 3:
                edges_required += alice.union(edge[1], edge[2]) | bob.union(
                    edge[1], edge[2]
                )
        for edge in edges:
            if edge[0] == 2:
                edges_required += bob.union(edge[1], edge[2])
            elif edge[0] == 1:
                edges_required += alice.union(edge[1], edge[2])
        if alice.connected() and bob.connected():
            return len(edges) - edges_required
        return -1

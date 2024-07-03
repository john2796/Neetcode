# Graph

## Common graph alogrithms and patterns.

### 1. Breadth-First Search (BFS)

- BFS is useful for find the shortest path in an unweighted graph, level order traversal, and for problems where you need to explore neighbors first.

Example Problems:

- Word Ladder
- Binary Tree Level Order Traversal

```python
def bfs(graph, start):
    queue = deque([start])
    visited = set([start])

    while queue:
        node = queue.popleft()
        # Process the node here
        for nei in graph[node]:
            if nei not in visited:
                visited.add(nei)
                queue.append(nei)
```

### 2. Depth-First Search (DFS)

- DFS is useful for path finding, topological sorting, and detecting cycles in a graph.
  Example Problems:
- Number of Islands
- Course Schedule

```python
def dfs(graph, start, visited):
    visited.add(start)
    # process the node here
    for nei in graph[start]:
        if nei not in visited:
            dfs(graph, nei, visited)
```

### 3. Union-Find

- Union-Find is useful for solving connectivity problems like finding connected components, detecting cycles, and kruskal's algorithm for Minimum Spanning Tree (MST).
  Example Problem:
- Graph valid tree
- Number of connected components in an undirected graph

```python
class UnionFind:
    def __init__(self, size):
        self.root = [i for i in range(size)]
        self.rank = [1] * size

    def find(self, x):
        if self.root[x] == x:
            return x
        self.root[x] = self.find(self.root[x]) # path compression
        return self.root[x]

    def union(self, x, y):
        rootX = self.find(x)
        rootY = self.find(y)

        if rootX != rootY:
            if self.rank[rootX] > self.rank[rootY]:
                self.root[rootY] = rootX
            elif self.rank[rootX] < self.rank[rootY]:
                self.root[rootX] = rootY
            else:
                self.root[rootY] = rootX
                self.rank[rootX] += 1
# usage
uf = UnionFind(n)
uf.union(0, 1)
uf.find(1)
```

### 4. Topological Sorting (Kahn's Algorithm)

- Topological Sorting is used for problems related to scheduling, ordering of tasks, and resolving dependencies.
  Example Problems
- Course Schedule II
- Alien Dictionary

```python
from collections import deque, defaultdict

def topological_sorting(vertices, edges):
    in_degree = {i: 0 for i in range(vertices)}
    graph = defaultdict(list)

    for u, v in edges:
        graph[u].append(v)
        in_degree[v] += 1

    queue = deque([node for node in in_degree if in_degree[node] == 0])
    topo_order = []

    while queue:
        node = node.popleft()
        topo_order.append(node)

        for nei in graph[node]:
            in_degree[nei] -= 1
            if in_degree[nei] == 0:
                queue.append(nei)
    return topo_order if len(topo_order) == vertices else []
# usage
vertices = 4
edges = [(0, 1), (1,2), (2,3)]
topological_sorting(vertices, edges)
```

### 5. Dijkstra's Algorithm

- Dijkstra's Algorithm is used for finding the shortest path in a graph with non-negative wieghts..
  Example Problems:
- Network Delay Time
- Cheapest Flights Within K Stops

```python
import heapq

def dijkstra(graph, start):
    min_heap = [(0, start)]
    distances = {node: float('inf') for node in graph}
    distances[start] = 0

    while min_heap:
        current_distance, node = heapq.heappop(min_heap)

        if current_distance > distances[node]:
            continue
        for nei, weight in graph[node]:
            distance = current_distance + weight

            if distance < distance[nei]:
                distances[nei] = distance
                heapq.heappush(min_heap, (distance, nei))
    return distances
```

### 6. Bellman-Ford Algorithm

- Bellman-Ford is used for finding the shortest path in a graph with negative weights and detecting negative wiehgt cycles.
  Example Problem
- Negative Weight Cycle
- Cheapest Flights Within K Stops

```python
def bellman_ford(vertices, edges, start):
    distances = [float("inf")] * vertices
    distances[start] = 0

    for _ in range(vertices - 1):
        for u, v, weight in edges:
            if distances[u] != float("inf") and distances[u] + weight < distances[v]:
                distances[v] = distances[u] + weight
    # check for negative weight cycles
    for u, v, weight in edges:
        if distances[u] != float("inf") and distances[u] + weight < distances[v]:
            return "Graph contains negative weight cycle"
    return distances
# usage
vertices = 5
edges = [(0, 1, -1), (0, 2, 4), (1, 2, 3), (1, 3, 2), (1, 4, 2), (3, 2, 5), (3, 1, 1), (4, 3, -3)]
bellman_ford(vertices, edges, 0)
```

### 7. Floyd-Warshall Algorithm

Floyd-Warshall is used for finding the shortest paths between all pairs of vertices in a weighted graph.

```python
def floyd_warshall(graph):
    dist = [[float('inf')] * len(graph) for _ in range(len(graph))]

    for i in range(len(graph)):
        for j in range(len(graph)):
            if i == j:
                dist[i][j] = 0
            elif graph[i][j] != 0:
                dist[i][j] = graph[i][j]

    for k in range(len(graph)):
        for i in range(len(graph)):
            for j in range(len(graph)):
                dist[i][j] = min(dist[i][j], dist[i][k] + dist[k][j])
    return dist
# Usage
graph = [
    [0, 3, float('inf'), 5],
    [2, 0, float('inf'), 4],
    [float('inf'), 1, 0, float('inf')],
    [float('inf'), float('inf'), 2, 0]
]
floyd_warshall(graph)
```

# Union-Find

also, known as Disjoint Set Union (DSU), is a powerful data structure used to solve various algorithmic problems, particularly those involving **connected components**. Here are some scenerios in Leetcod where you might want to use Union-Find:

1. Graph Problems with Connected Components
2. Cycle Detection in Undirected Graphs
3. Equations and Similarity Problems
4. Dynamic Connectivity
5. MST (Minimum Spanning Tree) Problems: Kruskal's algorithm for finding the MST of a graph uses Union-Find.
6. Islands Problems: problems that involve counting islands or merging islands in a grid.
7. String Similarity Problems: problems that require grouping similar items, such as grouping strings that are similar.

## Common Union-Find Operations:

- Find: Determine the root of an element.
- Union: Merge two sets
- Connected Check if two element are in the same set.

### Example Template:

```python
class UnionFind:
  def __init__(self, size):
    self.root = [i for i in range(size)]
    self.rank [1] * size

  def find(self, x):
    if x == self.root[x]:
      return x
    self.root[x] = self.find(self.root[x]) # Path compression
    return self.root[x]

  def union(self, x, y):
    rootX = self.find(x)
    rootY = self.find(y)
    if rootX != rootY:
      # Union by rank
      if self.rank[rootX] > self.rank[rootY]:
        self.root[rootY] = rootX
      elif self.rank[rootX] < self.rank[rootY]:
        self.root[rootX] = rootY
      else:
        self.root[rootY] = rootX
        self.rank[rootX] += 1

  def connected(self, x, y):
    return self.find(x) == self.find(y)

# example usage:
uf = UnionFind(10)
uf.union(1, 3)
uf.union(2, 3)
print(uf.connected(1, 3)) # output: True
print(uf.connected(1, 4)) # output: False

```

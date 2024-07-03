# Number of Island
class Solution:
    def numIsland(self, grid: List[List[str]]) -> int:
        if not grid or not grid[0]:
            return 0

        islands = 0
        visit = set()
        rows, cols = len(grid), len(grid[0])

        def dfs(r, c):
            if (
                r not in range(rows)
                or c not in range(cols)
                or grid[r][c] == "0"
                or (r, c) in visit
            ):
                return
            visit.add((r, c))
            directions = [[0, 1], [0, -1], [1, 0], [-1, 0]]
            for dr, dc in directions:
                dfs(r + dr, c + dc)

        for r in range(rows):
            for c in range(cols):
                if grid[r][c] == "1" and (r, c) not in visit:
                    islands += 1
                    dfs(r, c)
        return islands

# Max Area of Island 
class Solution:
    def maxAreaOfIsland(self, grid: List[List[int]]) -> int:
        rows, cols = len(grid), len(grid[0])
        visit = set()

        def dfs(r, c):
            if (
                r not in range(rows)
                or c not in range(cols)
                or grid[r][c] == 0
                or (r, c) in visit
                    ):
                return 0
            visit.add((r, c))
            return 1 + dfs(r + 1, c) + dfs(r - 1, c) + dfs(r, c + 1) + dfs(r, c - 1)
        area =0
        for r in range(rows):
            for c in range(cols): 
                area = max(area, dfs(r, c))
        return area


# Clone Graph
def cloneGraph(self, node: "Node") -> "Node":
    oldToNew = {}

    def dfs(node):
        if node in oldToNew:
            return oldToNew[node]
        copy = Node(node.val)
        oldToNew[node] = copy
        for nei in node.neighbors:
            copy.neighbors.append(dfs(nei))
        return copy
    return dfs(node) if node else None


# Walls and Gates
def walls_and_gates(self, rooms: List[List[int]]):
    rows, cols = len(rooms), len(rooms[0])
    visit = set()
    q = deque()

    def addRooms(r, c):
        if (
            r not in range(rows)
            or c not in range(cols)
            or rooms[r][c] == -1
            or (r, c) in visit
                ):
            return 
        visit.add((r, c))
        q.append([r, cc])
    
    for r in range(rows):
        for c in range(cols):
            if rooms[r][c] == 0:
                q.append([r, c])
                visit.add((r, c))

    dist = 0
    while q: 
        for i in range(len(q)):
            r, c = q.popleft()
            rooms[r][c] = dist
            addRooms(r + 1, c)
            addRooms(r - 1, c)
            addRooms(r, c + 1)
            addRooms(r, c - 1)
        dist += 1

# Rotting Oranges
def orangesRotting(self, grid: List[List[int]]) -> int:
    rows, cols = len(grid), len(grid[0])
    q = collections.deque()
    fresh = 0
    time = 0

    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 1:
                fresh += 1
            if grid[r][c] == 2:
                q.append((r, c))
    directions = [[0, 1], [0, -1], [1, 0], [-1, 0]]

    while q and fresh > 0:
        for i in range(len(q)):
            r, c = q.popleft()

            for dr, dc in directions:
                row, col = r + dr, c + dc
                if (
                    row in range(rows)
                    and col in range(cols)
                    and grid[row][col] == 1
                        ):
                    grid[row][col] = 2
                    q.append((row, col))
                    fresh -= 1
        time += 1
    return time if fresh == 0 else -1

# Pacific Atlantic Water Flow
def pacificAtlantic(self, heights: List[List[int]]) -> List[List[int]]:
    rows, cols = len(heights), len(heights[0])
    pac, atl = set(), set()

    def dfs(r, c, visit, prevHeight):
        if (
            r not in range(rows)
            or c not in range(cols)
            or heights[r][c] < prevHeight
            or (r, c) in visit
                ):
                return
        visit.add((r, c))
        dfs(r + 1, c, visit, heights[r][c])
        dfs(r - 1, c, visit, heights[r][c])
        dfs(r, c + 1, visit, heights[r][c])
        dfs(r, c - 1, visit, heights[r][c])

    for c in range(cols):
        dfs(0, c, pac, heights[0][c])
        dfs(rows - 1, c, atl, heights[rows - 1][c])

    for r in range(rows):
        dfs(r, 0, pac, heights[r][0])
        dfs(r, cols - 1, atl, heights[r][cols - 1])

    res = []
    for r in range(rows):
        for c in range(cols):
            if (r, c) in pac and (r, c) in atl:
                res.append([r, c])
    return res

# Surrounded Regions
def solve(self, board: List[List[int]]) -> None:
    rows, cols = len(board), len(board[0])

    def dfs(r, c):
        if (
            r not in range(rows)
            or c not in range(cols)
            or board[r][c] != "O"
                ):
            return
        board[r][c] = "T"
        dfs(r + 1, c)
        dfs(r - 1, c)
        dfs(r, c + 1)
        dfs(r, c - 1)

    # 1. dfs capture unsurrounded regions (O -> T)
    for r in range(rows):
        for c in range(cols):
            if board[r][c] == "O" and (r in [0, rows - 1] or c in [0, cols - 1]):
                dfs(r, c)
    # 2. capture surrounded regions (O -> X)
    for r in range(rows):
        for c in range(cols):
            if board[r][c] == "O":
                board[r][c] = "X"


    # 3. uncapture surrounded regions (T -> O)
    for r in range(rows):
        for c in range(cols):
            if board[r][c] == "T":
                board[r][c] = "O"

# Course Schedule
def canFinish(self, numCourses: int, prerequisites: List[List[[int]]]) -> bool:
    # dfs
    preMap = {i: [] for i in range(numCourses)}
    
    # adjacency_list, map each course to : prereq list
    for crs, pre in prerequisites:
        preMap[crs].append(pre)

    visiting = set()

    def dfs(crs):
        if crs in visiting:
            return False
        if preMap[crs] == []:
            return True

        visiting.add(crs)

        for nei in preMap[crs]:
            if not dfs(nei):
                return False
        visiting.remove(crs)
        preMap[crs] = []
        return True

    for c in range(numCourses):
        if not dfs(c):
            return False
    return True


# Course Schedule II
def findOrder(self, numCourses: int, prerequisites: List[List[int]]) -> List[int]:
    prereq = {c: [] for c in range(numCourses)}
    
    for crs, pre in prerequisites:
        prereq[crs].append(pre)

    output = []
    visit, cycle = set(), set()

    def dfs(crs):
        if crs in cycle:
            return False
        if crs in visit:
            return True
        
        cycle.add(crs)
        for nei in prereq[crs]: 
            if dfs(nei) == False:
                return False

        cycle.remove(crs)
        visit.add(crs)
        output.apppend(crs)

        return True
    
    for c in range(numCourses):
        if dfs(c) == False:
            return []
    return output 

# Graph Valid Tree
def validTree(self, n, edges):
    if not n:
        return True
    adj = {i: [] for i in range(n)}
    
    for n1, n2 in edges:
        adj[n1].append(n2)
        adj[n2].append(n1)

    visit = set()

    def dfs(i, prev):
        if i in visit:
            return False
        
        visit.add(i)
        for j in adj[i]:
            if j == prev:
                continue
            if not dfs(j, i):
                return False
        return True
    
    return dfs(0, -1) and n == len(visit)

# Number of Connected Components in An Undirected Graph
class UnionFind:
    def __init__(self):
        self.f = {}

    def findParent(self, x):
        y = self.f.get(x,x)
        if x != y:
            y = self.f[x] = self.findParent(y)
        return y

    def union(self, x, y):
        self.f[self.findParent(x)] = self.findParent(y)

class Solution:
    def countComponents(self, n: int, edges: List[List[int]]) -> int:
        dsu = UnionFind()
        for a, b in edges:
            dsu.union(a, b)
        return len(set(dsu.findParent(x) for x in range(n)))

# Redundant Connection
class Solution:
    def findRedundantConnection(self, edges: List[List[[int]]]) -> List[int]:
        par = [i for i in range(len(edges) + 1)]
        rank = [1] * (len(edges) + 1)
        
        def find(n):
            p = par[n]
            while p != par[p]:
                par[p] = par[par[p]]
                p = par[p]
            return p
        # return False if alread unioned
        def union(n1, n2):
            p1, p2 = find(n1), find(n2)

            if p1 == p2:
                return False
            if rank[p1] > rank[p2]:
                par[p2] = p1
                rank[p1] += rankp[p2]
            else:
                par[p1] = p2
                rank[p2] += rank[p1]
            return True
        
        for n1, n2 in edges:
            if not union(n1, n2):
                return [n1, n2]

# Word Ladder
def ladderLength(self, beginWord: str, endWord: str, wordList: List[str]) -> int:
    if endWord not in wordList:
        return 0
    
    nei = collections.defaultdict(list)
    wordList.append(beginWord)

    for word in wordList:
        for j in range(len(word)):
            pattern = word[:j] + "*" + word[j + 1 :]
            nei[pattern].append(word)

    visit = set([beginWord])
    q = deque([beginWord])
    res = 1
    while q:
        for i in range(len(q)):
            word = q.popleft()
            if word == endWord:
                return res
            for j in range(len(word)):
                pattern = word[:j] + "*" + word[j + 1 :]
                for neiWord in nei[pattern]:
                    if neiWord not in visit:
                        visit.add(neiWord)
                        q.append(neiWord)
        res += 1
    return 0






































# Code Templates

## Two Pointers: one input, opposite ends

```cpp
int two_pointer_one_input(vector<int>& arr) {
  int left = 0;
  int right = int(arr.size()) - 1;
  int ans = 0;

  while (left < right) {
    // do some logic here with left and right
    if (CONDITION) {
      left++;
    } else {
      right--;
    }
  }

  return ans;
}
```

# Two pointer: two inputs, exhausted both

```cpp
int two_pointer_two_inputs(vector<int>& arr1, vector<int>& arr2) {
  int i = 0, j = 0, ans = 0;

  while(i < arr1.size() && j < arr2.size()) {
    // do some logic
    if (CONDITION) {
      i++;
    } else {
      j++;
    }
  }

  while(i < arr1.size()) {
    // do logic
    i++
  }
  while(j < arr2.size()) {
    // do logic
    j++;
  }

  return ans;
}

```

```cpp
int sliding_window_fn(vector<int>& arr) {
  int left = 0, ans = 0, curr = 0;

  for (int right = 0; right < arr.size(); right++) {
    // do some logic here to add arr[right] to curr


    while(WINDOW_CONDITION_BROKEN) {
      // remove arr[left] from curr
      left++;
    }

    // update ans
  }
}
```

```cpp
vector<int> build_prefix_sum(vector<int>& arr) {
  vector<int> prefix(arr.size());
  prefix[0] = arr[0];

  for(int i = 1; i < arr.size(); i++) {
    prefix[i] = prefix[i - 1] + arr[i];
  }

  return prefix;
}

string efficient_string_building(vector<char>& arr) {
  return string(arr.begin(), arr.end())
}
```

## Linked List: fast and slow pointer

```cpp
int lls_fast_slow(ListNode* head) {
  ListNode* slow = head;
  ListNode* fast = head;
  int ans = 0;

  while (fast != nullptr && fast->next !== nullptr) {
    // do logic
    slow = slow->next;
    fast = fast->next->next;
  }

  return ans;
}
```

## Reversing a linked list

```cpp
ListNode* reverse_lls(ListNode* head) {
  ListNode* curr = head;
  ListNode* prev = nullptr;

  while (curr != nullptr) {
    ListNode* next_node = cur->next;
    curr->next = prev;
    prev = curr;
    curr = next_node;
  }
}
```

## Find number of subarrays that fit an exact criteria

```cpp
int find_number_subarrays(vector<int>& arr, int k){
  unordered_map<int, int> counts;
  counts[0] = 1;
  int ans = 0, curr = 0;

  for (int num: arr) {
    // do logic to change curr
    ans += counts[curr - k];
    counts[curr]++;
  }

  return ans;
}
```

## Monotonic increasing stack

```cpp
int monotonic_increasing_stack(vector<int>& arr) {
  stack<integer> stack;
  int ans = 0;

  for (int num: arr) {
    // for monotonic decreasing, just flip the > to <

    while (!stack.empty() && stack.top() > num) {
      // do logic

      stack.pop();
    }

    stack.push(num);
  }
}

```

## Binary Tree: DFS (recursive)

```cpp
int binary_tree_dfs(TreeNode* root) {
  if (root == nullptr) {
    return 0;
  }

  int ans = 0;
  // do logic

  dfs(root.left);
  dfs(root.right);
  return ans;
}
```

## Binary tree: DFS (iterative)

```cpp
int binary_tree_dfs_iterative(TreeNode* root) {
  stack<TreeNode*> stack;
  stack.push(root);
  int ans = 0;

  while(!stack.empty()) {
    TreeNode* node = stack.pop();
    stack.pop();

    // do logic

    if (node->left != nullptr) {
      stack.push(node->left);
    }
    if (node->right != nullptr) {
      stack.push(node->right);
    }
  }

  return ans;
}
```

## Binary tree: BFS

```cpp
int bfs_tree(TreeNode* root) {
  queue<TreeNode*> queue;
  queue.push(root);
  int ans = 0;


  while(!queue.empty()) {
    int current_length = queue.size();
    // do logic for current level

    for (int i = 0; i < current_length; i++) {
      TreeNode* node = queue.front();
      queue.pop();

      // do logic
      if(node->left != nullptr) {
        queue.push(node->left);
      }
      if(node->right != nullptr) {
        queue.push(node->right);
      }
    }
  }
  return ans;
}
```

## Graph: DFS (recursive)

- For the graph templates, assume the nodes are numbered from 0 to n-1 and the graph is given as an adjacency list.
  Depending on the problem, you may need to convert the input into an equivalent adjacency list before using templates.

```cpp
unordered_set<int> seen;
int fn(vector<vector<int>>& graph) {
    seen.insert(START_NODE);
    return dfs(START_NODOE, graph);
}
int dfs(int node, vector<vector<int>>& graph) {
  int ans = 0;
  // do some logic
  for (int neighbor: graph[node]) {
    if (seen.find(neighbor) == seen.end()) {
      seen.insert(neighbor);
      ans += dfs(neighbor, graph);
    }
  }

  return ans;
}
```

## Graph: DFS (iterative)

```cpp
int fn_dfs_iterative(vector<vector<int>>& graph) {
  stack<int> stack;
  unordered_set<int> seen;
  stack.push(START_NODE);
  seen.insert(START_NODE);
  int ans=0;

  while(!stack.empty()) {
    int node = stack.top();
    stack.pop();
    // do some logic
    for (int neighbor: graph[node]) {
      if (seen.find(neighbor) == seen.end()) {
        seen.insert(neighbor);
        stack.push(neighbor);
      }
    }
  }
}
```

## Graph : BFS

```cpp
int fn_graph_bfs(vector<vector<int>>& graph) {
  queue<int> queue;
  unordered_set<int> seen;
  queue.push(START_NODE);
  seen.insert(START_NODE);
  int ans = 0;

  while (!queue.empty()) {
    int node = queue.front();
    queue.pop();
    // do some logic
    for (int neighbor: graph[node]) {
      if (seen.find(neighbor) == seen.end()) {
        seen.insert(neighbor);
        queue.push(neighbor);
      }
    }
  }
}
```

## Find top k elements with heap

```cpp
vector<int> find_top_k_element_heap(vector<int>& arr, int k){
  priority_queue<int, CRITERIA> heap;
  for (int num: arr) {
    heap.push(num);

    if(heap.size() > k) {
      heap.pop();
    }
  }

  vector<int> ans;
  while (heap.size() > 0) {
    ans.push_back(heap.top());
    heap.pop();
  }

  return ans;
}
```

## Binary Search

```cpp
int binarySearch(vector<int>& arr, int target) {
  int left = 0;
  int right = int(arr.size()) - 1;

  while (left <= right) {
    int mid = left (right - left) / 2;
    if (arr[mid] == target) {
      // do something
      return mid;
    }
    if (arr[mid] > target) {
      right = mid - 1;
    } else {
      left = mid + 1;
    }
  }

  // left is the insertion point
  return left;
}
```

## Binary search: duplicate element, left-most insertion point

```cpp
int binarySearch(vector<int>& arr, int target){
  int left = 0;
  int right = arr.size();
  while(left < right){
    int mid = left + (right - left) / 2;
    if (arr[mid] >= target) {
      right = mid;
    } else{
      left = mid + 1;
    }
  }

  return left;
}

```

## Binary search: for greedy problems, if looking for minimum

```cpp
int binary_search_minimum(vector<int>& arr) {
  int left = MINIMUM_POSSIBLE_ANSWER;
  int right = MINIMUM_POSSIBLE_ANSWER;
  while (left <= right) {
    int mid = left + (right - left) / 2;
    if (check(mid)) {
      right = mid - 1;
    } else {
      left = mid + 1;
    }
  }

  return left;
}
```

## if looking for maximum.

```cpp
int fn_max_bs(vector<int>& arr) {
  int left = MINIMUM_POSSIBLE_ANSWER;
  int right = MAXIMUM_POSSIBLE_ANSWER;
  while (left <= right) {
    int mid = left + (right - left) / 2;
    if(check(mid)){
      left = mid + 1;
    } else {
      right = mid - 1;
    }
  }
  return right;
}

bool check(int x) {
  //  this function is implemented depending on the problem
  return BOOLEAN;
}
```

## backtracking

```cpp
int backtrack(STATE curr, OTHER_ARGUMENTS...){
  if(BASE_CASE) {
    // modify the answer
    return 0;
  }

  int ans = 0;
  for (ITERATE_OVER_INPUT) {
    // modify the current state
    ans += backtrack(curr, OTHER_ARGUMENTS...);
    // undo the modification of the current state
  }
  return ans;
}
```

## Dynamic programming: top-down memoization

```cpp
unordered_map<STATE, int> memo;
int fn(vector<int>& arr) {
    return dp(STATE_FOR_WHOLE_INPUT, arr);
}

int dp(STATE, vector<int>& arr) {
    if (BASE_CASE) {
       return 0;
    }

    if (memo.find(STATE) != memo.end()) {
      return memo[STATE];
    }

    int ans = RECURRENCE_RELATION(STATE);
    memo[STATE] = ans;
    return ans;
  }
```

## Build a trie

- note: using ga class is only necessary if you want to store data at each node.
  otherwise, you can implement a trie using only hash maps.

```cpp
struct TrieNode {
  int data;
  unordered_map<char, TrieNode*> children;
  TrieNode() : data(0), children(unordered_map<char, TrieNode*>()) {}
};

TrieNode* buildTrie(vector<string> words){
    TrieNode* root = new TrieNode();
    for (string word: words) {
        TrieNode* curr = root;
        for (char c: word) {
            if (cur->children.find(c) == cur->children.end()) {
                curr->children[c] = new TrieNode();
            }
            curr = curr->children[c];
        }
        // at this point, you have a full word at curr
        // you can perform more logic here to give curr an attribute if you want
    }
    return root;
}
```

## Dikstra's algorithm

```cpp
vector<int> distances(n, INT_MAX);
distances[source] = 0;
priority_queue<pair<int, int>, vector<pair<int, int>>, greater<pair<int, int>>> heap;
heap.push({0, source})

while(!heap.empty()) {
    int currDist = heap.top().first;
    int node = heap.top().second;
    heap.pop();

    if (currDist > distances[node]) {
        continue;
    }

    for(pair<int, int> edge: graph[node]) {
        int nei = edge.first;
        int weight = edge.second;
        int dist = currDist + weight;

        if (dist < distances[nei]) {
            distances[nei] = dist;
            heap.push({dist, nei});
        }
    }
}
```

## Task Scheduler - heap / priority queueu

- return the minimum numbers of intervals required to complete all tasks.

```python
def leastInterval(self, tasks: List[str], n: int) -> int:
  c = Counter(tasks)
  maxHeap = [-cnt for cnt in c.values()]
  heapq.heapify(maxHeap)
  time = 0
  q = deque() # [-cnt, iddleTime]
  while maxHeap or q:
    time += 1
    if maxHeap:
      cnt = 1 + heapq.heappop(maxHeap)
      if cnt:
        q.append([cnt, time + n])
    # iddleTime pass add the value back to heapq
    if q and q[0][1] == time:
      heapq.heappush(maxHeap, q.popleft()[0])
  return time
```

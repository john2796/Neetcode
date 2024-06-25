// Code Templates

// Two Pointers: one input, opposite ends
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

// Two pointer: two inputs, exhausted both 
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

// Linked List: fast and slow pointer 
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

// Reversing a linked list
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


// Find number of subarrays that fit an exact criteria
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

// Monotonic increasing stack

int monotonic_increasing_stack(vector<int>& arr) {
  stack<integer> stack;
  int ans = 0;

  for (int num: arr) {
    // for monotonic decreasing, just flip the > to <

    while (!stack.empty() && stack.top() > num) {
      // do logic

      stack.pop()
    }

    stack.push(num);
  }
}

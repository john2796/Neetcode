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

// // Two pointer 
// int two_pointer_two_inputs(vector<int>&)

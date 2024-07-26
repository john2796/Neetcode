# Dynamic Programming

## Climbing stairs

![](images/1.png)

- take 1 or 2 step

### recursion dfs and cache/memoization

- decision tree dfs: 2n ![](images/2.png)
- how to avoid repeated steps?
  - **cache/memoization**: time: O(n) - n = 1 to n-1 ![](images/3.png)

### DP solution

- solution depends on subproblem
- DP bottom up
- add 2 previous number to get current value, just like fibonacci.
  - ![](images/4.png)
  - ![](images/5.png)
  - ![](images/6.png)
- no memory need,use two variables the last two, shift variable
  - ![](images/7.png)

```python
class Solution:
    def climbStairs(self, n: int) -> int:
      one, two = 1, 1
      for i in range(n - 1):
        temp = one
        one = one + two
        two = temp
      return one
```

## Min Cost Climbing Stairs

![](climb-stairs/1.png)

- climb one or two steps
  - ![](climb-stairs/2.png)
    - jump from 10+15 = 25
    - jump from 15+0= 15

### Brute force: 2^n with cache O(n)

- start from index: 0
  - ![](climb-stairs/3.png)
- start from index: 1
  - ![](climb-stairs/4.png)

## DP solution

- use two single variables
  - ![](climb-stairs/5.png)
  - ![](climb-stairs/6.png)
  - ![](climb-stairs/7.png)
  - return the minimum of first 2 values

<details>
<summary>Solution</summary>
<br>
<!-- We need a space between the <br> and the content -->

![](climb-stairs/8.png)

</details>

## House Robber

![](house-robber/1.png)

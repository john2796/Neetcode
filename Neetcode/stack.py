# https://leetcode.com/problems/valid-parentheses/
class Solution:
    def isValid(self, s: str) -> bool:
        map = {")": "(", "]": "[", "}": "{"}

        stack = []

        for c in s:
            if c in map:
                if stack and stack[-1] == map[c]:
                    stack.pop()
                else:
                    return False
            else:
                stack.append(c)

        return True if not stack else False


# https://leetcode.com/problems/min-stack/
class MinStack:

    def __init__(self):
        self.s = []
        self.min_stack = []

    def push(self, val: int) -> None:
        self.s.append(val)
        min_val = min(val, self.min_stack[-1] if self.min_stack else val)
        self.min_stack.append(min_val)

    def pop(self) -> None:
        self.s.pop()
        self.min_stack.pop()

    def top(self) -> int:
        return self.s[-1]

    def getMin(self) -> int:
        return self.min_stack[-1]


# Your MinStack object will be instantiated and called as such:
# obj = MinStack()
# obj.push(val)
# obj.pop()
# param_3 = obj.top()
# param_4 = obj.getMin()


#  https://leetcode.com/problems/evaluate-reverse-polish-notation/
class Solution:
    def evalRPN(self, tokens: List[str]) -> int:
        s = []

        for t in tokens:
            if t == "+":
                v1, v2 = s.pop(), s.pop()
                s.append(v1 + v2)
            elif t == "-":
                v1, v2 = s.pop(), s.pop()
                s.append(v2 - v1)
            elif t == "*":
                v1, v2 = s.pop(), s.pop()
                s.append(int(v2) * int(v1))
            elif t == "/":
                v1, v2 = s.pop(), s.pop()
                s.append(int(float(v2) / v1))
            else:
                s.append(int(t))

        return s[0]

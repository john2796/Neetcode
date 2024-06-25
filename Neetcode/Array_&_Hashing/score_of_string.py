# https://leetcode.com/problems/score-of-a-string/description/?envType=daily-question&envId=2024-06-01

class Solution:
    def scoreOfString(self, s: str) -> int:
        # convert the character to ascii
        # subtract i - i + 1 
        # add all the values after ascii subtracted to neighbor

        total = 0
        
        for i in range(len(s) - 1):
            total += abs(ord(s[i]) - ord(s[i + 1]))
        return total

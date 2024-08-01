
# https://leetcode.com/problems/number-of-senior-citizens/?envType=daily-question&envId=2024-08-01
class Solution:
    def countSeniors(self, details: List[str]) -> int:
        # 11
        # 12 and 13
        res = 0
        for detail in details:
            age = detail[11:13]
            if int(age) > 60:
                res += 1
        return res
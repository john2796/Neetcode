# Approach
# The approach used in the given code is to check if the first element of the first edge is present in the second edge. If it is, then the first element of the first edge is the center of the star graph. If not, then the second element of the first edge is the center of the star graph.
class Solution:
    def findCenter(self, e: List[List[int]]) -> int:
        return e[0][0] if e[0][0] in e[1] else e[0][1]

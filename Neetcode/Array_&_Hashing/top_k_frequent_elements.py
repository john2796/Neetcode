# https://leetcode.com/problems/top-k-frequent-elements/description/


class Solution:
    # store count frequency in arr , loop in reverse to get most frequent when res len equal to k return values
    def topKFrequent(self, nums: List[int], k: int) -> List[int]:
        count = {}
        freq = [[] for i in range(len(nums) + 1)]

        for n in nums:
            count[n] = 1 + count.get(n, 0)
        for n, c in count.items():
            freq[c].append(n)
        # print(count, freq) #{1: 3, 2: 2, 3: 1} [[], [3], [2], [1], [], [], []]
        res = []
        for i in range(len(freq) - 1, 0, -1):
            # print(freq[i]) []
            # []
            # []
            # [1]
            # [2]
            # [3]
            for n in freq[i]:
                res.append(n)
                if len(res) == k:
                    return res

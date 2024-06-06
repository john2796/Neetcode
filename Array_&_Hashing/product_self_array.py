


def productExceptSelf(nums):
    res = [1] * (len(nums))

    for i in range(1, len(nums)):
        res[i] = nums[i - 1] * res[i - 1]

    # print(res) #[1,1,2,6]
    postfix = 1
    for i in range(len(nums) - 1, -1, -1):
        res[i] *= postfix
        postfix *= nums[i]
    return res


print(productExceptSelf([1,2,3,4]))

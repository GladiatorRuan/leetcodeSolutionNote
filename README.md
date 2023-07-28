# leetcodeSolutionNote
My personal leetCode solutions

## 一、动态规划
在动态规划中,最重要的是状态转移方程和初始化条件.这部分会针对性刷题
### 198. [打家劫舍](https://leetcode.cn/problems/house-robber/)
这道题的解题思路比较清晰,状态转移方程如下,假设DP数组代表当前能打劫到的最大收益,DP[i]为在第i间房子的时候最大的打劫收益,则状态转移方程如下:

$DP[i] = max(DP[i-1],DP[i-2] + nums[i])$

其中,nums为数组.解释一下这个解法的含义是.当你开始打劫第i间房间的时候,有2个选择:
1. 打劫,这个时候不能打劫前一间房子,因此受益是dp[i-2] + nums[i]
2. 选择不打劫, 那么就是不变DP[i-1]

所以能打取最优解就是上述公式
## 二、 图论算法
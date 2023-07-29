# leetcodeSolutionNote
My personal leetCode solutions

## 一、动态规划
在动态规划中,最重要的是状态转移方程和初始化条件.在动态规划中,初始化扮演者很重要的角色,即使得到了状态方程但是初始化错误,依旧得不到正确的结果.这部分会针对性刷题
### 198. [打家劫舍](https://leetcode.cn/problems/house-robber/)
这道题的解题思路比较清晰,状态转移方程如下,假设DP数组代表当前能打劫到的最大收益,DP[i]为在第i间房子的时候最大的打劫收益,则状态转移方程如下:

$DP[i] = max(DP[i-1],DP[i-2] + nums[i])$

其中,nums为数组.解释一下这个解法的含义是.当你开始打劫第i间房间的时候,有2个选择:
1. 打劫,这个时候不能打劫前一间房子,因此受益是dp[i-2] + nums[i]
2. 选择不打劫, 那么就是不变DP[i-1]

所以能得出最优解就是上述公式.现在考虑初始化问题. 由于我们需要依赖i-2,因此需要初始化前面两个值.即dp[0]和dp[1].当在第一个元素的时候,能获得的最大值就是nums[0], 当到第二个元素的时候, 由于不能够相邻的房间进行打劫,所以dp[1]只能选择一个偷,也就是获取最大利益的. 

$dp[1] = max(\,nums[0],\,nums[1])$

最终的实现代码:
```java
public int rob(int[] nums) {
        int lenOfNums = nums.length;
        if (lenOfNums == 1)
            return nums[0];
        if (lenOfNums == 2)
            return nums[0] > nums[1] ? nums[0] : nums[1];
        int[] dp = new int[lenOfNums];
        dp[0] = nums[0];
        dp[1] = nums[0] > nums[1] ? nums[0] : nums[1];
        for (int i = 2; i < nums.length; i++){
            dp[i] = Math.max(dp[i-1], dp[i-2] + nums[i]);
        }
        return dp[lenOfNums -1];
    }
```
同时这部分还是有一些优化空间,由于当前状态仅仅与一步之前的状态和2步之前的状态相关,所以可以使用2个固定的数滚动计算
### [213. 打家劫舍 II](https://leetcode.cn/problems/house-robber-ii/)
与前一题类似,但是由于用数组模拟环形链表.由于是环形的,所以选择了首就不能选择尾.因此第二部分其实相当于打家劫舍I中解题,只不过取值的范围不再是整个链表而是分2段取值,避开了首尾相接.

```java
public int rob(int[] nums) {
        if (nums.length == 1)
            return nums[0];
        else if (nums.length == 2)
            return Math.max(nums[0],nums[1]);
        else
            return Math.max(circleRob(0,nums.length-2,nums),circleRob(1,nums.length - 1,nums));
    }

    public int circleRob(int start, int end, int[] nums){
        int tmp;
        int pre = nums[start];
        int cur = nums[start] > nums[start + 1] ? nums[start] : nums[start+1];
        
        for (int i = start + 2; i <= end; i++){
            tmp = cur;
            cur = Math.max(pre + nums[i], cur);
            pre = tmp;
        }
        return cur;
    }
```
那么可以扩展一下,当相邻2个不能打劫,相邻k个不能打劫呢?




## 二、 图论算法
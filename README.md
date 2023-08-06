# leetcodeSolutionNote
这部分是个人leetcode题解,为了避免忘记解题策略,以及用于复习.题目来源于[leetcode](https://leetcode.cn)

## 一、数学
一些常见的数学题目,比如两数之和

### [136. 只出现一次的数字](https://leetcode.cn/problems/single-number/)
常规做法就是遍历以此,筛选出只是出现一次的元素即可,可以使用一个Map数据结构遍历一次获取元素的统计,再遍历一次筛选出出现频次就是1次的元素
```java
public int singleNumber(int[] nums) {
        Map<Integer,Integer> hs = new HashMap<>();
        for(int num : nums){
            hs.put(num, hs.getOrDefault(num, 0)+1);
        }

        for(int key:hs.keySet()){
            if(hs.get(key) == 1)
                return key;
        }
        return 0;
    }
```
但是这道题更期待的做法就是使用异或运算,由于a ^ a ^ b = b,a ^ a = 0,利用这个特性,可以找出这个元素
```java
public int singleNumber(int[] nums) {
        int result = nums[0];
        if(nums.length == 1)
            return result;

        for(int i = 1; i< nums.length; i++){
            result = result ^ nums[i];
        }
        return result;
    }
```

## 二、双指针

### [26. 删除有序数组中的重复项](https://leetcode.cn/problems/remove-duplicates-from-sorted-array/)
最常用的做法就是双指针,也就是一个指向当前位置,一个指向数组移动的位置
```java
public int removeDuplicates(int[] nums) {
        int N = 0,len = nums.length;
        for(int i = 1;i < len; i++){
            if(nums[i] != nums[N]){
                N++;
                nums[N] = nums[i];
            }
        }
        return N + 1;
    }
```

### [80. 删除有序数组中的重复项 II](https://leetcode.cn/problems/remove-duplicates-from-sorted-array-ii/)
与[26. 删除有序数组中的重复项](https://leetcode.cn/problems/remove-duplicates-from-sorted-array/)很相似,只不过现在最大的重复度是2,也就是允许至多2个元素重复. 方法其实和1类似,只不过由于重复度限制2,所以新增一个常数值来标记是否超过目前的重复度,因此双指针的情况相比之前就多了一种情况:
1. 如前,如果慢指针对应的元素与快指针的元素相等,则快慢指针均移动一个位置,同时对统计参数tmp进行复位操作,也就是使tmp =1
2. 如果慢指针与快指针元素重复,则需要判断重复度是否大于2,当有一个元素的时候,只允许再新增一个元素,因此统计的参数tmp <=1的时候可以继续使慢指针下一位等于快指针.
最终的代码如下:

```java
public int removeDuplicatesII(int[] nums) {
        int N = 0;
        int index = 1, tmp =1;
        while(index < nums.length){
            if (nums[N] != nums[index]){
                nums[N +1] = nums[index];
                N++;
                tmp = 1;
            }
            else if (nums[N] == nums[index] && tmp <=1) {
                nums[N + 1] = nums[index];
                N++;
                tmp++;
            }

            index++;

        }
        return N + 1;
    }

```
### [83. 删除排序链表中的重复元素](https://leetcode.cn/problems/remove-duplicates-from-sorted-list/)
这道题不知道为什么会在中难度的类似题之后,其实判断很简单,和数组的判断类似,如果慢指针的节点值与快指针的节点值相同,则将慢指针的next指针指向快指针的下一个节点,即删除重复元素,否则,挪动快慢指针
```java
public ListNode deleteDuplicates(ListNode head) {
        if (head == null)
            return null;
        ListNode slow = head;
        ListNode fast = head.next;
        while(fast != null){
            if (slow.val == fast.val){
                slow.next = fast.next;
            }
            else 
                slow = slow.next;
            fast = fast.next;
        }
        return head;
    }
```

### [82. 删除排序链表中的重复元素 II](https://leetcode.cn/problems/remove-duplicates-from-sorted-list-ii/)
这道题主要是删除有序链表的重复元素,与前一题不一样的地方在于,之前是保留了重复元素,只是让元素不重复,而这里则是直接删除重复的元素,举例子来说就是
[1,2,2,3,3,4] -> [1,4], 而不是[1,2,3,4]
考虑到该链表是有序链表,意味着链表是从小到大的or从大到小,相同值的Node肯定相连,因此加上一个标志值,即连续的节点的值,来找到最后一个节点,使得重复部分前一个,能够指向不重复节点的后一个.这个时候和去重复元素I类似,需要2个指针,快指针用于寻找重复节点,当确定不重复的时候,才移动慢节点
```java
public ListNode deleteDuplicates(ListNode head) {
       if (head ==null)
            return null;
        ListNode dummyHead = new ListNode(0);
        dummyHead.next = head;
        ListNode fast = dummyHead.next;
        ListNode slow = dummyHead;
        while(fast != null && fast.next != null){
            if (fast.val == fast.next.val ) {
                int x = fast.val;
                while (fast.next != null &&  fast.next.val == x) {
                    fast = fast.next;
                }
                slow.next = fast.next;
            }
            else{
                slow = slow.next;
            }
            fast = fast.next;
        }
        return dummyHead.next;
    }
```



### [167. 两数之和 II - 输入有序数组](https://leetcode.cn/problems/two-sum-ii-input-array-is-sorted/description/)
两数之和是leetcode的开门题目,常用的暴力法就可以解决,另外就是哈希表的方法,为了表示比较,列举两种办法

[1. 两数之和](https://leetcode.cn/problems/two-sum/)
解答如下
```java
public int[] twoSum(int[] nums, int target) {
        Map<Integer, Integer> map = new HashMap<>();
        for (int i = 0; i < nums.length; i++) {
            if (map.containsKey(target - nums[i]))
                return new int[]{i, map.get(target - nums[i])};
            map.put(nums[i], i);
        }
        return new int[]{-1, -1};
    }
```
现在回到这一道题,和之前不同的地方在于,这里输入数组是有序数组,但是要求了必须使用常量级额外的空间. 由于是有序数组,其实很容易想到使用双指针,其实这里面有一点贪心算法的思路,也就是当有序数组两个元素A + B  > target的时候,需要减少右端的值,当值 A+ B < target的时候需要右移动A,即比较小的值

```Java
public int[] twoSum(int[] numbers, int target) {
        int len = numbers.length;
        int low = 0, high = len -1;
        while (low <= high){
            if (numbers[low] + numbers[high] == target)
                return new int[]{low+1, high +1};
            else if (numbers[low] + numbers[high] < target)
                low++;
            else
                high--;
        }
        
        return new int[]{-1,-1};
    }
```
## 三、动态规划

在动态规划中,最重要的是状态转移方程和初始化条件.在动态规划中,初始化扮演者很重要的角色,即使得到了状态方程但是初始化错误,依旧得不到正确的结果.这部分会针对性刷题
### [121. 买卖股票的最佳时机](https://leetcode.cn/problems/best-time-to-buy-and-sell-stock/)

在本道题中,由于只能进行一次买卖,且时间具有一维的方向性,因此其实是一个一维的动态规划.定义dp数组,其中dp[i]代表在i步的时候获取的最大利润,则状态方程如下
$$
\,leftMin = Min(price[i], leftMin)\\
\,dp[i] = Max(dp[i],price[i] - leftMin)
$$
考虑初始化问题,当在第一步的时候,不做出买卖的利润最大,因此dp[0] = 0.因此代码为
```java
public int maxProfit(int[] prices) {
        int len = prices.length;
        int[] dp = new int[len];

        dp[0] = 0;
        int leftMin = prices[0];
        for(int i = 1; i < len; i++){
            leftMin = Math.min(leftMin, prices[i]);
            dp[i] = Math.max(dp[i-1],prices[i] - leftMin);
        }

        return dp[len-1];
    }

```
考虑到状态仅仅与前一个状态有关,因此只需要维持一个变量来表示这个,减少空间的使用
```java
public int maxProfit(int[] prices) {
        int len = prices.length;

        int maxProfit = 0;
        int leftMin = prices[0];
        for(int i = 1; i < len; i++){
            leftMin = Math.min(leftMin, prices[i]);
            maxProfit = Math.max(maxProfit,prices[i] - leftMin);
        }

        return maxProfit;
    }

````

### [122. 买卖股票的最佳时机 II](https://leetcode.cn/problems/best-time-to-buy-and-sell-stock-ii/)
相比之前的股票买卖I,由于可以在当天多次买卖,且限制至多持有一支股票,因此这道题可以使用贪心算法,也就是说只要有利差,就卖掉,否则不进行买卖操作,
$$
\, neighbor_profit = max(price[i]- price[i-1],0)\\
\, total_profit = total_profit + neighbor_profit
$$
考虑到我们现在要用动态规划解答,首先需要研究一下状态转移,手上持有股票状态是0和1,也就是要么持有一支,要么不持有:
1. 今天持有股票的累计最大收益,来源有2种可能,一是昨天持有股票,今天不交易,二是昨天没有持有股票,今天买入一支
2. 今天不持有股票的最大收益,来源也是2种可能,昨天不持有股票,今天不交易,昨天持有股票,今天卖出.

最终最大的收益肯定是手上不持有股票.因此,最开始的方案就是使用二维数组进行计算,第一维表示在第i步,第二维只有0和1这个值表示是否持有股票
```java
class Solution {
    public int maxProfit(int[] prices) {
        //二维动态方程解法
       int priceLength = prices.length;
       int[][] dp = new int[priceLength][2];
       //初始化
        dp[0][0] = 0;
        dp[0][1] = -prices[0];
        for (int i = 1; i < priceLength;i++){
            dp[i][0] = Math.max(dp[i-1][0],dp[i-1][1] + prices[i]);
            dp[i][1] = Math.max(dp[i-1][1],dp[i-1][0] - prices[i]);
        }
        return dp[priceLength-1][0];

    }
}
```
和其他动态规划的优化类似,由于最新的状态知与昨天有关,同时有0,1这2个状态,因此只需要4个变量来替代,2个代表昨天的持有与不持有状态,2个代表今天持有与不持有状态,当执行完昨天的动态之后,今天的动态就成为昨天,以此有如下当成
$$
\begin{aligned}
new0 &= max(old0,old1 + prices[i])\\
new1 &= max(old1,old0 - prices[i]) \\
old0 &= new0\\
old1 &= new1
\end{aligned}
$$
优化后的代码如下
```java
public int maxProfit(int[] prices) {
        int priceLength = prices.length;
       
       //初始化
        int new0 = 0, old0 = 0;
        int new1 =  -prices[0] , old1 = -prices[0];
        // int new0, new1;
        for (int i = 1; i < priceLength;i++){
            new0 = Math.max(old0,old1 + prices[i]);
            new1 = Math.max(old1,old0 - prices[i]);
            old0 = new0;
            old1 = new1;
        }
        return new0;
}
```
相比来说,代码结构差不多,但是能节省一点空间.
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

### [740. 删除并获得点数](https://leetcode.cn/problems/delete-and-earn/)

这道题是打家劫舍的升级版本,一种解法是统计所有的元素去重,求和,并且对元素排序,但是这种情况下需要判断元素是否相连等.这种情况有点像打家劫舍题目的情况,当抢某一个点的时候,则不能抢相邻的点,这个时候,如果我们把元素的值作为对应的位置,同时把该元素相同的的加和,作为该点的值,即转换成打家劫舍的问题了198. 打家劫舍


```java 
class Solution {
    public int deleteAndEarn(int[] nums) {
        int maxValue = 0;
        int lenOfnums = nums.length;
        if (lenOfnums == 1)
            return nums[0];

        //找到最大的值，取一个连续的情况，然后仿照打家劫舍的方式处理
        for (int i = 0; i < lenOfnums; i++){
            maxValue = Math.max(maxValue,nums[i]);
        }

        int[] AusNums = new int[maxValue + 1];
        AusNums[0] = 0;


        Arrays.fill(AusNums,0);
        for (int element : nums){
            AusNums[element] += element;
        }

        return rob(AusNums);
        
    }

    public int rob(int[] nums) {
        if(nums.length == 0)
            return 0;
        int[] dp = new int[nums.length];
        dp[0] = nums[0];
        if(nums.length < 2)
            return nums[0];
        dp[1] = (nums[0]>nums[1]) ? nums[0]:nums[1];
        for(int i=2;i<nums.length;i++)
            dp[i] = ((nums[i] + dp[i-2]) > dp[i-1] ? (nums[i]+dp[i-2]) : dp[i-1]);
        return dp[nums.length - 1];
    }
}

```

### [304. 二维区域和检索 - 矩阵不可变](https://leetcode.cn/problems/range-sum-query-2d-immutable/description/)
   在leetcode有一些题利用的是前缀和方法，很多时候都用的是图形法解，一开始还是有点费解的，特别是对于我这样不太擅长图形转化成公式的人来说。本文用的单纯的数学推导得到递推公式，理解的基础上直接用代码实现公式

计算公式如下:
$$
S[row1,row2][col1,col2] = \Sigma_{row1}^{row2}\Sigma_{col1}^{col2} Matix[i][j]
$$
其中$i\in[row1,row2],j\in[col1,col2]$

使用伪代码如下
```java
int sum = 0;
for i from row1 -> row2
    for j from col1 -> col2
        sum = sum + Matrix[i][j]
return sum
```
但是在上述求解过程中不难发现求解S[i][j]会反复计算Matrix[0][0],Matrix[0][1],Matrix[1][0]...求解区域越大前面的元素被求解调用的次数越多，其实不难发现就是存在重复子问题，可以考虑动态规划，借用之前图形就是
![avatar](graphs/%E5%8C%BA%E5%9F%9F%E5%92%8C%E6%B1%82%E8%A7%A3%E7%A4%BA%E6%84%8F%E5%9B%BE.png)

对应的式子是$S[D] =S[D_{total}] -S[A] - S[B] + S[C]$.
其中$S[D_{total}]$为D区域右下角坐标到原点的求和。由上式不难发现，子问题被存储在数组S中，避免了反复对Matrix的求和。
针对S[D]求解，计算公式为$S[D_x][D_y] = \Sigma_{0}^{D_x}\Sigma_{0}^{D_y} Matix[i][j]$,这个问题一样，反复求解之前数据，思路和之前一样，如果直接求解，就是遍历$[0,D_x],[0,D_y]$区间，针对求和公式分解，定义$S[k_x][k_y]$为坐标点k的，由原点到该坐标的方块区域内坐标点求和
$$
\begin{align*}  S[k_x][k_y] &= \Sigma_{0}^{k_x}\Sigma_{0}^{k_y} Matix[i][j] \\                    &= \Sigma_{0}^{k_x-1}\Sigma_{0}^{k_y-1}Matrix[i][j] +  \Sigma_{0}^{k_x-1}Matrix[i][k_y]  + \Sigma_{0}^{k_y-1}Matrix[k_x][j] + Matrix[k_x][k_y]\\ &=  \Sigma_{0}^{k_x-1}\Sigma_{0}^{k_y}Matrix[i][j] +   \Sigma_{0}^{k_x-1}\Sigma_{0}^{k_y-1}Matrix[i][j]  -\Sigma_{0}^{k_x}\Sigma_{0}^{k_y-1}Matrix[i][j] + Matrix[k_x][k_y]\\ &=S[k_x-1][k_y] + S[k_x][k_y-1] - S[k_x -1][k_y-1] + Matrix[k_x][k_y] \end{align*}
$$
因为$S[k_x-1][k_y]  =\Sigma_{0}^{k_x-1}\Sigma_{0}^{k_y}Matrix[i][j] =  \Sigma_{0}^{k_x-1}Matrix[i][k_y] +  \Sigma_{0}^{k_x-1}\Sigma_{0}^{k_y-1}Matrix[i][j]$,同理对于$S[k_x][k_y-1]$不难推导得到,而$S[k_x -1][k_y-1]  = \Sigma_{0}^{k_x-1}\Sigma_{0}^{k_y-1}Matrix[i][j]$即为重叠区域，上述递推式子即只需要每次加上$Matrix[i][j]$即可，而无需反复求和，利用空间换时间

总结起来就是两个递推式
$$
\begin{equation}   \begin{cases} 1、S[k_x][k_y] = S[k_x-1][k_y] + S[k_x][k_y-1] - S[k_x -1][k_y-1] + Matrix[k_x][k_y] \\ 2、S[D_{shadow}] =S[D_x][D_y] -S[A_x][A_y] - S[B_x][B_y] + S[C_x][C_y] \end{cases} \end{equation} 
$$

最终答案如下:
```java
class NumMatrix {
    private int[][] dp;
    public NumMatrix(int[][] matrix) {
        // int m = matrix.length;
        // int n = matrix[0].length;此处会报错,因为matrix没有matrix[0]
        // System.out.println(matrix.length);
        // System.out.println(matrix[0].length);
        if(matrix.length ==0 || matrix[0].length ==0) return;
        dp = new int[matrix.length+1][matrix[0].length+1];
        for(int i = 0;i<matrix.length;i++ )
            for(int j= 0;j<matrix[0].length;j++)
                dp[i+1][j+1] = dp[i][j+1]+dp[i+1][j] - dp[i][j] + matrix[i][j];
    }
    
    public int sumRegion(int row1, int col1, int row2, int col2) {
        return dp[row2+1][col2+1] - dp[row1][col2+1] - dp[row2+1][col1] + dp[row1][col1];
    }
}
```

## 四、贪心算法
在前一节动态规划中,我们还曾使用过贪心算法来计算,这里我们从分发饼干开始计算

### [455. 分发饼干](https://leetcode.cn/problems/assign-cookies/)
先满足最小胃口的,然后逐渐满足胃口更大的,这样才能达到最多的孩子吃饱,所以需要对饼干和胃口排序,做升序处理之后,使用2个指针来计算. 与之前的双指针不一样的地方在于,过去双指针大部分时候主要是针对一个数组或者一个线性表,而本次则是对于胃口和饼干分别使用一个指针来移动位置进行比较. 不难发现一个事实,只有当前面的小孩满足胃口之后,我们才需要继续找下一个孩子的胃口并与饼干做比较.简单来说,由于两个数组都是升序的
1. 当饼干份量不满足前一个孩子的胃口的时候,我们需要继续寻找更大份量的饼干,即饼干的指针往后移动一次
2. 当饼干满足当前小孩胃口的时候,由于饼干已经用于该小孩,我们需要对小孩 & 饼干的指针均向后移动一次
```java
    public int findContentChildren(int[] g, int[] s) {
        Arrays.sort(g);
        Arrays.sort(s);
        int child = 0,cookie = 0;
        while (child < g.length && cookie < s.length){
            if (g[child] <= s[cookie])
                ++child;
            ++cookie;
        }
        return child;
    }
```

## 回溯算法
回溯算法其实也算是一种暴力算法,只不过存在一些剪枝操作. 回溯算法一般的结构为:
```Java
void backtracking(params){
    if(终止条件){
        存放结果;
        return;
    }

    for(本层元素){
        处理节点;
        backtracking(params,一般是下一个选择);
        回溯,还原状态
    }
}
```
回溯算法里面经典的就是N皇后,全排列等问题.
首先从最简单的树图的中序遍历开始
### [94. 二叉树的中序遍历](https://leetcode.cn/problems/binary-tree-inorder-traversal/description/)
其实只需要dfs即可,判断好终止条件
```java
/**
 * Definition for a binary tree node.
 * public class TreeNode {
 *     int val;
 *     TreeNode left;
 *     TreeNode right;
 *     TreeNode() {}
 *     TreeNode(int val) { this.val = val; }
 *     TreeNode(int val, TreeNode left, TreeNode right) {
 *         this.val = val;
 *         this.left = left;
 *         this.right = right;
 *     }
 * }
 */
class Solution {
    public List<Integer> inorderTraversal(TreeNode root) {
        List<Integer> res = new ArrayList<>();
        inorderTraversal(root,res);

        return res;
    }
    
    private void inorderTraversal(TreeNode root,List<Integer> res){
        if (root == null)
            return ;
        
        inorderTraversal(root.left,res);
        res.add(root.val);
        inorderTraversal(root.right,res);
    }
}
```
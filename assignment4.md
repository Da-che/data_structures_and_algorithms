# Assignment #4: 位操作、栈、链表、堆和NN

Updated 1203 GMT+8 Mar 10, 2025

2025 spring, Complied by <mark>陈宣之 生命科学学院</mark>



> **说明：**
>
> 1. **解题与记录：**
>
>    对于每一个题目，请提供其解题思路（可选），并附上使用Python或C++编写的源代码（确保已在OpenJudge， Codeforces，LeetCode等平台上获得Accepted）。请将这些信息连同显示“Accepted”的截图一起填写到下方的作业模板中。（推荐使用Typora https://typoraio.cn 进行编辑，当然你也可以选择Word。）无论题目是否已通过，请标明每个题目大致花费的时间。
>
> 2. **提交安排：**提交时，请首先上传PDF格式的文件，并将.md或.doc格式的文件作为附件上传至右侧的“作业评论”区。确保你的Canvas账户有一个清晰可见的头像，提交的文件为PDF格式，并且“作业评论”区包含上传的.md或.doc附件。
>
> 3. **延迟提交：**如果你预计无法在截止日期前提交作业，请提前告知具体原因。这有助于我们了解情况并可能为你提供适当的延期或其他帮助。 
>
> 请按照上述指导认真准备和提交作业，以保证顺利完成课程要求。



## 1. 题目

### 136.只出现一次的数字

bit manipulation, https://leetcode.cn/problems/single-number/



<mark>请用位操作来实现，并且只使用常量额外空间。</mark>



代码：

```python
class Solution:
    def singleNumber(self, nums: List[int]) -> int:
        ans=0
        for i in nums:
            ans^=i
        return ans
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20250318000337078](C:\Users\20157\AppData\Roaming\Typora\typora-user-images\image-20250318000337078.png)



### 20140:今日化学论文

stack, http://cs101.openjudge.cn/practice/20140/



思路：



代码：

```python
literature=input()
stack=[]
for i in literature:
    if i!="]":
        stack.append(i)
    else:
        temp=[]
        while stack[-1]!="[":
            temp.append(stack.pop())
        stack.pop()
        num=[]
        while temp and temp[-1].isdigit():
            num.append(temp.pop())
        num=int("".join(num))
        temp.reverse()
        temp=''.join(temp)
        stack.append(temp*num)
print(''.join(stack))
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20250318003225365](C:\Users\20157\AppData\Roaming\Typora\typora-user-images\image-20250318003225365.png)



### 160.相交链表

linked list, https://leetcode.cn/problems/intersection-of-two-linked-lists/



思路：

n*m遍历超时

a，b同时分别走两条路后走对方路径，若有交点，则在交点处两指针步数相同，a=b=intersectionnode，若无交点，则在第二次走完时a=b=None

代码：

```python
1.（超时）
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution:
    def getIntersectionNode(self, headA: ListNode, headB: ListNode) -> Optional[ListNode]:
        a=headA
        b=headB
        ini=headB
        while a:
            while b:
                if a==b:
                    return a
                else:
                    b=b.next
            b=ini
            a=a.next
        return None
2.（Leecode答案）
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution:
    def getIntersectionNode(self, headA: ListNode, headB: ListNode) -> Optional[ListNode]:
        a=headA
        b=headB
        while a!=b:
            a=a.next if a else headB
            b=b.next if b else headA
        return a
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20250318111836474](C:\Users\20157\AppData\Roaming\Typora\typora-user-images\image-20250318111836474.png)



### 206.反转链表

linked list, https://leetcode.cn/problems/reverse-linked-list/



思路：

寒假写过的模板题

代码：

```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def reverseList(self, head: Optional[ListNode]) -> Optional[ListNode]:
        prev=None
        current=head
        while current:
            next_node=current.next
            current.next=prev
            prev=current
            current=next_node
        return prev
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20250318112001649](C:\Users\20157\AppData\Roaming\Typora\typora-user-images\image-20250318112001649.png)



### 3478.选出和最大的K个元素

heap, https://leetcode.cn/problems/choose-k-elements-with-maximum-sum/



思路：

不知道为什么一开始用了倒序，每一个i都要再扫描一遍所有j

顺序扫描，先扫描的nums1一定小于等于后扫描的，q维护包括当前位置的最大的k个nums2，val维护和，若i与刚刚扫描的位置nums1相等，则ans也相等，若不相等，则ans为上一步的val

代码：

```python
1.（超时）
class Solution:
    import heapq
    def findMaxSum(self, nums1: List[int], nums2: List[int], k: int) -> List[int]:
        answer=[0]*len(nums1)
        nums1=sorted(enumerate(nums1),key=lambda x:x[1],reverse=True)
        for i in range(len(nums1)):
            q=[0]*k
            for j in range(i+1,len(nums1)):
                if nums1[j][1]<nums1[i][1]:
                    heapq.heappush(q,-nums2[nums1[j][0]])
                else:
                    heapq.heappush(q,0)
            ans=0
            for l in range(k):
                ans+=heapq.heappop(q)*(-1)
            answer[nums1[i][0]]=ans
        return answer
2.
class Solution:
    import heapq
    def findMaxSum(self, nums1: List[int], nums2: List[int], k: int) -> List[int]:
        n=len(nums1)
        ans=[0]*n
        pack=sorted((x,y,idx) for idx,(x,y) in enumerate(zip(nums1,nums2)))
        q=[]
        val=0
        for i,(x,y,idx) in enumerate(pack):
            if i>=1 and x==pack[i-1][0]:
                ans[idx]=ans[pack[i-1][2]]
            else:
                ans[idx]=val
            val+=y
            heapq.heappush(q,y)
            if len(q)>k:
                val-=heapq.heappop(q)
        return ans
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20250318104728427](C:\Users\20157\AppData\Roaming\Typora\typora-user-images\image-20250318104728427.png)



### Q6.交互可视化neural network

https://developers.google.com/machine-learning/crash-course/neural-networks/interactive-exercises

**Your task:** configure a neural network that can separate the orange dots from the blue dots in the diagram, achieving a loss of less than 0.2 on both the training and test data.

**Instructions:**

In the interactive widget:

1. Modify the neural network hyperparameters by experimenting with some of the following config settings:
   - Add or remove hidden layers by clicking the **+** and **-** buttons to the left of the **HIDDEN LAYERS** heading in the network diagram.
   - Add or remove neurons from a hidden layer by clicking the **+** and **-** buttons above a hidden-layer column.
   - Change the learning rate by choosing a new value from the **Learning rate** drop-down above the diagram.
   - Change the activation function by choosing a new value from the **Activation** drop-down above the diagram.
2. Click the Play button above the diagram to train the neural network model using the specified parameters.
3. Observe the visualization of the model fitting the data as training progresses, as well as the **Test loss** and **Training loss** values in the **Output** section.
4. If the model does not achieve loss below 0.2 on the test and training data, click reset, and repeat steps 1–3 with a different set of configuration settings. Repeat this process until you achieve the preferred results.

给出满足约束条件的<mark>截图</mark>，并说明学习到的概念和原理。





## 2. 学习总结和收获

<mark>如果发现作业题目相对简单，有否寻找额外的练习题目，如“数算2025spring每日选做”、LeetCode、Codeforces、洛谷等网站上的题目。</mark>

stack比较熟悉，heap除了Dijstra还用不习惯，位运算，链表都是新东西，需要再花点时间熟悉，有很多小技巧现在还没法自己想出来，比如链表加双指针，heap只扫描一遍就能维护好最优（有点像dp？）。

到现在都还没在数算花很多时间，除了作业没做什么题，急急急。终于把高党结束了，希望能抽出一些时间花在数算上。










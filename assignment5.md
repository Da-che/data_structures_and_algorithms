# Assignment #5: 链表、栈、队列和归并排序

Updated 1348 GMT+8 Mar 17, 2025

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

### LC21.合并两个有序链表

linked list, https://leetcode.cn/problems/merge-two-sorted-lists/

思路：

初始化头节点与运行节点，双指针list1和list2，若均非空，值小的连接在运行节点之后，指针进入下一位，若有一指针为空，运行节点连接非空指针节点，返回头节点以后部分

代码：

```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def mergeTwoLists(self, list1: Optional[ListNode], list2: Optional[ListNode]) -> Optional[ListNode]:
        head=temp=ListNode(0)
        while list1 and list2:
            if list1.val<=list2.val:
                temp.next=list1
                list1=list1.next
            else:
                temp.next=list2
                list2=list2.next
            temp=temp.next
        if list1:
            temp.next=list1
        else:
            temp.next=list2
        return head.next
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20250325093351852](C:\Users\20157\AppData\Roaming\Typora\typora-user-images\image-20250325093351852.png)



### LC234.回文链表

linked list, https://leetcode.cn/problems/palindrome-linked-list/

<mark>请用快慢指针实现。</mark>

（寒假写的）快慢指针找到中间节点，翻转中间节点以后的链表，比较前后链表，若相等则为回文

代码：

```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def isPalindrome(self, head: Optional[ListNode]) -> bool:
        if not head or not head.next:
            return True
        else:
            fast,slow=head,head
            while fast and fast.next:
                slow=slow.next
                fast=fast.next.next
            prev=None
            while slow:
                next_node=slow.next
                slow.next=prev
                prev=slow
                slow=next_node
            lft,rgt=head,prev
            while rgt:
                if lft.val!=rgt.val:
                    return False
                rgt=rgt.next
                lft=lft.next
            return True
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20250325094125255](C:\Users\20157\AppData\Roaming\Typora\typora-user-images\image-20250325094125255.png)



### LC1472.设计浏览器历史记录

doubly-lined list, https://leetcode.cn/problems/design-browser-history/

<mark>请用双链表实现。</mark>



代码：

```python
class ListNode:
    def __init__(self,val=None,next=None,prev=None):
        self.val=val
        self.next=next
        self.prev=prev

class BrowserHistory:

    def __init__(self, homepage: str):
        self.cur=ListNode(homepage)
        

    def visit(self, url: str) -> None:
        new=ListNode(url)
        self.cur.next=new
        new.prev=self.cur
        self.cur=self.cur.next


    def back(self, steps: int) -> str:
        while steps>0 and self.cur.prev:
            self.cur=self.cur.prev
            steps-=1
        return self.cur.val


    def forward(self, steps: int) -> str:
        while steps>0 and self.cur.next:
            self.cur=self.cur.next
            steps-=1
        return self.cur.val



# Your BrowserHistory object will be instantiated and called as such:
# obj = BrowserHistory(homepage)
# obj.visit(url)
# param_2 = obj.back(steps)
# param_3 = obj.forward(steps)
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20250325102113441](C:\Users\20157\AppData\Roaming\Typora\typora-user-images\image-20250325102113441.png)



### 24591: 中序表达式转后序表达式

stack, http://cs101.openjudge.cn/practice/24591/

思路：

stack存储运算符号与括号，扫描到数字即存入ans，扫描到运算符，将栈顶所有优先级更高的运算符弹出存入ans，再将该运算符压入栈，若扫描到右括号，弹出栈顶所有运算符存入ans直到左括号，最后将剩下的数字与运算符存入ans输出

代码：

```python
symbols={"+":1,"-":1,"*":2,"/":2}
n=int(input())
for i in range(n):
    equation=input()
    stack=[]
    ans=[]
    num=""
    for j in equation:
        if j.isdigit() or j==".":
            num+=j
        else:
            if num:
                ans.append(num)
                num=""
            if j in symbols.keys():
                while stack and stack[-1] in symbols.keys() and symbols[j]<=symbols[stack[-1]]:
                    ans.append(stack.pop())
                stack.append(j)
            elif j=="(":
                stack.append(j)
            elif j==")":
                while stack and stack[-1]!="(":
                    ans.append(stack.pop())
                stack.pop()
    if num:
        ans.append(num)
    while stack:
        ans.append(stack.pop())
    print(" ".join(ans))
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20250325110051445](C:\Users\20157\AppData\Roaming\Typora\typora-user-images\image-20250325110051445.png)



### 03253: 约瑟夫问题No.2

queue, http://cs101.openjudge.cn/practice/03253/

<mark>请用队列实现。</mark>



代码：

```python
from collections import deque
def josephus(n,p,m):
    q=deque(range(n,0,-1))
    q.rotate(p-1)
    ans=[]
    while q:
        q.rotate(m-1)
        ans.append(q.pop())
    return ",".join(list(map(str,ans)))

while True:
    n,p,m=map(int,input().split())
    if n==0 and p==0 and m==0:
        break
    print(josephus(n,p,m))
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20250325112028123](C:\Users\20157\AppData\Roaming\Typora\typora-user-images\image-20250325112028123.png)



### 20018: 蚂蚁王国的越野跑

merge sort, http://cs101.openjudge.cn/practice/20018/

思路：

经典Merge sort基础上ans计算后比前大的个数

代码：

```python
n=int(input())
ants=[]
for _ in range(n):
    ants.append(int(input()))
ants.reverse()
ans=0

def merge_sort(arr):
    global ans
    if len(arr) <= 1:
        return arr
    mid = len(arr) // 2
    left = arr[:mid]
    right = arr[mid:]
    left = merge_sort(left)
    right = merge_sort(right)
    return merge(left,right)

def merge(left,right):
    global ans
    result=[]
    i=j=0
    while i<len(left) and j<len(right):
        if left[i]<=right[j]:
            result.append(left[i])
            i+=1
        else:
            result.append(right[j])
            j+=1
            ans+=len(left)-i
    result+=left[i:]
    result+=right[j:]
    return result


merge_sort(ants)
print(ans)
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20250325114123266](C:\Users\20157\AppData\Roaming\Typora\typora-user-images\image-20250325114123266.png)



## 2. 学习总结和收获

<mark>如果发现作业题目相对简单，有否寻找额外的练习题目，如“数算2025spring每日选做”、LeetCode、Codeforces、洛谷等网站上的题目。</mark>

还是要多练习，通过写链表题现在已经大致熟悉了OOP写法。感觉stack还是难，后序计算还能写，中序转后序就混乱了。Merge sort在学计概的时候勉强弄懂，现在第二次学还是比较得心应手。这学期两门实验课，实验报告写得头大，每日选做进度非常缓慢。










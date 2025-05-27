# Assignment #7: 20250402 Mock Exam

Updated 1624 GMT+8 Apr 2, 2025

2025 spring, Complied by <mark>陈宣之 生命科学学院</mark>



> **说明：**
>
> 1. **⽉考**：AC<mark>3</mark> 。考试题⽬都在“题库（包括计概、数算题目）”⾥⾯，按照数字题号能找到，可以重新提交。作业中提交⾃⼰最满意版本的代码和截图。
>
> 2. **解题与记录：**
>
>    对于每一个题目，请提供其解题思路（可选），并附上使用Python或C++编写的源代码（确保已在OpenJudge， Codeforces，LeetCode等平台上获得Accepted）。请将这些信息连同显示“Accepted”的截图一起填写到下方的作业模板中。（推荐使用Typora https://typoraio.cn 进行编辑，当然你也可以选择Word。）无论题目是否已通过，请标明每个题目大致花费的时间。
>
> 3. **提交安排：**提交时，请首先上传PDF格式的文件，并将.md或.doc格式的文件作为附件上传至右侧的“作业评论”区。确保你的Canvas账户有一个清晰可见的头像，提交的文件为PDF格式，并且“作业评论”区包含上传的.md或.doc附件。
>
> 4. **延迟提交：**如果你预计无法在截止日期前提交作业，请提前告知具体原因。这有助于我们了解情况并可能为你提供适当的延期或其他帮助。 
>
> 请按照上述指导认真准备和提交作业，以保证顺利完成课程要求。



## 1. 题目

### E05344:最后的最后

http://cs101.openjudge.cn/practice/05344/



思路：

建立双向循环链表（纯纯给自己找麻烦），kill（）实现节点删除，然后模拟

代码：

```python
class Node:
    def __init__(self, value=0):
        self.prev = self
        self.next = self
        self.value = value

    def kill(self):
        if self.next == self:
            return None
        elif self.next.next == self:
            self.next.prev,self.next.next = self.next,self.next
            return self.next
        else:
            self.prev.next, self.next.prev = self.next, self.prev
            return self.next


class Solution:
    def generate(self,n1):
        the_node = Node(1)
        for i in range(2, n1 + 1):
            node1 = Node(i)
            node1.prev,node1.next,the_node.next.prev,the_node.next = the_node, the_node.next, node1, node1
            the_node = node1
        return the_node.next

    def josephus(self,node1,k1):
        ans1 = []
        while node1.next != node1:
            for i in range(k1-1):
                node1 = node1.next
            ans1.append(node1.value)
            node1 = node1.kill()
        return ans1

if __name__ == '__main__':
    s = Solution()
    n,k = map(int,input().split())
    node = s.generate(n)
    ans = s.josephus(node,k)
    print(*ans)
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20250407161231091](C:\Users\20157\AppData\Roaming\Typora\typora-user-images\image-20250407161231091.png)



### M02774: 木材加工

binary search, http://cs101.openjudge.cn/practice/02774/



思路：

经典二分搜索

代码：

```python
n,k=map(int,input().split())
lengths=[]
for i in range(n):
    lengths.append(int(input()))
lengths.sort()
def judge(lg,k,lengths):
    cnt=0
    for i in lengths:
        cnt+=i//lg
    if cnt>=k:
        return True
    else:
        return False

lo,hi=1,lengths[-1]
while lo<=hi:
    mid=(lo+hi)//2
    if judge(mid,k,lengths):
        lo=mid+1
    else:
        hi=mid-1

print(hi)
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20250407161032805](C:\Users\20157\AppData\Roaming\Typora\typora-user-images\image-20250407161032805.png)



### M07161:森林的带度数层次序列存储

tree, http://cs101.openjudge.cn/practice/07161/



思路：

初始化多叉树，将节点先存在nodes：List[TreeNode]，度存在degrees：Dict[TreeNode,int]。开始建树：scan扫描nodes，q=deque（）存已扫描过的，popleft出正在处理的节点current，从degrees中找到度，继续扫描该个数的节点加入children并存入q，建好后返回root。然后就是后序遍历递归输出。

代码：

```python
from typing import List
from collections import deque
class TreeNode:
    def __init__(self,val):
        self.val = val
        self.children = []

class Solution:
    def generate(self,saving:List[str]):
        degrees=dict()
        nodes=[]
        for i in range(0,len(saving),2):
            node=TreeNode(saving[i])
            nodes.append(node)
            degrees[node]=int(saving[i+1])
        q=deque()
        root=nodes[0]
        q.append(root)
        scan=1
        while q:
            current=q.popleft()
            for i in range(degrees[current]):
                current.children.append(nodes[scan+i])
                q.append(current.children[-1])
            scan+=degrees[current]
        return root

    def postorder(self,node:TreeNode,result:List[str])->List[str]:
        for child in node.children:
            self.postorder(child,result)
        result.append(str(node.val))
        return result

if __name__ == '__main__':
    n=int(input())
    res=[]
    for _ in range(n):
        saving=input().split()
        root=Solution().generate(saving)
        res+=Solution().postorder(root,[])
    print(*res)
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20250407222209758](C:\Users\20157\AppData\Roaming\Typora\typora-user-images\image-20250407222209758.png)



### M18156:寻找离目标数最近的两数之和

two pointers, http://cs101.openjudge.cn/practice/18156/



思路：

首尾双指针，大了右指针左移，小了左指针右移

代码：

```python
target=int(input())
arr=list(map(int,input().split()))
arr.sort()
lft,rgt=0,len(arr)-1
ans=arr[lft]+arr[rgt]
while lft<rgt-1:
    if arr[lft]+arr[rgt]>target:
        rgt-=1
    elif arr[lft]+arr[rgt]<target:
        lft+=1
    else:
        ans=arr[lft]+arr[rgt]
        break
    if abs(arr[lft]+arr[rgt]-target)<abs(ans-target) or \
            abs(arr[lft]+arr[rgt]-target)==abs(ans-target) \
            and arr[lft]+arr[rgt]<ans:
        ans=arr[lft]+arr[rgt]
print(ans)
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20250407224135834](C:\Users\20157\AppData\Roaming\Typora\typora-user-images\image-20250407224135834.png)



### M18159:个位为 1 的质数个数

sieve, http://cs101.openjudge.cn/practice/18159/



思路：

埃氏筛打表

代码：

```python
judge=[1]*10001
for i in range(2,10001):
    for j in range(2*i,10001,i):
        judge[j]=0
for i in range(2,10001):
    if str(i)[-1]!="1":
        judge[i]=0


n=int(input())
for cnt in range(n):
    a=int(input())
    ans=[]
    for i in range(2,a):
        if judge[i]:
            ans.append(i)
    print("Case",cnt+1,":",sep="")
    if ans:
        print(*ans)
    else:
        print("NULL")
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20250407161159609](C:\Users\20157\AppData\Roaming\Typora\typora-user-images\image-20250407161159609.png)



### M28127:北大夺冠

hash table, http://cs101.openjudge.cn/practice/28127/



思路：

defaultdict加set计数，然后sort排序

代码：

```python
from collections import defaultdict
n=int(input())
AC=defaultdict(set)
update=defaultdict(int)
for i in range(n):
    team,question,result=input().split(",")
    if result=="yes":
        AC[team].add(question)
    update[team]+=1

AC_update=defaultdict(tuple)
for team in update.keys():
    AC_update[team]=(len(AC[team]),update[team])
teams=sorted(AC_update.items(),key=lambda x:x[0])
teams.sort(key=lambda x:(x[1][0],-x[1][1]),reverse=True)
if len(teams)>12:
    for i in range(12):
        print(i+1,teams[i][0],teams[i][1][0],teams[i][1][1])
else:
    for i in range(len(teams)):
        print(i+1,teams[i][0],teams[i][1][0],teams[i][1][1])
```



代码运行截图 ==（AC代码截图，至少包含有"Accepted"）==

![image-20250407161006100](C:\Users\20157\AppData\Roaming\Typora\typora-user-images\image-20250407161006100.png)



## 2. 学习总结和收获

<mark>如果发现作业题目相对简单，有否寻找额外的练习题目，如“数算2025spring每日选做”、LeetCode、Codeforces、洛谷等网站上的题目。</mark>

月考有点过于放松了，当练习写了属于是（）。

约瑟夫问题用链表死磕，而且自找麻烦写的双向循环链表，考场上有些小bug没debug出来。

加上对我来说有些超纲的多叉树，这两题可以算是我第一次独立写OOP，好消息是可以写得自认为比较规范了，坏消息是都没有AC（难蚌），花了很多时间然后先放着了。

剩下一个小时左右，木材加工是经典二分查找，数质数是埃氏筛打表，都很快就解决了。北大夺冠用的defaultdict存，set计数，因为sort的lambda写法不太能同时排序两个字典，卡了一会儿，于是先合并把数据塞到一起。两数之和后来发现其实很简单，但一看到这种只给一个数列的，看起来像是greedy的题就发怵，写了首位两个指针后迟迟不敢往下写，属于是结束后自己研究半天都不觉得是简单题（害怕）

总而言之，前半学期练习的少，只有理论没有实践，只AC3是可预见的。期中过后大概就有比较充裕的时间追每日一练了。

看群里截图，怎么感觉数算大佬比计概还多（







